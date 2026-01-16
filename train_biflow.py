import argparse
import builtins
import pathlib
import os
import torch
import torch.amp
import torch.utils.data
import torchvision as tv
from tqdm import tqdm

# 假设原有的模块依然存在
import biflow
import transformer_flow
import utils     


def main(args):
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)
    
    # 获取数据
    data, num_classes = utils.get_data(args.dataset, args.img_size, args.data)

    def print_log(*args, **kwargs):
        if dist.local_rank == 0:
            builtins.print(*args, **kwargs)

    print_log(f'{" BiFlow Training Config " :-^80}')

    # -------------------------------------------------------
    # 初始化 FID 评估器
    # -------------------------------------------------------
    fid = utils.FID(reset_real_features=False, normalize=True).to('cuda')
    fid_stats_file = args.data / f'{args.dataset}_{args.img_size}_fid_stats.pth'
    if fid_stats_file.exists():
        print_log(f"Loading FID stats from {fid_stats_file}")
        fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=False))
    else:
        raise FileNotFoundError(f'FID stats file "{fid_stats_file}" not found, run prepare_fid_stats.py.')
    dist.barrier()
    
    # -------------------------------------------------------
    # 初始化 Teacher Model (根据 samply.py 的配置)
    # -------------------------------------------------------
    # 硬编码配置或从 args 获取。这里根据你的需求参考 afhq_model_8_768_8_8_0.07.pth
    teacher_config = {
        'in_channels': 3,
        'img_size': args.img_size, # 256 for AFHQ
        'patch_size': 8,
        'channels': 768,
        'num_blocks': 8,
        'layers_per_block': 8,
        'num_classes': num_classes, # AFHQ=3
        'nvp': True # 假设 Teacher 是 NVP
    }
    
    print_log(f"Initializing Teacher with config: {teacher_config}")
    teacher_model = transformer_flow.Model(**teacher_config)
    
    # 加载 Teacher 权重
    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print_log(f"Loading Teacher Checkpoint from: {args.teacher_ckpt}")
        ckpt = torch.load(args.teacher_ckpt, map_location='cpu')# 直接load，无需多言
        teacher_model.load_state_dict(ckpt)
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found at {args.teacher_ckpt}")
    
    teacher_model.to('cuda')
    
    # -------------------------------------------------------
    # 初始化 BiFlow (Student)
    # -------------------------------------------------------
    model = biflow.BiFlow(teacher_model=teacher_model).to('cuda')
    
    
    if dist.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])
    else:
        model = model
    # 打印参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"Student Model Parameters: {n_params / 1e6:.2f} M")

    # -------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------
    # 注意：只优化 requires_grad 的参数 (即 Student 部分)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler()

    # 数据加载器
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print_log(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict']) # 加载 Student 权重
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print_log(f"=> no checkpoint found at '{args.resume}'")

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train() # Teacher 是 eval 模式，Student 是 train 模式
        # 注意：BiFlow 里的 self.teacher.eval() 应该在 init 里调用，或者在这里显式调用
        if dist.distributed:
            model.module.teacher.eval()
        else:
            model.teacher.eval() 

        pbar = tqdm(loader, desc=f'Epoch {epoch}', disable=dist.rank != 0)
        
        loss_acc = 0
        recon_acc = 0
        align_acc = 0
        
        for i, (x, y) in enumerate(pbar):
            x = x.cuda()

            # if i == 100:
            #     break
            if args.dataset in ['imagenet64']:
                y = None
            else:
                y = y.cuda()

            optimizer.zero_grad()

            # BiFlow Forward + AMP
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                total_loss, _, loss_recon, loss_align = model(x, y)

            # AMP + GradScaler 反向与梯度裁剪
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Logging
            loss_acc += total_loss.item()
            recon_acc += loss_recon.item()
            align_acc += loss_align.item()

            if dist.rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss_acc/(i+1):.4f}',
                    'Recon': f'{recon_acc/(i+1):.4f}',
                    'Align': f'{align_acc/(i+1):.4f}'
                })
        scheduler.step()

        # -------------------------------------------------------
        # Validation / Sampling (1-NFE) + FID
        # -------------------------------------------------------
        if (epoch + 1) % args.sample_freq == 0:
            print_log("Sampling with 1-NFE Student for FID...")
            model.eval()
            with torch.no_grad():
                # 随机采样噪声 z，维度要匹配 Teacher 的 patchify 输出
                n_patches = (args.img_size // teacher_config['patch_size']) ** 2
                in_chans = teacher_config['in_channels'] * (teacher_config['patch_size']**2)

                # 每个进程生成一部分样本，然后在进程间拼接
                local_batch = 16 // dist.world_size if dist.world_size > 0 else 16
                z_sample = torch.randn(local_batch, n_patches, in_chans, device='cuda')
                teacher_var = model.module.teacher.var if dist.distributed else model.teacher.var
                z_sample = z_sample * teacher_var.sqrt()

                save_dir = pathlib.Path(args.logdir) / f'epoch_{epoch + 1}/student_samples'
                save_dir.mkdir(parents=True, exist_ok=True)
                save_dir_teacher = pathlib.Path(args.logdir) / f'epoch_{epoch + 1}/teacher_samples'
                save_dir_teacher.mkdir(parents=True, exist_ok=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    samples = model.module.sample_1nfe(z_sample,sample_dir = save_dir) if dist.distributed else model.sample_1nfe(z_sample,sample_dir = save_dir) #先测一下non-guidance
                    samples_teacher = model.module.teacher.reverse(z_sample, None, guidance=0,sample_dir = save_dir_teacher) if dist.distributed else model.teacher.reverse(z_sample, None, guidance=0,sample_dir = save_dir_teacher)


                # 聚合所有进程的样本，用于 FID 计算
                samples_all = dist.gather_concat(samples.detach())
                fid.update(0.5 * (samples_all.clip(min=-1, max=1) + 1), real=False)

                samples_all_t = dist.gather_concat(samples_teacher.detach())


                # 仅在 rank 0 上保存可视化样本
                if dist.rank == 0:
                    tv.utils.save_image(samples_all[:16], save_dir / 'student_samples_1nfe.png', normalize=True, nrow=4)
                    tv.utils.save_image(samples_all_t[:16], save_dir_teacher / 'teacher_samples.png', normalize=True, nrow=4)

            # 计算并打印 FID，只在 rank 0 上执行
            if dist.rank == 0:
                fid_score = fid.compute().item()
                fid.reset()
                print_log(f"Epoch {epoch+1} FID: {fid_score:.4f}")

                # 保存 Checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args,
                }, pathlib.Path(args.logdir) / 'latest.pth')
            dist.barrier()


            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiFlow Student Training')
    # 数据集参数
    parser.add_argument('--dataset', default='afhq', type=str)
    parser.add_argument('--data', default='data', type=pathlib.Path)
    parser.add_argument('--img_size', default=256, type=int)

    # 训练参数
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--sample_freq', default=500, type=int)
    parser.add_argument('--logdir', default='runs/biflow_train', type=str)
    parser.add_argument('--resume', default='True', type=str)

    # Teacher Checkpoint 路径 (必须提供)
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help='Path to the pretrained Teacher .pth file (e.g., afhq_model_8_768_8_8_0.07.pth)')

    args = parser.parse_args()

    # 自动创建输出目录
    pathlib.Path(args.logdir).mkdir(parents=True, exist_ok=True)

    main(args)