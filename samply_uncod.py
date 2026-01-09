import os
import torch
import torchvision as tv
import transformer_flow 
import utils
import pathlib
utils.set_random_seed(0)
notebook_output_path = pathlib.Path('runs/notebook')

# specify the following parameters to match the model config
dataset = 'afhq'
num_classes = {'imagenet': 1000, 'imagenet64': 0, 'afhq': 3}[dataset]
img_size = 256
channel_size = 3

batch_size = 16
patch_size = 8
channels = 768
blocks = 8
layers_per_block = 8
noise_std = 0.07

device = 'cuda'

model_name = f'{patch_size}_{channels}_{blocks}_{layers_per_block}_{noise_std:.2f}'
ckpt_file = notebook_output_path / f'{dataset}_model_{model_name}.pth'
print(f'Loading model from {notebook_output_path}')


# we can download a pretrained model, comment this out if testing your own checkpoints
# os.system(f'wget https://ml-site.cdn-apple.com/models/tarflow/afhq256/afhq_model_8_768_8_8_0.07.pth -q -P {notebook_output_path}')

sample_dir = notebook_output_path / f'{dataset}_uncond_samples_{model_name}'
sample_dir.mkdir(exist_ok=True, parents=True)

fixed_noise = torch.randn(batch_size, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
if num_classes:
    fixed_y = torch.randint(num_classes, (batch_size,), device=device)
else:
    fixed_y = None

model = transformer_flow.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
             num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt_file))
print('checkpoint loaded!')

guided_samples = {}
with torch.no_grad(): #搞明白模型的输入和输出是什么
    for guidance in [0]: #完全不使用label，做uncondition sampling
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            samples = model.reverse(fixed_noise, None, guidance,sample_dir=sample_dir)
            guided_samples[guidance] = samples
        tv.utils.save_image(samples, sample_dir / f'samples_guidance_{guidance:.2f}.png', normalize=True, nrow=4)
        print(f'guidance {guidance} sampling complete')


# finally we denoise the samples
for p in model.parameters():
    p.requires_grad = False
    
# remember the loss is mean, whereas log prob is sum
lr = batch_size * img_size ** 2 * channel_size * noise_std ** 2
for guidance, sample in guided_samples.items():
    x = torch.clone(guided_samples[guidance]).detach()
    x.requires_grad = True
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        z, outputs, logdets = model(x, fixed_y)
    loss = model.get_loss(z, logdets)
    p_model = torch.exp(-loss) #这个值对应了p_model(x)的概率密度函数
    grad = torch.autograd.grad(loss, [x])[0] #对应论文中2.5. Score Based Denoising
    x.data = x.data - lr * grad
    samples = x
    print(f'guidance {guidance} denoising complete')
    tv.utils.save_image(samples, sample_dir / f'samples_guidance_{guidance:.2f}_denoised.png', normalize=True, nrow=4)