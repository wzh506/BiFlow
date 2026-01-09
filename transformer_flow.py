#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
from typing import Union
import torchvision as tv
from pathlib import Path
class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
        self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample: #采样的时候才开启，存储前面历史的k,v,避免重复计算
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: Union[torch.Tensor, None] = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2) #这是learnable的位置编码
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2) #这是learnable的位置编码,感觉是过拟合呀
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches))) #存为常数变量，下三角矩阵

    def forward(self, x: torch.Tensor, y: Union[torch.Tensor, None] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x) #第一层好像是不变的，见论文2.2. Block Autoregressive Flows 中permutations的部分
        pos_embed = self.permutation(self.pos_embed, dim=0) # 为啥每一层要反向？
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1) #小于0的地方为1，其它为0
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0) #处理drop label的情况,这里对应Guidance
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)
        # self.attn_mask2=torch.tril(self.attn_mask, diagonal=-1)
        # o_x = x.clone()
        # for block in self.attn_blocks:
        #     o_x = block(o_x, self.attn_mask2) 
        # o_x = self.proj_out(o_x)  #感觉可以这样改，但是忽略了残差连接，所以还是有问题哈
        for block in self.attn_blocks:
            x = block(x, self.attn_mask) 
        x = self.proj_out(x) #这里理一下思路，这里输出xtorch.Size([4, 1024, 384])
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1) 
        # 上面这里再剖析一下，这里作用和attention_mask应该是不一样得。但是后面chunk了，移位感觉没用吧
        if self.nvp:
            xa, xb = x.chunk(2, dim=-1) #输入就是192维度的，输出变成384维度的，然后一分为二变回原来的通道1
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype) #为啥取复数，函数怎么来的？
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: Union[torch.Tensor, None] = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension （<i的逆向操作,i=0时进去的只有i=1）
        x = self.proj_in(x_in) + pos_embed[i : i + 1] #x是上一步生成的结果
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: Union[torch.Tensor, None] = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1) 
        for i in range(x.size(1) - 1): #自回归采样，只能一个个计算每个batch内的(因为要用到每一步新生成的x),(只能预测i-1个，第0个轮流坐庄)
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            if guidance > 0 and guide_what: #guidance机制，对应论文中2.6. Guidance
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0] #为x[:, i + 1]赋值(应为这次用的是x<(i+1)生成的)
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C'), T是 patch数量，C'是每个patch的维度"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size) #一点不带处理是吧
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W),这一步其实没有任何运算，只是序列位置重新排布而已"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)
    # 真实数据x被映射到一个高斯分布，           outputsize                   patch
    def forward(
        self, x: torch.Tensor, y: Union[torch.Tensor, None] = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x)
        outputs = []
        logdets = torch.zeros((), device=x.device) #这个是计算的行列式吗？
        for block in self.blocks: #除了第一层，其他每层都是Flip
            x, logdet = block(x, y)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)
        # self.var = (1 - VAR_LR) * self.var + VAR_LR * z2_detached, 目前的VAR
    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean() #最小化这个函数即可

    def reverse( #这是这类模型的特色，会附带一个reverse函数
        self,
        x: torch.Tensor,
        y: Union[torch.Tensor, None] = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        sample_dir: Path = Path("./"),
    ) -> Union[torch.Tensor, list[torch.Tensor]]:#按照原来的顺序倒着运行即可
        (sample_dir / f'temp_{guidance}').mkdir(parents=True, exist_ok=True)
        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt() # 用目前的self.var进行缩放,因为训练时有实时更新var.不过肉眼看不出来
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance) #guidance就是y label
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)
        for i, img in enumerate(seq):
            tv.utils.save_image(img, sample_dir / f'temp_{guidance}' / f'step_{i:02d}.png', normalize=True, nrow=4)

        if not return_sequence:
            return x
        else:
            return seq
# tv.utils.save_image(self.unpatchify(x), sample_dir/'temp' / f'noise.png', normalize=True, nrow=4)
# tv.utils.save_image(self.unpatchify(x), sample_dir / f'noise.png', normalize=True, nrow=4)
