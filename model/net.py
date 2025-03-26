import math
import warnings
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from .simple_diffusion import ResnetBlock
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from model.dformer.encoders.DFormer_2 import DFormer_Large as DFormer
from einops import rearrange, reduce


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialAttention(nn.Module):
    """
    The implementation of spatial attention mechanism.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)
        return weight_map


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x, timesteps, cond_img):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = cond_img.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, timesteps, cond_img):
        x = self.forward_features(x, timesteps, cond_img)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d
import torch.nn as nn


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class BlockFFT(nn.Module):
    def __init__(self, dim, h, w, time_embed_dim, groups=8, ):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, scale_shift=None):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Attention2(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class FFTconv(nn.Module):
    def __init__(self, num_channels):
        super(FFTconv, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.convFFT = nn.Sequential(
            nn.Conv2d(2 * num_channels, 2 * num_channels, 3, padding=1),
            nn.BatchNorm2d(2 * num_channels),
            nn.ReLU(),
            nn.Conv2d(2 * num_channels, num_channels, 3, padding=1)
        )
        self.sa = SpatialAttention(kernel_size=7)
        self.ca = ChannelAttention(64)

    def forward(self, x0):
        b, c, h, w = x0.shape
        x = x0.float()
        y = torch.fft.fft2(x)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y1 = F.relu(self.conv1(y_f))
        y1 = self.convFFT(y1)
        y = self.conv_v(y1)
        y = self.ca(y) * y
        y = self.sa(y) * y
        rgb_M = torch.sigmoid(y)
        y1 = y1 * rgb_M
        y1 = torch.fft.ifft2(y1, s=(h, w)).float()
        return x0 + y1


class CA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth, t):
        x = torch.cat((rgb, depth), dim=1)
        x = self.max_pool(x)
        x = self.fc1(x)
        max_out = self.fc2(self.relu1(x))
        out = max_out
        depth = depth.mul(self.sigmoid(out))
        return depth


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CA_SA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_SA_Enhance, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, x, c, t):
        x_d = self.self_CA_Enhance(c, x, t)
        sa = self.self_SA_Enhance(x_d, t)
        depth_enhance = x.mul(sa)
        return depth_enhance


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(reduction_dim * 5, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        # edge_features = F.interpolate(edge, x_size[2:],
        #                               mode='bilinear', align_corners=True)
        # edge_features = self.edge_conv(edge_features)
        # out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        out = self.down(out)
        return out


class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_c4_2 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim // 8)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        #
        # self.linear_d4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_d3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_d2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_d1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)
        #
        # self.linear_fuse_d = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
        #                                 norm_cfg=dict(type='BN', requires_grad=True))
        # self.linear_fuse34_d = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                                   norm_cfg=dict(type='BN', requires_grad=True))
        # self.linear_fuse2_d = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                                  norm_cfg=dict(type='BN', requires_grad=True))
        # self.linear_fuse1_d = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                                  norm_cfg=dict(type='BN', requires_grad=True))

        # self.rgbt_fuse = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                             norm_cfg=dict(type='BN', requires_grad=True))

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            resnet_block(1, embedding_dim // 8, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 4, kernel_size=7, padding=3,
                       stride=2,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # CrossAtt(out_chns=embedding_dim),
            # resnet_block(embedding_dim // 8, embedding_dim // 4, time_emb_dim=self.time_embed_dim),
            # CrossAtt(out_chns=embedding_dim),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim, kernel_size=7, padding=3, stride=2,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up_c4 = Upsample(c4_in_channels, embedding_dim // 8, factor=32)

        self.CSA = CA_SA_Enhance(64)

        # self.decouple1 = nn.Sequential(
        #     nn.GroupNorm(num_groups=min(embedding_dim * 2 // 4, 8), num_channels=embedding_dim * 2),
        #     nn.Conv2d(embedding_dim * 2, embedding_dim * 2, 3, padding=1),
        #     BlockFFT(embedding_dim * 2, 88, 88, self.time_embed_dim, 11),
        self.decouple1 = nn.Sequential(
            nn.GroupNorm(num_groups=min(embedding_dim // 4, 8), num_channels=embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            BlockFFT(embedding_dim, 88, 88, self.time_embed_dim, 11),
        )

        self.decouple2 = nn.Sequential(
            nn.GroupNorm(num_groups=min(embedding_dim // 4, 8), num_channels=embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            BlockFFT(embedding_dim, 88, 88, self.time_embed_dim, 11),
        )

        # self.mid_block1 = resnet_block(embedding_dim * 2, embedding_dim * 2, time_emb_dim=self.time_embed_dim)
        # self.mid_attn = Residual(PreNorm(embedding_dim * 2, Attention2(embedding_dim * 2, )))
        # self.mid_block2 = resnet_block(embedding_dim * 2, embedding_dim * 2, time_emb_dim=self.time_embed_dim)

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 2, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.aspp = _AtrousSpatialPyramidPoolingModule(256)

        # self.up_RGB = nn.Sequential(
        #     ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                norm_cfg=dict(type='BN', requires_grad=True)),
        #     # resnet_block(embedding_dim, embedding_dim),
        #     Upsample(embedding_dim, embedding_dim // 4, factor=2),
        #     ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
        #                norm_cfg=dict(type='BN', requires_grad=True)),
        #     Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
        #     ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
        #                norm_cfg=dict(type='BN', requires_grad=True)),
        # )

        self.pred1 = nn.Sequential(
            Upsample(embedding_dim // 2, embedding_dim // 4, factor=2),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 4, self.num_classes, kernel_size=1)
        )
        self.pred2 = nn.Sequential(
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 4, self.num_classes, kernel_size=1)
        )
        self.pred3 = nn.Sequential(
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs_1, inputs_2, timesteps, x):

        # ([1, 1, 352, 352])

        # ([1, 256, 88, 88])
        # ([1, 256, 88, 88])
        # ([1, 256, 88, 88])

        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        down_fea = []
        up_fea = []

        ############## MLP decoder on C1-C4 ###########
        f_1 = self.MLP_decoder(inputs_1)
        # f_1 = self.aspp(f_1)
        _c4 = self.up_c4(inputs_1[-1])
        ##############################################

        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                x = self.CSA(x, _c4, t)
            # elif isinstance(blk, CrossAtt):
            #     x = blk(x,inputs_1[])
            else:
                x = blk(x)
            down_fea.append(x)

        ############## MLP decoder on C1-C4 ###########
        # f_1 = self.MLP_decoder(inputs_1)
        # f_2 = self.MLP_decoder_d(inputs_1)
        # f12 = self.rgbt_fuse(torch.cat([f_1, f_2], dim=1))
        # fusion x_feat and x then transposed conv
        # RGB = torch.cat([f1, x], dim=1)
        f_1 = f_1 + self.decouple1(f_1)
        x = down_fea.pop() + self.decouple2(x)
        # f_1 = self.fft(f_1)
        # x = self.fft(x)
        x1 = torch.cat([f_1, x], dim=1)
        # x2 = torch.cat([f_2, x], dim=1)
        x = x1
        # x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        # x = self.mid_block2(x, t)
        # x = x + self.decouple1(x)
        # x = torch.cat([f_1,f_2, x], dim=1)
        # down_fea.pop()
        for blk in self.up:
            # 此处增加残差链接
            # x = torch.cat([x, down_fea.pop()], dim=1)
            if isinstance(blk, Upsample):
                x = blk(x)
                x = torch.cat([x, down_fea.pop()], dim=1)
                up_fea.append(x)
            else:
                x = blk(x)
            # up_fea.append(x)
        # [10, 256, 88, 88]
        # [10, 64, 176, 176]
        # [10, 64, 176, 176]
        # [10, 32, 352, 352]
        # [10, 32, 352, 352]
        # [10, 32, 352, 352]
        x1 = self.pred1(up_fea[0])[:, :-1, :, :]
        x2 = self.pred2(up_fea[1])[:, :-1, :, :]
        x3 = self.pred3(x)[:, :-1, :, :]
        return x1, x2, x3

    def fft(self, x, rate=0.25):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        # todo
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv

    def mlp_decode_c4(self, inputs):
        c1, c2, c3, c4 = inputs
        # 1 ([1, 96, 88, 88])
        # 2 ([1, 192, 44, 44])
        # 3 ([1, 288, 22, 22])
        # 4 ([1, 576, 11, 11])
        n, _, h, w = c4.shape
        _c4 = self.linear_c4_2(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        return _c4

    def MLP_decoder(self, inputs):

        c1, c2, c3, c4 = inputs
        # 1 ([1, 96, 88, 88])
        # 2 ([1, 192, 44, 44])
        # 3 ([1, 288, 22, 22])
        # 4 ([1, 576, 11, 11])
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # torch.Size([1, 256, 88, 88])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # torch.Size([1, 256, 88, 88])

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))

        return _c

    def MLP_decoder_d(self, inputs):

        c1, c2, c3, c4 = inputs
        # 1 ([1, 96, 88, 88])
        # 2 ([1, 192, 44, 44])
        # 3 ([1, 288, 22, 22])
        # 4 ([1, 576, 11, 11])
        n, _, h, w = c4.shape
        _c4 = self.linear_d4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_d3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_d2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # torch.Size([1, 256, 88, 88])
        _c1 = self.linear_d1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # torch.Size([1, 256, 88, 88])

        L34 = self.linear_fuse34_d(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2_d(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1_d(torch.cat([L2, _c1], dim=1))

        return _c


class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        # self.RGB_encoder = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        ck = "E:\hr\CK\OneDrive_1_2024-3-12\\DFormer_Large.pth.tar"
        self.RGBD_encoder = DFormer(drop_path_rate=0.1, norm_cfg=dict(type='BN', requires_grad=True), pretrained=ck)
        # self.RGBD_encoder.init_weights("E:\hr\CK\OneDrive_1_2024-3-12\\DFormer_Base.pth.tar")
        # self.T_encoder = pvt_v2_b4_m(in_chans=mask_chans, mask_chans=3)
        self.decode_head = Decoder(dims=[96, 192, 288, 576], dim=256, class_num=class_num, mask_chans=mask_chans)
        # self.rgbt_decoder = rgbt_decoder(dim=128)
        # self._init_weights()  # load pretrain

    def forward(self, x, timesteps, cond_img):
        if x.shape[1] == 2:
            x_cut1 = x[:, 1:, :, :]
            x_cut2 = x[:, :1, :, :]
        else:
            x_cut1 = x_cut2 = x
        T = cond_img[:, -1:, :, :].repeat(1, 3, 1, 1)
        RGB = cond_img[:, :-1, :, :]
        features_1 = self.RGBD_encoder(RGB, T, x_cut1)
        # features_1 = self.RGB_encoder(x_cut1, timesteps, cond_img[:, :-1, :, :])
        # features_x = self.T_encoder(None, timesteps, x_cut1)
        # pred = self.rgbt_decoder(features_1, None)

        # T = cond_img[:, -1:, :, :].repeat(1, 3, 1, 1)
        # features_2 = self.T_encoder(x_cut2, timesteps, T)
        # pred = self.rgbt_decoder(features_1, features_2, timesteps)
        pred = self.decode_head(features_1, None, timesteps, x_cut1)
        return pred

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',

        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load(
            r"E:\hr\CK\pvt_v2_b4_m.pth")  # for save mem
        model_dict = self.RGB_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.RGB_encoder.load_state_dict(model_dict, strict=False)

        # model_dict = self.T_encoder.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.T_encoder.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass


