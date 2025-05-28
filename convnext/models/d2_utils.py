import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from itertools import repeat
import collections.abc

import timm
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_,  DropPath

GELU_COEF = 1.702
SQRT2_OVER_2 = 0.7071067811865475244

# From timm
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    

class Affine2(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.alpha_sym = nn.Parameter(torch.ones(dim//2))
        self.alpha_asym = nn.Parameter(torch.ones(dim//2))
        self.beta = None
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim//2))

    def forward(self, xs):
        if self.beta is not None:
            return (
                self.alpha_sym * xs[0] + self.beta,
                self.alpha_asym * xs[1]
            )
        return (
            self.alpha_sym * xs[0],
            self.alpha_asym * xs[1]
        )

class Dropout2(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, xs):
        return self.dropout(xs[0]), self.dropout(xs[1])

def my_gelu(x):
    return x * torch.sigmoid(GELU_COEF * x)

class GELU2Module(nn.Module):
    def forward(self, xs):
        y1 = F.gelu(SQRT2_OVER_2*(xs[0] + xs[1]))
        y2 = F.gelu(SQRT2_OVER_2*(xs[0] - xs[1]))
        return SQRT2_OVER_2*(y1 + y2), SQRT2_OVER_2*(y1 - y2)

class Linear2(nn.Module):
    def __init__(self, input_channels, output_channels, bias=True):
        super().__init__()
        self.in_features = input_channels
        self.out_features = output_channels
        self.linear_sym = nn.Linear(
            input_channels//2, output_channels//2, bias=bias)
        self.linear_asym = nn.Linear(
            input_channels//2, output_channels//2, bias=False)
    def forward(self, xs):
        ys0 = self.linear_sym(xs[0])
        ys1 = self.linear_asym(xs[1])
        return ys0, ys1

class LayerNorm2(nn.Module):
    def __init__(self, channels, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        self.ln_sym = nn.LayerNorm(
            channels//2, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
        self.ln_asym = nn.LayerNorm(
            channels//2, eps=eps, elementwise_affine=elementwise_affine, bias=False)
    def forward(self, xs):
        return self.ln_sym(xs[0]), self.ln_asym(xs[1])
        # return SQRT2_OVER_2 * self.ln_sym(xs[0]), SQRT2_OVER_2 * self.ln_asym(xs[1])  # 34357508

class LayerNorm2v2(nn.Module):
    def __init__(self, channels, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        self.scaling = Affine2(channels, bias=bias) if elementwise_affine else nn.Identity()
        self.eps = eps
    def forward(self, xs):
        std = SQRT2_OVER_2 * torch.sqrt(
            xs[0].var(dim=-1, unbiased=False, keepdim=True)
            + xs[1].var(dim=-1, unbiased=False, keepdim=True)
            + self.eps
        )
        xs = (
            (xs[0] - xs[0].mean(dim=-1, keepdim=True)) / std,
            (xs[1] - xs[1].mean(dim=-1, keepdim=True)) / std
        )
        return self.scaling(xs)


class Mlp2(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=GELU2Module,
        norm_layer=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = Linear2

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = Dropout2(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = Dropout2(drop_probs[1])

    def forward(self, xs):
        xs = self.fc1(xs)
        xs = self.act(xs)
        xs = self.drop1(xs)
        xs = self.norm(xs)
        xs = self.fc2(xs)
        xs = self.drop2(xs)
        return xs

def drop_path2(xs,
               drop_prob: float = 0.,
               training: bool = False,
               scale_by_keep: bool = True):
    """ Modified from timm """
    if drop_prob == 0. or not training:
        return xs
    x1, x2 = xs
    keep_prob = 1 - drop_prob
    shape = (x1.shape[0],) + (1,) * (x1.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x1.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x1 * random_tensor, x2 * random_tensor

class DropPath2(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, xs):
        return drop_path2(xs, self.drop_prob, self.training, self.scale_by_keep)

class FlattenAndPermuteBCHWGrid(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        if W % 2 != 0: raise NotImplementedError()
        self.H, self.W = H, W

        # define permutation that puts symmetrically placed tokens 
        # exactly H*W//2 apart
        idx = torch.arange(H*W).reshape(H, W)
        left_idx = idx[:, :W//2].flatten()
        right_idx = idx[:, W//2:].flip(1).flatten()
        self.permutation = nn.Parameter(
            torch.cat([left_idx, right_idx]),
            requires_grad=False,
        )

    def forward(self, im):
        B, C, H, W = im.shape
        assert self.H == H, "incorrect height"
        assert self.W == W, "incorrect width"
        return im.flatten(2)[:, :, self.permutation].transpose(1, 2)

class SymOrAsymConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias,
                 sym=True,
                ):
        super().__init__()
        if bias and not sym:
            raise ValueError()
        self.kernel_size = kernel_size
        self.stride = stride
        self.sym = sym
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        if kernel_size[1] % 2 != 0:
            raise NotImplementedError("Odd kernel sizes not yet implemented")
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1] // 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.sym:
            weight = torch.cat([self.weight, self.weight.flip(-1)], dim=-1)
        else:
            weight = torch.cat([self.weight, -self.weight.flip(-1)], dim=-1)
        return nn.functional.conv2d(x, weight, self.bias, stride=self.stride) 

class PatchEmbed2(nn.Module):
    output_fmt: timm.layers.format.Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 output_fmt=None,
                 bias=True,
                 strict_img_size=True,
                 dynamic_img_pad=False,
                ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if embed_dim % 2 != 0:
            raise ValueError()

        if dynamic_img_pad:
            raise NotImplementedError()

        if output_fmt is not None:
            raise NotImplementedError()

        self.flatten = flatten
        if flatten:
            self.flatten_and_permute = FlattenAndPermuteBCHWGrid(
                self.grid_size[0], self.grid_size[1])
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj_sym = SymOrAsymConv2d(in_chans,
                                        embed_dim//2,
                                        kernel_size=self.patch_size,
                                        stride=self.patch_size,
                                        bias=bias,
                                        sym=True,
                                       )
        self.proj_asym = SymOrAsymConv2d(in_chans,
                                         embed_dim//2,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         bias=False,
                                         sym=False,
                                        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def _init_img_size(self, img_size):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                timm.layers.trace_utils._assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                )
                timm.layers.trace_utils._assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                )
        x1 = self.proj_sym(x)
        x2 = self.proj_asym(x)
        if self.flatten:
            x1 = self.flatten_and_permute(x1)
            x2 = self.flatten_and_permute(x2)
        x1, x2 = self.norm((x1, x2))
        # return x1, x2
        return SQRT2_OVER_2 * x1, SQRT2_OVER_2 * x2

class PatchLinear2(nn.Module):
    def __init__(self, num_patches_side, bias=True):
        super().__init__()
        self.linear2 = Linear2(
            num_patches_side**2, num_patches_side**2, bias=bias)
    def forward(self, xs):
        x_sym_0, x_sym_1 = torch.tensor_split(xs[0], 2, dim=1)
        x_asym_0, x_asym_1 = torch.tensor_split(xs[1], 2, dim=1)
        x_sym_0 = x_sym_0.transpose(1, 2)
        x_sym_1 = x_sym_1.transpose(1, 2)
        x_asym_0 = x_asym_0.transpose(1, 2)
        x_asym_1 = x_asym_1.transpose(1, 2)
        x_sym_sym = (x_sym_0 + x_sym_1)
        x_sym_asym = (x_sym_0 - x_sym_1)
        x_asym_sym = (x_asym_0 + x_asym_1)
        x_asym_asym = (x_asym_0 - x_asym_1)
        x_sym_sym, x_sym_asym = self.linear2((x_sym_sym, x_sym_asym))
        x_asym_sym, x_asym_asym = self.linear2((x_asym_sym, x_asym_asym))
        ys0_0 = 0.5 * (x_sym_sym + x_sym_asym)
        ys0_1 = 0.5 * (x_sym_sym - x_sym_asym)
        ys1_0 = 0.5 * (x_asym_sym + x_asym_asym)
        ys1_1 = 0.5 * (x_asym_sym - x_asym_asym)
        return (
            torch.cat([ys0_0, ys0_1], dim=2).transpose(1, 2),
            torch.cat([ys1_0, ys1_1], dim=2).transpose(1, 2),
        )

class layers_scale_mlp_blocks2(nn.Module):
    def __init__(self,
                 dim,
                 drop=0.,
                 drop_path=0.,
                 act_layer=GELU2Module,
                 init_values=1e-4,
                 num_patches=196):
        super().__init__()
        self.norm1 = Affine2(dim)
        self.attn = PatchLinear2(int(math.sqrt(num_patches)))  # transpose inside this
        self.drop_path = DropPath2(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Affine2(dim)
        self.mlp = Mlp2(in_features=dim,
                        hidden_features=int(4.0 * dim),
                        act_layer=act_layer,
                        drop=drop)
        self.gamma_1 = Affine2(dim, bias=False)
        self.gamma_1.alpha_sym.data = init_values*torch.ones_like(
            self.gamma_1.alpha_sym.data)
        self.gamma_1.alpha_asym.data = init_values*torch.ones_like(
            self.gamma_1.alpha_asym.data)
        self.gamma_2 = Affine2(dim, bias=False)
        self.gamma_2.alpha_sym.data = init_values*torch.ones_like(
            self.gamma_2.alpha_sym.data)
        self.gamma_2.alpha_asym.data = init_values*torch.ones_like(
            self.gamma_2.alpha_asym.data)

    def forward(self, xs):
        x1, x2 = self.drop_path(self.gamma_1(self.attn(self.norm1(xs))))
        xs = (xs[0] + x1, xs[1] + x2)
        x1, x2 = self.drop_path(self.gamma_2(self.mlp(self.norm2(xs))))
        return xs[0] + x1, xs[1] + x2

