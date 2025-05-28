# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from .d2_utils import (
    SQRT2_OVER_2,
    GELU2Module, Affine2, Linear2, DropPath2, PatchEmbed2, SymOrAsymConv2d,
)

class SymOrAsymDepthwiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size=7, padding=3, sym=True, bias=True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise NotImplementedError("Even kernel sizes not implemented yet")
        if sym:
            self.weight = nn.Parameter(torch.empty(
                channels, 1, kernel_size, (kernel_size // 2) + 1))
        else:
            self.weight = nn.Parameter(torch.empty(
                channels, 1, kernel_size, kernel_size // 2))
            self.zero_center = nn.Parameter(torch.zeros(
                channels, 1, kernel_size, 1), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
        else:
            self.bias = None
        self.reset_parameters()

        self.sym = sym
        self.padding = padding
        self.groups = channels  # depthwise

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
            weight = torch.cat(
                [self.weight, self.weight[..., :-1].flip(-1)], dim=-1)
        else:
            weight = torch.cat(
                [self.weight, self.zero_center, -self.weight.flip(-1)], dim=-1)
        return F.conv2d(x, weight, self.bias, padding=self.padding, groups=self.groups) 

class DepthwiseConv2d2(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError()
        self.sym_to_sym = SymOrAsymDepthwiseConv2d(
            dim//4, sym=True, bias=True)
        self.sym_to_asym = SymOrAsymDepthwiseConv2d(
            dim//4, sym=False, bias=False)
        self.asym_to_sym = SymOrAsymDepthwiseConv2d(
            dim//4, sym=False, bias=True)
        self.asym_to_asym = SymOrAsymDepthwiseConv2d(
            dim//4, sym=True, bias=False)
    def forward(self, xs):
        # TODO: rewrite so that there are 4 values in xs and don't do chunking here?
        x11, x12 = xs[0].chunk(2, dim=1)
        x21, x22 = xs[1].chunk(2, dim=1)
        return (
            torch.hstack((self.sym_to_sym(x11), self.asym_to_sym(x21))),
            torch.hstack((self.sym_to_asym(x12), self.asym_to_asym(x22)))
        )

class SymmetricDepthwiseConv2d2(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError()
        self.sym_to_sym = SymOrAsymDepthwiseConv2d(
            dim//2, sym=True, bias=True)
        self.asym_to_asym = SymOrAsymDepthwiseConv2d(
            dim//2, sym=True, bias=False)
    def forward(self, xs):
        return (
            self.sym_to_sym(xs[0]),
            self.asym_to_asym(xs[1])
        )

class SymOrAsym2x2Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, sym=True, bias=True, stride=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, 2, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.sym = sym
        self.stride = stride

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
            weight = torch.cat(
                [self.weight, self.weight], dim=-1)
        else:
            weight = torch.cat(
                [self.weight, -self.weight], dim=-1)
        return F.conv2d(x, weight, self.bias, stride=self.stride) 

class DownsampleConv2d2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError()
        if output_dim % 2 != 0:
            raise ValueError()
        self.sym_to_sym = SymOrAsym2x2Conv2d(
            input_dim//2, output_dim//2, sym=True, bias=True, stride=2)
        self.sym_to_asym = SymOrAsym2x2Conv2d(
            input_dim//2, output_dim//2, sym=False, bias=False, stride=2)
        self.asym_to_sym = SymOrAsym2x2Conv2d(
            input_dim//2, output_dim//2, sym=False, bias=False, stride=2)  # bias not needed as it is included in sym_to_sym
        self.asym_to_asym = SymOrAsym2x2Conv2d(
            input_dim//2, output_dim//2, sym=True, bias=False, stride=2)
    def forward(self, xs):
        return (
            self.sym_to_sym(xs[0]) + self.asym_to_sym(xs[1]),
            self.sym_to_asym(xs[0]) + self.asym_to_asym(xs[1])
        )

class InvariantDownsampleConv2d2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear2 = Linear2(input_dim, output_dim)
    def forward(self, xs):
        # input is assumed to be NHWC
        xs = [
            0.25*(x[:, ::2, ::2] + x[:, ::2, 1::2] + x[:, 1::2, ::2] + x[:, 1::2, 1::2])
            for x in xs
        ]
        return self.linear2(xs)

class Affine2ChannelsFirst(nn.Module):
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
                self.alpha_sym[None, :, None, None] * xs[0] + self.beta[None, :, None, None],
                self.alpha_asym[None, :, None, None] * xs[1]
            )
        return (
            self.alpha_sym[None, :, None, None] * xs[0],
            self.alpha_asym[None, :, None, None] * xs[1]
        )

class Linear2ChannelsFirst(nn.Module):
    def __init__(self, input_channels, output_channels, bias=True):
        super().__init__()
        self.in_features = input_channels
        self.out_features = output_channels
        self.linear_sym = nn.Conv2d(
            input_channels//2, output_channels//2, bias=bias, kernel_size=1)
        self.linear_asym = nn.Conv2d(
            input_channels//2, output_channels//2, bias=False, kernel_size=1)
    def forward(self, xs):
        ys0 = self.linear_sym(xs[0])
        ys1 = self.linear_asym(xs[1])
        return ys0, ys1
            
class ConvNeXtBlock2(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, channels_last_input=True):
        super().__init__()
        self.dwconv = DepthwiseConv2d2(dim, kernel_size=7, padding=3)
        self.norm = LayerNorm2LastOrFirst(dim, eps=1e-6)
        self.pwconv1 = Linear2(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = GELU2Module()
        self.pwconv2 = Linear2(4 * dim, dim)
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = (
                Affine2(dim, bias=False)
            )
            self.gamma.alpha_sym.data = layer_scale_init_value*torch.ones_like(
                self.gamma.alpha_sym.data)
            self.gamma.alpha_asym.data = layer_scale_init_value*torch.ones_like(
                self.gamma.alpha_asym.data)
        self.drop_path = DropPath2(drop_path) if drop_path > 0. else nn.Identity()
        self.channels_last_input = channels_last_input

    def forward(self, xs):
        if self.channels_last_input:
            inputs = xs
            xs = (xs[0].permute(0, 3, 1, 2), xs[1].permute(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
        else:
            inputs = (xs[0].permute(0, 2, 3, 1), xs[1].permute(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
        xs = self.dwconv(xs)
        xs = (xs[0].permute(0, 2, 3, 1), xs[1].permute(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
        xs = self.norm(xs)
        xs = self.pwconv1(xs)
        xs = self.act(xs)
        xs = self.pwconv2(xs)
        if self.gamma is not None:
            xs = self.gamma(xs)

        xs = self.drop_path(xs)
        return (inputs[0] + xs[0], inputs[1] + xs[1])

class ConvNeXtBlock2ChannelsFirst(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (1) as we find it faster in PyTorch with torch.compile
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = DepthwiseConv2d2(dim, kernel_size=7, padding=3)
        # self.dwconv = SymmetricDepthwiseConv2d2(dim, kernel_size=7, padding=3)  # This is slightly faster (but less expressive)
        self.norm = LayerNorm2LastOrFirst(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = Linear2ChannelsFirst(dim, 4 * dim)
        self.act = GELU2Module()
        self.pwconv2 = Linear2ChannelsFirst(4 * dim, dim)
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = (
                Affine2ChannelsFirst(dim, bias=False)
            )
            self.gamma.alpha_sym.data = layer_scale_init_value*torch.ones_like(
                self.gamma.alpha_sym.data)
            self.gamma.alpha_asym.data = layer_scale_init_value*torch.ones_like(
                self.gamma.alpha_asym.data)
        self.drop_path = DropPath2(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, xs):
        inputs = xs
        xs = self.dwconv(xs)
        xs = self.norm(xs)
        xs = self.pwconv1(xs)
        xs = self.act(xs)
        xs = self.pwconv2(xs)
        if self.gamma is not None:
            xs = self.gamma(xs)

        xs = self.drop_path(xs)
        return (inputs[0] + xs[0], inputs[1] + xs[1])

class ConvNeXt2(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 img_size=224,  # TODO: img_size is not really required as input here
                 use_symmetrized_asym_feats=False,
                 **kwargs,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            PatchEmbed2(
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=dims[0],
                patch_size=4,
                flatten=False
            ),
            LayerNorm2LastOrFirst(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2LastOrFirst(dims[i], eps=1e-6, data_format="channels_first"),
                DownsampleConv2d2(dims[i], dims[i+1]),
                # InvariantDownsampleConv2d2(dims[i], dims[i+1]),  # Faster, less expressive, current implementation channels last
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            # stage = nn.Sequential(
            #     *[ConvNeXtBlock2(dim=dims[i], drop_path=dp_rates[cur + j], 
            #                      layer_scale_init_value=layer_scale_init_value,
            #                      # channels_last_input=j>0,
            #                      channels_last_input=(i>0 or j>0),
            #                     ) for j in range(depths[i])]
            # )
            stage = nn.Sequential(
                *[ConvNeXtBlock2ChannelsFirst(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.use_symmetrized_asym_feats = use_symmetrized_asym_feats
        if use_symmetrized_asym_feats:
            self.final_dim = dims[-1]
        else:
            self.final_dim = dims[-1] // 2
        self.norm = nn.LayerNorm(self.final_dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(self.final_dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, SymOrAsymConv2d, SymOrAsymDepthwiseConv2d, SymOrAsym2x2Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, xs):
        for i in range(4):
            xs = self.downsample_layers[i](xs)
            xs = self.stages[i](xs)
        if self.use_symmetrized_asym_feats:
            return self.norm(torch.cat((xs[0].mean([-2, -1]), torch.abs(xs[1]).mean([-2, -1])), dim=-1))
        return self.norm(xs[0].mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        # if self.use_symmetrized_asym_feats:
        #     return self.norm(torch.cat((xs[0].mean([1, 2]), torch.abs(xs[1]).mean([1, 2])), dim=-1))
        # return self.norm(xs[0].mean([1, 2])) # global average pooling, (N, H, W, C) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm2LastOrFirst(nn.Module):
    r""" Equivariant LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(
        self,
        channels,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        data_format="channels_last",
    ):
        super().__init__()
        if bias and not elementwise_affine:
            raise NotImplementedError()
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError() 
        if data_format == "channels_last":
            self.scaling = (
                Affine2(channels, bias=bias) if elementwise_affine 
                else nn.Identity()
            )
        elif data_format == "channels_first":
            self.scaling = (
                Affine2ChannelsFirst(channels, bias=bias) if elementwise_affine 
                else nn.Identity()
            )
        self.channel_dim = -1 if data_format == "channels_last" else 1
        self.eps = eps

    def forward(self, xs):
        xs = (
            xs[0] - xs[0].mean(dim=self.channel_dim, keepdim=True),
            xs[1] - xs[1].mean(dim=self.channel_dim, keepdim=True)
        )
        s = SQRT2_OVER_2 * torch.sqrt(
            xs[0].pow(2).mean(dim=self.channel_dim, keepdim=True)
            + xs[1].pow(2).mean(dim=self.channel_dim, keepdim=True)
            + self.eps
        )
        xs = (xs[0] / s, xs[1] / s)
        return self.scaling(xs)
        # std = SQRT2_OVER_2 * torch.sqrt(
        #     xs[0].var(dim=self.channel_dim, unbiased=False, keepdim=True)
        #     + xs[1].var(dim=self.channel_dim, unbiased=False, keepdim=True)
        #     + self.eps
        # )
        # xs = (
        #     (xs[0] - xs[0].mean(dim=self.channel_dim, keepdim=True)) / std,
        #     (xs[1] - xs[1].mean(dim=self.channel_dim, keepdim=True)) / std
        # )
        # return self.scaling(xs)



@register_model
def d2_convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt2(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt2(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
