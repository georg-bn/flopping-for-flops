# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .d2_convnext import (
    ConvNeXtBlock2, SymOrAsymDepthwiseConv2d, ConvNeXtBlock2ChannelsFirst
)
from .d2_utils import (
    SQRT2_OVER_2,
    GELU2Module, Affine2, Linear2, DropPath2, PatchEmbed2, SymOrAsymConv2d,
)

class ConvNeXtIsotropic2(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depth=18, dim=384, drop_path_rate=0., 
                 layer_scale_init_value=0, head_init_scale=1.,
                 img_size=224,  # TODO: img_size is not really required as input here
                 use_symmetrized_asym_feats=False,
                 **kwargs,
                 ):
        super().__init__()

        self.stem = PatchEmbed2(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=dim,
            patch_size=16,
            flatten=False,
        )
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock2ChannelsFirst(
                dim=dim,
                drop_path=dp_rates[i], 
                layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(depth)])

        self.use_symmetrized_asym_feats = use_symmetrized_asym_feats
        if use_symmetrized_asym_feats:
            self.final_dim = dim
        else:
            self.final_dim = dim // 2
        self.norm = nn.LayerNorm(self.final_dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(self.final_dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, SymOrAsymConv2d, SymOrAsymDepthwiseConv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, xs):
        xs = self.stem(xs)
        xs = self.blocks(xs)
        if self.use_symmetrized_asym_feats:
            return self.norm(torch.cat((xs[0].mean([-2, -1]), torch.abs(xs[1]).mean([-2, -1])), dim=-1))
        return self.norm(xs[0].mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def d2_convnext_isotropic_small(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic2(depth=18, dim=384, **kwargs)
    if pretrained:                                     
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_isotropic_base(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic2(depth=18, dim=768, **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_convnext_isotropic_large(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic2(depth=36, dim=1024, **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
