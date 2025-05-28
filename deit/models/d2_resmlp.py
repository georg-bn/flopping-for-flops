import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import timm
from timm.models.layers import trunc_normal_,  DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .d2_utils import (
    my_gelu,
    Affine,
    SymOrAsymConv2d,
    SQRT2_OVER_2,
    to_2tuple,
)

__all__ = [
    'resmlp_12_d2',
    'resmlp_24_d2',
    'resmlp_equi_d2_B_24',
    'resmlp_equi_d2_L_24'
]


class Affine22(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.alpha_sym = nn.Parameter(torch.ones(dim//2))
        self.alpha_asym = nn.Parameter(torch.ones(dim//2))
        self.beta = None
        if bias:
            self.beta_sym = nn.Parameter(torch.zeros(dim//2))
            self.beta_asym = nn.Parameter(torch.zeros(dim//2))

    def forward(self, xs):
        if self.beta is not None:
            return (
                self.alpha_sym * xs[0] + self.beta_sym,
                self.alpha_sym * xs[1],
                self.alpha_asym * xs[2],
                self.alpha_asym * xs[3] + self.beta_asym,
            )
        return (
            self.alpha_sym * xs[0],
            self.alpha_sym * xs[1],
            self.alpha_asym * xs[2],
            self.alpha_asym * xs[3]
        )

class Dropout22(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, xs):
        return self.dropout(xs[0]), self.dropout(xs[1]), self.dropout(xs[2]), self.dropout(xs[3])

class GELU22Module(nn.Module):
    def forward(self, xs):
        y1 = my_gelu(0.5*(xs[0] + xs[1] + xs[2] + xs[3]))
        y2 = my_gelu(0.5*(xs[0] + xs[1] - xs[2] - xs[3]))
        y3 = my_gelu(0.5*(xs[0] - xs[1] + xs[2] - xs[3]))
        y4 = my_gelu(0.5*(xs[0] - xs[1] - xs[2] + xs[3]))
        return (
            0.5 * (y1 + y2 + y3 + y4),
            0.5 * (y1 + y2 - y3 - y4),
            0.5 * (y1 - y2 + y3 - y4),
            0.5 * (y1 - y2 - y3 + y4)
        )

class LayerNorm22(nn.Module):
    def __init__(self, channels, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        self.ln_sym = nn.LayerNorm(
            channels//2, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
        self.ln_asym = nn.LayerNorm(
            channels//2, eps=eps, elementwise_affine=elementwise_affine, bias=False)
    def forward(self, xs):
        return (
            self.ln_sym(xs[0]),
            self.ln_asym(xs[1]),
            self.ln_asym(xs[2]),
            self.ln_sym(xs[3])
            # SQRT2_OVER_2 * self.ln_sym(xs[0]),
            # SQRT2_OVER_2 * self.ln_asym(xs[1]),
            # SQRT2_OVER_2 * self.ln_asym(xs[2]),
            # SQRT2_OVER_2 * self.ln_sym(xs[3])
        )


class LayerNorm22v2(nn.Module):
    def __init__(self, channels, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        self.scaling = Affine22(channels, bias=bias) if elementwise_affine else nn.Identity()
        self.eps = eps
    def forward(self, xs):
        std = 0.5 * torch.sqrt(
            xs[0].var(dim=-1, unbiased=False, keepdim=True)
            + xs[1].var(dim=-1, unbiased=False, keepdim=True)
            + xs[2].var(dim=-1, unbiased=False, keepdim=True)
            + xs[3].var(dim=-1, unbiased=False, keepdim=True)
            + self.eps
        )
        xs = (
            (xs[0] - xs[0].mean(dim=-1, keepdim=True)) / std,
            (xs[1] - xs[1].mean(dim=-1, keepdim=True)) / std,
            (xs[2] - xs[2].mean(dim=-1, keepdim=True)) / std,
            (xs[3] - xs[3].mean(dim=-1, keepdim=True)) / std
        )
        return self.scaling(xs)

class Mlp22(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=GELU22Module,
        norm_layer=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = Linear22

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = Dropout22(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = Dropout22(drop_probs[1])

    def forward(self, xs):
        xs = self.fc1(xs)
        xs = self.act(xs)
        xs = self.drop1(xs)
        xs = self.norm(xs)
        xs = self.fc2(xs)
        xs = self.drop2(xs)
        return xs

def drop_path22(xs,
                drop_prob: float = 0.,
                training: bool = False,
                scale_by_keep: bool = True):
    """ Modified from timm """
    if drop_prob == 0. or not training:
        return xs
    x1, x2, x3, x4 = xs
    keep_prob = 1 - drop_prob
    shape = (x1.shape[0],) + (1,) * (x1.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x1.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x1 * random_tensor, x2 * random_tensor, x3 * random_tensor, x4 * random_tensor

class DropPath22(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, xs):
        return drop_path22(xs, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class FlattenAndPermuteBCHWGridToFourier(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        if W % 2 != 0: raise NotImplementedError()
        self.H, self.W = H, W

        # define permutation that puts symmetrically placed tokens 
        # exactly H*W//2 apart
        idx = torch.arange(H*W).reshape(H, W)
        self.left_idx = nn.Parameter(idx[:, :W//2].flatten(), requires_grad=False)
        self.right_idx = nn.Parameter(idx[:, W//2:].flip(1).flatten(), requires_grad=False)

    def forward(self, im):
        B, C, H, W = im.shape
        assert self.H == H, "incorrect height"
        assert self.W == W, "incorrect width"
        im_BLC = im.flatten(2).transpose(1, 2)
        return (
            SQRT2_OVER_2*(im_BLC[:, self.left_idx] + im_BLC[:, self.right_idx]),
            SQRT2_OVER_2*(im_BLC[:, self.left_idx] - im_BLC[:, self.right_idx]),
        )

class Linear22(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_sym = nn.Linear(
            in_channels // 2,
            out_channels // 2,
            bias=bias,
        )
        self.lin_asym = nn.Linear(
            in_channels // 2,
            out_channels // 2,
            bias=bias,
        )

    def forward(self, xs):
        x_sym_sym, x_sym_asym, x_asym_sym, x_asym_asym = xs
        x_sym_sym = self.lin_sym(x_sym_sym)
        x_asym_asym = self.lin_asym(x_asym_asym)
        x_sym_asym = F.linear(
            x_sym_asym, self.lin_sym.weight, bias=None)
        x_asym_sym = F.linear(
            x_asym_sym, self.lin_asym.weight, bias=None)
        return x_sym_sym, x_sym_asym, x_asym_sym, x_asym_asym


class PatchLinear22(nn.Module):
    def __init__(self, num_patches_side, bias=True):
        super().__init__()
        self.lin_sym = nn.Linear(
            (num_patches_side**2) // 2,
            (num_patches_side**2) // 2,
            bias=bias,
        )
        self.lin_asym = nn.Linear(
            (num_patches_side**2) // 2,
            (num_patches_side**2) // 2,
            bias=bias,
        )

    def forward(self, xs):
        x_sym_sym, x_sym_asym, x_asym_sym, x_asym_asym = xs
        x_sym_sym = self.lin_sym(
            x_sym_sym.transpose(1, 2)).transpose(1, 2)
        x_asym_asym = self.lin_asym(
            x_asym_asym.transpose(1, 2)).transpose(1, 2)
        x_sym_asym = F.linear(
            x_sym_asym.transpose(1, 2),
            self.lin_asym.weight,
            bias=None,
        ).transpose(1, 2)
        x_asym_sym = F.linear(
            x_asym_sym.transpose(1, 2),
            self.lin_sym.weight,
            bias=None,
        ).transpose(1, 2)
        return x_sym_sym, x_sym_asym, x_asym_sym, x_asym_asym

class PatchEmbed22(nn.Module):
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

        if not flatten or output_fmt is not None:
            raise NotImplementedError()
        else:
            self.flatten = flatten
            self.flatten_and_permute_to_fourier = FlattenAndPermuteBCHWGridToFourier(
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
            x11, x12 = self.flatten_and_permute_to_fourier(x1)
            x21, x22 = self.flatten_and_permute_to_fourier(x2)
        x11, x12, x21, x22 = self.norm((x11, x12, x21, x22))
        return x11, x12, x21, x22

class layers_scale_mlp_blocks22(nn.Module):
    def __init__(self,
                 dim,
                 drop=0.,
                 drop_path=0.,
                 act_layer=GELU22Module,
                 init_values=1e-4,
                 num_patches=196):
        super().__init__()
        self.norm1 = Affine22(dim)
        self.attn = PatchLinear22(int(math.sqrt(num_patches)))  # transpose inside this
        self.drop_path = DropPath22(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Affine22(dim)
        self.mlp = Mlp22(in_features=dim,
                         hidden_features=int(4.0 * dim),
                         act_layer=act_layer,
                         drop=drop)
        self.gamma_1 = Affine22(dim, bias=False)
        self.gamma_1.alpha_sym.data = init_values*torch.ones_like(
            self.gamma_1.alpha_sym.data)
        self.gamma_1.alpha_asym.data = init_values*torch.ones_like(
            self.gamma_1.alpha_asym.data)
        self.gamma_2 = Affine22(dim, bias=False)
        self.gamma_2.alpha_sym.data = init_values*torch.ones_like(
            self.gamma_2.alpha_sym.data)
        self.gamma_2.alpha_asym.data = init_values*torch.ones_like(
            self.gamma_2.alpha_asym.data)

    def forward(self, xs):
        x1, x2, x3, x4 = self.drop_path(self.gamma_1(self.attn(self.norm1(xs))))
        xs = (xs[0] + x1, xs[1] + x2, xs[2] + x3, xs[3] + x4)
        x1, x2, x3, x4 = self.drop_path(self.gamma_2(self.mlp(self.norm2(xs))))
        return xs[0] + x1, xs[1] + x2, xs[2] + x3, xs[3] + x4


class resmlp_models_equi_d2_22rep(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 drop_rate=0.,
                 Patch_layer=PatchEmbed22,
                 act_layer=GELU22Module,
                 drop_path_rate=0.0,
                 init_scale=1e-4,
                 use_symmetrized_asym_feats=False,
                 **kwargs):
        super().__init__()



        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=int(in_chans),
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks22(dim=embed_dim,
                                      drop=drop_rate,
                                      drop_path=dpr[i],
                                      act_layer=act_layer,
                                      init_values=init_scale,
                                      num_patches=num_patches,
                                     )
            for i in range(depth)])


        self.use_symmetrized_asym_feats = use_symmetrized_asym_feats
        self.final_dim = embed_dim if use_symmetrized_asym_feats else embed_dim//2
        self.norm = Affine(self.final_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(self.final_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.final_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        for i , blk in enumerate(self.blocks):
            x  = blk(x)

        if self.use_symmetrized_asym_feats:
            x = torch.cat([
                torch.cat([x[0], x[3]], dim=1),
                torch.abs(torch.cat([x[1], x[2]], dim=1)),
            ], dim=2)
        else:
            x = torch.cat([x[0], x[3]], dim=1)  # get invariant features only
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)

        return x[:, 0]

    def forward(self, x):
        x  = self.forward_features(x)
        x = self.head(x)
        return x 

@register_model
def resmlp_equi_d2_12(pretrained=False,dist=False, **kwargs):
    model = resmlp_models_equi_d2_22rep(
        patch_size=16, embed_dim=384, depth=12,
        Patch_layer=PatchEmbed22,
        init_scale=0.1,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model
  
@register_model
def resmlp_equi_d2_24(pretrained=False,dist=False,dino=False, **kwargs):
    model = resmlp_models_equi_d2_22rep(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed22,
        init_scale=1e-5,**kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def resmlp_equi_d2_B_24(pretrained=False,dist=False, in_22k = False, **kwargs):
    model = resmlp_models_equi_d2_22rep(
        # patch_size=8,
        patch_size=16,
        embed_dim=768, depth=24,
        Patch_layer=PatchEmbed22,
        init_scale=1e-6,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth"
        elif in_22k:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth"
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    
    return model

@register_model
def resmlp_equi_d2_L_24(pretrained=False,dist=False, in_22k = False, **kwargs):
    model = resmlp_models_equi_d2_22rep(
        # patch_size=8,
        patch_size=16,
        embed_dim=1280, depth=24,
        Patch_layer=PatchEmbed22,
        init_scale=1e-6,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth"
        elif in_22k:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth"
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    
    return model


if __name__ == "__main__":
    pass
