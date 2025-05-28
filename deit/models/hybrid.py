# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .vit import vit_models, Layer_scale_init_Block, Attention
from .d2_vit import Attention2, LayerNorm2v2, Layer_scale_init_Block2 
from .d2_utils import SQRT2_OVER_2, Mlp2, GELU2Module, PatchEmbed2
import torch.functional as F

class hybrid_vit_models(nn.Module):
    """ S2-equivariant in early layers Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_scale=1e-4,
                 use_attn_mixer=False,
                 use_fused_attn=False,
                 **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim//2)),
            nn.Parameter(torch.zeros(1, 1, embed_dim//2), requires_grad=False)
        ])

        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches//2, embed_dim//2)),
            nn.Parameter(torch.zeros(1, num_patches//2, embed_dim//2))
        ])

        dpr = [drop_path_rate for i in range(depth)]
        self.depth = depth
        self.blocks = nn.ModuleList([
            Layer_scale_init_Block2(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=partial(LayerNorm2v2, eps=1e-6),
                act_layer=GELU2Module,
                Attention_block=Attention2,
                Mlp_block=Mlp2,
                init_values=init_scale,
                use_attn_mixer=use_attn_mixer,
                use_fused_attn=use_fused_attn,
            )
            if i < depth // 2
            else
            Layer_scale_init_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                Attention_block=Attention,
                Mlp_block=Mlp,
                init_values=init_scale,
                use_fused_attn=use_fused_attn,
            )
            for i in range(depth)]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.feature_info = [
            dict(num_chs=embed_dim, reduction=0, module='head')
        ]

        self.final_dim = embed_dim
        self.head = (
            nn.Linear(self.final_dim, num_classes) 
            if num_classes > 0 else nn.Identity()
        )

        for p in self.pos_embed:
            trunc_normal_(p, std=.02)
        for p in self.cls_token:
            if p.requires_grad:
                trunc_normal_(p, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed.0', 'pos_embed.1', 'cls_token.0', 'cls_token.1'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.final_dim, num_classes) 
            if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x1, x2 = self.patch_embed(x)

        cls_tokens1 = self.cls_token[0].expand(B, -1, -1)
        cls_tokens2 = self.cls_token[1].expand(B, -1, -1)
        
        x1 = x1 + torch.cat((self.pos_embed[0], self.pos_embed[0]), dim=1)
        x2 = x2 + torch.cat((self.pos_embed[1], -self.pos_embed[1]), dim=1)
        
        x1 = torch.cat((cls_tokens1, x1), dim=1)
        x2 = torch.cat((cls_tokens2, x2), dim=1)
        xs = (x1, x2)
            
        for i, blk in enumerate(self.blocks[:self.depth//2]):
            xs = blk(xs)

        x = torch.cat(
            (SQRT2_OVER_2 * (xs[0] + xs[1]),
             SQRT2_OVER_2 * (xs[0] - xs[1])),
            dim=-1,
        )
        for i, blk in enumerate(self.blocks[self.depth//2:]):
            x = blk(x) 

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(
                x,
                p=float(self.dropout_rate),
                training=self.training,
            )
        x = self.head(x)
        
        return x

@register_model
def hybrid_deit_tiny_patch16_LS(pretrained=False,
                         img_size=224,
                         pretrained_21k = False,
                         **kwargs):
    model = hybrid_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    
    return model
    
    
@register_model
def hybrid_deit_small_patch16_LS(pretrained=False,
                          img_size=224,
                          pretrained_21k = False,
                          **kwargs):
    model = hybrid_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()

    return model

@register_model
def hybrid_deit_medium_patch16_LS(pretrained=False,
                                  img_size=224,
                                  pretrained_21k = False,
                                  **kwargs):
    model = hybrid_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers = Layer_scale_init_Block,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model 

@register_model
def hybrid_deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = hybrid_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def hybrid_deit_large_patch16_LS(pretrained=False,
                                 img_size=224,
                                 pretrained_21k = False,
                                 **kwargs):
    model = hybrid_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def hybrid_deit_huge_patch14_LS(pretrained=False,
                                img_size=224,
                                pretrained_21k = False,
                                **kwargs):
    model = hybrid_vit_models(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def hybrid_deit_huge_patch14_52_LS(pretrained=False,
                            img_size=224,
                            pretrained_21k = False,
                            **kwargs):
    model = hybrid_vit_models(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=52,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model

### EXTRA MODELS FOR DEBUGGING

class WrappedAttention2(Attention2):
    def forward(self, x):
        xs = x.tensor_split(2, dim=-1)
        return torch.cat(super().forward(xs), dim=-1)

class debug_ultra_hybrid_vit_models(vit_models):
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)

        xs = x.tensor_split(2, dim=-1)
            
        for i , blk in enumerate(self.blocks):
            xs = blk(xs)
            
        xs = self.norm(xs)
        return torch.cat((xs[0][:, 0], xs[1][:, 0]), dim=-1)

@register_model
def debug_hybrid_deit_tiny_patch16_LS(pretrained=False,
                                      img_size=224,
                                      pretrained_21k = False,
                                      **kwargs):
    model = vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        Attention_block=WrappedAttention2,
        **kwargs)
    
    return model

@register_model
def debug_ultra_hybrid_deit_tiny_patch16_LS(pretrained=False,
                                            img_size=224,
                                            pretrained_21k = False,
                                            **kwargs):
    model = debug_ultra_hybrid_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers=Layer_scale_init_Block2,
        Attention_block=Attention2,
        Mlp_block=Mlp2,
        act_layer=GELU2Module,
        **kwargs)
    
    return model
