import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .d2_utils import (
    Linear2, Dropout2, Mlp2, Affine2, DropPath2, PatchEmbed2, GELU2Module,
    to_2tuple, SQRT2_OVER_2, LayerNorm2v2
)

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath


class Attention2(nn.Module):
    # modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_attn_mixer=False,
                 fused_attn=False,
                ):
        if use_attn_mixer and fused_attn:
            raise NotImplementedError("Can't use mixed attn weights with fused attn")
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads  # 3435712
        # head_dim = dim // (2*num_heads)  # 3435713
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = Linear2(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear2(dim, dim)
        self.proj_drop = Dropout2(proj_drop)

        self.use_attn_mixer = use_attn_mixer
        if use_attn_mixer:
            # self.attn_mixer = nn.Parameter(torch.ones(4))  # 3436603
            self.attn_mixer = nn.Parameter(torch.tensor([1., 0.]))

        self.fused_attn = fused_attn


    def forward(self, xs):
        x1, x2 = xs
        B, N, C = x1.shape

        qkv1, qkv2 = self.qkv(xs)
        # https://github.com/huggingface/pytorch-image-models/blob/cb4cea561a3f39bcd6a3105c72d7e0b2b928bf44/timm/models/vision_transformer.py#L92

        if self.fused_attn:
            qkv1 = qkv1.reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )
            qkv2 = qkv2.reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )
            qkv = torch.cat((qkv1, qkv2), dim=-1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x1, x2 = x.chunk(2, dim=-1)
        else:
            qkv1 = qkv1.reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
            qkv2 = qkv2.reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)

            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            
            q1 = q1 * self.scale
            q2 = q2 * self.scale

            attn1 = (q1 @ k1.transpose(-2, -1))
            attn2 = (q2 @ k2.transpose(-2, -1))
            # attn2 = torch.abs(q2 @ k2.transpose(-2, -1))

            if self.use_attn_mixer:
                attn = self.attn_mixer[0] * attn1 + self.attn_mixer[1] * attn2
            else:
                attn = attn1 + attn2
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # It would also be possible to have shared attention here:
            # attn = attn1 + attn2  #3434060
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # attn1 = attn2 = attn

            # attn1 = attn1.softmax(dim=-1)  #3434048
            # attn1 = self.attn_drop(attn1)
            # attn2 = attn2.softmax(dim=-1)
            # attn2 = self.attn_drop(attn2)

            # x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
            # x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)

            x1 = (attn @ v1)
            x2 = (attn @ v2)
        xs = (x1.transpose(1, 2).reshape(B, N, C),
              x2.transpose(1, 2).reshape(B, N, C))
        xs = self.proj(xs)
        xs = self.proj_drop(xs)
        return xs

class Layer_scale_init_Block2(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=GELU2Module,
                 norm_layer=LayerNorm2v2,
                 Attention_block=Attention2,
                 Mlp_block=Mlp2,
                 init_values=1e-4,
                 use_attn_mixer=False,
                 use_fused_attn=False,
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_attn_mixer=use_attn_mixer,
            fused_attn=use_fused_attn,
        )
        self.drop_path = DropPath2(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp_block(in_features=dim,
                             hidden_features=mlp_hidden_dim,
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

class d2_vit_models(nn.Module):
    """ D2-equivariant Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
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
                 norm_layer=LayerNorm2v2,
                 block_layers=Layer_scale_init_Block2,
                 Patch_layer=PatchEmbed2,
                 act_layer=GELU2Module,
                 Attention_block=Attention2,
                 Mlp_block=Mlp2,
                 init_scale=1e-4,
                 use_symmetrized_asym_feats=False,
                 use_attn_mixer=False,
                 use_fused_attn=False,
                 **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
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
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                Attention_block=Attention_block,
                Mlp_block=Mlp_block,
                init_values=init_scale,
                use_attn_mixer=use_attn_mixer,
                use_fused_attn=use_fused_attn,
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        self.feature_info = [
            dict(num_chs=embed_dim, reduction=0, module='head')
        ]

        self.use_symmetrized_asym_feats = use_symmetrized_asym_feats
        if use_symmetrized_asym_feats:
            self.final_dim = embed_dim
        else:
            self.final_dim = embed_dim // 2
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
            
        for i , blk in enumerate(self.blocks):
            xs = blk(xs)
            
        xs = self.norm(xs)

        if self.use_symmetrized_asym_feats:
            return torch.cat((xs[0][:, 0], torch.abs(xs[1][:, 0])), dim=-1)
            # return torch.cat((xs[0].mean(dim=1), torch.abs(xs[1]).mean(dim=1)), dim=-1)
        else:
            return xs[0][:, 0]
            # return xs[0].mean(dim=1)

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

# Model types from DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
# TODO: check that these are the correct models!

@register_model
def d2_deit_tiny_patch16_LS(pretrained=False,
                            img_size=224,
                            pretrained_21k = False,
                            **kwargs):
    model = d2_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers=Layer_scale_init_Block2,
        **kwargs)
    
    return model
    
    
@register_model
def d2_deit_small_patch16_LS(pretrained=False,
                             img_size=224,
                             pretrained_21k = False,
                             **kwargs):
    model = d2_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers=Layer_scale_init_Block2,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def d2_deit_medium_patch16_LS(pretrained=False,
                              img_size=224,
                              pretrained_21k = False,
                              **kwargs):
    model = d2_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers = Layer_scale_init_Block2,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model 

@register_model
def d2_deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = d2_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers=Layer_scale_init_Block2,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def d2_deit_large_patch16_LS(pretrained=False,
                          img_size=224,
                          pretrained_21k = False,
                          **kwargs):
    model = d2_vit_models(
        img_size = img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers=Layer_scale_init_Block2,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def d2_deit_huge_patch14_LS(pretrained=False,
                         img_size=224,
                         pretrained_21k = False,
                         **kwargs):
    model = d2_vit_models(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers = Layer_scale_init_Block2,
        **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model
    
@register_model
def d2_deit_huge_patch14_52_LS(pretrained=False,
                            img_size=224,
                            pretrained_21k = False,
                            **kwargs):
    model = d2_vit_models(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=52,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm2v2, eps=1e-6),
        block_layers = Layer_scale_init_Block2,
        **kwargs)

    return model
    

if __name__ == "__main__":
    pass
