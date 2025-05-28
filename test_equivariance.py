import torch
import torch.nn as nn
import math

DEVICE = 'cuda'

from deit.models.d2_resmlp import (
    PatchLinear22,
    Linear22,
    Affine22,
    GELU22Module,
    LayerNorm22,
    LayerNorm22v2,
    layers_scale_mlp_blocks22,
    PatchEmbed22,
    FlattenAndPermuteBCHWGridToFourier,
    resmlp_equi_d2_24,
)

from deit.models.d2_utils import (
    LayerNorm2v2,
    PatchEmbed2,
)

from deit.models.d2_vit import (
    Attention2,
    Layer_scale_init_Block2,
    d2_deit_medium_patch16_LS,
    d2_deit_tiny_patch16_LS,
)

from deit.models.vit_inv_early_models import (
    d2_inv_early_deit_medium_patch16_LS
)

from convnext.models.d2_convnext import (
    DepthwiseConv2d2,
    ConvNeXtBlock2,
    ConvNeXtBlock2ChannelsFirst,
    LayerNorm2LastOrFirst,
    InvariantDownsampleConv2d2,
    DownsampleConv2d2,
    d2_convnext_small,
    d2_convnext_base,
)

from convnext.models.d2_convnext_isotropic import (
    d2_convnext_isotropic_base,
)

from deit.models.vit import (
    Attention,
    Layer_scale_init_Block,
    deit_medium_patch16_LS,
    deit_tiny_patch16_LS,
)

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

def simple_invariance_test_resmlp():
    device = DEVICE

    net = resmlp_equi_d2_24(use_symmetrized_asym_feats=True).to(device)

    x = torch.randn((31, 3, 224, 224), device=device)
    x2 = torch.randn((31, 3, 224, 224), device=device)
    x_flip = x.clone().flip(-1)
    y1 = net(x)
    y2 = net(x2)
    y1_flip = net(x_flip)
    assert not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), "Bad test, these should be different"
    assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple invariance test of ResMLP architecture passed!")

def simple_equivariance_test_resmlp():
    device = DEVICE

    B = 1 if DEVICE == 'cpu' else 64
    C = 768
    side = 14
    flatten_and_permute_to_fourier = FlattenAndPermuteBCHWGridToFourier(side, side)
    patch_embed22 = PatchEmbed22(patch_size=224//side, embed_dim=C).to(device)

    patch_linear22 = PatchLinear22(side, bias=True).to(device)
    aff22 = Affine22(C, bias=True).to(device)
    gelu22 = GELU22Module()
    ln22 = LayerNorm22(C).to(device)
    block22 = layers_scale_mlp_blocks22(C).to(device)
    linear22 = Linear22(C, C, bias=True).to(device)
    net22 = nn.Sequential(block22, patch_linear22, gelu22, aff22, ln22, gelu22, linear22)

    im = torch.randn((B, 3, 224, 224), device=device)
    x11, x12, x21, x22 = patch_embed22(im)
    x11_flip, x12_flip, x21_flip, x22_flip = patch_embed22(im.flip(-1))

    y11, y12, y21, y22 = net22((x11, x12, x21, x22))
    y11_flip, y12_flip, y21_flip, y22_flip = net22((x11_flip, x12_flip, x21_flip, x22_flip))
    assert torch.allclose(y11, y11_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    assert torch.allclose(y12, -y12_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
    assert not torch.allclose(y12, y12_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
    assert torch.allclose(y21, -y21_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
    assert torch.allclose(y22, y22_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple equivariance test for ResMLP components passed!")

def simple_equivariance_test_vit():
    device = DEVICE
    B = 1 if device == 'cpu' else 64
    C = 768
    side = 14

    with torch.inference_mode():
        x1 = torch.randn((B, 1+side**2, C//2), device=device)
        x2 = torch.randn((B, 1+side**2, C//2), device=device)
        x22 = torch.randn((B, 1+side**2, C//2), device=device)
        attn2 = Attention2(C, qkv_bias=True, fused_attn=True).to(device)
        block2 = Layer_scale_init_Block2(C, num_heads=12, use_fused_attn=True).to(device)

        net2 = nn.Sequential(attn2, block2)

        y1, y2 = net2((x1, x2))
        y12, y22 = net2((x1, x22))
        y1_flip, y2_flip = net2((x1, -x2))

        assert not torch.allclose(y1, y12, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert not torch.allclose(y2, y22, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
        assert not torch.allclose(y2, y2_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
        assert torch.allclose(y2, -y2_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"

    print("Simple equivariance test for ViT components passed!")

def simple_equivariance_test_convnext_stem():
    device = DEVICE
    B = 1 if device == 'cpu' else 32
    C = 384
    side = 64

    with torch.inference_mode():
        x1 = torch.randn((B, 3, side, side), device=device)
        x2 = torch.randn((B, 3, side, side), device=device)
        net2 = nn.Sequential(
            PatchEmbed2(img_size=side, in_chans=3, embed_dim=C, patch_size=4, flatten=False),
            LayerNorm2LastOrFirst(C, eps=1e-6, data_format="channels_first")
        ).to(device)

        y1, y2 = net2(x1)
        y12, y22 = net2(x2)
        y1_flip, y2_flip = net2(x1.flip(-1))

        assert not torch.allclose(y1, y12, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert not torch.allclose(y2, y22, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert torch.allclose(y1, y1_flip.flip(-1), atol=1e-5, rtol=1e-5), "Not invariant"
        assert not torch.allclose(y2, y2_flip.flip(-1), atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
        assert torch.allclose(y2, -y2_flip.flip(-1), atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"

    print("Simple equivariance test for ConvNeXt stem passed!")

class PermuterFirstToLast(nn.Module):
    def forward(self, xs):
        return (xs[0].permute(0, 2, 3, 1), xs[1].permute(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)

class PermuterLastToFirst(nn.Module):
    def forward(self, xs):
        return (xs[0].permute(0, 3, 1, 2), xs[1].permute(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

def simple_equivariance_test_convnext():
    device = DEVICE
    B = 1 if device == 'cpu' else 32
    C = 384
    D = 512
    side = 64

    with torch.inference_mode():
        x1 = torch.randn((B, side, side, C//2), device=device)
        x2 = torch.randn((B, side, side, C//2), device=device)
        x22 = torch.randn((B, side, side, C//2), device=device)
        for net2 in (
            nn.Sequential(PermuterLastToFirst(), DepthwiseConv2d2(C, C), PermuterFirstToLast()),
            nn.Sequential(PermuterLastToFirst(), ConvNeXtBlock2(C, layer_scale_init_value=1, channels_last_input=False)),
            ConvNeXtBlock2(C, layer_scale_init_value=1),
            InvariantDownsampleConv2d2(C, D),
            nn.Sequential(
                PermuterLastToFirst(),
                ConvNeXtBlock2ChannelsFirst(C, layer_scale_init_value=1),
                PermuterFirstToLast(),
            ),
            nn.Sequential(
                PermuterLastToFirst(),
                DownsampleConv2d2(C, D),
                PermuterFirstToLast(),
            ),
        ):
            net2 = net2.to(device)
            y1, y2 = net2((x1, x2))
            y12, y22 = net2((x1, x22))
            y1_flip, y2_flip = net2((x1.flip(-2), -x2.flip(-2)))

            if not isinstance(net2, InvariantDownsampleConv2d2):
                assert not torch.allclose(y1, y12, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
            assert not torch.allclose(y2, y22, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
            assert torch.allclose(y1, y1_flip.flip(-2), atol=1e-5, rtol=1e-5), "Not invariant"
            assert not torch.allclose(y2, y2_flip.flip(-2), atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
            assert torch.allclose(y2, -y2_flip.flip(-2), atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"

    print("Simple equivariance test for ConvNeXt components passed!")

def simple_invariance_test_convnext():
    device = DEVICE

    with torch.inference_mode():
        net = d2_convnext_base(
            use_symmetrized_asym_feats=True, layer_scale_init_value=1,
        ).to(device)

        x = torch.randn((31, 3, 224, 224), device=device)
        x_flip = x.clone().flip(-1)
        x2 = x.clone().flip(-2)
        y1 = net(x)
        y1_flip = net(x_flip)
        y2 = net(x2)
        assert not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), "Bad test, these should be different"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple invariance test of convnext architecture passed!")

def simple_invariance_test_convnext_isotropic():
    device = DEVICE

    with torch.inference_mode():
        net = d2_convnext_isotropic_base(
            use_symmetrized_asym_feats=True, layer_scale_init_value=1,
        ).to(device)

        x = torch.randn((31, 3, 224, 224), device=device)
        x_flip = x.clone().flip(-1)
        x2 = x.clone().flip(-2)
        y1 = net(x)
        y1_flip = net(x_flip)
        y2 = net(x2)
        assert not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), "Bad test, these should be different"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple invariance test of convnext-isotropic architecture passed!")

def simple_equivariance_test_vit():
    device = DEVICE
    B = 1 if device == 'cpu' else 256
    C = 768
    side = 14

    with torch.inference_mode():
        x1 = torch.randn((B, 1+side**2, C//2), device=device)
        x2 = torch.randn((B, 1+side**2, C//2), device=device)
        x22 = torch.randn((B, 1+side**2, C//2), device=device)
        attn2 = Attention2(C, qkv_bias=True, fused_attn=True).to(device)
        block2 = Layer_scale_init_Block2(C, num_heads=12, use_fused_attn=True, init_values=1).to(device)

        net2 = nn.Sequential(attn2, block2)

        y1, y2 = net2((x1, x2))
        y12, y22 = net2((x1, x22))
        y1_flip, y2_flip = net2((x1, -x2))

        assert not torch.allclose(y1, y12, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert not torch.allclose(y2, y22, atol=1e-5, rtol=1e-5), "Bad test, these should not be equal"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
        assert not torch.allclose(y2, y2_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"
        assert torch.allclose(y2, -y2_flip, atol=1e-5, rtol=1e-5), "Not (-1)-equivariant"

    print("Simple equivariance test for ViT components passed!")

def simple_invariance_test_vit():
    device = DEVICE

    with torch.inference_mode():
        net = d2_deit_medium_patch16_LS(
            use_symmetrized_asym_feats=True, use_fused_attn=True, init_scale=1,
        ).to(device)

        x = torch.randn((31, 3, 224, 224), device=device)
        x_flip = x.clone().flip(-1)
        x2 = x.clone().flip(-2)
        y1 = net(x)
        y1_flip = net(x_flip)
        y2 = net(x2)
        assert not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), "Bad test, these should be different"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple invariance test of ViT architecture passed!")

def simple_invariance_test_inv_early_vit():
    device = DEVICE

    with torch.inference_mode():
        net = d2_inv_early_deit_medium_patch16_LS(
            use_fused_attn=True, init_scale=1,
        ).to(device)
        x = torch.randn((31, 3, 224, 224), device=device)
        x_flip = x.clone().flip(-1)
        x2 = x.clone().flip(-2)
        y1 = net(x)
        y1_flip = net(x_flip)
        y2 = net(x2)
        assert not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), "Bad test, these should be different"
        assert torch.allclose(y1, y1_flip, atol=1e-5, rtol=1e-5), "Not invariant"
    print("Simple invariance test of early invariarized ViT architecture passed!")

def norm_comparison_patchembed():
    print("comparing norms of patch-embed output")
    device = DEVICE

    net = PatchEmbed(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768).to(device)
    net_equi = PatchEmbed2(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768).to(device)

    x = torch.randn((5, 3, 224, 224), device=device)
    x2 = x.clone().flip(-2)  # same numbers different order
    y1 = net(x).flatten(start_dim=1)
    y2 = net(x2).flatten(start_dim=1)
    y1_equi = net_equi(x)
    y2_equi = net_equi(x2)
    y1_equi = torch.cat(y1_equi, dim=-1).flatten(start_dim=1)
    y2_equi = torch.cat(y2_equi, dim=-1).flatten(start_dim=1)
    print(torch.norm(y1, dim=1).mean())
    print(torch.norm(y1_equi, dim=1).mean())
    print(torch.norm(y1-y2, dim=1).mean())
    print(torch.norm(y1_equi-y2_equi, dim=1).mean())

def norm_comparison_layernorm():
    print("comparing norms of layernorm output")
    device = DEVICE

    net = nn.LayerNorm(
        768,
        elementwise_affine=False,
        bias=False,
        eps=1e-10,
    ).to(device)
    net_equi = LayerNorm2v2(
        768,
        elementwise_affine=False,
        bias=False,
        eps=1e-10,
    ).to(device)
    net_equi22 = LayerNorm22v2(
        768,
        elementwise_affine=False,
        bias=False,
        eps=1e-10,
    ).to(device)

    x = torch.randn((5, 145, 768), device=device)
    x2 = torch.randn((5, 145, 768), device=device)
    xs = (x[..., :384].clone(), x[..., 384:].clone())
    xs2 = (x2[..., :384].clone(), x2[..., 384:].clone())
    xss = (x[..., :192].clone(), x[..., 192:384].clone(), x[..., 384:576].clone(), x[..., 576:].clone())
    xss2 = (x2[..., :192].clone(), x2[..., 192:384].clone(), x2[..., 384:576].clone(), x2[..., 576:].clone())

    y1 = net(x).flatten(start_dim=1)
    print(net(x)[0,0].norm() / math.sqrt(768))
    y2 = net(x2).flatten(start_dim=1)

    y1_equi = net_equi(xs)
    print((math.sqrt(y1_equi[0][0,0].norm()**2 + y1_equi[1][0,0].norm()**2)) / math.sqrt(768))
    y2_equi = net_equi(xs2)
    y1_equi = torch.cat(y1_equi, dim=-1).flatten(start_dim=1)
    y2_equi = torch.cat(y2_equi, dim=-1).flatten(start_dim=1)

    y1_equi22 = net_equi22(xss)
    print((math.sqrt(y1_equi22[0][0,0].norm()**2 + y1_equi22[1][0,0].norm()**2 + y1_equi22[2][0,0].norm()**2 + y1_equi22[3][0,0].norm()**2)) / math.sqrt(768))
    y2_equi22 = net_equi22(xss2)
    y1_equi22 = torch.cat(y1_equi22, dim=-1).flatten(start_dim=1)
    y2_equi22 = torch.cat(y2_equi22, dim=-1).flatten(start_dim=1)

    print(torch.norm(y1, dim=1).mean())
    print(torch.norm(y1_equi, dim=1).mean())
    print(torch.norm(y1_equi22, dim=1).mean())
    print(torch.norm(y1-y2, dim=1).mean())
    print(torch.norm(y1_equi-y2_equi, dim=1).mean())
    print(torch.norm(y1_equi22-y2_equi22, dim=1).mean())

def norm_comparison_block():
    print("comparing norms of block output")
    device = DEVICE

    net = Layer_scale_init_Block(
        dim=768,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        init_values=1e-4,
        drop=.1,
    ).to(device)
    net_equi = Layer_scale_init_Block2(
        dim=768,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        init_values=1e-4,
        drop=.1,
    ).to(device)

    x = torch.randn((5, 145, 768), device=device)
    x2 = torch.randn((5, 145, 768), device=device)
    xs = (x[..., :384].clone(), x[..., 384:].clone())
    xs2 = (x2[..., :384].clone(), x2[..., 384:].clone())

    y1 = net(x).flatten(start_dim=1)
    y2 = net(x2).flatten(start_dim=1)

    y1_equi = net_equi(xs)
    y2_equi = net_equi(xs2)
    y1_equi = torch.cat(y1_equi, dim=-1).flatten(start_dim=1)
    y2_equi = torch.cat(y2_equi, dim=-1).flatten(start_dim=1)
    print(torch.norm(y1, dim=1).mean())
    print(torch.norm(y1_equi, dim=1).mean())
    print(torch.norm(y1-y2, dim=1).mean())
    print(torch.norm(y1_equi-y2_equi, dim=1).mean())

def norm_comparison_attention():
    print("comparing norms of attention output")
    device = DEVICE

    net = Attention(
        dim=768,
        num_heads=12,
        qkv_bias=True,
        fused_attn=True,
    ).to(device)
    net_equi = Attention2(
        dim=768,
        num_heads=12,
        qkv_bias=True,
        fused_attn=True,
    ).to(device)

    x = torch.randn((5, 145, 768), device=device)
    x2 = torch.randn((5, 145, 768), device=device)
    xs = (x[..., :384].clone(), x[..., 384:].clone())
    xs2 = (x2[..., :384].clone(), x2[..., 384:].clone())

    y1 = net(x).flatten(start_dim=1)
    y2 = net(x2).flatten(start_dim=1)

    y1_equi = net_equi(xs)
    y2_equi = net_equi(xs2)
    y1_equi = torch.cat(y1_equi, dim=-1).flatten(start_dim=1)
    y2_equi = torch.cat(y2_equi, dim=-1).flatten(start_dim=1)
    print(torch.norm(y1, dim=1).mean())
    print(torch.norm(y1_equi, dim=1).mean())
    print(torch.norm(y1-y2, dim=1).mean())
    print(torch.norm(y1_equi-y2_equi, dim=1).mean())

def norm_comparison_vit():
    print("comparing norms of vit output")
    device = DEVICE

    net = deit_tiny_patch16_LS(num_classes=1000).to(device)
    net_equi = d2_deit_tiny_patch16_LS(num_classes=1000, use_symmetrized_asym_feats=True).to(device)

    x = torch.randn((5, 3, 224, 224), device=device)
    x2 = torch.randn((5, 3, 224, 224), device=device)

    y1 = net(x)
    y2 = net(x2)
    y1_equi = net_equi(x)
    y2_equi = net_equi(x2)

    print(torch.norm(y1, dim=1).median())
    print(torch.norm(y1_equi, dim=1).median())
    print(torch.norm(y1-y2, dim=1).median())
    print(torch.norm(y1_equi-y2_equi, dim=1).median())

if __name__ == "__main__":
    # norm_comparison_patchembed()
    # norm_comparison_attention()
    # norm_comparison_layernorm()
    # norm_comparison_block()
    # norm_comparison_vit()
    simple_invariance_test_resmlp()
    simple_equivariance_test_resmlp()
    simple_equivariance_test_vit()
    simple_invariance_test_vit()
    simple_invariance_test_inv_early_vit()
    simple_equivariance_test_convnext_stem()
    simple_equivariance_test_convnext()
    simple_invariance_test_convnext()
    simple_invariance_test_convnext_isotropic()
