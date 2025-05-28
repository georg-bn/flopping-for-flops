import torch
from fvcore.nn import FlopCountAnalysis
import time
from tabulate import tabulate
from timm.models import create_model
import torch.amp
from tqdm import tqdm
import argparse
import deit.utils as utils

import deit.models.vit
import deit.models.resmlp
import deit.models.hybrid
import deit.models.d2_vit
import deit.models.d2_resmlp
import convnext.models.convnext_isotropic
import convnext.models.d2_convnext_isotropic


WARMUP_ITERATIONS = 10
NUM_ITERATIONS = 100
BATCH_SIZE = 64

headers = ["Model Name", "Params (10⁶)", "Throughput (im/s)", "FLOPS (10⁹)", "Peak Mem (MB)"]

# Model parameters cover the largest batch size that fits into memory
_MODEL_PARAMS = {
    "convnext_isotropic_small": BATCH_SIZE,
    "convnext_isotropic_base": BATCH_SIZE,
    "convnext_isotropic_large": BATCH_SIZE,
    
    "d2_convnext_isotropic_small": BATCH_SIZE,
    "d2_convnext_isotropic_base": BATCH_SIZE,
    "d2_convnext_isotropic_large": BATCH_SIZE,

    "resmlp_12": BATCH_SIZE,
    "resmlp_24": BATCH_SIZE,
    "resmlpB_24": BATCH_SIZE,
    "resmlpL_24": BATCH_SIZE,

    "resmlp_equi_d2_12": BATCH_SIZE,
    "resmlp_equi_d2_24": BATCH_SIZE,
    "resmlp_equi_d2_B_24": BATCH_SIZE,
    "resmlp_equi_d2_L_24": BATCH_SIZE,

    "deit_small_patch16_LS": BATCH_SIZE,
    "deit_base_patch16_LS": BATCH_SIZE,
    "deit_large_patch16_LS": BATCH_SIZE,
    "deit_huge_patch14_LS": BATCH_SIZE,

    "d2_deit_small_patch16_LS": BATCH_SIZE,
    "d2_deit_base_patch16_LS": BATCH_SIZE,
    "d2_deit_large_patch16_LS": BATCH_SIZE,
    "d2_deit_huge_patch14_LS": BATCH_SIZE,

    "hybrid_deit_huge_patch14_LS": BATCH_SIZE,
    "hybrid_deit_large_patch16_LS": BATCH_SIZE,
    "hybrid_deit_base_patch16_LS": BATCH_SIZE,
}

def compute_peak_mem(model, batch_size=8, device='cuda', amp=True):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.amp.autocast(device_type='cuda', enabled=amp):
        model(x)
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)  # in MB
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_memory

def compute_throughput(model, x):
    timing = []
    for _ in range(WARMUP_ITERATIONS):
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            model(x)

    torch.cuda.synchronize()
    for _ in range(NUM_ITERATIONS):
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            start = time.time()
            model(x)
            torch.cuda.synchronize()
            timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    
    return timing.mean()


@torch.no_grad()
def compute_complexity(model_name, args):
    torch.cuda.empty_cache()

    device = 'cuda'

    model = create_model(model_name, pretrained=False, use_symmetrized_asym_feats=args.symmetrized_asym_feats )
    model.eval()
    model.to(device)

    B = _MODEL_PARAMS[model_name]
    x = torch.randn(B, 3, 224, 224, device=device)

    params = sum(p.numel() for p in model.parameters())

    flops = FlopCountAnalysis(model, x)
    utils.jits(flops)
    flops_per_image = flops.total() / B / 1e9  # in GFLOPs

    if args.compile:
        model = torch.compile(model, mode='max-autotune')
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            y = model(x)

    peak_memory = compute_peak_mem(model, batch_size=B)
    imgs_per_sec = B/compute_throughput(model, x)

    return [model_name, f"{params/1e6:.4f}", f"{imgs_per_sec:.0f}", f"{flops_per_image:.1f}", f"{peak_memory:.0f}"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute model complexity')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--symmetrized-asym-feats', action='store_true', help='')
    parser.set_defaults(symmetrized_asym_feats=False)
    args = parser.parse_args()

    data = []
    for model_name in tqdm(_MODEL_PARAMS.keys(), desc='Computing model complexity'):
        data.append(compute_complexity(model_name, args))

    print(tabulate(data, headers=headers, tablefmt="grid"))