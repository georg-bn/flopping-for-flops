import argparse
import itertools
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from models_v2 import (
    deit_tiny_patch16_LS,
    deit_small_patch16_LS,
    deit_medium_patch16_LS,
    deit_large_patch16_LS,
    deit_huge_patch14_LS,
)
from s2_models_v2 import (
    s2_deit_tiny_patch16_LS,
    s2_deit_small_patch16_LS,
    s2_deit_medium_patch16_LS,
    s2_deit_large_patch16_LS,
    s2_deit_huge_patch14_LS,
)
from torch.profiler import profile, record_function, ProfilerActivity

# see https://github.com/pytorch/pytorch/issues/100253#issuecomment-2068178828
def trace_handler(p, save_folder, save_prefix):
    output = p.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=20)
    # print(output)
    p.export_chrome_trace(
        os.path.join(
            save_folder,
            save_prefix + "trace_" + str(p.step_num) + ".json",
        )
    )

def profile_model(model,
                  input_batch,
                  save_folder,
                  save_prefix,
                  tuple_output=False,
                  backward=False,
                  num_iterations=50,
                  device="cuda",
                 ):
    with torch.inference_mode(not backward):
        for _ in range(5):
            y = model(input_batch)
            if tuple_output and backward:
                y[0].sum().backward()
            elif backward:
                y.sum().backward()
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=2,
        # ),
        # on_trace_ready=lambda p: trace_handler(p, save_folder, save_prefix),
        # profile_memory=True,
        # with_stack=True,
        # with_flops=True,
        # experimental_config=torch._C._profiler._ExperimentalConfig(
        #     verbose=True,
        # ),
    ) as p:
        with torch.inference_mode(not backward):
            for _ in range(num_iterations):
                y = model(input_batch)
                if tuple_output and backward:
                    y[0].sum().backward()
                elif backward:
                    y.sum().backward()
                # p.step()
        torch.cuda.synchronize(device)
    print(p.key_averages().table(
        sort_by="self_cuda_time_total",
        max_src_column_width=None,
        max_name_column_width=1000,
        row_limit=-1,
    ))
    # save_path = os.path.join(
    #     save_folder,
    #     f"{save_prefix}stacks"
    # )
    # p.export_stacks(save_path, metric="self_cuda_time_total")
    return p

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    benchmark_backward = False
    if benchmark_backward:
        print("Benchmarking forward+backward")
    else:
        print("Benchmarking only forward")

    device = "cuda"

    batch = 128
    print(f"batch size: {batch}")
    x = torch.randn(batch, 3, 224, 224, device=device)

    for precision in ["high"]:  # 'highest', 'high', 'medium'
        print(f"SETTING PRECISION: {precision}")
        torch.set_float32_matmul_precision(precision)

        for name, comp, fused_attn in itertools.product(
            ["standard", "equi"],
            [False, True],
            [False, True],
        ):
            print(f"Profiling {name}")
            if name == "standard":
                net = deit_large_patch16_LS(use_fused_attn=fused_attn).to(device)
            elif name == "equi":
                net = s2_deit_large_patch16_LS(use_fused_attn=fused_attn).to(device)
            else:
                raise ValueError()
            print(net)
            print(f"Profiling {name}")
            if comp:
                torch._dynamo.reset()
                net = torch.compile(net,
                                    # backend="inductor",
                                    mode="max-autotune",
                                    dynamic=False,
                                   ).to(device)
                print("Compiled model!")
            else:
                print("NON-compiled model!")
            if fused_attn:
                print("Using fused attn!")
            else:
                print("NOT using fused attn!")
            torch.cuda.synchronize(device)
            p = profile_model(net,
                              x,
                              save_folder=args.save_folder,
                              save_prefix=f"{name}_comp{comp}",
                              tuple_output=False,
                              backward=benchmark_backward,
                             )
            torch.cuda.synchronize(device)
