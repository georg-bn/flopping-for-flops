import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from resmlp_models import resmlp_24
from s2_resmlp_models import resmlp_equi_s2_24, PatchLinear2, Linear2, \
        PatchLinear22, Linear22
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

# class PatchLinear(nn.Linear):
#     def __init__(self, in_c, out_c, bias=True):
#         super().__init__(in_c, out_c, bias)

#     def forward(self, x):
#         if self.bias is not None:
#             return torch.einsum(
#                 "...ab,...ac,c->...cb",
#                 x, self.weight, self.bias,
#             )
#         return torch.einsum(
#             "...ab,...ac->...cb",
#             x, self.weight,
#         )

class PatchLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
    def forward(self, x):
        return self.lin(x.transpose(1, 2)).transpose(1, 2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    device = "cuda"

    batch = 1024
    print(f"batch size: {batch}")
    # x = torch.randn(batch, 3, 224, 224, device=device)
    C = 768
    side = 14
    x = torch.randn(batch, side*side, C, device=device)
    # x = x.transpose(1, 2).clone().contiguous()
    x1, x2 = torch.tensor_split(x, 2, dim=2)
    x1 = x1.clone().contiguous()
    x2 = x2.clone().contiguous()
    x11, x12 = torch.tensor_split(x1, 2, dim=1)
    x11 = x11.clone().contiguous()
    x12 = x12.clone().contiguous()
    x21, x22 = torch.tensor_split(x2, 2, dim=1)
    x21 = x21.clone().contiguous()
    x22 = x22.clone().contiguous()

    for precision in ["high"]:  # 'highest', 'high', 'medium'
        print(f"SETTING PRECISION: {precision}")
        torch.set_float32_matmul_precision(precision)

        for name, comp in [
            ("standard", False),
            ("equi", False),
            ("standard", True),
            ("equi", True),
        ]:
            print(f"Profiling {name}")
            if name == "standard":
                # net = resmlp_24().to(device)
                net = PatchLinear(side*side, side*side).to(device)
                # net = nn.Linear(C, C).to(device)
            elif name == "equi":
                # net = resmlp_equi_s2_24().to(device)
                net = PatchLinear22(side).to(device)
                # net = Linear22(C, C).to(device)
            else:
                raise ValueError()
            if comp:
                torch._dynamo.reset()
                net = torch.compile(net,
                                    # backend="inductor",
                                    mode="max-autotune",
                                    dynamic=False,
                                   ).to(device)
                print("Compiled model!")
            torch.cuda.synchronize(device)
            p = profile_model(net,
                              # x,
                              x if name == "standard" else (x11, x12, x21, x22),
                              save_folder=args.save_folder,
                              save_prefix=f"{name}_comp{comp}",
                              tuple_output=(name == "equi"),
                              backward=False,
                             )
            torch.cuda.synchronize(device)
