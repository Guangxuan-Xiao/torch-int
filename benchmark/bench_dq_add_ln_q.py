import torch
from icecream import ic
from torch_int.functional.fused import dq_add_layernorm_q_py, dq_add_layernorm_q_cpp
from utils import bench_func_latency
import numpy as np


def residual_layernorm_fp16(input, residual, ln):
    residual_out = input + residual
    ln_out = ln(residual_out)
    return residual_out, ln_out


@torch.no_grad()
def bench_dq_add_layernorm_q():
    B, L, H = 1, 512, 12288
    input_int32 = torch.randint(-65536, 65536,
                                (B, L, H), dtype=torch.int32).cuda()
    input_scale_fp = 0.01
    residual_input_fp = torch.randn(B, L, H).cuda()
    layernorm = torch.nn.LayerNorm(H).cuda()
    gamma = layernorm.weight
    beta = layernorm.bias
    eps = layernorm.eps
    args = (input_int32, input_scale_fp,
            residual_input_fp, gamma, beta, eps)
    input_fp16 = torch.randn(B, L, H, dtype=torch.float16, device='cuda')
    residual_fp16 = torch.randn(B, L, H, dtype=torch.float16, device='cuda')
    ln_fp16 = torch.nn.LayerNorm(H, dtype=torch.float16).cuda()
    print("fp16")
    bench_func_latency(residual_layernorm_fp16, (input_fp16,
                       residual_fp16, ln_fp16), num_iter=10000)
    print("int8 + fp32 py")
    bench_func_latency(dq_add_layernorm_q_py, args, num_iter=10000)
    print("int8 cpp + fp32 cpp")
    bench_func_latency(dq_add_layernorm_q_cpp, args, num_iter=10000)
    residual_input_fp = torch.randn(B, L, H).cuda().half()
    gamma = layernorm.weight.half()
    beta = layernorm.bias.half()
    args = (input_int32, input_scale_fp,
            residual_input_fp, gamma, beta, eps)
    print("int8 + fp16 py")
    bench_func_latency(dq_add_layernorm_q_py, args, num_iter=10000)
    print("int8 + fp16 cpp")
    bench_func_latency(dq_add_layernorm_q_cpp, args, num_iter=10000)


if __name__ == '__main__':
    bench_dq_add_layernorm_q()
