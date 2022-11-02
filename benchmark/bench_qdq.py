import torch
from torch_int.functional.quantization import dynamic_quantize_activation_per_token_min_max, dequantize_activation_w_per_channel_a_per_token
from torch_int._CUDA import quantize_activation_per_token, dequantize_activation_per_token
import argparse
from icecream import ic
from utils import bench_func_latency


@torch.no_grad()
def bench_qdq():
    SEQ_LEN, CIN, COUT = 256, 12288, 12288
    TIMES = 10000
    act = torch.randn((SEQ_LEN, CIN), dtype=torch.float16, device='cuda')
    q_act = torch.randint(-127, 127, (SEQ_LEN, COUT),
                          dtype=torch.int8, device='cuda')
    w_scale = torch.randn((1, COUT), dtype=torch.float16, device='cuda')
    a_scale = torch.randn((SEQ_LEN, 1), dtype=torch.float16, device='cuda')

    # for _ in range(TIMES):
    # dynamic_quantize_activation_per_token_min_max(act)
    # dequantize_activation_w_per_channel_a_per_token(q_act, w_scale, a_scale)
    t_q_py = bench_func_latency(
        dynamic_quantize_activation_per_token_min_max, (act.clone(), ), TIMES)
    ic(t_q_py)
    t_dq_py = bench_func_latency(
        dequantize_activation_w_per_channel_a_per_token, (q_act.clone(), w_scale.clone(), a_scale.clone()), TIMES)
    ic(t_dq_py)

    t_q_c = bench_func_latency(
        quantize_activation_per_token, (act.clone(), ), TIMES)
    ic(t_q_c)
    t_dq_c = bench_func_latency(dequantize_activation_per_token, (q_act.clone(
    ), w_scale.clone(), a_scale.clone()), TIMES)
    ic(t_dq_c)


if __name__ == '__main__':
    bench_qdq()
