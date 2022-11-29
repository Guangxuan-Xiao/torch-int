import torch
from torch_int.functional.quantization import dynamic_quantize_activation_per_token_absmax, dequantize_activation_w_per_channel_a_per_token
from torch_int._CUDA import quantize_activation_per_token, dequantize_activation_per_token
import argparse
from icecream import ic

@torch.no_grad()
def test_qdq():
    SEQ_LEN, C = 4, 4
    act = torch.randn((SEQ_LEN, C), dtype=torch.float16, device='cuda')
    w_scale = torch.randn((1, C), dtype=torch.float16, device='cuda')
    act_ref = act * w_scale

    ic(act)
    q_act_py, a_scale_py = dynamic_quantize_activation_per_token_absmax(act.clone())
    q_act_c, a_scale_c = quantize_activation_per_token(act.clone())

    ic(torch.allclose(q_act_py, q_act_c))
    ic(q_act_py, q_act_c)
    ic(torch.allclose(a_scale_py, a_scale_c))
    ic(a_scale_py, a_scale_c)

    dq_act_py = dequantize_activation_w_per_channel_a_per_token(q_act_py, w_scale, a_scale_py)
    dq_act_c = dequantize_activation_per_token(q_act_c, w_scale, a_scale_c)

    ic(torch.allclose(dq_act_py, dq_act_c))
    ic(dq_act_py, dq_act_c)

    ic(act_ref)
    ic(torch.allclose(dq_act_py, act_ref, atol=1e-1))
    ic(torch.allclose(dq_act_c, act_ref, atol=1e-1))

if __name__ == '__main__':
    test_qdq()
