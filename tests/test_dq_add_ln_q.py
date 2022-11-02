import torch
from torch_int._CUDA import dq_add_layernorm_q
from icecream import ic


def ref_dq_add_layernorm_q(input, input_scale, residual_input, gamma, beta, eps):
    residual_output_fp32 = torch.add(residual_input, input, alpha=input_scale)
    ln_output_fp32 = torch.nn.functional.layer_norm(
        residual_output_fp32, residual_output_fp32.shape[-1:], gamma, beta, eps)
    return ln_output_fp32


@torch.no_grad()
def test_dq_add_layernorm_q():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-65536, 65535, (N,), dtype=torch.int32)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float()
    linear.bias.data = bias.float()
    y_gt = linear(x.float())
    y = linear_a8_w8_b32_o32(x.cuda(), weight.cuda(), bias.cuda())
    ic(torch.allclose(y_gt, y.float().cpu(), atol=1e-3))


if __name__ == '__main__':
    test_dq_add_layernorm_q()
