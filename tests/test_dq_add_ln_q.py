import torch
from icecream import ic
from torch_int.functional.fused import dq_add_layernorm_q_py, dq_add_layernorm_q_cpp


@torch.no_grad()
def test_dq_add_layernorm_q():
    B, L, H = 2, 3, 4
    input_int32 = torch.randint(-65536, 65536, (B, L, H), dtype=torch.int32)
    input_scale_fp = 0.01
    residual_input_fp = torch.randn(B, L, H)
    layernorm = torch.nn.LayerNorm(H)
    gamma = layernorm.weight
    beta = layernorm.bias
    eps = layernorm.eps
    py_output = dq_add_layernorm_q_py(
        input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps)
    ic(py_output)
    cpp_output = dq_add_layernorm_q_cpp(
        input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps)
    ic(cpp_output)
    ic(torch.allclose(py_output[0], cpp_output[0]))
    ic(torch.allclose(py_output[1], cpp_output[1]))


if __name__ == '__main__':
    test_dq_add_layernorm_q()
