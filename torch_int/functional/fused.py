import torch
from torch_int._CUDA import dq_add_layernorm_q


def dq_add_layernorm_q_py(input_int32, input_scale_fp32, residual_input_fp32, gamma, beta, eps):
    residual_output_fp32 = torch.add(
        residual_input_fp32, input_int32, alpha=input_scale_fp32)
    # layernorm is applied to the last dimension
    ln_output_fp32 = torch.nn.functional.layer_norm(
        residual_output_fp32, residual_output_fp32.shape[-1:], gamma, beta, eps)
    ln_output_int8 = ln_output_fp32.clamp(-128, 127).to(torch.int8)
    return residual_output_fp32, ln_output_int8


def dq_add_layernorm_q_cpp(input_int32, input_scale_fp32, residual_input_fp32, gamma, beta, eps):
    return dq_add_layernorm_q(input_int32, input_scale_fp32, residual_input_fp32, gamma, beta, eps)
