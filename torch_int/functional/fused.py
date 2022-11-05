import torch
from torch_int._CUDA import dq_add_layernorm_q


def dq_add_layernorm_q_py(input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps):
    residual_output_fp = torch.add(
        residual_input_fp, input_int32, alpha=input_scale_fp)
    # layernorm is applied to the last dimension
    ln_output_fp = torch.nn.functional.layer_norm(
        residual_output_fp, residual_output_fp.shape[-1:], gamma, beta, eps)
    ln_output_int8 = ln_output_fp.clamp(-128, 127).round().to(torch.int8)
    return residual_output_fp, ln_output_int8


def dq_add_layernorm(input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps):
    residual_output_fp = torch.add(
        residual_input_fp, input_int32, alpha=input_scale_fp)
    # layernorm is applied to the last dimension
    ln_output_fp = torch.nn.functional.layer_norm(
        residual_output_fp, residual_output_fp.shape[-1:], gamma, beta, eps)
    return residual_output_fp, ln_output_fp


def dq_add_layernorm_q_cpp(input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps):
    return dq_add_layernorm_q(input_int32, input_scale_fp, residual_input_fp, gamma, beta, eps)
