import torch
from ..functional.fused import dq_add_layernorm_q_cpp


class DQ_Add_LayerNorm_Q(torch.nn.Module):
    def __init__(self, dim, input_scale: float, eps=1e-5):
        super().__init__()
        self.input_scale = input_scale
        self.eps = eps
        self.register_buffer('gamma', torch.ones(dim, dtype=torch.float32))
        self.register_buffer('beta', torch.zeros(dim, dtype=torch.float32))

    def forward(self, residual_input_fp32, input_int32):
        # input_int32: [B, L, H] int32
        # residual_input_fp32: [B, L, H] fp32
        # return residual_output_fp32, ln_output_int8
        return dq_add_layernorm_q_cpp(
            input_int32, self.input_scale, residual_input_fp32,
            self.gamma, self.beta, self.eps)
