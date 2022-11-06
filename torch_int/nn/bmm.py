import torch
from .._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t


class BMM_S8T_S8N_S8T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int8
        return bmm_s8t_s8n_s8t(a, b, self.alpha)

    @staticmethod
    def from_io_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_S8T_S8N_S8T(1.0)
        bmm_module.alpha = a_scale * b_scale / output_scale
        bmm_module.a_scale = a_scale
        bmm_module.b_scale = b_scale
        bmm_module.output_scale = output_scale
        return bmm_module


class BMM_S8T_S8N_S32T(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_s32t(a, b)
