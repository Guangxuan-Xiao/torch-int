import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t


def bmm_i8_o8(a, b, scale):
    # a: [B, L, H] int8
    # b: [B, L, H] int8
    # scale: float
    # return: [B, L, H] int8 = a * b.T * scale
    return bmm_s8t_s8n_s8t(a, b, scale)


def bmm_i8_o32(a, b):
    # a: [B, L, H] int8
    # b: [B, L, H] int8
    # return: [B, L, H] int32 = a * b.T
    return bmm_s8t_s8n_s32t(a, b)
