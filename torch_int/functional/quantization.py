import torch

def quantize_per_tensor_min_max(t, use_zero_point=False):
    min, max = torch.min(t), torch.max(t)


def quantize_per_channel_min_max(t, axis, use_zero_point=False):
    pass

def dequantize_per_tensor_min_max(t, scale, zero_point=0):
    pass

def dequantize_per_channel_min_max(t, axis, scale, zero_point=0):
    pass
