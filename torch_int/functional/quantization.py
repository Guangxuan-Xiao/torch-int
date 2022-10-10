import torch


@torch.no_grad()
def dynamic_quantize_per_tensor_min_max(t, use_zero_point=False):
    min, max = torch.min(t), torch.max(t)


@torch.no_grad()
def dynamic_quantize_per_channel_min_max(t, axis, use_zero_point=False):
    pass

@torch.no_grad()
def dequantize_per_tensor(t, scale, zero_point=0):
    pass

@torch.no_grad()
def dequantize_per_channel(t, axis, scale, zero_point=0):
    pass

@torch.no_grad()
def static_quantize_per_tensor(t, scale, zero_point=0):
    pass


@torch.no_grad()
def static_quantize_per_channel(t, axis, scale, zero_point=0):
    pass
