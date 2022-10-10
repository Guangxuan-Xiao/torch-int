import torch
import numpy as np


def _get_weight_per_channel_scales(w, n_bit=8, k_near_zero_tolerance=1e-6):
    # NOTICE: the zero point for w is always chosen as 0, so it is actually a symmetric quantization
    def _extract_min_max_from_weight(weights):
        dim_size = weights.shape[0]

        if weights.max() == weights.min():  # all the elements are the same?
            mins = np.zeros(dim_size)
            maxs = np.zeros(dim_size)
            single_value = weights.min().item()
            if single_value < 0.:
                mins[:] = single_value
                maxs[:] = -single_value
            elif single_value > 0.:
                mins[:] = -single_value
                maxs[:] = single_value
            else:
                mins[:] = maxs[:] = single_value
            return torch.from_numpy(mins).to(weights.device), torch.from_numpy(maxs).to(weights.device)
        else:
            weights = weights.reshape(weights.shape[0], -1)
            mins = weights.min(dim=1)[0]
            maxs = weights.max(dim=1)[0]
            maxs = torch.max(mins.abs(), maxs.abs())
            mins = -maxs
            return mins, maxs

    def _expand_very_small_range(mins, maxs):
        k_smallest_half_range = k_near_zero_tolerance / 2
        if (maxs - mins).min() > k_near_zero_tolerance:
            return mins, maxs
        else:
            for i in range(len(mins)):
                mins[i] = min(mins[i], -k_smallest_half_range)
                maxs[i] = max(maxs[i], k_smallest_half_range)
            return mins, maxs

    mins, maxs = _extract_min_max_from_weight(w)
    mins, maxs = _expand_very_small_range(mins, maxs)
    assert (mins + maxs).max().float() < 1e-9, (mins +
                                                maxs).max().float()  # symmetric
    return maxs / (2 ** (n_bit - 1) - 1)


@torch.no_grad()
def quantize_weight_per_channel_min_max(w):
    scales = _get_weight_per_channel_scales(
        w, n_bit=8).reshape(-1, *([1] * (len(w.shape)-1)))
    w_q = (w / scales).to(torch.int8)
    return w_q, scales


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_min_max_zeropoint(t):
    max_val = t.max()[0]
    min_val = t.min()[0]
    quant_min = -127
    quant_max = 127
    nudged_scale = (max_val - min_val) / (quant_max - quant_min)
    zp = (max_val + min_val) / 2
    zp = (zp / nudged_scale).round() * nudged_scale
    t -= zp
    max_val = (max_val - min_val) / 2

    max_val = torch.clamp(max_val, min=1e-8) / (2 ** 7 - 1)
    q_act = (t / max_val).to(torch.int8)
    return q_act, max_val, zp


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_min_max(t):
    max_val = t.abs().max()
    max_val = torch.clamp(max_val, min=1e-8) / (2 ** 7 - 1)
    q_act = (t / max_val).to(torch.int8)
    return q_act, max_val


@torch.no_grad()
def dynamic_quantize_activation_per_token_min_max(t):
    max_val = t.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8) / (2 ** 7 - 1)
    q_act = (t / max_val).to(torch.int8)
    return q_act, max_val


@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_token(q_act, w_scales, a_scales):
    # q_act: [batch_size, seq_len, dim]
    # w_scales: [dim]
    # a_scales: [batch_size, seq_len, 1]
    return q_act.to(torch.float16) * w_scales.reshape(1, 1, -1) * a_scales


@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_tensor(q_act, w_scales, a_scales):
    # q_act: [batch_size, seq_len, dim]
    # w_scales: [dim]
    # a_scales: [1]
    return q_act.to(torch.float16) * w_scales.reshape(1, 1, -1) * a_scales


@torch.no_grad()
def static_quantize_per_tensor(t, scale, zero_point=0):
    raise NotImplementedError


@torch.no_grad()
def static_quantize_per_channel(t, axis, scale, zero_point=0):
    raise NotImplementedError
