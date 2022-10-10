import torch
from .._CUDA import gemm_cutlass as gemm
from ..functional.quantization import (
    quantize_weight_per_channel_min_max,
    dynamic_quantize_activation_per_tensor_min_max,
    dynamic_quantize_activation_per_token_min_max,
    dequantize_activation_w_per_channel_a_per_token,
    dequantize_activation_w_per_channel_a_per_tensor,
)


class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Int8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randint(-127, 127, (self.out_features,
                                    self.in_features), dtype=torch.int8)
        if bias:
            self.bias = torch.zeros(
                (1, self.out_features), dtype=torch.float16)
        else:
            self.register_parameter('bias', None)
        self.weight_scales = torch.ones(self.out_features, dtype=torch.float16)
        self.activation_quantizer = dynamic_quantize_activation_per_token_min_max
        self.activation_dequantizer = dequantize_activation_w_per_channel_a_per_token

    def to(self, *args, **kwargs):
        super(Int8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x, x_scale = self.activation_quantizer(x)
        q_y = gemm(q_x, self.weight)
        y = self.activation_dequantizer(q_y, self.weight_scales, x_scale)
        y = y + self.bias
        return y

    @staticmethod
    def from_float(module):
        assert isinstance(module, torch.nn.Linear)
        new_module = Int8Linear(
            module.in_features, module.out_features, module.bias is not None)
        new_module.weight, new_module.weight_scales = quantize_weight_per_channel_min_max(module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module
