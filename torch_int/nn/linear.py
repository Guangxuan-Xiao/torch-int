import torch
from .._CUDA import gemm_cutlass as gemm
from .._CUDA import linear_a8_w8_b32_o32, linear_relu_a8_w8_b8_o8, linear_a8_w8_b8_o8, linear_a8_w8_b32_o32_with_scaling
from ..functional.quantization import (
    quantize_weight_per_channel_min_max,
    dynamic_quantize_activation_per_tensor_min_max,
    dynamic_quantize_activation_per_token_min_max,
    dequantize_activation_w_per_channel_a_per_token,
    dequantize_activation_w_per_channel_a_per_tensor,
    fake_quantize_activation_per_tensor_min_max,
    fake_quantize_activation_per_token_min_max,
)


class W8A8B32O32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int32, requires_grad=False))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b32_o32(x, self.weight, self.bias)
        y = y.view(*x_shape[:-1], -1)
        return y


class W8A8B32O32LinearWithScaling(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int32, requires_grad=False))
        self.out_scale = 1.0
        self.bias_scale = 1.0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b32_o32_with_scaling(
            x, self.weight, self.bias, self.out_scale, self.bias_scale)
        y = y.view(*x_shape[:-1], -1)
        return y


class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.out_scale = 1.0
        self.bias_scale = 1.0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b8_o8(x, self.weight, self.bias,
                               self.out_scale, self.bias_scale)
        y = y.view(*x_shape[:-1], -1)
        return y


class W8A8B8O8LinearReLU(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.out_scale = 1.0
        self.bias_scale = 1.0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_relu_a8_w8_b8_o8(x, self.weight, self.bias,
                                    self.out_scale, self.bias_scale)
        y = y.view(*x_shape[:-1], -1)
        return y


class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token'):
        super(Int8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)
        self.register_buffer('weight_scales', torch.ones(
            self.out_features, dtype=torch.float16, requires_grad=False))
        if act_quant == 'per_token':
            self.activation_quantizer = dynamic_quantize_activation_per_token_min_max
            self.activation_dequantizer = dequantize_activation_w_per_channel_a_per_token
        elif act_quant == 'per_tensor':
            self.activation_quantizer = dynamic_quantize_activation_per_tensor_min_max
            self.activation_dequantizer = dequantize_activation_w_per_channel_a_per_tensor
        else:
            raise ValueError('act_quant must be "per_token" or "per_tensor"')

    def to(self, *args, **kwargs):
        super(Int8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        q_x, x_scale = self.activation_quantizer(x)
        q_y = gemm(q_x, self.weight)
        y = self.activation_dequantizer(q_y, self.weight_scales, x_scale)
        y = y.view(*x_shape[:-1], -1)
        if self.bias is not None:
            y += self.bias
        return y

    @staticmethod
    def from_float(module, act_quant='per_token'):
        assert isinstance(module, torch.nn.Linear)
        new_module = Int8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant)
        new_module.weight, new_module.weight_scales = quantize_weight_per_channel_min_max(
            module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return super().__repr__() + f'({self.in_features}, {self.out_features}, bias={self.bias is not None}, act_quant={self.activation_quantizer.__name__})'


class W8A16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(W8A16Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)
        self.register_buffer('weight_scales', torch.ones(
            self.out_features, dtype=torch.float16, requires_grad=False))

    def to(self, *args, **kwargs):
        super(W8A16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        weight_fp16 = self.weight.to(torch.float16)
        weight_fp16.mul_(self.weight_scales.view(-1, 1))
        y = torch.functional.F.linear(x, weight_fp16, self.bias)
        del weight_fp16
        return y

    @staticmethod
    def from_float(module):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A16Linear(
            module.in_features, module.out_features, module.bias is not None)
        new_module.weight, new_module.weight_scales = quantize_weight_per_channel_min_max(
            module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return super().__repr__() + f'({self.in_features}, {self.out_features}, bias={self.bias is not None})'


class W8FakeA8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token'):
        super(W8FakeA8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)
        self.register_buffer('weight_scales', torch.ones(
            self.out_features, dtype=torch.float16, requires_grad=False))

        if act_quant == 'per_token':
            self.activation_fake_quantizer = fake_quantize_activation_per_token_min_max
        elif act_quant == 'per_tensor':
            self.activation_fake_quantizer = fake_quantize_activation_per_tensor_min_max
        else:
            raise ValueError('act_quant must be "per_token" or "per_tensor"')

    def to(self, *args, **kwargs):
        super(W8FakeA8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        weight_fp16 = self.weight.to(torch.float16)
        weight_fp16.mul_(self.weight_scales.view(-1, 1))
        x = self.activation_fake_quantizer(x)
        y = torch.functional.F.linear(x, weight_fp16, self.bias)
        del weight_fp16
        return y

    @staticmethod
    def from_float(module, act_quant='per_token'):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8FakeA8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant)
        new_module.weight, new_module.weight_scales = quantize_weight_per_channel_min_max(
            module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return super().__repr__() + f'({self.in_features}, {self.out_features}, bias={self.bias is not None})'
