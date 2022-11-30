import torch
from .._CUDA import (linear_a8_w8_b32_o32,
                     linear_relu_a8_w8_b8_o8,
                     linear_a8_w8_b8_o8,
                     linear_a8_w8_b32_o32_with_scaling,
                     linear_a8_w8_bfp32_ofp32
                     )
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)


class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

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
                               self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8Linear(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B8O8LinearReLU(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

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
                                    self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = W8A8B8O8LinearReLU(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B32O32LinearWithoutScaling(torch.nn.Module):
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


class W8A8B32O32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

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
            x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B32O32Linear(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        module.bias = module.bias.float()
        bias_scale = module.bias.abs().max() / (2**31 - 1)
        int32_bias = (module.bias / bias_scale).round().to(torch.int32)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int32_bias
        int8_module.a = alpha
        int8_module.b = beta
        int8_module.input_scale = input_scale
        int8_module.output_scale = output_scale
        int8_module.weight_scale = weight_scale
        int8_module.bias_scale = bias_scale
        return int8_module


class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        self.bias = self.bias.to(torch.float32)
        y = linear_a8_w8_bfp32_ofp32(
            x, self.weight, self.bias, self.a.item(), 1)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32Linear(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        int8_module.bias = module.bias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module


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
        weight_fp16.mul_(self.weight_scales)
        y = torch.functional.F.linear(x, weight_fp16, self.bias)
        del weight_fp16
        return y

    @staticmethod
    def from_float(module, weight_quant='per_channel'):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A16Linear(
            module.in_features, module.out_features, module.bias is not None)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.weight_scales = quantize_weight_per_channel_absmax(
                module.weight)
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.weight_scales = quantize_per_tensor_absmax(module.weight)
        else:
            raise ValueError(
                'weight_quant must be "per_channel" or "per_tensor"')
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
            self.activation_fake_quantizer = fake_quantize_activation_per_token_absmax
        elif act_quant == 'per_tensor':
            self.activation_fake_quantizer = fake_quantize_activation_per_tensor_absmax
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
        new_module.weight, new_module.weight_scales = quantize_weight_per_channel_absmax(
            module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return super().__repr__() + f'({self.in_features}, {self.out_features}, bias={self.bias is not None})'
