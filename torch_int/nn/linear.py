import torch
from .._CUDA import gemm

class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Int8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randint(-127, 127, (self.in_features, self.out_features), dtype=torch.int8)
        if bias:
            self.bias = torch.zeros(out_features, dtype=torch.float16)
        else:
            self.register_parameter('bias', None)

    def to(self, *args, **kwargs):
        super(Int8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, input):
        # A, B, transa, transb
        return gemm(input, self.weight, False, False) + self.bias

    @staticmethod
    def from_float(module):
        assert isinstance(module, torch.nn.Linear)
        new_module = Int8Linear(module.in_features, module.out_features, module.bias is not None)
        new_module.weight = module.weight.to(torch.int8)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.int8)
        return new_module
