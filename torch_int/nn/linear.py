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
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.randint(-127, 127, (self.in_features, self.out_features), dtype=torch.int8)
        if self.bias is not None:
            self.bias.data.zero_()

    def to(self, *args, **kwargs):
        super(Int8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, input):
        return gemm(input, self.weight, False, False) + self.bias
