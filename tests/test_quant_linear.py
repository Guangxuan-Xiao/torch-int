from multiprocessing import dummy
import torch
from torch_int.nn import Int8Linear

@torch.no_grad()
def test_quant_linear():
    linear = torch.nn.Linear(16, 32).cuda().half()
    int8_linear = Int8Linear.from_float(linear).cuda()
    dummy_input = torch.randn(16, 16).cuda().half()
    print('Float linear')
    y = linear(dummy_input)
    print(y)
    print('Int8 linear')
    q_y = int8_linear(dummy_input)
    print(q_y)
    print('MSE')
    print(torch.mean((y - q_y) ** 2))

if __name__ == '__main__':
    test_quant_linear()
