import torch
from torch_int.nn import Int8Linear
from utils import bench_model
import argparse

@torch.no_grad()
def bench_quant_linear():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--C1', type=int, default=12288)
    parser.add_argument('--C2', type=int, default=12288)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()
    bench_quant_linear()
