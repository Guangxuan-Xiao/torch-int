import torch
from torch_int._CUDA import linear_a8_w8_b32_o32, linear_relu_a8_w8_b8_o8, linear_a8_w8_b8_o8
from utils import bench_func_latency
import argparse


def bench_linear_a8_w8_b32_o32(precision, seq_len, c1, c2):
    if precision == 'int8':
        dummy_input = torch.randint(-127, 127,
                                    (seq_len, c1), dtype=torch.int8).cuda()
        weight = torch.randint(-127, 127, (c2, c1), dtype=torch.int8).cuda()
        bias = torch.randint(-65535, 65535, (c2,), dtype=torch.int32).cuda()
        args = (dummy_input, weight, bias)
        fn = linear_a8_w8_b32_o32
    elif precision == 'fp16':
        dummy_input = torch.randn(seq_len, c1).half().cuda()
        model = torch.nn.Linear(c1, c2).half().cuda()
        args = (dummy_input,)
        fn = model.forward
    else:
        raise NotImplementedError
    bench_func_latency(fn, args, num_iter=2000)


def bench_linear_a8_w8_b8_o8(precision, seq_len, c1, c2):
    if precision == 'int8':
        dummy_input = torch.randint(-127, 127,
                                    (seq_len, c1), dtype=torch.int8).cuda()
        weight = torch.randint(-127, 127, (c2, c1), dtype=torch.int8).cuda()
        bias = torch.randint(-127, 127, (c2,), dtype=torch.int8).cuda()
        args = (dummy_input, weight, bias, 0.001, 0.01)
        fn = linear_a8_w8_b8_o8
    elif precision == 'fp16':
        dummy_input = torch.randn(seq_len, c1).half().cuda()
        model = torch.nn.Linear(c1, c2).half().cuda()
        args = (dummy_input,)
        fn = model.forward
    else:
        raise NotImplementedError
    bench_func_latency(fn, args, num_iter=2000)


def bench_linear_relu_a8_w8_b8_o8(precision, seq_len, c1, c2):
    if precision == 'int8':
        dummy_input = torch.randint(-127, 127,
                                    (seq_len, c1), dtype=torch.int8).cuda()
        weight = torch.randint(-127, 127, (c2, c1), dtype=torch.int8).cuda()
        bias = torch.randint(-127, 127, (c2,), dtype=torch.int8).cuda()
        args = (dummy_input, weight, bias, 0.001, 0.01)
        fn = linear_relu_a8_w8_b8_o8
    elif precision == 'fp16':
        dummy_input = torch.randn(seq_len, c1).half().cuda()
        model = torch.nn.Sequential(
            torch.nn.Linear(c1, c2).half(),
            torch.nn.ReLU().half()
        ).cuda()
        args = (dummy_input,)
        fn = model.forward
    else:
        raise NotImplementedError
    bench_func_latency(fn, args, num_iter=2000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--C1', type=int, default=12288)
    parser.add_argument('--C2', type=int, default=12288)
    parser.add_argument('--precision', type=str, default='int8')
    parser.add_argument('--func', type=str, default='linear_a8_w8_b32_o32')
    args = parser.parse_args()

    SEQ_LEN = args.seq_len
    C1, C2 = args.C1, args.C2
    print('SEQ_LEN = ', SEQ_LEN)
    print('C1 = ', C1)
    print('C2 = ', C2)
    print('precision = ', args.precision)
    if args.func == 'linear_a8_w8_b32_o32':
        bench_linear_a8_w8_b32_o32(args.precision, SEQ_LEN, C1, C2)
    elif args.func == 'linear_a8_w8_b8_o8':
        bench_linear_a8_w8_b8_o8(args.precision, SEQ_LEN, C1, C2)
    elif args.func == 'linear_relu_a8_w8_b8_o8':
        bench_linear_relu_a8_w8_b8_o8(args.precision, SEQ_LEN, C1, C2)
    else:
        raise NotImplementedError
