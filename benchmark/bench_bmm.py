import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t
from utils import bench_func_latency
import argparse


def bench_bmm(precision, batch_size, seq_len, hidden_dim):
    if precision == 'int8':
        a = torch.randint(-128, 127, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int8).cuda()
        b = torch.randint(-128, 127, (batch_size, seq_len,
                          hidden_dim), dtype=torch.int8).cuda()
        scale = 0.01
        args = (a, b, scale)
        fn = bmm_s8t_s8n_s8t
    elif precision == 'fp16':
        a = torch.randn(batch_size, seq_len, hidden_dim).half().cuda()
        b = torch.randn(batch_size, seq_len,
                        hidden_dim).half().cuda().transpose(1, 2)
        args = (a, b)
        fn = torch.bmm
    else:
        raise NotImplementedError
    bench_func_latency(fn, args, num_iter=5000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=12288)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()
    print(f'B={args.batch_size}, L={args.seq_len}, H={args.hidden_dim}, precision={args.precision}')
    bench_bmm(args.precision, args.batch_size, args.seq_len, args.hidden_dim)
