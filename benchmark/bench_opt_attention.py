import torch
from torch_int.models.opt import Int8OPTAttention
from transformers.models.opt.modeling_opt import OPTAttention
from utils import bench_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=12288)
    parser.add_argument('--num-heads', type=int, default=96)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()

    B, L, D, H = args.batch_size, args.seq_len, args.hidden_dim, args.num_heads
    print(
        f'batch_size: {B}, seq_len: {L}, hidden_dim: {D}, num_heads: {H}, precision: {args.precision}')
    if args.precision == 'int8':
        dummy_input = torch.randint(-127, 127,
                                    (B, L, D), dtype=torch.int8)
        model = Int8OPTAttention(D, H)
    elif args.precision == 'fp16':
        dummy_input = torch.randn(B, L, D, dtype=torch.float16)
        model = OPTAttention(D, H).half()
    else:
        raise NotImplementedError
    bench_model(model, dummy_input, num_iter=100)
