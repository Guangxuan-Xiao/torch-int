import torch
from torch_int.models.opt import Int8OPTDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from utils import bench_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=12288)
    parser.add_argument('--ffn-dim', type=int, default=12288 * 4)
    parser.add_argument('--num-attention-heads', type=int, default=96)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()

    print(args)
    B, L, D = args.batch_size, args.seq_len, args.hidden_size
    if args.precision == 'int8':
        residual = torch.randn(B, L, D, dtype=torch.float16)
        hidden_states = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (B, L, D), dtype=torch.int32)
        inputs = (residual, hidden_states)
        model = Int8OPTDecoderLayer(args)
    elif args.precision == 'fp16':
        inputs = (torch.randn(B, L, D, dtype=torch.float16),)
        args.dropout = 0
        args.attention_dropout = 0
        args.do_layer_norm_before = True
        args.activation_function = 'relu'
        model = OPTDecoderLayer(args).half()
    else:
        raise NotImplementedError
    bench_model(model, inputs, num_iter=100)
