from ast import arg
import torch
from torch_int.nn import Int8Linear
from utils import bench_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=32768)  # 32K
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()

    SEQ_LEN = args.seq_len
    HIDDEN_SIZE = args.hidden_dim
    dummy_input = torch.randn(SEQ_LEN, HIDDEN_SIZE).half()

    if args.precision == 'int8':
        model_int8 = Int8Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        print("Int8Linear:")
        t_int8, m_int8 = bench_model(model_int8, dummy_input.to(torch.int8))
    elif args.precision == 'fp16':
        model_fp16 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE).half()
        print("FP16Linear:")
        t_fp16, m_fp16 = bench_model(model_fp16, dummy_input)
    else:
        raise NotImplementedError

    # Results on V100:
    # Int8Linear:
    # Average inference time: 5.63 ms
    # Peak memory usage: 1052.06 MB
    # FP16Linear:
    # Average inference time: 6.90 ms
    # Peak memory usage: 3088.12 MB
    # Int8Linear is 1.23x faster than FP16Linear
    # Int8Linear uses 2.94x less memory than FP16Linear
