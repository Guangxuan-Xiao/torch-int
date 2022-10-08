from ast import arg
import torch
from torch_int.nn import Int8Linear
from utils import bench_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--C1', type=int, default=12288)
    parser.add_argument('--C2', type=int, default=12288)
    parser.add_argument('--precision', type=str, default='int8')
    args = parser.parse_args()

    SEQ_LEN = args.seq_len
    C1, C2 = args.C1, args.C2

    if args.precision == 'int8':
        dummy_input = torch.randint(-127, 127, (SEQ_LEN, C1))
        model_int8 = Int8Linear(C1, C2)
        print("Int8Linear:")
        t_int8, m_int8 = bench_model(model_int8, dummy_input)
    elif args.precision == 'fp16':
        dummy_input = torch.randn(SEQ_LEN, C1).half()
        model_fp16 = torch.nn.Linear(C1, C2).half()
        print("FP16Linear:")
        t_fp16, m_fp16 = bench_model(model_fp16, dummy_input)
    else:
        raise NotImplementedError

    # Results on V100:
    # Int8Linear:
    # Average inference time: 5.62 ms
    # Peak memory usage: 1052.06 MB
    # FP16Linear:
    # Average inference time: 6.98 ms
    # Peak memory usage: 2064.06 MB
