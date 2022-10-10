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
    print('SEQ_LEN = ', SEQ_LEN)
    print('C1 = ', C1)
    print('C2 = ', C2)
    print('precision = ', args.precision)
    if args.precision == 'int8':
        dummy_input = torch.randint(-127, 127, (SEQ_LEN, C1), dtype=torch.int8)
        model = Int8Linear(C1, C2)
    elif args.precision == 'fp16':
        dummy_input = torch.randn(SEQ_LEN, C1).half()
        model = torch.nn.Linear(C1, C2).half()
    else:
        raise NotImplementedError
    bench_model(model, dummy_input, num_iter=10000)
    # Results on V100:
    # Int8Linear:
    # Average inference time: 5.62 ms
    # Peak memory usage: 1052.06 MB
    # FP16Linear:
    # Average inference time: 6.98 ms
    # Peak memory usage: 2064.06 MB
