import torch
from torch_int.nn import Int8Linear
from utils import bench_model
import argparse
from transformers import AutoModelForCausalLM
from ellm.tools.quantize_int import quantize_model_int
from tqdm import trange
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-125m')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--q', action='store_true')
    parser.add_argument('--a_bit', type=int, default=8)
    args = parser.parse_args()
    model_name = "/dataset/opt/{}".format(args.model)
    TIMES = args.num_iter
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16)

    if args.q:
        quantize_model_int(model, a_bit=args.a_bit)
        print(model)
    dummy_input = torch.randint(
        0, 50257, (args.batch_size, args.seq_len), dtype=torch.long)

    dummy_input = dummy_input.cuda()
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        print("benchmark")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in trange(TIMES):
            model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        print(
            f"Average inference time: {start.elapsed_time(end) / TIMES:.2f} ms")
