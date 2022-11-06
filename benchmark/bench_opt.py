import torch
from torch_int.nn.opt import Int8OPTDecoder, Int8OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoder, OPTConfig, OPTForCausalLM
from utils import bench_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-125m')
    parser.add_argument('--precision', type=str, default='int8-fp32')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--model-path-prefix', type=str,
                        default='benchmark/opt_configs')
    args = parser.parse_args()

    print(args)
    model_path = f'{args.model_path_prefix}/{args.model}'
    config = OPTConfig.from_pretrained(model_path)
    input_ids = torch.randint(0, config.vocab_size,
                              (args.batch_size, args.seq_len))
    if args.precision == 'int8-fp32':
        model = Int8OPTForCausalLM(config)
    elif args.precision == 'int8-fp16':
        model = Int8OPTForCausalLM(config).half()
    elif args.precision == 'fp16':
        model = OPTForCausalLM(config).half()
    else:
        raise NotImplementedError
    bench_model(model, (input_ids, ), num_iter=500)
