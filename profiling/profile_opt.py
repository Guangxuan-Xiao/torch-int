import torch
from torch_int.models.opt import Int8OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig, OPTForCausalLM
import argparse
from utils import profile_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-13b')
    parser.add_argument('--precision', type=str, default='int8')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()

    print(args)
    model_path = f'/dataset/opt/{args.model}'
    config = OPTConfig.from_pretrained(model_path)
    input_ids = torch.randint(0, config.vocab_size,
                              (args.batch_size, args.seq_len))
    export_path = f'log/profiling/{args.model}/{args.precision}/B{args.batch_size}-L{args.seq_len}'
    if args.precision == 'int8-fp32':
        model = Int8OPTForCausalLM(config)
    elif args.precision == 'int8-fp16':
        model = Int8OPTForCausalLM(config).half()
    elif args.precision == 'fp16':
        model = OPTForCausalLM(config).half()
    else:
        raise NotImplementedError
    profile_model(model, (input_ids, ), export_path)
