import torch
from torch_int.models.opt import Int8OPTDecoder, Int8OPTForCausalLM, Int8OPTModel
from transformers.models.opt.modeling_opt import OPTDecoder, OPTConfig, OPTForCausalLM, OPTModel
from transformers import AutoTokenizer
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T

from utils import bench_model
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt-13b')
    parser.add_argument('--precision', type=str, default='int8-fp16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--model-path-prefix', type=str,
                        default='/dataset/opt')
    args = parser.parse_args()

    print(args)
    model_path = f'{args.model_path_prefix}/{args.model}'
    config = OPTConfig.from_pretrained(model_path)
    with open('benchmark/example_text.txt', 'r') as f:
        example_text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(example_text, return_tensors='pt')
    # expand the input to the desired sequence length and batch size
    input_ids = input_ids[:, :args.seq_len]
    input_ids = input_ids.repeat(args.batch_size, 1)

    if args.precision == 'int8-fp32':
        model_path += '-int8'
        model = Int8OPTModel.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.float32)
    elif args.precision == 'int8-fp16':
        model_path += '-int8'
        model = Int8OPTModel.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.half)
    elif args.precision == 'fp16':
        model = OPTModel.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.half)
    elif args.precision == 'llm_int8':
        model = OPTModel.from_pretrained(
            model_path, device_map='auto', load_in_8bit=True)
    elif args.precision == 'llm_int8_0':
        model = OPTModel.from_pretrained(
            model_path, device_map='auto', load_in_8bit=True)
        from bitsandbytes.nn import Linear8bitLt
        for name, module in model.named_modules():
            if isinstance(module, Linear8bitLt):
                module.state.threshold = 0
    elif args.precision == 'int8-fp16-dynamic-a-token':
        model_path += '-int8'
        model = Int8OPTModel.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.half)
        for name, module in model.named_modules():
            if isinstance(module, (W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU)):
                module.old_forward = module.forward

                def get_new_forward(self):
                    def new_forward(input):
                        absmax = input.abs().max(dim=-1, keepdim=True)[0]
                        output = self.old_forward(input)
                        return output
                    return new_forward
                module.forward = get_new_forward(module)
    elif args.precision == 'int8-fp16-dynamic-a-tensor':
        model_path += '-int8'
        model = Int8OPTModel.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.half)
        for name, module in model.named_modules():
            if isinstance(module, (W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU)):
                module.old_forward = module.forward

                def get_new_forward(self):
                    def new_forward(input):
                        absmax = input.abs().max()
                        output = self.old_forward(input)
                        return output
                    return new_forward
                module.forward = get_new_forward(module)
    else:
        raise NotImplementedError
    bench_model(model, (input_ids, ), num_iter=1000)
