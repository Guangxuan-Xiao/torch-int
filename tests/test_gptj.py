import torch
from torch_int.models.gptj import Int8GPTJForCausalLM, Int8GPTJBlock, Int8GPTJMLP, Int8GPTJAttention, Int8GPTJModel
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJModel, GPTJConfig, GPTJForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from icecream import ic
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
# from transformers import GPTJTok
from datasets import load_dataset
from tqdm import tqdm
import json
import copy

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate2(self, model):
        model.eval()
        # The task is to predict the last token of the input.
        total, hit = 0, 0
        idx = 0
        pbar = tqdm(self.dataset, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids.cuda())
            idx += 1
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            pbar.set_postfix({'acc': hit / total})
        acc = hit / total
        return acc

    @torch.no_grad()
    def evaluate(self, modelX, model):
        model.eval()
        # The task is to predict the last token of the input.
        idx = 0
        total, hit = 0, 0
        hit2 = 0
        pbar = tqdm(self.dataset, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids.to('cuda'))
            outputs2 = modelX(input_ids.to('cuda'))
            model.transformer.d.clear()
            modelX.transformer.d.clear()
            idx += 1
            last_token_logits = outputs.logits[:, -2, :]
            last_token_logits = outputs2.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            pred2 = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            hit2 += (pred == label).sum().item()
            pbar.set_postfix({'acc': hit / total, 'accX': hit2 / total})
        acc = hit / total
        return acc

MP = "/home/iman/fgg/smoothquant/SF/codegen-350M-multiX.pt"
@torch.no_grad()
def test_opt():
    dataset = load_dataset('lambada', split='validation[:1000]')
    dataset = dataset.shuffle(seed=42)
    checkpoint = "moyix/codegen-350M-multi-gptj"
    # checkpoint = "Salesforce/codegen-350M-multi"
    config = GPTJConfig.from_pretrained('moyix/codegen-350M-multi-gptj')
    model = GPTJForCausalLM.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = 'auto').cuda()
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-multi')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')
    dlsj = "./tests/model_dec_scales.json"
    decoder_layer_scales = []
    with open(dlsj, 'r') as fp:
        decoder_layer_scales = json.load(fp)
    # these layers will not be quantized
    layers_to_keep = list(range(13))
    int8_model = Int8GPTJForCausalLM.from_float(model, decoder_layer_scales, k = layers_to_keep)
    acc = evaluator.evaluate2(int8_model.to('cuda'))
    ic(acc)


if __name__ == '__main__':
    test_opt()
