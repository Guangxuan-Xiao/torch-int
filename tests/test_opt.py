import torch
from torch_int.models.opt import Int8OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from icecream import ic
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm


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
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last token of the input.
        total, hit = 0, 0
        pbar = tqdm(self.dataset, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            # label is the last token which is not the padding token
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            pbar.set_postfix({'acc': hit / total})
        acc = hit / total
        return acc


@torch.no_grad()
def test_opt():
    dataset = load_dataset('lambada', split='validation[:1000]')
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-13b')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')
    int8_model_path = '/dataset/opt/opt-13b-smoothquant'
    # precision = 'fp16'
    precision = 'int8'
    if precision == 'int8':
        model = Int8OPTForCausalLM.from_pretrained(int8_model_path,
                                                   device_map='auto', torch_dtype=torch.float16)
    else:
        model = OPTForCausalLM.from_pretrained('facebook/opt-13b',
                                               device_map='auto',
                                               torch_dtype=torch.float16)
    acc = evaluator.evaluate(model)
    ic(acc)


if __name__ == '__main__':
    test_opt()
