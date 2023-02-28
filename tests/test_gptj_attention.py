import torch
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJConfig
from torch_int.models.gptj import Int8GPTJAttention 
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from typing import Tuple
from icecream import ic
from functools import partial

def store_act(module, x, y, act_dict, name):
    # print(f"{name}: {y.mean()}")
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_gptj_attention():
    B, L, D, H = 1, 32, 128, 1    
    x = torch.randn(B, L, D)
    x_scale = x.abs().max() / 127
    config = GPTJConfig()
    config.n_embd = D
    config.n_head = H
    config.rotary_dim = None
    attn = GPTJAttention(config)
    attn.eval()
    act_dict = {}
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = attn(x)
    y = y[0]

    q_output_scale = act_dict['q_proj'][1].abs().max() / 127
    k_output_scale = act_dict['k_proj'][1].abs().max() / 127
    v_output_scale = act_dict['v_proj'][1].abs().max() / 127
    out_input_scale = act_dict['out_proj'][0].abs().max() / 127
    int8_attn = Int8GPTJAttention.from_float(
        attn, x_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale).cuda()
    int8_attn.eval()
    q_act_dict = {}
    for name, module in int8_attn.named_modules():
        if isinstance(module, (W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU)):
            module.register_forward_hook(
                partial(store_act, act_dict=q_act_dict, name=name))
    q_x = (x / x_scale).round().to(torch.int8)
    y_hat = int8_attn(q_x.cuda())[0].cpu()

    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_gptj_attention()
