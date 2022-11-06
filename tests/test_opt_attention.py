import torch
from transformers.models.opt.modeling_opt import OPTAttention
from torch_int.nn.opt import Int8OPTAttention
from torch_int.nn.linear import W8A8B32O32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from typing import Tuple
from icecream import ic
from functools import partial


def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_opt_attention():
    B, L, D, H = 1, 16, 16, 1
    x = torch.randn(B, L, D)
    x_scale = x.abs().max() / 127
    attn = OPTAttention(D, H)
    attn.eval()
    act_dict = {}
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = attn(x)[0]

    y_scale = y.abs().max() / 127

    q_output_scale = act_dict['q_proj'][1].abs().max() / 127
    k_output_scale = act_dict['k_proj'][1].abs().max() / 127
    v_output_scale = act_dict['v_proj'][1].abs().max() / 127
    out_input_scale = act_dict['out_proj'][0].abs().max() / 127
    int8_attn = Int8OPTAttention.from_float(
        attn, x_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale, y_scale).cuda()
    int8_attn.eval()
    q_act_dict = {}
    for name, module in int8_attn.named_modules():
        if isinstance(module, (W8A8B32O32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU)):
            module.register_forward_hook(
                partial(store_act, act_dict=q_act_dict, name=name))
    q_x = (x / x_scale).round().to(torch.int8)
    q_y = int8_attn(q_x.cuda())[0].cpu()
    # ic(q_y)
    y_hat = q_y * y_scale
    # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)
    for key in q_act_dict:
        q_x, q_y = q_act_dict[key]
        x, y = act_dict[key]
        if key == 'q_proj':
            y *= attn.scaling
        x_scale = x.abs().max() / 127
        y_scale = y.abs().max() / 127
        r2_x = (q_x.cpu() * x_scale - x).pow(2).mean() / x.pow(2).mean()
        r2_y = (q_y.cpu() * y_scale - y).pow(2).mean() / y.pow(2).mean()
        ic(key, r2_x, x_scale, r2_y, y_scale)

if __name__ == '__main__':
    test_opt_attention()
