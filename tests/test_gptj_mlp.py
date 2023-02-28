import torch
from transformers.models.gptj.modeling_gptj import GPTJMLP, GPTJConfig
from torch_int.models.gptj import Int8GPTJMLP 
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from typing import Tuple
from icecream import ic
from functools import partial
from torch_int.nn.fused import LayerNormQ
from torch.nn import LayerNorm

def store_act(module, x, y, act_dict, name):
    # print(f"{name}: {y.mean()}")
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_gptj_mlp():
    B, L, D, H = 1, 16, 32, 1    
    x = torch.randn(B, L, D)*40
    x = torch.clamp(x, -127, 127)
    x_scale = x.abs().max() / 127
    config = GPTJConfig()
    config.n_embd = D
    config.n_head = H
    intermediate_size = 4*D
    config.rotary_dim = None
    mlp = GPTJMLP(intermediate_size, config)
    mlp.eval()
    act_dict = {}
    for name, module in mlp.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = mlp(x)
    y = y[0]

    fc_in_scale = act_dict['fc_in'][0].abs().max() / 127
    fc_out_scale = act_dict['fc_out'][0].abs().max() / 127
    int8_mlp = Int8GPTJMLP.from_float(
        mlp, fc_in_scale, fc_out_scale).cuda()
    int8_mlp.eval()
    q_x = x.round().to(torch.int8)
    y_hat = int8_mlp(q_x.cuda()).cpu()
    print(y_hat.shape)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_gptj_mlp()
