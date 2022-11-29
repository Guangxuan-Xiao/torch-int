import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
from torch_int.models.opt import Int8OPTDecoderLayer
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
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
def test_opt_decoder_layer():
    config = OPTConfig.from_pretrained('facebook/opt-125m')
    B, L, D, H = 1, 256, config.hidden_size, config.num_attention_heads

    x = torch.randn(B, L, D)
    layer = OPTDecoderLayer(config)
    layer.eval()
    act_dict = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = layer(x)[0]

    attn_input_scale = act_dict['self_attn.q_proj'][0].abs().max() / 127
    q_output_scale = act_dict['self_attn.q_proj'][1].abs().max() / 127
    k_output_scale = act_dict['self_attn.k_proj'][1].abs().max() / 127
    v_output_scale = act_dict['self_attn.v_proj'][1].abs().max() / 127
    out_input_scale = act_dict['self_attn.out_proj'][0].abs().max() / 127
    fc1_input_scale = act_dict['fc1'][0].abs().max() / 127
    fc2_input_scale = act_dict['fc2'][0].abs().max() / 127
    int8_layer = Int8OPTDecoderLayer.from_float(
        layer, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale, fc1_input_scale, fc2_input_scale).cuda()
    int8_layer.eval()

    y_hat = int8_layer(x.cuda())[0].cpu()

    # # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_opt_decoder_layer()
