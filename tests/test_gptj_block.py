import torch
from transformers.models.gptj.modeling_gptj import GPTJBlock, GPTJConfig
from torch_int.models.gptj import Int8GPTJBlock 
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from typing import Tuple
from icecream import ic
from functools import partial
import matplotlib.pyplot as plt

def store_act(module, x, y, act_dict, name):
    # print(f"{name}: {y.mean()}")
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_gptj_block():
    config : GPTJConfig = GPTJConfig.from_pretrained('Salesforce/codegen-350M-mono')
    B, L, D, H = 1, 256, config.n_embd, config.n_head
    x = torch.randn(B, L, D)*10
    blk = GPTJBlock(config)
    blk.eval()
    act_dict = {}
    for name, module in blk.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
        if isinstance(module, torch.nn.LayerNorm):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))

    y = blk(x)
    y = y[0].cpu()
    print(act_dict.keys())
    # exit(0)
    ln1_input_scale = act_dict['ln_1'][1].abs().max() / 127
    attn_input_scale = act_dict['attn.q_proj'][0].abs().max() / 127
    q_output_scale =   act_dict['attn.q_proj'][1].abs().max() / 127
    k_output_scale =   act_dict['attn.k_proj'][1].abs().max() / 127
    v_output_scale =   act_dict['attn.v_proj'][1].abs().max() / 127
    out_input_scale =  act_dict['attn.out_proj'][0].abs().max() / 127
    fc1_input_scale = act_dict['mlp.fc_in'][0].abs().max() / 127
    fc2_input_scale = act_dict['mlp.fc_out'][0].abs().max() / 127
    int8_blk = Int8GPTJBlock.from_float(
        blk, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale, fc1_input_scale, fc2_input_scale).cuda()
    int8_blk.eval()
    q_act_dict = {}

    y_hat = int8_blk(x.cuda())[0].cpu()
    # rd = blk.dbgi
    # md = int8_blk.dbgi
    # RN = 256
    # ra = rd['atto'].cpu().flatten()[:RN]
    # ma = md['attoX'].cpu().flatten()[:RN]
    # rf = rd['ffn'].cpu().flatten()[:RN]
    # mf = md['ffnX'].cpu().flatten()[:RN]
    # rr = rd['resi'].cpu().flatten()[:RN]
    # mr = md['resiX'].cpu().flatten()[:RN]
    #
    # plt.plot(ra.flatten())
    # print(f"MAX: a:{ra.abs().max()} f:{rf.abs().max()} r:{rr.abs().max()+0.0000001}")
    # plt.plot(ma - ra, color='r')
    # plt.savefig("Xa.jpg", dpi=300)
    # plt.cla()
    # # plt.plot(rf)
    # plt.plot(mf - rf, color='r')
    # plt.savefig("Xf.jpg", dpi=300)
    # plt.cla()
    # # plt.plot(rr.flatten())
    # plt.plot(mr - rr, color='r')
    # plt.savefig("Xr.jpg", dpi=300)

    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_gptj_block()
