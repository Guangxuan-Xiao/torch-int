import torch
from transformers.models.opt.modeling_opt import OPTDecoder, OPTConfig
from torch_int.models.opt import Int8OPTDecoder
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
    B, L, D, H = 1, 255, config.hidden_size, config.num_attention_heads

    x = torch.randint(0, config.vocab_size, (B, L)).cuda()
    decoder = OPTDecoder(config).cuda()
    decoder.eval()
    act_dict = {}
    for name, module in decoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = decoder(x)[0]

    decoder_layer_scales = []
    for idx in range(config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[f"layers.{idx}.self_attn.q_proj"][0].abs(
        ).max() / 127
        scale_dict["q_output_scale"] = act_dict[f"layers.{idx}.self_attn.q_proj"][1].abs(
        ).max() / 127
        scale_dict["k_output_scale"] = act_dict[f"layers.{idx}.self_attn.k_proj"][1].abs(
        ).max() / 127
        scale_dict["v_output_scale"] = act_dict[f"layers.{idx}.self_attn.v_proj"][1].abs(
        ).max() / 127
        scale_dict["out_input_scale"] = act_dict[f"layers.{idx}.self_attn.out_proj"][0].abs(
        ).max() / 127
        scale_dict["fc1_input_scale"] = act_dict[f"layers.{idx}.fc1"][0].abs(
        ).max() / 127
        scale_dict["fc2_input_scale"] = act_dict[f"layers.{idx}.fc2"][0].abs(
        ).max() / 127
        decoder_layer_scales.append(scale_dict)

    int8_decoder = Int8OPTDecoder.from_float(decoder, decoder_layer_scales).cuda()
    int8_decoder.eval()

    y_hat = int8_decoder(x.cuda())[0]

    # # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_opt_decoder_layer()
