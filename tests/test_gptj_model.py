import torch
from transformers.models.gptj.modeling_gptj import GPTJModel, GPTJConfig
from torch_int.models.gptj import Int8GPTJModel
from icecream import ic
from functools import partial


def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_gptj_model_layer():
    config = GPTJConfig.from_pretrained('Salesforce/codegen-350M-mono')

    B, L, D, H = 1, 256, config.n_embd, config.n_head

    x = torch.randint(0, config.vocab_size, (B, L))
    model = GPTJModel(config)
    model.eval()
    act_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = model(x)[0].cuda()
    decoder_layer_scales = []
    for idx in range(config.n_layer):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[f"h.{idx}.attn.q_proj"][0].abs(
        ).max() / 127
        scale_dict["q_output_scale"] = act_dict[f"h.{idx}.attn.q_proj"][1].abs(
        ).max() / 127
        scale_dict["k_output_scale"] = act_dict[f"h.{idx}.attn.k_proj"][1].abs(
        ).max() / 127
        scale_dict["v_output_scale"] = act_dict[f"h.{idx}.attn.v_proj"][1].abs(
        ).max() / 127
        scale_dict["out_input_scale"] = act_dict[f"h.{idx}.attn.out_proj"][0].abs(
        ).max() / 127
        scale_dict["fc1_input_scale"] = act_dict[f"h.{idx}.mlp.fc_in"][0].abs(
        ).max() / 127
        scale_dict["fc2_input_scale"] = act_dict[f"h.{idx}.mlp.fc_out"][0].abs(
        ).max() / 127
        decoder_layer_scales.append(scale_dict)

    int8_model = Int8GPTJModel.from_float(model, decoder_layer_scales).cuda()
    int8_model.eval()

    y_hat = int8_model(x.cuda())[0]

    # # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_gptj_model_layer()
