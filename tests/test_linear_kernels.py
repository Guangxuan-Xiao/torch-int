import torch
from torch_int._CUDA import linear_a8_w8_b32_o32, linear_relu_a8_w8_b8_o8, linear_a8_w8_b8_o8
from icecream import ic


@torch.no_grad()
def test_quant_linear_a8_w8_b32_o32():
    B, M, N = 128, 1024, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-65536, 65535, (N,), dtype=torch.int32)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float()
    linear.bias.data = bias.float()
    y_gt = linear(x.float())
    y = linear_a8_w8_b32_o32(x.cuda(), weight.cuda(), bias.cuda())
    ic(torch.allclose(y_gt, y.float().cpu(), atol=1e-3))


@torch.no_grad()
def test_quant_linear_a8_w8_b8_o8():
    B, M, N = 128, 128, 128
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    output_scale, bias_scale = 0.01, 0.02
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * output_scale
    linear.bias.data = bias.float() * bias_scale
    y_gt = linear(x.float()).clamp(-128, 127).long()
    ic(y_gt)
    y = linear_a8_w8_b8_o8(x.cuda(), weight.cuda(),
                           bias.cuda(), output_scale, bias_scale)
    ic(y)
    ic(torch.allclose(y_gt, y.float(), atol=1e-3))


@torch.no_grad()
def test_quant_linear_relu_a8_w8_b8_o8():
    B, M, N = 128, 128, 128
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8).cuda()
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8).cuda()
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8).cuda()
    output_scale, bias_scale = 0.1, 0.2
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * output_scale
    linear.bias.data = bias.float() * bias_scale
    linear.cuda()
    y_gt = linear(x.float()).clamp(min=0)
    y = linear_a8_w8_b8_o8(x, weight, bias, output_scale, bias_scale)
    ic(y_gt)
    ic(y)
    ic(torch.allclose(y_gt, y.float(), atol=1e-3))


if __name__ == '__main__':
    print('test_quant_linear_a8_w8_b32_o32')
    test_quant_linear_a8_w8_b32_o32()
    # print('test_quant_linear_a8_w8_b8_o8')
    # test_quant_linear_a8_w8_b8_o8()
    # print('test_quant_linear_relu_a8_w8_b8_o8')
    # test_quant_linear_relu_a8_w8_b8_o8()
