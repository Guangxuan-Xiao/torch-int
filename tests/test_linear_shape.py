import torch
from torch_int._CUDA import linear_a8_w8_b8_o8, linear_a8_w8_bfp32_ofp32
from icecream import ic



@torch.no_grad()
def test_quant_linear_a8_w8_bfp32_ofp32():
    B, M, N = 1, 16, 4
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randn(N, dtype=torch.float32)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.001, 1
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    y_gt = linear(x.float())
    y = linear_a8_w8_bfp32_ofp32(
        x.cuda(), weight.cuda(), bias.cuda(), alpha, beta)
    print(y.shape)
    ic(torch.allclose(y_gt, y.cpu(), atol=0.5))


@torch.no_grad()
def test_quant_linear_a8_w8_b8_o8():
    B, M, N = 1, 16, 16
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.001, 0.01
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    y_gt = linear(x.float()).clamp(-128, 127).round().long()
    y = linear_a8_w8_b8_o8(x.cuda(), weight.cuda(),
                           bias.cuda(), alpha, beta).cpu().long()
    ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=1))


if __name__ == '__main__':
    print('test_quant_linear_a8_w8_bfp32_ofp32')
    test_quant_linear_a8_w8_bfp32_ofp32()
    print('test_quant_linear_a8_w8_b8_o8')
    test_quant_linear_a8_w8_b8_o8()