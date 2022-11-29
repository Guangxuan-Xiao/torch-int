import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_f32t
from icecream import ic

@torch.no_grad()
def test_bmm_s8t_s8n_f32t_shape():
    # used by attn_prob x value
    B, M, K, N = 1, 1, 16, 4
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8).cuda()
    scale = 0.001
    c = bmm_s8t_s8n_f32t(a, b, scale)
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)) * scale
    ic(torch.mean((c_gt - c) ** 2))

@torch.no_grad()
def test_bmm_s8t_s8n_s8t_shape():
    # used by attn_prob x value
    B, M, K, N = 1, 1, 16, 16
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8)
    scale = 0.001
    c = bmm_s8t_s8n_s8t(a.cuda(), b.cuda(), scale)
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)) * scale
    c_gt = c_gt.clamp(-128, 127).round().to(torch.int8)
    ic(torch.allclose(c_gt, c.cpu()))
    
if __name__ == '__main__':
    test_bmm_s8t_s8n_f32t_shape()
    test_bmm_s8t_s8n_s8t_shape()