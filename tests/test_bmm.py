import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t
from icecream import ic


@torch.no_grad()
def test_bmm():
    B, M, N = 8, 512, 12288
    a = torch.randint(-128, 127, (B, M, N), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, M, N), dtype=torch.int8)
    scale = 0.01
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)) * scale
    c_gt = c_gt.clamp(-128, 127).round()
    c = bmm_s8t_s8n_s8t(a.cuda(), b.cuda(), scale)
    ic(torch.allclose(c_gt, c.float().cpu(), atol=1e-3))


if __name__ == '__main__':
    test_bmm()
