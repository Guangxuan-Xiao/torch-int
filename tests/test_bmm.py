import torch
from torch_int._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t
from icecream import ic


@torch.no_grad()
def test_bmm_s8t_s8n_s8t():
    # used by attn_prob x value
    B, M, K, N = 1, 512, 512, 32
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8)
    scale = 0.001
    c = bmm_s8t_s8n_s8t(a.cuda(), b.cuda(), scale)
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)) * scale
    c_gt = c_gt.clamp(-128, 127).round().to(torch.int8)
    ic(torch.allclose(c_gt, c.cpu()))


@torch.no_grad()
def test_bmm_s8t_s8n_s8t_2():
    # used by attn_prob x value
    B, M, K, N = 1, 32, 32, 16
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, K, N), dtype=torch.int8)
    scale = 0.0001
    c_gt = torch.bmm(a.float(), b.float()) * scale
    c_gt = c_gt.clamp(-128, 127).round().to(torch.int8)
    ic(c_gt)
    b1 = b.cuda().transpose(1, 2)
    c1 = bmm_s8t_s8n_s8t(a.cuda(), b1, scale)
    ic(c1)
    ic(torch.mean((c_gt.float() - c1.cpu().float()) ** 2))
    b2 = b.cuda().transpose(1, 2).contiguous()
    c2 = bmm_s8t_s8n_s8t(a.cuda(), b2, scale)
    ic(c2)
    ic(torch.mean((c_gt.float() - c2.cpu().float()) ** 2))
    b3 = b.cuda().transpose(1, 2).contiguous().clone()
    c3 = bmm_s8t_s8n_s8t(a.cuda(), b3, scale)
    ic(c3)
    ic(torch.mean((c_gt.float() - c3.cpu().float()) ** 2))


# (A_row V_row)_row ^ T = (V_row ^T A_row ^T)_row = (V^T_row A_col)_row
# (A_row V_row)_row = (A_row V_col ^T)_row
@torch.no_grad()
def test_bmm_s8t_s8n_s32t():
    # used by query x key
    B, M, K, N = 1, 512, 512, 12288
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (B, N, K), dtype=torch.int8)
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)
                     ).round().to(torch.int32)
    c = bmm_s8t_s8n_s32t(a.cuda(), b.cuda())
    ic(torch.allclose(c_gt, c.cpu()))


if __name__ == '__main__':
    # print('test_bmm_s8t_s8n_s8t')
    # test_bmm_s8t_s8n_s8t()
    print('test_bmm_s8t_s8n_s8t_2')
    test_bmm_s8t_s8n_s8t_2()
    # print('test_bmm_s8t_s8n_s32t')
    # test_bmm_s8t_s8n_s32t()
