
import torch
from torch_int import gemm


def test_gemm():
    RANGE = 127
    M, N, K = 128, 128, 128
    a = torch.randint(-RANGE, RANGE, (M, K), dtype=torch.int8)
    b = torch.randint(-RANGE, RANGE, (K, N), dtype=torch.int8)

    a = a.to(torch.int8).cuda()
    b = b.to(torch.int8).cuda()

    c_gt = torch.mm(a.float(), b.float())

    transa, transb = False, False
    c = gemm(a, b, transa, transb)

    print(torch.allclose(c_gt, c.float(), atol=1e-3))

    transa, transb = True, False
    c = gemm(a.T.contiguous(), b, transa, transb)
    print(torch.allclose(c_gt, c.float(), atol=1e-3))

    transa, transb = False, True
    c = gemm(a, b.T.contiguous(), transa, transb)
    print(torch.allclose(c_gt, c.float(), atol=1e-3))

    transa, transb = True, True
    c = gemm(a.T.contiguous(), b.T.contiguous(), transa, transb)
    print(torch.allclose(c_gt, c.float(), atol=1e-3))


if __name__ == '__main__':
    test_gemm()
