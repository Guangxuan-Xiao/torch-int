
import torch
from torch_int import gemm

def test_gemm():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    print(a)
    print(b)
    c_gt = torch.mm(a, b)
    print(c_gt)

    a = a.to(torch.int8).cuda()
    b = b.to(torch.int8).cuda()
    print(a)
    print(b)
    c = gemm(a, b)
    print(c)

if __name__ == '__main__':
    test_gemm()
