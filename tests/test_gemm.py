
import torch
from torch_int import gemm

def test_gemm():
    a = torch.randint(-128, 127, (4, 8), dtype=torch.int8)
    b = torch.randint(-128, 127, (8, 12), dtype=torch.int8)

    a = a.to(torch.int8).cuda()
    b = b.to(torch.int8).cuda()
    print(a)
    print(b)
    transa, transb = False, False
    c = gemm(a, b, transa, transb)
    print(c)
    print(torch.allclose(c.float(), torch.mm(a.float(), b.float())))

    transa, transb = True, False
    c = gemm(a.T, b, transa, transb)
    print(c)

    transa, transb = False, True
    c = gemm(a, b.T, transa, transb)
    print(c)

    transa, transb = True, True
    c = gemm(a.T, b.T, transa, transb)
    print(c)

if __name__ == '__main__':
    test_gemm()
