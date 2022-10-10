
import torch
from torch_int._CUDA import gemm_cutlass as gemm
import torch.backends.cudnn as cudnn
from utils import bench_func


def test_gemm():
    RANGE = 127
    M, N, K = 128, 128, 128

    a = torch.randint(-RANGE, RANGE, (M, K), dtype=torch.int8)
    b = torch.randint(-RANGE, RANGE, (K, N), dtype=torch.int8)

    a = a.to(torch.int8).cuda()
    b = b.to(torch.int8).cuda()

    c_gt = torch.mm(a.float(), b.float())

    c = gemm(a, b.T.contiguous())

    print('C = A * B')
    print(torch.allclose(c_gt, c.float(), atol=1e-3))


def bench_gemm():
    RANGE = 127
    M, N, K = 256, 12288 * 4, 12288 # OPT-175B FC1

    a = torch.randint(-RANGE, RANGE, (M, K), dtype=torch.int8)
    b = torch.randint(-RANGE, RANGE, (K, N), dtype=torch.int8)

    a = a.to(torch.int8).cuda()
    b = b.to(torch.int8).cuda()

    transa, transb = False, False
    print('C = A * B')
    bench_func(gemm, (a, b), num_iter=100)

    transa, transb = True, False
    print('C = A.T * B')
    bench_func(gemm, (a.T.contiguous(), b), num_iter=100)

    transa, transb = False, True
    print('C = A * B.T')
    bench_func(gemm, (a, b.T.contiguous()), num_iter=100)

    transa, transb = True, True
    print('C = A.T * B.T')
    bench_func(gemm, (a.T.contiguous(), b.T.contiguous()), num_iter=100)


if __name__ == '__main__':
    test_gemm()
    # bench_gemm()
