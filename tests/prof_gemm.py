import torch
from torch_int._CUDA import gemm_cutlass, gemm_cublas

L, C1, C2 = 512, 12288, 12288
TIMES = 100000

torch.cuda.synchronize()
a, b = torch.randint(-127, 127, (L, C1), device='cuda:0'), torch.randint(-127, 127, (C1, C2), device='cuda:0')
a, b = a.to(torch.int8), b.to(torch.int8)

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()
# torch.cuda.synchronize()
# for _ in range(TIMES):
#     gemm_cublas(a, b, False, False)
# torch.cuda.synchronize()
# end.record()
# print('Cublas: ', start.elapsed_time(end))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
torch.cuda.synchronize()
for _ in range(TIMES):
    gemm_cutlass(a, b)
torch.cuda.synchronize()
end.record()
print('Cutlass: ', start.elapsed_time(end))


a, b = torch.randn(L, C1, device='cuda:0').half(), torch.randn(C1, C2, device='cuda:0').half()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
torch.cuda.synchronize()
for _ in range(TIMES):
    torch.mm(a, b)
torch.cuda.synchronize()
end.record()
print('Pytorch: ', start.elapsed_time(end))
