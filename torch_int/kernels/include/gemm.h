#ifndef GEMM_H
#define GEMM_H
#include <torch/types.h>

torch::Tensor gemm_cublas(torch::Tensor A, torch::Tensor B, bool transa,
                          bool transb);

torch::Tensor gemm_cutlass(torch::Tensor input, torch::Tensor weight);


#endif // !GEMM_H
