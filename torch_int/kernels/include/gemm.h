#ifndef GEMM_H
#define GEMM_H
#include "cublas_v2.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/torch.h>

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

torch::Tensor gemm_cublas(torch::Tensor A, torch::Tensor B, bool transa,
                          bool transb);

torch::Tensor gemm_cutlass(torch::Tensor input, torch::Tensor weight);


#endif // !GEMM_H
