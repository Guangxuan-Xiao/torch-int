#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor gemm(torch::Tensor A, torch::Tensor B, bool transa, bool transb) {
  // Input: A, B are 2D signed 8-bit integer tensors
  // Output: C is 2D signed 32-bit integer tensor

  const int alpha = 1;
  const int beta = 0; // C = alpha * A * B + beta * C

  cublasStatus_t stat;

  cudaDataType_t Atype = CUDA_R_8I;
  cudaDataType_t Btype = CUDA_R_8I;
  cudaDataType_t Ctype = CUDA_R_32I;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("CUBLAS initialization failed");
  }

  torch::Device device = A.device();

  cublasOperation_t transa_ = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb_ = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

  int n = transa ? A.size(1) : A.size(0);
  int k = transa ? A.size(0) : A.size(1);
  int m = transb ? B.size(0) : B.size(1);

  int lda = transa ? n : k;
  int ldb = transb ? k : m;
  int ldc = m;

  torch::Tensor C =
      torch::zeros({n, m}, torch::dtype(torch::kInt).device(device));

  // Note that CuBLAS assumes column-major matrices, so we actually pass the
  // transposes of A and B
  stat = cublasGemmEx(handle, transb_, transa_, m, n, k, &alpha, B.data_ptr(),
                      Btype, ldb, A.data_ptr(), Atype, lda, &beta, C.data_ptr(),
                      Ctype, ldc, computeType, algo);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("CUBLAS GEMM failed, error code: " +
                             std::string(cublasGetStatusString(stat)));
  }

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, "GEMM (CUDA)");
}