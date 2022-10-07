#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"
#include <cuda.h>
#include <torch/torch.h>


torch::Tensor igemm(torch::Tensor A, torch::Tensor B)
 {
    // Input: A, B are 2D signed 8-bit integer tensors
    // Output: C is 2D signed 32-bit integer tensor
    // B is transposed
    
    const int alpha = 1;
    const int beta = 0; // C = alpha * A * B + beta * C


    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(0);
    
    cublasStatus_t stat;
    
    cudaDataType_t Atype = CUDA_R_8I;
    cudaDataType_t Btype = CUDA_R_8I;
    cudaDataType_t Ctype = CUDA_R_32I;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
    }
    
    // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    torch::Tensor C = torch::zeros({m, n}, torch::dtype(torch::kInt).device(torch::kCUDA));

    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A.data_ptr(), Atype, m, B.data_ptr(), Btype, k, &beta, C.data_ptr(), Ctype, m, computeType, algo);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUBLAS GEMM failed, error code: " + std::string(cublasGetStatusString(stat)));
    }

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("igemm", &igemm, "igemm (CUDA)");
}