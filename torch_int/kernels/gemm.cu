#include "include/gemm.h"
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
                          bool transb) {
  // Input: A, B are 2D signed 8-bit integer tensors
  // Output: C is 2D signed 32-bit integer tensor

  const int alpha = 1;
  const int beta = 0; // C = alpha * A * B + beta * C

  cublasStatus_t stat;

  cudaDataType_t Atype = CUDA_R_8I;
  cudaDataType_t Btype = CUDA_R_8I;
  cudaDataType_t Ctype = CUDA_R_32I;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

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

torch::Tensor gemm_cutlass(torch::Tensor input, torch::Tensor weight) {

  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 16>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  auto out = torch::empty({M, N}, torch::dtype(torch::kInt32).device(device));

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,            // <- reference to matrix C on device
      out_ref,   // <- reference to matrix D on device
      {alpha, beta},
      1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass gemm failed, error code");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass gemm failed, error code");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass gemm failed, error code");
  }

  return out;
}

