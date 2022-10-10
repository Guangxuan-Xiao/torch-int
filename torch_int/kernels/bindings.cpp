#include <torch/extension.h>
#include <torch/types.h>
#include "include/gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_cutlass", &gemm_cutlass, "GEMM (CUTLASS)");
  m.def("gemm_cublas", &gemm_cublas, "GEMM (CUTLASS)");
}