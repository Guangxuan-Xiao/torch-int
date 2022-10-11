#include <torch/extension.h>
#include "include/gemm.h"
#include "include/quantization.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_cutlass", &gemm_cutlass, "GEMM (CUTLASS)");
  m.def("gemm_cublas", &gemm_cublas, "GEMM (CUBLAS)");
  m.def("quantize_activation_per_tensor", &quantize_activation_per_tensor, "Quantize activation per tensor");
  m.def("quantize_activation_per_token", &quantize_activation_per_token, "Quantize activation per token");
  m.def("dequantize_activation_per_tensor", &dequantize_activation_per_tensor, "Dequantize activation per tensor");
  m.def("dequantize_activation_per_token", &dequantize_activation_per_token, "Dequantize activation per token");
}
