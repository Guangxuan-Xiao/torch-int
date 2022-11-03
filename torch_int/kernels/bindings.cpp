#include "include/bmm.h"
#include "include/fused.h"
#include "include/gemm.h"
#include "include/linear.h"
#include "include/quantization.h"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_cutlass", &gemm_cutlass, "GEMM (CUTLASS)");
  m.def("gemm_cublas", &gemm_cublas, "GEMM (CUBLAS)");
  m.def("quantize_activation_per_tensor", &quantize_activation_per_tensor,
        "Quantize activation per tensor");
  m.def("quantize_activation_per_token", &quantize_activation_per_token,
        "Quantize activation per token");
  m.def("dequantize_activation_per_tensor", &dequantize_activation_per_tensor,
        "Dequantize activation per tensor");
  m.def("dequantize_activation_per_token", &dequantize_activation_per_token,
        "Dequantize activation per token");
  m.def("linear_relu_a8_w8_b8_o8", &linear_relu_a8_w8_b8_o8,
        "Linear ReLU (INT8)");
  m.def("linear_a8_w8_b32_o32", &linear_a8_w8_b32_o32, "Linear (INT32)");
  m.def("linear_a8_w8_b8_o8", &linear_a8_w8_b8_o8, "Linear (INT8)");
  m.def("dq_add_layernorm_q", &dq_add_layernorm_q,
        "DQ + Add + LayerNorm (INT8)");
  m.def("bmm_s8t_s8n_s8t", &bmm_s8t_s8n_s8t, "BMM (INT8) A x B.T");
}
