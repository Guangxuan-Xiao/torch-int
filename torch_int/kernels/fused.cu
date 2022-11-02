#include "include/fused.h"
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

std::tuple<torch::Tensor,
           torch::Tensor> // (residual_output (FP32), ln_output (INT8))
dq_add_layernorm_q_int32_fp32_int8(
    torch::Tensor input,          // INT32
    float input_scale,            // FP32
    torch::Tensor residual_input, // FP32
    torch::Tensor gamma,          // FP32
    torch::Tensor beta            // FP32
    ) // The output scale has already been fused into gamma and beta
{
  // residual_output = residual_input + input * input_scale
  auto residual_output_fp32 = torch::add(residual_input, input, input_scale);

  auto ln_output_fp32 =
      torch::layer_norm(residual_output_fp32, {residual_output_fp32.size(-1)},
                        gamma, beta, 1e-5, false);
  ln_output_fp32.clamp_(-128, 127);
  auto ln_output_int8 = ln_output_fp32.to(torch::kChar);
  return std::make_tuple(residual_output_fp32, ln_output_int8);
}