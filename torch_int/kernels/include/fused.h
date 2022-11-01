#ifndef FUSED_H
#define FUSED_H

#include <torch/types.h>

std::tuple<torch::Tensor, torch::Tensor> // (residual_output, ln_output)
dq_add_layernorm_q_int32_fp32_int8(torch::Tensor input,          // INT32
                                   float input_scale,            // FP32
                                   torch::Tensor residual_input, // FP32
                                   torch::Tensor gamma,          // FP32
                                   torch::Tensor beta            // FP32
); // The output scale has already been fused into gamma and beta

#endif // FUSED_H