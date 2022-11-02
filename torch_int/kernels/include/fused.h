#ifndef FUSED_H
#define FUSED_H

#include <torch/types.h>

std::tuple<torch::Tensor,
           torch::Tensor> // (residual_output (FP32), ln_output (INT8))
dq_add_layernorm_q(torch::Tensor input,          // INT32
                   float input_scale,            // FP32
                   torch::Tensor residual_input, // FP32
                   torch::Tensor gamma,          // FP32
                   torch::Tensor beta,           // FP32
                   float epsilon                 // FP32
); // The output scale has already been fused into gamma and beta

#endif // FUSED_H