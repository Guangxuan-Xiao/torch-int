#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32(torch::Tensor input,  // INT8
                                   torch::Tensor weight, // INT8
                                   torch::Tensor bias    // INT32
);

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32_with_scaling(torch::Tensor input,  // INT8
                                                torch::Tensor weight, // INT8
                                                torch::Tensor bias,   // INT32
                                                float output_scale,   // FP32
                                                float bias_scale      // FP32
);

// used by fc1, return INT8
torch::Tensor linear_relu_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
                                      torch::Tensor bias,   // INT8
                                      float output_scale,   // FP32
                                      float bias_scale      // FP32
);

// used by q_proj, k_proj, v_proj, return INT8
torch::Tensor linear_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                 torch::Tensor weight, // INT8
                                 torch::Tensor bias,   // INT8
                                 float output_scale,   // FP32
                                 float bias_scale      // FP32
);

#endif // LINEAR_HS