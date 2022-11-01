#ifndef LINEAR_H
#define LINEAR_H

// used by fc1
torch::Tensor linear_relu_a8_w8_b32_o8(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // INT32
                                       float output_scale    // FP32
);

// used by out_proj and fc2
torch::Tensor linear_a8_w8_b32_o32(torch::Tensor input,  // INT8
                                   torch::Tensor weight, // INT8
                                   torch::Tensor bias    // INT32
);

// used by q_proj, k_proj, v_proj
torch::Tensor linear_a8_w8_b32_o8(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor bias,   // FP32
                                  float output_scale    // FP32
);

#endif // LINEAR_H