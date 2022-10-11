#ifndef QUANTIZATION_H
#define QUANTIZATION_H
#include <torch/types.h>

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_tensor(torch::Tensor input);

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_token(torch::Tensor input);

torch::Tensor dequantize_activation_per_tensor(torch::Tensor quantized,
                                               torch::Tensor w_scale,
                                               torch::Tensor a_scale);

torch::Tensor dequantize_activation_per_token(torch::Tensor quantized,
                                              torch::Tensor w_scale,
                                              torch::Tensor a_scale);

#endif // !QUANTIZATION_H