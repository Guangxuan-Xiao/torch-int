#include "include/quantization.h"
#include <torch/torch.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_tensor(torch::Tensor input) {
  auto max_val = input.abs().max();
  max_val.clamp_(1e-8).div_(127.0);
  input.div_(max_val).round_().clamp_(-128, 127);
  auto quantized = input.to(torch::kInt8);
  return std::make_tuple(quantized, max_val);
}

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_token(torch::Tensor input) {
  auto max_val = std::get<0>(input.abs().max(-1, true));
  max_val.clamp_(1e-8).div_(127.0);
  input.div_(max_val).round_().clamp_(-128, 127);
  auto quantized = input.to(torch::kInt8);
  return std::make_tuple(quantized, max_val);
}

torch::Tensor dequantize_activation_per_tensor(torch::Tensor quantized,
                                               torch::Tensor w_scale,
                                               torch::Tensor a_scale) {
  auto dtype = a_scale.dtype();
  quantized = quantized.to(torch::kFloat);
  quantized.mul_(w_scale.reshape({1, -1})).mul_(a_scale);
  return quantized.to(dtype);
}

torch::Tensor dequantize_activation_per_token(torch::Tensor quantized,
                                              torch::Tensor w_scale,
                                              torch::Tensor a_scale) {
  auto dtype = a_scale.dtype();
  quantized = quantized.to(torch::kFloat);
  quantized.mul_(w_scale.reshape({1, -1})).mul_(a_scale.reshape({-1, 1}));
  return quantized.to(dtype);
}
