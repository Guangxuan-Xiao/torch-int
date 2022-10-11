#include "include/quantization.h"
#include <torch/torch.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_tensor(torch::Tensor input) {
  auto max_val = input.max();
  max_val = torch::clamp(max_val, 1e-8) / 127.0;
  auto quantized =
      torch::clamp(input / max_val, -128, 127).round().to(torch::kInt8);
  return std::make_tuple(quantized, max_val);
}

std::tuple<torch::Tensor, torch::Tensor>
quantize_activation_per_token(torch::Tensor input) {
  auto max_val = std::get<0>(input.max(1, true));
  max_val = torch::clamp(max_val, 1e-8) / 127.0;
  auto quantized =
      torch::clamp(input / max_val, -128, 127).round().to(torch::kInt8);
  return std::make_tuple(quantized, max_val);
}

torch::Tensor dequantize_activation_per_tensor(torch::Tensor quantized,
                                               torch::Tensor w_scale,
                                               torch::Tensor a_scale) {
  auto dtype = a_scale.dtype();
  quantized = quantized.to(torch::kFloat);
  auto dequantized = quantized * w_scale.reshape({1, -1}) * a_scale;
  return dequantized.to(dtype);
}

torch::Tensor dequantize_activation_per_token(torch::Tensor quantized,
                                              torch::Tensor w_scale,
                                              torch::Tensor a_scale) {
  auto dtype = a_scale.dtype();
  quantized = quantized.to(torch::kFloat);
  auto dequantized =
      quantized * w_scale.reshape({1, -1}) * a_scale.reshape({-1, 1});
  return dequantized.to(dtype);
}
