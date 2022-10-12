#ifndef FUSED_H
#define FUSED_H

#include <torch/types.h>

torch::Tensor fused_qdq_linear(torch::Tensor input, torch::Tensor weight,
                               torch::Tensor bias, torch::Tensor scale,
                               torch::Tensor zero_point);

#endif // FUSED_H