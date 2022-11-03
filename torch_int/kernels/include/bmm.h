#ifndef BMM_H
#define BMM_H
#include <torch/types.h>
torch::Tensor bmm_s8t_s8n_s8t(torch::Tensor A, torch::Tensor B,
                              float output_scale);

torch::Tensor bmm_s8t_s8n_s32t(torch::Tensor A, torch::Tensor B);

#endif // BMM_H