#ifndef BMM_H
#define BMM_H
#include <torch/types.h>
torch::Tensor bmm_s8t_s8n_f32t(torch::Tensor A, torch::Tensor B, float alpha);

torch::Tensor baddmm_s8t_s8n_f32t(torch::Tensor A, torch::Tensor B,
                                  torch::Tensor bias, float alpha, float beta);

torch::Tensor bmm_s8t_s8n_s8t(torch::Tensor A, torch::Tensor B, float alpha);

torch::Tensor bmm_s8t_s8n_s32t(torch::Tensor A, torch::Tensor B);

#endif // BMM_H