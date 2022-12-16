import torch
import math

from torch import nn
from torch.nn import functional as F

from transformers.models.bloom.modeling_bloom import (
    BloomConfig,
    BloomForCausalLM,
    BloomModel,
    BloomPreTrainedModel,
    BloomAttention,
    BloomMLP,
    BloomGelu,
    BloomBlock
)

from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)


class Int8BloomAttention(nn.Module):

    _split_heads = BloomAttention._split_heads
    _merge_heads = BloomAttention._merge_heads

    def __init__(self, config: BloomConfig):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.query_key_value = W8A8B8O8Linear(
            self.hidden_size, 3 * self.hidden_size)
        self.dense = W8A8BFP32OFP32Linear(self.hidden_size, self.hidden_size)

    @staticmethod
    @torch.no_grad()
    def from_float(module: BloomAttention,
                   input_scale: float,
                   qkv_output_scale: float,
                   dense_input_scale: float):
        int8_module = BloomAttention(module.embed_dim, module.num_heads)

        int8_module.query_key_value = W8A8B8O8Linear.from_float(
            module.query_key_value, input_scale, qkv_output_scale)
        int8_module.dense = W8A8BFP32OFP32Linear.from_float(
            module.dense, dense_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            qkv_output_scale, qkv_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, qkv_output_scale, dense_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = self.query_key_value(hidden_states)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        # matmul_result = alibi.baddbmm(
        #     batch1=query_layer,
        #     batch2=key_layer,
        #     beta=self.beta,
        #     alpha=self.inv_norm_factor,
        # )
        matmul_result = alibi * self.beta + self.qk_bmm(
            query_layer, key_layer) * self.inv_norm_factor

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(
            attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        output_tensor = output_tensor + residual

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class Int8BloomMLP(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = intermediate_output + residual

        return output


class Int8BloomBlock(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        if config.apply_residual_connection_post_layernorm:
            raise NotImplementedError('Not implemented yet.')

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
