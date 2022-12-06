import torch
from torch import nn
from transformers.models.gptj.modeling_gptj import (
    GPTJConfig,
    GPTJForCausalLM,
    GPTJModel,
    GPTJPreTrainedModel,
    GPTJAttention,
    GPTJMLP,
    GPTJBlock,
    BaseModelOutputWithPast
)

from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class Int8GPTJAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, n_embd, n_head, max_position_embeddings, rotary_dim = None):
        super().__init__()

        max_positions = max_position_embeddings
        self.embed_dim = n_embd
        self.num_attention_heads  = n_head
        self.head_dim = n_embd // n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )

        if (self.head_dim * self.num_attention_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.attention_weight_scale = 1.0
        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)
        self.k_proj = W8A8B8O8Linear(n_embd, n_embd)
        self.v_proj = W8A8B8O8Linear(n_embd, n_embd)
        self.q_proj = W8A8B8O8Linear(n_embd, n_embd)
        self.out_proj = W8A8BFP32OFP32Linear(n_embd, n_embd)
        self.rotary_dim = None
        if rotary_dim is not None:
            self.rotary_dim = rotary_dim
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: GPTJAttention,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8GPTJAttention(module.embed_dim, module.num_attention_heads, module.bias.shape[3], module.rotary_dim)
        # Fuse the scaling into the q_proj output scale
        q_output_scale = q_output_scale * module.scale_attn
        # TODO: GPTJ has no bias, find a way to elide these later 
        module.q_proj.bias = torch.nn.Parameter(torch.zeros(module.embed_dim, dtype=float))
        module.v_proj.bias = torch.nn.Parameter(torch.zeros(module.embed_dim, dtype=float))
        module.k_proj.bias = torch.nn.Parameter(torch.zeros(module.embed_dim, dtype=float))
        module.out_proj.bias = torch.nn.Parameter(torch.zeros(module.embed_dim, dtype=float))
        module.q_proj.weight *= module.scale_attn
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale)
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale)
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale)
        return int8_module

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            # (batch, blocks, head, block_length, head_features)
            return tensor.permute(0, 1, 3, 2, 4)
        elif len(tensor.shape) == 4:
            # (batch, head, seq_length, head_features)
            return tensor.permute(0, 2, 1, 3)
        elif len(tensor.shape) == 3:
            return tensor.permute(1, 0, 2)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        elif len(tensor.shape) == 3:
            tensor = tensor.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + \
            (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length -
                                query_length: key_length, :key_length].to(torch.bool)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.int8)
        key = key.to(torch.int8)

        # attn_weights = torch.matmul(query, key.transpose(-1, -2))
        proj_shape = (self.bsz * self.num_attention_heads, -1, self.head_dim)
        key = key.view(*proj_shape)
        query = self._shape(
            query, self.tgt_len, 1).view(*proj_shape)
        attn_weights = self.qk_bmm(query, key)

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(
            mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_weights = attn_weights.to(value.dtype)
        attn_weights.mul_(127).round_()
        attn_weights = attn_weights.to(torch.int8)
        # attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # attn_output = torch.matmul(attn_weights, value)
        value = value[0]
        attn_weights = attn_weights[0]
        # print(attn_weights.shape)
        # print(value.shape)
        attn_output = self.pv_bmm(attn_weights, value)
        value = value.view(1, *value.shape)
        attn_weights = attn_weights.view(1, *attn_weights.shape)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        self.bsz, self.tgt_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(
            query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(
            key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(
            value, self.num_attention_heads, self.head_dim, False)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class Int8GPTJMLP(nn.Module):
    # in MLP: intermediate_size= 4 * embed_dim
    def __init__(self, intermediate_size, embed_dim):
        super().__init__()

        self.fc1 = W8A8B8O8LinearGELU(embed_dim, intermediate_size)
        self.fc2 = W8A8BFP32OFP32Linear(intermediate_size, embed_dim)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    @staticmethod
    def from_float(module: GPTJMLP, fc1_input_scale: float, fc2_input_scale: float):
        int8_module = Int8GPTJMLP(
            module.fc_in.out_features, module.fc_in.in_features)
        int8_module.fc1 = W8A8B8O8LinearGELU.from_float(
            module.fc_in, fc1_input_scale)
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(
            module.fc_out, fc2_input_scale)
        return int8_module


class Int8GPTJBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = LayerNormQ(config.n_embd)
        self.attn = Int8GPTJAttention(config)
        self.mlp = Int8GPTJMLP(inner_dim, config.n_embd)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

    @staticmethod
    def from_float(module, attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        int8_module = Int8GPTJBlock(config)
        int8_module.mlp = Int8GPTJMLP.from_float(
            module.mlp, fc1_input_scale, fc2_input_scale)
        int8_module.ln_1 = LayerNormQ.from_float(module.ln_1, attn_input_scale)
        int8_module.attn = Int8GPTJAttention.from_float(
            module.attn, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)


class Int8GPTJModel(GPTJPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([Int8PTJBlock(config)
                               for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    get_input_embeddings = GPTJModel.get_input_embeddings
    set_input_embeddings = GPTJModel.set_input_embeddings
    forward = GPTJModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8GPTJModel(module.config)
        int8_module.h = nn.ModuleList(
            [Int8GPTJBlock.from_float(mm, decoder_layer_scales) for mm in module.h])
        return int8_module


class Int8GPTJForCausalLM(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Int8GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8GPTJForCausalLM(module.config)
        int8_module.transformer = Int8GPTJModel(config, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    get_input_embeddings = GPTJForCausalLM.get_input_embeddings
    set_input_embeddings = GPTJForCausalLM.set_input_embeddings
    get_output_embeddings = GPTJForCausalLM.get_output_embeddings
    set_output_embeddings = GPTJForCausalLM.set_output_embeddings
    forward = GPTJForCausalLM.forward
    prepare_inputs_for_generation = GPTJForCausalLM.prepare_inputs_for_generation
    _reorder_cache = GPTJForCausalLM._reorder_cache
    parallelize = GPTJForCausalLM.parallelize
    deparallelize = GPTJForCausalLM.deparallelize
