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

@torch.no_grad()
def quantize_per_tensor_absmax(t):
    scale = t.abs().max() / 127
    if not t.is_cuda:
        # half rounding is not supported on CPU
        t = t.float()
    # use inplace operation to save memory
    t.div_(scale).round_()
    t_q = t.to(torch.int8)
    return t_q, scale

from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
from transformers.activations import ACT2FN

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
    x_ = x.to(torch.float32)
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    r = ((x_.to(torch.float) * cos) + (rotate_every_two(x_.to(torch.float)) * sin))
    r = r.clamp(-128, 127).to(torch.int8)
    return r


class Int8GPTJAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, n_embd, n_head, max_position_embeddings, rotary_dim = None):
        super().__init__()  
        max_positions = max_position_embeddings
        self.max_position = max_positions
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
        # scale_h = module.head_dim**-0.5
        ## scaling
        # qoo = q_output_scale
        # q_output_scale = q_output_scale * scale_h 
        # module.q_proj.weight *= scale_h
        # qs2 = q_output_scale * scale_h
        ## scaling
        # TODO: GPTJ has no bias, find a way to elide these later 
        module.q_proj.bias = torch.nn.Parameter(torch.zeros((1,module.embed_dim), dtype=module.q_proj.weight.dtype))
        module.v_proj.bias = torch.nn.Parameter(torch.zeros((1,module.embed_dim), dtype=module.v_proj.weight.dtype))
        module.k_proj.bias = torch.nn.Parameter(torch.zeros((1,module.embed_dim), dtype=module.k_proj.weight.dtype))
        module.out_proj.bias = torch.nn.Parameter(torch.zeros((1,module.embed_dim), dtype=module.out_proj.weight.dtype))
        module.cuda()
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale)
        wc = module.k_proj.weight.clone()
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale)
        int8_weight, weight_scale = quantize_per_tensor_absmax(wc)
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale)
        int8_module.v_proj.requires_grad = False
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)
        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        # print(f"{v_output_scale}/{out_input_scale}")
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
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
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
                                query_length: key_length, :key_length].to(torch.bool).cuda()
        
        # key = key.transpose(-1, -2)
        proj_shape = (self.bsz * self.num_attention_heads, -1, self.head_dim)
        key = key.reshape(*proj_shape)
        query = query.view(*proj_shape)
        query = query.contiguous()
        key = key.contiguous()
        attn_weights = self.qk_bmm(query, key)
        attn_weights = attn_weights.view(self.bsz, self.num_attention_heads, self.tgt_len, key_length)
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
        attn_weights.mul_(127).round_()
        attn_weights = attn_weights.to(torch.int8)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_weights = attn_weights.view(self.bsz * self.num_attention_heads, -1, self.tgt_len).contiguous()
        value = value.transpose(2,3)
        value = value.reshape(self.num_attention_heads * self.bsz, self.head_dim, self.tgt_len).contiguous()
        attn_output = self.pv_bmm(attn_weights, value)
        attn_weights = attn_weights.view(self.bsz, self.num_attention_heads, self.tgt_len, key_length)
        attn_output = attn_output.view(self.bsz, self.num_attention_heads, self.tgt_len, self.head_dim)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        self.bsz, self.tgt_len, _ = hidden_states.size()
        # self.out_proj.cuda()
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

            key = torch.cat([k_rot, k_pass.to(torch.int8)], dim=-1)
            query = torch.cat([q_rot, q_pass.to(torch.int8)], dim=-1)
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
        attn_output = attn_output.contiguous()
        attn_output = self.out_proj(attn_output)

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
        # hidden_states = hidden_states.to(torch.float)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states 

    @staticmethod
    def from_float(module: GPTJMLP, fc1_input_scale: float, fc2_input_scale: float):
        int8_module = Int8GPTJMLP(
            module.fc_in.out_features, module.fc_in.in_features)
        int8_module.fc1 = W8A8B8O8LinearGELU.from_float(
            module.fc_in, fc1_input_scale, fc2_input_scale)
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(
            module.fc_out, fc2_input_scale)
        return int8_module


class Int8GPTJBlock(nn.Module):
    def __init__(self, inner_dim, n_embd, n_head, max_position_embeddings, rotary_dim = None):
        super().__init__()
        self.ln_1 = LayerNormQ(n_embd)
        self.attn = Int8GPTJAttention(n_embd, n_head, max_position_embeddings, rotary_dim)
        self.mlp = Int8GPTJMLP(inner_dim, n_embd)

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
        inner_dim = module.mlp.fc_out.in_features
        n_embd = module.ln_1.normalized_shape[0]
        int8_module = Int8GPTJBlock(inner_dim, n_embd, module.attn.num_attention_heads, module.attn.bias.shape[0], module.attn.rotary_dim)
        int8_module.mlp = Int8GPTJMLP.from_float(
            module.mlp, fc1_input_scale, fc2_input_scale)
        int8_module.ln_1 = LayerNormQ.from_float(module.ln_1, attn_input_scale)
        int8_module.attn = Int8GPTJAttention.from_float(
            module.attn, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)
        return int8_module


class Int8GPTJModel(GPTJPreTrainedModel):
    # TODO: have to add padding!
    def __init__(self, config):
        self.d = {}
        super().__init__(config)
        n_layer = config.n_layer
        inner_dim = 4 * config.n_embd
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Identity()
        self.padding_idx = config.pad_token_id
        # self.h = nn.ModuleList()
        self.h = nn.ModuleList([Int8GPTJBlock(inner_dim, self.embed_dim, config.n_head, config.n_positions, config.rotary_dim)
                               for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    get_input_embeddings = GPTJModel.get_input_embeddings
    set_input_embeddings = GPTJModel.set_input_embeddings
    old_forward = GPTJModel.forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        from torch.nn.functional import pad
        input_len = input_ids.shape[1]
        if input_len % 16 != 0:
            padding_len = 16 - input_len % 16
            input_ids =  pad(input_ids, (0, padding_len), value=self.padding_idx)
            if attention_mask is not None:
                attention_mask = pad(attention_mask, (0, padding_len), value=0)
        output = self.old_forward(input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
        if input_len % 16 != 0:
            output.last_hidden_state = output.last_hidden_state[:,:input_len, :]
        return output

    @staticmethod
    def from_float(module : GPTJModel, decoder_layer_scales, k = None):
        config = GPTJConfig(vocab_size=module.vocab_size, n_embd=module.embed_dim, n_layer=len(module.h), rotary_dim=module.h[0].attn.rotary_dim
        , n_inner=4*module.embed_dim)
        int8_module = Int8GPTJModel(config)
        for i, layer in enumerate(module.h):
            if k is not None and i in k:
                int8_module.h[i] = layer
            else:
                int8_module.h[i] = Int8GPTJBlock.from_float(layer, **decoder_layer_scales[i])
        int8_module.ln_f = module.ln_f.to(torch.float)
        int8_module.wte = module.wte
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
    def from_float(module, decoder_layer_scales, k = None):
        int8_module = Int8GPTJForCausalLM(module.config)
        int8_module.transformer = Int8GPTJModel.from_float(module.transformer, decoder_layer_scales, k)
        int8_module.lm_head = module.lm_head.to(torch.float)
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
