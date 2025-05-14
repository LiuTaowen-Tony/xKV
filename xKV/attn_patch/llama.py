from typing import Optional, Tuple
import types
import torch.nn as nn
import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaForCausalLM
)

from transformers.cache_utils import Cache
from ..customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache

def xKV_llama_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        is_prefill = q_len > 128 # assume auto-regressive, set 128 for multi-turn #FIXME
        #NOTE(brian1009): Skip the RoPE on key and only apply onto query for now.
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if is_prefill: # prefilling
                #NOTE(brian1009): In our customized cache, we will perform different kind of compression methods
                # To boost performance, we will not use the kv returned from the cache, but instead use the original KV
                assert isinstance(past_key_value, FakeLayerMergingCache)
                past_key_value.update(key_states, value_states, self.layer_idx, mode='prefill', cos=cos, sin=sin)
                key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
            else:
                key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, mode='decode')
    
        if self.config._attn_implementation != "sdpa":
            raise ValueError("Only sdpa is supported for now")

        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



def enable_llama_xKV_eval(
    model: LlamaForCausalLM,
):
    for i, layer in enumerate(model.model.layers):
        module = layer.self_attn
        
        if not isinstance(module, LlamaAttention):
            raise ValueError("Only LlamaAttention is supported for now")
        
        module.forward = types.MethodType(
            xKV_llama_forward, module
        )