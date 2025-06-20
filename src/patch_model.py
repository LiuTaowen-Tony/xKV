from collections.abc import Callable
import dataclasses
import torch
from typing import Tuple, Optional, List
import transformers
from src.compressor import KVCompressor

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    Cache,
)


@dataclasses.dataclass
class KVCacheConfig:
    """Configuration for the compressed KV cache system."""

    num_layers: int
    seq_window_size: int
    enable_decompression: bool = True


class CompressedKVCache(Cache):
    """
    A KV cache that uses trained compression models to reduce memory usage.

    The cache operates in two phases:
    1. Accumulation: New KV pairs are stored in scratch buffers per layer
    2. Compression: When scratch buffers reach the window size, they are compressed
    """

    def __init__(self, compressor_model: KVCompressor, config: KVCacheConfig):
        self.compressor = compressor_model
        self.config = config
        self.device = next(compressor_model.parameters()).device

        # Scratch buffers for accumulating KV pairs before compression
        # List: layer_idx -> [batch_size, seq_len, hidden_dim]
        self.scratch_keys: List[Optional[torch.Tensor]] = [None] * config.num_layers
        self.scratch_values: List[Optional[torch.Tensor]] = [None] * config.num_layers

        #  b, l, s, h (to be compressed)
        # Compressed storage
        self.compressed_cache: Optional[torch.Tensor] = None

        # Decompressed cache for fast access
        self.decompressed_keys: Optional[torch.Tensor] = None
        self.decompressed_values: Optional[torch.Tensor] = None
        self.cached_seq_length: int = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if 
        return self.cached_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return 16384

    def add_kv_to_layer(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int) -> None:
        """Add new key-value pairs to the scratch buffer for a specific layer."""
        # [b, s, h]
        # only update cached_seq_length for the first layer
        if layer_idx == 0:
            self.cached_seq_length += keys.shape[1]

        if self.scratch_keys[layer_idx] is None:
            self.scratch_keys[layer_idx] = keys
            self.scratch_values[layer_idx] = values
        else:
            self.scratch_keys[layer_idx] = torch.cat([self.scratch_keys[layer_idx], keys], dim=1)
            self.scratch_values[layer_idx] = torch.cat(
                [self.scratch_values[layer_idx], values], dim=1
            )

    def should_compress(self) -> bool:
        """Check if scratch buffers have enough data to compress."""
        if self.scratch_keys[0] is None:
            return False
        seq_len = self.scratch_keys[0].shape[1]
        return seq_len >= self.config.seq_window_size

    def compress_and_update_cache(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compress accumulated KV pairs and update the cache.

        Returns:
            compressed_data: The newly compressed KV data
            decompressed_keys: Updated decompressed keys (if decompression enabled)
            decompressed_values: Updated decompressed values (if decompression enabled)
        """
        # Collect all layers' scratch data
        all_keys, all_values = self._collect_scratch_data()

        # Split into compressible portion and remainder
        compressible_keys, remaining_keys, compressible_values, remaining_values = (
            self._split_by_window_size(all_keys, all_values)
        )

        # Compress the compressible portion
        compressed_data = self.compressor.compress(compressible_keys, compressible_values)

        # Update compressed cache
        self._update_compressed_cache(compressed_data)

        # Update scratch buffers with remainder
        self._update_scratch_buffers(remaining_keys, remaining_values)

        # Update decompressed cache if needed
        decompressed_keys, decompressed_values = None, None
        if self.config.enable_decompression:
            decompressed_keys, decompressed_values = self._update_decompressed_cache(
                compressed_data
            )

        return compressed_data, decompressed_keys, decompressed_values

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the cached key-value pairs for a specific layer."""
        return (
            self.decompressed_keys[:, layer_idx, :, :],
            self.decompressed_values[:, layer_idx, :, :],
        )

    def get_decompressed_kv_cache_concat_scratch_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the cached key-value pairs for a specific layer."""
        if self.decompressed_keys is None:
            return self.scratch_keys[layer_idx], self.scratch_values[layer_idx]
        
        # otherwise concat the decompressed cache with the scratch buffers
        decompressed_keys, decompressed_values = self.get_layer_cache(layer_idx)
        # [b, s, h]
        return (
            torch.cat([self.scratch_keys[layer_idx], decompressed_keys], dim=1),
            torch.cat([self.scratch_values[layer_idx], decompressed_values], dim=1),
        )

    def _collect_scratch_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect all scratch data across layers into tensors."""
        # Stack along layer dimension: [batch_size, num_layers, seq_len, hidden_dim]
        keys = torch.stack(self.scratch_keys, dim=1)
        values = torch.stack(self.scratch_values, dim=1)
        return keys, values

    def _split_by_window_size(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split tensors into compressible portion and remainder based on window size."""
        # keys: b, l, s, h
        seq_len = keys.shape[2] # seq_len dimension
        compressible_len = (seq_len // self.config.seq_window_size) * self.config.seq_window_size

        compressible_keys = keys[:, :, :compressible_len, :]
        remaining_keys = keys[:, :, compressible_len:, :]
        compressible_values = values[:, :, :compressible_len, :]
        remaining_values = values[:, :, compressible_len:, :]

        return compressible_keys, remaining_keys, compressible_values, remaining_values

    def _update_compressed_cache(self, new_compressed_data: torch.Tensor) -> None:
        """Update the compressed cache with new data."""
        if self.compressed_cache is None:
            self.compressed_cache = new_compressed_data
        else:
            self.compressed_cache = torch.cat([self.compressed_cache, new_compressed_data], dim=2)

    def _update_scratch_buffers(
        self, remaining_keys: torch.Tensor, remaining_values: torch.Tensor
    ) -> None:
        """Update scratch buffers with remaining data after compression."""
        # Split back into per-layer lists
        remaining_keys_list = remaining_keys.unbind(dim=1)  # Split along layer dimension
        remaining_values_list = remaining_values.unbind(dim=1)

        for layer_idx in range(self.config.num_layers):
            self.scratch_keys[layer_idx] = remaining_keys_list[layer_idx]
            self.scratch_values[layer_idx] = remaining_values_list[layer_idx]

    def _update_decompressed_cache(
        self, compressed_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the decompressed cache with newly compressed data."""
        new_keys, new_values = self.compressor.decompress(compressed_data)

        if self.decompressed_keys is None:
            self.decompressed_keys = new_keys
            self.decompressed_values = new_values
        else:
            self.decompressed_keys = torch.cat([self.decompressed_keys, new_keys], dim=2)
            self.decompressed_values = torch.cat([self.decompressed_values, new_values], dim=2)

        return self.decompressed_keys, self.decompressed_values


    def _reset_cache(self) -> None:
        """Reset the cache to its initial state."""
        self.compressed_cache = None
        self.decompressed_keys = None
        self.decompressed_values = None
        self.scratch_keys = [None] * self.config.num_layers
        self.scratch_values = [None] * self.config.num_layers


def create_patched_attention_forward(
    original_attention: torch.nn.Module, kv_cache: CompressedKVCache, layer_idx: int
) -> Callable:
    """Create a patched forward function for attention layers."""

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[CompressedKVCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Patched forward pass that uses compressed KV cache.
        """
        # Compute query, key, value projections
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, original_attention.head_dim)

        query_states = original_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = original_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Add new KV pairs to cache
        kv_cache.add_kv_to_layer(key_states, value_states, layer_idx)

        # Compress if this is the last layer and we have enough data
        if layer_idx == kv_cache.config.num_layers - 1 and kv_cache.should_compress():
            kv_cache.compress_and_update_cache()

        # Retrieve cached KV pairs for attention computation
        # cached_keys, cached_values = kv_cache.get_layer_cache(layer_idx)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Select attention implementation
        if original_attention.config._attn_implementation == "eager":
            attention_fn = eager_attention_forward
        else:
            if original_attention.config._attn_implementation == "sdpa" and output_attentions:
                raise ValueError("SDPA does not support output_attentions")
            attention_fn = ALL_ATTENTION_FUNCTIONS[original_attention.config._attn_implementation]

        # Compute attention
        attn_output, attn_weights = attention_fn(
            original_attention,
            query_states,
            cached_keys,
            cached_values,
            attention_mask,
            dropout=(
                0.0 if not original_attention.training else original_attention.attention_dropout
            ),
            scaling=original_attention.scaling,
            **kwargs,
        )

        # Project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = original_attention.o_proj(attn_output)

        return attn_output, attn_weights

    return patched_forward


def patch_model_with_compressed_cache(
    model: transformers.LlamaForCausalLM, compressor: torch.nn.Module, config: KVCacheConfig
) -> transformers.LlamaForCausalLM:
    """
    Patch a LLaMA model to use compressed KV cache.

    Args:
        model: The LLaMA model to patch
        compressor: The compression model
        config: Configuration for the KV cache

    Returns:
        The patched model
    """
    # Create the compressed KV cache
    kv_cache = CompressedKVCache(compressor, config)

    # Patch each attention layer
    for layer_idx, layer in enumerate(model.model.layers):
        patched_forward = create_patched_attention_forward(layer.self_attn, kv_cache, layer_idx)
        layer.self_attn.forward = patched_forward

    # Store cache reference on model for external access
    model._compressed_kv_cache = kv_cache

    return model
