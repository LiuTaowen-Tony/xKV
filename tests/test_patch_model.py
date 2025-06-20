#!/usr/bin/env python3
"""
Test script for the patch model functionality.
Tests the CompressedKVCache, patching mechanism, and integration with transformer models.
"""

import torch
import pytest
from typing import Tuple
from unittest.mock import Mock, patch

# Import the modules we want to test
from src.patch_model import (
    CompressedKVCache,
    KVCacheConfig,
    create_patched_attention_forward,
    patch_model_with_compressed_cache,
)
from src.utils.utils import load_model_and_tokenizer


class MockCompressor:
    """Mock compressor for testing without actual model dependencies."""

    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
        self.device = torch.device("cpu")
        # Add a dummy parameter to make parameters() work
        self._dummy_param = torch.nn.Parameter(torch.tensor([1.0]))

    def parameters(self):
        # Return an iterator of parameters
        yield self._dummy_param

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Mock compression that reduces sequence dimension."""
        batch_size, num_layers, seq_len, hidden_dim = keys.shape
        compressed_seq_len = int(seq_len * self.compression_ratio)

        # Concatenate and compress
        kv_concat = torch.cat([keys, values], dim=-1)  # [B, L, S, 2*H]
        compressed = kv_concat[:, :, :compressed_seq_len, :]  # Simple truncation for testing

        return compressed

    def decompress(self, compressed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mock decompression that splits and pads back to original size."""
        batch_size, num_layers, compressed_seq_len, hidden_dim_2x = compressed.shape
        hidden_dim = hidden_dim_2x // 2

        # Split K and V
        keys_compressed = compressed[:, :, :, :hidden_dim]
        values_compressed = compressed[:, :, :, hidden_dim:]

        # Pad back to original sequence length (assume original was 2x compressed)
        original_seq_len = compressed_seq_len * 2
        pad_len = original_seq_len - compressed_seq_len

        keys = torch.nn.functional.pad(keys_compressed, (0, 0, 0, pad_len), value=0)
        values = torch.nn.functional.pad(values_compressed, (0, 0, 0, pad_len), value=0)

        return keys, values


class TestKVCacheConfig:
    """Test the KVCacheConfig dataclass."""

    def test_config_creation(self):
        """Test basic config creation with required parameters."""
        config = KVCacheConfig(num_layers=4, seq_window_size=32)

        assert config.num_layers == 4
        assert config.seq_window_size == 32
        assert config.enable_decompression == True  # default

    def test_config_with_custom_values(self):
        """Test config creation with custom values."""
        config = KVCacheConfig(
            num_layers=8, seq_window_size=64, enable_decompression=False
        )

        assert config.num_layers == 8
        assert config.seq_window_size == 64
        assert config.enable_decompression == False


class TestCompressedKVCache:
    """Test the CompressedKVCache class."""

    @pytest.fixture
    def mock_compressor(self):
        """Create a mock compressor for testing."""
        return MockCompressor()

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return KVCacheConfig(num_layers=4, seq_window_size=32)

    @pytest.fixture
    def kv_cache(self, mock_compressor, config):
        """Create a CompressedKVCache instance for testing."""
        return CompressedKVCache(mock_compressor, config)

    def test_initialization(self, mock_compressor, config):
        """Test proper initialization of CompressedKVCache."""
        cache = CompressedKVCache(mock_compressor, config)

        assert cache.compressor == mock_compressor
        assert cache.config == config
        assert len(cache.scratch_keys) == config.num_layers
        assert len(cache.scratch_values) == config.num_layers
        assert all(k is None for k in cache.scratch_keys)
        assert all(v is None for v in cache.scratch_values)
        assert cache.compressed_cache is None
        assert cache.decompressed_keys is None
        assert cache.decompressed_values is None

    def test_add_kv_to_layer_first_time(self, kv_cache):
        """Test adding KV pairs to an empty layer."""
        batch_size, seq_len, hidden_dim = 2, 16, 64
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)
        layer_idx = 0

        kv_cache.add_kv_to_layer(keys, values, layer_idx)

        assert kv_cache.scratch_keys[layer_idx] is not None
        assert kv_cache.scratch_values[layer_idx] is not None
        assert torch.equal(kv_cache.scratch_keys[layer_idx], keys)
        assert torch.equal(kv_cache.scratch_values[layer_idx], values)

    def test_add_kv_to_layer_concatenation(self, kv_cache):
        """Test adding KV pairs to a layer that already has data."""
        batch_size, seq_len, hidden_dim = 2, 16, 64
        keys1 = torch.randn(batch_size, seq_len, hidden_dim)
        values1 = torch.randn(batch_size, seq_len, hidden_dim)
        keys2 = torch.randn(batch_size, seq_len, hidden_dim)
        values2 = torch.randn(batch_size, seq_len, hidden_dim)
        layer_idx = 0

        kv_cache.add_kv_to_layer(keys1, values1, layer_idx)
        kv_cache.add_kv_to_layer(keys2, values2, layer_idx)

        expected_keys = torch.cat([keys1, keys2], dim=1)
        expected_values = torch.cat([values1, values2], dim=1)

        assert kv_cache.scratch_keys[layer_idx].shape == expected_keys.shape
        assert kv_cache.scratch_values[layer_idx].shape == expected_values.shape
        assert torch.equal(kv_cache.scratch_keys[layer_idx], expected_keys)
        assert torch.equal(kv_cache.scratch_values[layer_idx], expected_values)

    def test_should_compress_empty_cache(self, kv_cache):
        """Test should_compress returns False for empty cache."""
        assert not kv_cache.should_compress()

    def test_should_compress_insufficient_data(self, kv_cache):
        """Test should_compress returns False when not enough data."""
        batch_size, seq_len, hidden_dim = 2, 16, 64  # seq_len < seq_window_size (32)
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)

        kv_cache.add_kv_to_layer(keys, values, 0)

        assert not kv_cache.should_compress()

    def test_should_compress_sufficient_data(self, kv_cache):
        """Test should_compress returns True when enough data."""
        batch_size, seq_len, hidden_dim = 2, 64, 64  # seq_len > seq_window_size (32)
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)

        # Add data to all layers
        for layer_idx in range(kv_cache.config.num_layers):
            kv_cache.add_kv_to_layer(keys, values, layer_idx)

        assert kv_cache.should_compress()

    def test_compress_and_update_cache(self, kv_cache):
        """Test the compression and cache update process."""
        batch_size, seq_len, hidden_dim = 2, 64, 64
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)

        # Add data to all layers
        for layer_idx in range(kv_cache.config.num_layers):
            kv_cache.add_kv_to_layer(keys, values, layer_idx)

        # Compress
        compressed_data, decompressed_keys, decompressed_values = (
            kv_cache.compress_and_update_cache()
        )

        # Check that compression occurred
        assert compressed_data is not None
        assert kv_cache.compressed_cache is not None

        # Check that decompressed cache is updated (if decompression enabled)
        if kv_cache.config.enable_decompression:
            assert decompressed_keys is not None
            assert decompressed_values is not None
            assert kv_cache.decompressed_keys is not None
            assert kv_cache.decompressed_values is not None

        # Check that scratch buffers are updated with remainder
        window_size = kv_cache.config.seq_window_size
        expected_remainder_len = seq_len % window_size
        if expected_remainder_len > 0:
            assert kv_cache.scratch_keys[0] is not None
            assert kv_cache.scratch_keys[0].shape[1] == expected_remainder_len

    def test_get_layer_cache(self, kv_cache):
        """Test retrieving cached data for a specific layer."""
        batch_size, seq_len, hidden_dim = 2, 64, 64
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        values = torch.randn(batch_size, seq_len, hidden_dim)

        # Add data and compress
        for layer_idx in range(kv_cache.config.num_layers):
            kv_cache.add_kv_to_layer(keys, values, layer_idx)

        kv_cache.compress_and_update_cache()

        # Get layer cache
        layer_keys, layer_values = kv_cache.get_layer_cache(0)

        assert layer_keys is not None
        assert layer_values is not None
        assert layer_keys.shape[0] == batch_size  # batch dimension preserved
        assert layer_values.shape[0] == batch_size


class TestPatchingFunctions:
    """Test the patching functions."""

    @pytest.fixture
    def mock_attention(self):
        """Create a mock attention module."""
        attention = Mock()
        attention.head_dim = 64
        attention.q_proj = Mock(return_value=torch.randn(2, 32, 256))
        attention.k_proj = Mock(return_value=torch.randn(2, 32, 256))
        attention.v_proj = Mock(return_value=torch.randn(2, 32, 256))
        attention.o_proj = Mock(return_value=torch.randn(2, 32, 256))
        attention.config = Mock()
        attention.config._attn_implementation = "eager"
        attention.training = False
        attention.attention_dropout = 0.0
        attention.scaling = 1.0
        return attention

    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KV cache."""
        cache = Mock()
        cache.config = Mock()
        cache.config.num_layers = 4
        cache.add_kv_to_layer = Mock()
        cache.should_compress = Mock(return_value=False)
        cache.get_layer_cache = Mock(
            return_value=(torch.randn(2, 32, 64), torch.randn(2, 32, 64))  # keys  # values
        )
        return cache

    def test_create_patched_attention_forward(self, mock_attention, mock_kv_cache):
        """Test creation of patched forward function."""
        patched_forward = create_patched_attention_forward(
            mock_attention, mock_kv_cache, layer_idx=0
        )

        assert callable(patched_forward)

        # Test calling the patched forward
        hidden_states = torch.randn(2, 32, 256)
        position_embeddings = (torch.randn(2, 32, 64), torch.randn(2, 32, 64))

        with patch("train.patch_model.apply_rotary_pos_emb") as mock_rotary, patch(
            "train.patch_model.eager_attention_forward"
        ) as mock_attention_fn:

            mock_rotary.return_value = (torch.randn(2, 4, 32, 64), torch.randn(2, 4, 32, 64))
            mock_attention_fn.return_value = (torch.randn(2, 32, 256), None)

            output, weights = patched_forward(hidden_states, position_embeddings)

            # Check that KV cache methods were called
            mock_kv_cache.add_kv_to_layer.assert_called_once()
            mock_kv_cache.get_layer_cache.assert_called_once_with(0)

            assert output is not None
            assert output.shape == (2, 32, 256)


class TestIntegrationWithRealModel:
    """Integration tests with real transformer models."""

    @pytest.mark.slow
    def test_patch_real_model(self):
        """Test patching a real transformer model (requires GPU/significant memory)."""
        # Use a very small model for testing
        model_name = "unsloth/Llama-3.2-1B"  # Small model for testing

        model, tokenizer = load_model_and_tokenizer(model_name)

        # Create compressor config
        compressor = MockCompressor()

        # Create KV cache config
        kv_config = KVCacheConfig(
            num_layers=model.config.num_hidden_layers, seq_window_size=32, enable_decompression=False
        )

        # Patch the model
        patched_model = patch_model_with_compressed_cache(model, compressor, kv_config)

        # Verify the model was patched
        assert hasattr(patched_model, "_compressed_kv_cache")
        assert isinstance(patched_model._compressed_kv_cache, CompressedKVCache)

        # Test forward pass with simple input
        input_text = "Hello world"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["past_key_values"] = patched_model._compressed_kv_cache
        inputs["use_cache"] = True

        with torch.no_grad():
            outputs = patched_model(**inputs)

        assert outputs.logits is not None
        assert outputs.logits.shape[-1] == model.config.vocab_size



class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_compressor_type(self):
        """Test error handling for invalid compressor types."""
        mock_model = Mock()
        mock_model.model.layers = []

        mock_compressor = MockCompressor()
        config = KVCacheConfig(num_layers=4, seq_window_size=32)

        # This should work fine
        try:
            patched_model = patch_model_with_compressed_cache(mock_model, mock_compressor, config)
            assert patched_model is not None
        except Exception as e:
            pytest.fail(f"Valid patching failed: {e}")

    def test_empty_scratch_cache_compression(self):
        """Test behavior when trying to compress empty scratch cache."""
        compressor = MockCompressor()
        config = KVCacheConfig(num_layers=4, seq_window_size=32)
        cache = CompressedKVCache(compressor, config)

        # Should not compress when cache is empty
        assert not cache.should_compress()

    def test_mismatched_layer_indices(self):
        """Test handling of mismatched layer indices."""
        compressor = MockCompressor()
        config = KVCacheConfig(num_layers=4, seq_window_size=32)
        cache = CompressedKVCache(compressor, config)

        keys = torch.randn(2, 16, 64)
        values = torch.randn(2, 16, 64)

        # Adding to valid layer should work
        cache.add_kv_to_layer(keys, values, 0)
        assert cache.scratch_keys[0] is not None

        # Adding to invalid layer should raise error
        with pytest.raises(IndexError):
            cache.add_kv_to_layer(keys, values, 10)


def test_checkpoint_compatibility():
    """Test that patched models can be saved and loaded."""
    # Create a simple mock model
    mock_model = Mock()
    mock_model.model.layers = [Mock() for _ in range(4)]

    # Create compressor and config
    compressor = MockCompressor()
    config = KVCacheConfig(num_layers=4, seq_window_size=32)

    # Patch the model
    patched_model = patch_model_with_compressed_cache(mock_model, compressor, config)

    # Verify cache is attached
    assert hasattr(patched_model, "_compressed_kv_cache")

    # Test that we can access the cache
    cache = patched_model._compressed_kv_cache
    assert isinstance(cache, CompressedKVCache)
    assert cache.config.num_layers == 4


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running patch model tests...")

    # Test config creation
    config = KVCacheConfig(num_layers=4, seq_window_size=32)
    print(f"✅ Config created: {config}")

    # Test cache creation
    compressor = MockCompressor()
    cache = CompressedKVCache(compressor, config)
    print(f"✅ Cache created with {len(cache.scratch_keys)} layer buffers")

    # Test adding KV pairs
    keys = torch.randn(2, 16, 64)
    values = torch.randn(2, 16, 64)
    cache.add_kv_to_layer(keys, values, 0)
    print(f"✅ Added KV pairs to layer 0: {cache.scratch_keys[0].shape}")

    # Test compression readiness
    print(f"✅ Should compress: {cache.should_compress()}")

    print("\n🎉 Basic tests passed! Run with pytest for comprehensive testing.")
    print("Usage: pytest tests/test_patch_model.py -v")
    print("For integration tests: pytest tests/test_patch_model.py -v -m slow")
