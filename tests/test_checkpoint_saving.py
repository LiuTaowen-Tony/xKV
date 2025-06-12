#!/usr/bin/env python3
"""
Test script to verify that checkpoint saving and loading works correctly
with the new compressor-only saving approach.
"""

import torch
import tempfile
import os
from train.model import ConvolutionalCompressor, ConvolutionalCompressorConfig
from train.kv_lightning_module import KVCompressorLightningModule
from utils import load_model_and_tokenizer


def test_checkpoint_saving():
    """Test that checkpoints are saved and loaded correctly."""
    print("🧪 Testing checkpoint saving and loading...")

    # Load a small model for testing
    print("Loading base model...")
    model, tokenizer = load_model_and_tokenizer("unsloth/Llama-3.2-1B-Instruct")

    # Create compressor config
    config = ConvolutionalCompressorConfig(
        num_layers=4,  # Use fewer layers for testing
        hidden_dim=2048,
        kv_dim=512,
        layer_stride=2,
        seq_stride=2,
        kernel_size=3,
        hidden_channels=1024,
        activation="gelu",
        dropout_rate=0.1,
    )

    # Create Lightning module
    print("Creating Lightning module...")
    lightning_module = KVCompressorLightningModule(
        config=config,
        base_model=model,
        learning_rate=1e-4,
        compressor_type="convolutional",
    )

    # Get original compressor parameters for comparison
    original_params = {
        name: param.clone()
        for name, param in lightning_module.compressor.named_parameters()
    }

    print(
        f"Original compressor has {sum(p.numel() for p in lightning_module.compressor.parameters())} parameters"
    )

    # Save checkpoint to temporary file
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_file:
        checkpoint_path = tmp_file.name

    try:
        print(f"Saving checkpoint to {checkpoint_path}...")

        # Create a dummy checkpoint
        checkpoint = {
            "state_dict": lightning_module.state_dict(),
            "hyper_parameters": {
                "config": config,
                "learning_rate": 1e-4,
                "compressor_type": "convolutional",
            },
        }

        # Apply the on_save_checkpoint method
        checkpoint = lightning_module.on_save_checkpoint(checkpoint)

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Check checkpoint size and contents
        checkpoint_size = os.path.getsize(checkpoint_path)
        print(f"Checkpoint size: {checkpoint_size / (1024*1024):.2f} MB")

        # Verify that base_model is not in the checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        state_dict_keys = list(loaded_checkpoint["state_dict"].keys())

        has_base_model = any(key.startswith("base_model.") for key in state_dict_keys)
        has_kv_collector = any(
            key.startswith("kv_collector.") for key in state_dict_keys
        )
        has_compressor = any(key.startswith("compressor.") for key in state_dict_keys)

        print(f"Checkpoint contains:")
        print(f"  - Base model parameters: {has_base_model}")
        print(f"  - KV collector parameters: {has_kv_collector}")
        print(f"  - Compressor parameters: {has_compressor}")
        print(
            f"  - Excluded components: {loaded_checkpoint.get('excluded_components', 'None')}"
        )

        assert not has_base_model, "Checkpoint should not contain base_model parameters"
        assert (
            not has_kv_collector
        ), "Checkpoint should not contain kv_collector parameters"
        assert has_compressor, "Checkpoint should contain compressor parameters"

        # Test loading the compressor directly
        print("\nTesting direct compressor loading...")
        from train.eval_perplexity_rnn_trained import load_trained_compressor

        loaded_compressor = load_trained_compressor(checkpoint_path, device="cpu")

        print(
            f"Loaded compressor has {sum(p.numel() for p in loaded_compressor.parameters())} parameters"
        )

        # Verify parameters match
        loaded_params = {
            name: param for name, param in loaded_compressor.named_parameters()
        }

        params_match = True
        for name, original_param in original_params.items():
            if name in loaded_params:
                if not torch.allclose(original_param, loaded_params[name], atol=1e-6):
                    print(f"Parameter {name} does not match!")
                    params_match = False
            else:
                print(f"Parameter {name} missing in loaded model!")
                params_match = False

        if params_match:
            print("✅ All parameters match!")
        else:
            print("❌ Parameter mismatch detected!")
            return False

        # Test that we can use the loaded compressor
        print("\nTesting compressor functionality...")

        # Create dummy input
        batch_size, num_layers, seq_len, kv_dim = 1, 4, 32, 512
        k_dummy = torch.randn(batch_size, 1, seq_len, num_layers * kv_dim)
        v_dummy = torch.randn(batch_size, 1, seq_len, num_layers * kv_dim)

        # Test compression and decompression
        with torch.no_grad():
            compressed = loaded_compressor.compress(k_dummy, v_dummy)
            k_recon, v_recon = loaded_compressor.decompress(
                compressed, batch_size, seq_len
            )

            print(f"Input shapes: K={k_dummy.shape}, V={v_dummy.shape}")
            print(f"Compressed shape: {compressed.shape}")
            print(f"Reconstructed shapes: K={k_recon.shape}, V={v_recon.shape}")

            # Calculate reconstruction error
            k_mse = torch.nn.functional.mse_loss(k_recon, k_dummy).item()
            v_mse = torch.nn.functional.mse_loss(v_recon, v_dummy).item()

            print(f"Reconstruction MSE: K={k_mse:.6f}, V={v_mse:.6f}")

        print("✅ Checkpoint saving and loading test passed!")
        return True

    finally:
        # Clean up temporary file
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    success = test_checkpoint_saving()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        exit(1)
