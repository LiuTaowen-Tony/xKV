#!/usr/bin/env python3
"""
Perplexity evaluation script that runs in RNN mode with trained KV compression models.
This script processes sequences token by token (auto-regressive mode) and calculates
perplexity using trained compression models from the train directory.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# Add the root directory to the path
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from utils import load_model_and_tokenizer, set_seed
from train.model import (
    ConvolutionalCompressor,
    ConvolutionalCompressorConfig,
    EnhancedConvolutionalCompressor,
    EnhancedConvolutionalCompressorConfig,
    VAEConvolutionalCompressor,
    VAEConvolutionalCompressorConfig,
    Dual1DConvolutionalCompressor,
    Dual1DConvolutionalCompressorConfig,
)
from train.kv_lightning_module import KVCompressorLightningModule
from train.kv_cache_collector import KVCacheCollector


class TrainedKVCache:
    """
    Custom KV cache that uses trained compression models.
    Replaces the standard cache with compressed storage.
    """

    def __init__(
        self, compressor_model, max_batch_size: int = 1, max_seq_len: int = 2048
    ):
        self.compressor = compressor_model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.compressed_cache = {}  # Store compressed representations per layer
        self.cache_positions = {}  # Track cache positions per layer
        self.device = next(compressor_model.parameters()).device

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states using compression.

        Args:
            key_states: New key states [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_position: Position tensor for cache updates

        Returns:
            Updated key and value states
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Initialize cache for this layer if not exists
        if layer_idx not in self.compressed_cache:
            self.compressed_cache[layer_idx] = None
            self.cache_positions[layer_idx] = 0

        # For the first token or when cache is empty, store directly
        if self.compressed_cache[layer_idx] is None:
            # Reshape for compression: [batch_size, 1, seq_len, num_heads * head_dim]
            k_reshaped = (
                key_states.transpose(1, 2).contiguous().view(batch_size, 1, seq_len, -1)
            )
            v_reshaped = (
                value_states.transpose(1, 2)
                .contiguous()
                .view(batch_size, 1, seq_len, -1)
            )

            # Compress and store
            if hasattr(self.compressor, "compress"):
                if isinstance(self.compressor, VAEConvolutionalCompressor):
                    compressed, mu, logvar, shape_info = self.compressor.compress(
                        k_reshaped, v_reshaped
                    )
                    self.compressed_cache[layer_idx] = (compressed, shape_info, "vae")
                else:
                    compressed = self.compressor.compress(k_reshaped, v_reshaped)
                    if isinstance(compressed, tuple):  # For dual1d compressor
                        self.compressed_cache[layer_idx] = (
                            compressed[0],
                            compressed[1],
                            "dual1d",
                        )
                    else:
                        self.compressed_cache[layer_idx] = (
                            compressed,
                            (batch_size, seq_len),
                            "standard",
                        )

            self.cache_positions[layer_idx] = seq_len
            return key_states, value_states

        else:
            # For subsequent tokens, decompress, append, and recompress
            # This is a simplified approach - in practice, you might want more efficient incremental updates

            # Decompress existing cache
            cache_data = self.compressed_cache[layer_idx]
            if cache_data[2] == "vae":
                compressed, shape_info, _ = cache_data
                k_cached, v_cached = self.compressor.decompress(compressed, shape_info)
            elif cache_data[2] == "dual1d":
                compressed, shape_info, _ = cache_data
                k_cached, v_cached = self.compressor.decompress(compressed, shape_info)
            else:
                compressed, shape_info, _ = cache_data
                k_cached, v_cached = self.compressor.decompress(
                    compressed, shape_info[0], shape_info[1]
                )

            # Reshape cached data back to original format
            cached_seq_len = k_cached.shape[2]
            k_cached = k_cached.view(
                batch_size, cached_seq_len, num_heads, head_dim
            ).transpose(1, 2)
            v_cached = v_cached.view(
                batch_size, cached_seq_len, num_heads, head_dim
            ).transpose(1, 2)

            # Concatenate with new states
            k_combined = torch.cat([k_cached, key_states], dim=2)
            v_combined = torch.cat([v_cached, value_states], dim=2)

            # Reshape for compression
            combined_seq_len = k_combined.shape[2]
            k_reshaped = (
                k_combined.transpose(1, 2)
                .contiguous()
                .view(batch_size, 1, combined_seq_len, -1)
            )
            v_reshaped = (
                v_combined.transpose(1, 2)
                .contiguous()
                .view(batch_size, 1, combined_seq_len, -1)
            )

            # Recompress and store
            if isinstance(self.compressor, VAEConvolutionalCompressor):
                compressed, mu, logvar, shape_info = self.compressor.compress(
                    k_reshaped, v_reshaped
                )
                self.compressed_cache[layer_idx] = (compressed, shape_info, "vae")
            else:
                compressed = self.compressor.compress(k_reshaped, v_reshaped)
                if isinstance(compressed, tuple):  # For dual1d compressor
                    self.compressed_cache[layer_idx] = (
                        compressed[0],
                        compressed[1],
                        "dual1d",
                    )
                else:
                    self.compressed_cache[layer_idx] = (
                        compressed,
                        (batch_size, combined_seq_len),
                        "standard",
                    )

            self.cache_positions[layer_idx] = combined_seq_len
            return k_combined, v_combined

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the sequence length for a given layer."""
        return self.cache_positions.get(layer_idx, 0)

    def clear(self):
        """Clear all cached data."""
        self.compressed_cache.clear()
        self.cache_positions.clear()


class PerplexityEvaluatorRNNTrained:
    """
    Perplexity evaluator that runs in RNN mode with trained compression models.
    Processes sequences token by token to calculate perplexity.
    """

    def __init__(self, model, tokenizer, compressor_model, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.compressor_model = compressor_model
        self.device = device
        self.model.eval()
        self.compressor_model.eval()

        # Patch the model to use our trained compression cache
        self._patch_model_with_trained_cache()

    def _patch_model_with_trained_cache(self):
        """Patch the model to use trained compression cache."""
        # This is a simplified patching approach
        # In practice, you might need more sophisticated integration
        self.trained_cache = TrainedKVCache(self.compressor_model)

        # Store original forward methods
        self.original_forwards = {}

        # Patch attention layers to use compressed cache
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "self_attn"):
                self.original_forwards[layer_idx] = layer.self_attn.forward
                layer.self_attn.forward = self._create_patched_forward(
                    layer.self_attn, layer_idx
                )

    def _create_patched_forward(self, attention_module, layer_idx):
        """Create a patched forward method for attention layers."""
        original_forward = self.original_forwards[layer_idx]

        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):

            # Call original forward but intercept KV states
            outputs = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,  # Don't use standard cache
                output_attentions=output_attentions,
                use_cache=False,
                cache_position=cache_position,
                **kwargs,
            )

            if use_cache:
                # Extract key and value states from the attention computation
                # This is model-specific and might need adjustment
                bsz, q_len, _ = hidden_states.size()

                # Get query, key, value projections
                query_states = attention_module.q_proj(hidden_states)
                key_states = attention_module.k_proj(hidden_states)
                value_states = attention_module.v_proj(hidden_states)

                # Reshape to attention heads
                query_states = query_states.view(
                    bsz, q_len, attention_module.num_heads, attention_module.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz,
                    q_len,
                    attention_module.num_key_value_heads,
                    attention_module.head_dim,
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz,
                    q_len,
                    attention_module.num_key_value_heads,
                    attention_module.head_dim,
                ).transpose(1, 2)

                # Update compressed cache
                key_states, value_states = self.trained_cache.update(
                    key_states, value_states, layer_idx, cache_position
                )

                # Create a dummy past_key_value for compatibility
                past_key_value = (key_states, value_states)

                return (
                    outputs[0],
                    outputs[1] if len(outputs) > 1 else None,
                    past_key_value,
                )

            return outputs

        return patched_forward

    def calculate_perplexity_rnn(
        self, text: str, max_length: Optional[int] = None, stride: int = 512
    ) -> Dict[str, float]:
        """
        Calculate perplexity in RNN mode (token by token processing).

        Args:
            text: Input text to evaluate
            max_length: Maximum sequence length to process
            stride: Stride for sliding window (if text is too long)

        Returns:
            Dictionary containing perplexity metrics
        """
        # Tokenize the input text
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)

        if max_length is not None and input_ids.size(1) > max_length:
            # Use sliding window approach for long sequences
            return self._calculate_perplexity_sliding_window(
                input_ids, max_length, stride
            )
        else:
            return self._calculate_perplexity_single_sequence(input_ids)

    def _calculate_perplexity_single_sequence(
        self, input_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate perplexity for a single sequence in RNN mode.
        """
        seq_len = input_ids.size(1)
        total_log_likelihood = 0.0
        total_tokens = 0

        # Clear cache
        self.trained_cache.clear()

        with torch.no_grad():
            # Process token by token in RNN mode
            for i in tqdm(range(1, seq_len), desc="Processing tokens", leave=False):
                # Get current token
                current_token = input_ids[:, i : i + 1]  # Shape: [1, 1]

                if i == 1:
                    # First step: use the first token as input
                    input_token = input_ids[:, :i]  # Shape: [1, i]
                else:
                    # Subsequent steps: use only the current token
                    input_token = input_ids[:, i - 1 : i]  # Shape: [1, 1]

                # Forward pass with compressed cache
                outputs = self.model(
                    input_ids=input_token, use_cache=True, return_dict=True
                )

                # Get logits for the current position
                logits = outputs.logits[:, -1, :]  # Shape: [1, vocab_size]

                # Calculate log probability of the target token
                log_probs = F.log_softmax(logits, dim=-1)
                target_token = current_token.squeeze(0)  # Shape: [1]
                token_log_prob = log_probs[0, target_token].item()

                total_log_likelihood += token_log_prob
                total_tokens += 1

        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = torch.exp(-torch.tensor(avg_log_likelihood)).item()

        return {
            "perplexity": perplexity,
            "avg_log_likelihood": avg_log_likelihood,
            "total_tokens": total_tokens,
            "sequence_length": seq_len,
        }

    def _calculate_perplexity_sliding_window(
        self, input_ids: torch.Tensor, max_length: int, stride: int
    ) -> Dict[str, float]:
        """
        Calculate perplexity using sliding window for long sequences.
        """
        seq_len = input_ids.size(1)
        total_log_likelihood = 0.0
        total_tokens = 0

        with torch.no_grad():
            for begin_loc in tqdm(range(0, seq_len, stride), desc="Processing windows"):
                end_loc = min(begin_loc + max_length, seq_len)

                # Extract current window
                input_window = input_ids[:, begin_loc:end_loc]

                # Calculate perplexity for this window
                window_result = self._calculate_perplexity_single_sequence(input_window)

                # Accumulate results (weighted by number of tokens)
                window_tokens = window_result["total_tokens"]
                window_log_likelihood = (
                    window_result["avg_log_likelihood"] * window_tokens
                )

                total_log_likelihood += window_log_likelihood
                total_tokens += window_tokens

                # Break if we've reached the end
                if end_loc == seq_len:
                    break

        # Calculate overall perplexity
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = torch.exp(-torch.tensor(avg_log_likelihood)).item()

        return {
            "perplexity": perplexity,
            "avg_log_likelihood": avg_log_likelihood,
            "total_tokens": total_tokens,
            "sequence_length": seq_len,
        }

    def evaluate_dataset(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: Optional[int] = None,
        max_length: Optional[int] = 1024,
        stride: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on a dataset.
        """
        logger.info(f"Loading dataset: {dataset_name} ({dataset_config}) - {split}")

        # Load dataset
        try:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {}

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info(f"Evaluating {len(dataset)} samples")

        all_perplexities = []
        all_log_likelihoods = []
        total_tokens = 0

        for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
            text = sample.get("text", "")

            # Skip empty texts
            if not text.strip():
                continue

            try:
                result = self.calculate_perplexity_rnn(
                    text=text, max_length=max_length, stride=stride
                )

                all_perplexities.append(result["perplexity"])
                all_log_likelihoods.append(result["avg_log_likelihood"])
                total_tokens += result["total_tokens"]

                # Log progress periodically
                if (i + 1) % 10 == 0:
                    avg_ppl = np.mean(all_perplexities)
                    logger.info(
                        f"Processed {i+1} samples, avg perplexity: {avg_ppl:.4f}"
                    )

            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

        # Calculate final statistics
        if all_perplexities:
            mean_perplexity = np.mean(all_perplexities)
            median_perplexity = np.median(all_perplexities)
            std_perplexity = np.std(all_perplexities)
            mean_log_likelihood = np.mean(all_log_likelihoods)
        else:
            mean_perplexity = float("inf")
            median_perplexity = float("inf")
            std_perplexity = 0.0
            mean_log_likelihood = float("-inf")

        results = {
            "mean_perplexity": mean_perplexity,
            "median_perplexity": median_perplexity,
            "std_perplexity": std_perplexity,
            "mean_log_likelihood": mean_log_likelihood,
            "total_tokens": total_tokens,
            "num_samples": len(all_perplexities),
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
        }

        return results


def load_trained_compressor(
    checkpoint_path: str, device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a trained compressor model from checkpoint.

    Args:
        checkpoint_path: Path to the Lightning checkpoint
        device: Device to load the model on

    Returns:
        Loaded compressor model
    """
    logger.info(f"Loading trained compressor from: {checkpoint_path}")

    # Load the checkpoint to get hyperparameters and determine compressor type
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "hyper_parameters" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain hyperparameters"
        )

    hparams = checkpoint["hyper_parameters"]

    # Get compressor configuration and type
    config = hparams["config"]
    compressor_type = hparams.get("compressor_type", "convolutional")

    logger.info(f"Loading compressor type: {compressor_type}")
    logger.info(f"Compressor config: {config}")

    # Create the compressor model directly based on type
    if compressor_type == "convolutional":
        compressor = ConvolutionalCompressor(config)
    elif compressor_type == "enhanced_convolutional":
        compressor = EnhancedConvolutionalCompressor(config)
    elif compressor_type == "vae_convolutional":
        compressor = VAEConvolutionalCompressor(config)
    elif compressor_type == "dual1d_convolutional":
        compressor = Dual1DConvolutionalCompressor(config)
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    # Load only the compressor state dict
    state_dict = checkpoint["state_dict"]

    # Filter to get only compressor parameters
    compressor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("compressor."):
            # Remove 'compressor.' prefix to match the compressor model's parameter names
            new_key = key[len("compressor.") :]
            compressor_state_dict[new_key] = value

    # Load the state dict into the compressor
    compressor.load_state_dict(compressor_state_dict, strict=True)

    # Move to device and set to eval mode
    compressor.to(device)
    compressor.eval()

    logger.info(
        f"Successfully loaded compressor with {sum(p.numel() for p in compressor.parameters())} parameters"
    )

    return compressor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity in RNN mode with trained compression models"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Base model to load"
    )
    parser.add_argument(
        "--compressor_checkpoint",
        type=str,
        required=True,
        help="Path to trained compressor checkpoint",
    )
    parser.add_argument(
        "--flash2", action="store_true", help="Whether to use flash-attention2"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name to evaluate on",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )

    # Evaluation arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length to process",
    )
    parser.add_argument(
        "--stride", type=int, default=512, help="Stride for sliding window"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output arguments
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file to save results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    logger.info("Starting perplexity evaluation in RNN mode with trained compression")
    logger.info(f"Arguments: {vars(args)}")

    # Load base model and tokenizer
    logger.info(f"Loading base model: {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, use_flash_attn2=args.flash2
    )

    # Load trained compressor
    compressor = load_trained_compressor(args.compressor_checkpoint)

    # Create evaluator
    evaluator = PerplexityEvaluatorRNNTrained(model, tokenizer, compressor)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        max_length=args.max_length,
        stride=args.stride,
    )

    # Log results
    logger.info("Evaluation completed!")
    logger.info("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save results if output file specified
    if args.output_file:
        # Add metadata to results
        results["args"] = vars(args)
        results["model_name"] = args.model_name_or_path
        results["compressor_checkpoint"] = args.compressor_checkpoint

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {args.output_file}")

    # Print final summary
    print("\n" + "=" * 50)
    print("PERPLEXITY EVALUATION SUMMARY (TRAINED COMPRESSION)")
    print("=" * 50)
    print(f"Base Model: {args.model_name_or_path}")
    print(f"Compressor: {args.compressor_checkpoint}")
    print(f"Dataset: {args.dataset_name} ({args.dataset_config})")
    print(f"Mean Perplexity: {results.get('mean_perplexity', 'N/A'):.4f}")
    print(f"Median Perplexity: {results.get('median_perplexity', 'N/A'):.4f}")
    print(f"Total Tokens: {results.get('total_tokens', 'N/A')}")
    print(f"Samples Processed: {results.get('num_samples', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
