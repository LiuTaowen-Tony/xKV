#!/usr/bin/env python3
"""
Perplexity evaluation script that runs in RNN mode with xKV attention compression.
This script processes sequences token by token (auto-regressive mode) and calculates
perplexity using the compressed attention mechanism.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# Add the root directory to the path
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from utils import (
    load_model_and_tokenizer,
    apply_kv_compress_patch,
    add_common_args,
    set_seed,
)
from xKV.patch import KVCompress
from xKV.configurations import generate_consecutive_xKV_config


class PerplexityEvaluatorRNN:
    """
    Perplexity evaluator that runs in RNN mode with xKV compression.
    Processes sequences token by token to calculate perplexity.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

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

        # Clear any existing cache
        if hasattr(self.model, "past_key_values"):
            self.model.past_key_values = None

        with torch.no_grad():
            # Process token by token in RNN mode
            past_key_values = None

            for i in tqdm(range(1, seq_len), desc="Processing tokens", leave=False):
                # Get current token and previous context
                current_token = input_ids[:, i : i + 1]  # Shape: [1, 1]

                if i == 1:
                    # First step: use the first token as input
                    input_token = input_ids[:, :i]  # Shape: [1, i]
                else:
                    # Subsequent steps: use only the current token
                    input_token = input_ids[:, i - 1 : i]  # Shape: [1, 1]

                # Forward pass with past key values (RNN mode)
                outputs = self.model(
                    input_ids=input_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                # Update past key values for next iteration
                past_key_values = outputs.past_key_values

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
                trg_len = end_loc - begin_loc

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

        Args:
            dataset_name: Name of the dataset to load
            dataset_config: Configuration of the dataset
            split: Dataset split to use
            max_samples: Maximum number of samples to evaluate
            max_length: Maximum sequence length
            stride: Stride for sliding window

        Returns:
            Dictionary containing evaluation results
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity in RNN mode with xKV compression"
    )

    # Add common arguments (model, xKV options, etc.)
    add_common_args(parser)

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

    logger.info("Starting perplexity evaluation in RNN mode")
    logger.info(f"Arguments: {vars(args)}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, use_flash_attn2=args.flash2
    )

    # Apply xKV compression if enabled
    if args.xKV:
        logger.info("Applying xKV compression patch")
        model = apply_kv_compress_patch(model, args)
        logger.info("xKV compression applied successfully")
    else:
        logger.info("Running without xKV compression")

    # Create evaluator
    evaluator = PerplexityEvaluatorRNN(model, tokenizer)

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
        results["xKV_enabled"] = args.xKV

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {args.output_file}")

    # Print final summary
    print("\n" + "=" * 50)
    print("PERPLEXITY EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name} ({args.dataset_config})")
    print(f"xKV Compression: {'Enabled' if args.xKV else 'Disabled'}")
    print(f"Mean Perplexity: {results.get('mean_perplexity', 'N/A'):.4f}")
    print(f"Median Perplexity: {results.get('median_perplexity', 'N/A'):.4f}")
    print(f"Total Tokens: {results.get('total_tokens', 'N/A')}")
    print(f"Samples Processed: {results.get('num_samples', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
