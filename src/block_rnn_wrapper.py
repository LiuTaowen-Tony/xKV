
from src.kv_cache_collector import KVCacheCollector
from src.patch_model import CompressedKVCache, KVCacheConfig, patch_model_with_compressed_cache
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)



class BlockRNNWrapper:
    """
    Perplexity evaluator that runs in RNN mode with trained compression models.
    Processes sequences token by token to calculate perplexity.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compressor_model,
        kv_cache_config: KVCacheConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.compressor_model = compressor_model
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.model.eval()
        self.compressor_model.eval()

        # Patch the model to use our trained compression cache
        self.patched_model = self._patch_model_with_trained_cache()

    def _patch_model_with_trained_cache(self):
        """Patch the model to use trained compression cache."""
        logger.info("Patching model with trained compression cache...")

        # Use the improved patch_model_with_compressed_cache function
        patched_model = patch_model_with_compressed_cache(
            self.model, self.compressor_model, self.kv_cache_config
        )

        logger.info("Model patching completed successfully")
        return patched_model

    def _compute_per_token_lprobs_by_blocks(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss by blocks, this is a more memory efficient way to compute the loss.
        """
        seq_len = input_ids.size(1)
        # Reset cache before processing
        self.patched_model._reset_cache()
        block_size = self.kv_cache_config.seq_window_size
        lprobs_accumulator = None

        with torch.no_grad():
            # Process token by token in RNN mode
            for i in tqdm(range(1, seq_len), desc="Processing tokens", leave=False):
                # Get current token
                current_block = input_ids[:, i : i + block_size]  # Shape: [b, window_size]
                # Forward pass with compressed cache
                outputs = self.patched_model(
                    input_ids=current_block, use_cache=True, return_dict=True,  past_key_values=self.patched_model._compressed_kv_cache
                )
                logits = outputs.logits # Shape: [b, block_size, vocab_size]
                lprobs = F.log_softmax(logits, dim=-1)
                if lprobs_accumulator is None:
                    lprobs_accumulator = lprobs # Shape: [b, block_size, vocab_size]
                else:
                    lprobs_accumulator = torch.cat([lprobs_accumulator, lprobs], dim=1)
        return lprobs_accumulator

    def _compute_logits_by_blocks(
        self, input_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute the logits for a sequence of input ids, this might go OOM if the sequence is too long.
        """
        seq_len = input_ids.size(1)
        # Reset cache before processing
        self.patched_model._reset_cache()
        block_size = self.kv_cache_config.seq_window_size
        logits_accumulator = None

        with torch.no_grad():
            # Process token by token in RNN mode
            for i in tqdm(range(1, seq_len), desc="Processing tokens", leave=False):
                # Get current token
                current_block = input_ids[:, i : i + block_size]  # Shape: [b, window_size]

                # Forward pass with compressed cache
                outputs = self.patched_model(
                    input_ids=current_block, use_cache=True, return_dict=True,  past_key_values=self.patched_model._compressed_kv_cache
                )

                # Get logits for the current position
                if logits_accumulator is None:
                    logits_accumulator = outputs.logits # Shape: [b, block_size, vocab_size]
                else:
                    logits_accumulator = torch.cat([logits_accumulator, outputs.logits], dim=1)

        return logits_accumulator