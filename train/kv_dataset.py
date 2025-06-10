import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any, List
import numpy as np
from .kv_cache_collector import KVCacheCollector


class KVCacheDataset(Dataset):
    """
    Dataset that generates KV cache data from text using a frozen base model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        num_samples: int = 1000,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Initialize the KV cache dataset.
        
        Args:
            model: The frozen base model to extract KV cache from
            tokenizer: Tokenizer for the model
            dataset_name: Name of the dataset to use
            dataset_config: Configuration of the dataset
            num_samples: Number of samples to use
            max_length: Maximum sequence length
            split: Dataset split to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and prepare dataset
        print(f"Loading dataset: {dataset_name}/{dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        # Filter out empty texts and take only requested number of samples
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)  # Minimum text length
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length,
                return_tensors=None  # Keep as lists for now
            )
        
        print("Tokenizing dataset...")
        self.dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"Dataset ready with {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - input_ids: Input token IDs
            - attention_mask: Attention mask
        """
        item = self.dataset[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        } 