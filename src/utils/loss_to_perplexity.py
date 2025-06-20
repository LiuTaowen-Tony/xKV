#!/usr/bin/env python3
"""
Loss to Perplexity Evaluation Utilities

This module provides functions to convert between loss values and perplexity metrics,
as well as utilities for evaluating language model performance.

Perplexity is defined as the exponential of the average negative log-likelihood:
- Loss = -log_likelihood / num_tokens  (average negative log-likelihood)
- Perplexity = exp(Loss) = exp(-log_likelihood / num_tokens)

Lower perplexity indicates better model performance.
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
import math


def loss_to_perplexity(loss: Union[float, torch.Tensor, np.ndarray]) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Convert loss (negative log-likelihood) to perplexity.
    
    Args:
        loss: Loss value(s) - can be a single value, tensor, or array
        
    Returns:
        Perplexity value(s) in the same format as input
        
    Examples:
        >>> loss_to_perplexity(2.0)
        7.38905609893065
        
        >>> loss_to_perplexity(torch.tensor([1.0, 2.0, 3.0]))
        tensor([2.7183, 7.3891, 20.0855])
    """
    if isinstance(loss, torch.Tensor):
        return torch.exp(loss)
    elif isinstance(loss, np.ndarray):
        return np.exp(loss)
    else:
        return math.exp(float(loss))


def perplexity_to_loss(perplexity: Union[float, torch.Tensor, np.ndarray]) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Convert perplexity to loss (negative log-likelihood).
    
    Args:
        perplexity: Perplexity value(s) - can be a single value, tensor, or array
        
    Returns:
        Loss value(s) in the same format as input
        
    Examples:
        >>> perplexity_to_loss(7.389)
        1.9999...
        
        >>> perplexity_to_loss(torch.tensor([2.718, 7.389, 20.086]))
        tensor([1.0000, 2.0000, 3.0000])
    """
    if isinstance(perplexity, torch.Tensor):
        return torch.log(perplexity)
    elif isinstance(perplexity, np.ndarray):
        return np.log(perplexity)
    else:
        return math.log(float(perplexity))


def log_likelihood_to_perplexity(
    log_likelihood: Union[float, torch.Tensor, np.ndarray], 
    num_tokens: Union[int, torch.Tensor, np.ndarray]
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Convert log-likelihood to perplexity.
    
    Args:
        log_likelihood: Total log-likelihood (can be negative)
        num_tokens: Number of tokens used to compute the log-likelihood
        
    Returns:
        Perplexity value
        
    Examples:
        >>> log_likelihood_to_perplexity(-100.0, 50)
        7.38905609893065
    """
    # Perplexity = exp(-log_likelihood / num_tokens)
    avg_neg_log_likelihood = -log_likelihood / num_tokens
    return loss_to_perplexity(avg_neg_log_likelihood)


def calculate_perplexity_from_logits(
    logits: torch.Tensor, 
    target_ids: torch.Tensor, 
    ignore_index: int = -100,
    reduction: str = 'mean'
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Calculate perplexity from model logits and target token IDs.
    
    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs of shape [batch_size, seq_len]
        ignore_index: Token ID to ignore in loss calculation (e.g., padding tokens)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        
    Returns:
        Dictionary containing:
        - 'perplexity': Perplexity value
        - 'loss': Cross-entropy loss
        - 'log_likelihood': Total log-likelihood
        - 'num_tokens': Number of tokens used in calculation
    """
    # Shift logits and targets for causal language modeling
    # Predict next token, so shift targets to align with logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    # Flatten for loss calculation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Calculate cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    token_losses = loss_fn(shift_logits, shift_labels)
    
    # Mask out ignored tokens
    valid_tokens = (shift_labels != ignore_index)
    valid_losses = token_losses[valid_tokens]
    
    if len(valid_losses) == 0:
        return {
            'perplexity': float('inf'),
            'loss': float('inf'),
            'log_likelihood': float('-inf'),
            'num_tokens': 0
        }
    
    # Calculate statistics
    if reduction == 'mean':
        avg_loss = valid_losses.mean()
    elif reduction == 'sum':
        avg_loss = valid_losses.sum() / len(valid_losses)
    else:  # reduction == 'none'
        avg_loss = valid_losses
    
    num_tokens = len(valid_losses)
    total_log_likelihood = -valid_losses.sum().item()
    
    # Calculate perplexity
    if reduction == 'none':
        perplexity = torch.exp(avg_loss)
    else:
        perplexity = torch.exp(avg_loss).item()
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss.item() if reduction != 'none' else avg_loss,
        'log_likelihood': total_log_likelihood,
        'num_tokens': num_tokens
    }


def evaluate_sequence_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: str = 'cuda',
    max_length: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate perplexity for a single text sequence.
    
    Args:
        model: Language model
        tokenizer: Tokenizer corresponding to the model
        text: Input text to evaluate
        device: Device to run evaluation on
        max_length: Maximum sequence length (truncate if longer)
        
    Returns:
        Dictionary containing perplexity metrics
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids, labels=input_ids)
        
        # Calculate perplexity from logits
        logits = outputs.logits
        results = calculate_perplexity_from_logits(logits, input_ids)
        
        # Add sequence length info
        results['sequence_length'] = input_ids.size(1)
        results['text_length'] = len(text)
        
    return results


def batch_evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: Optional[int] = None
) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluate perplexity for a batch of text sequences.
    
    Args:
        model: Language model
        tokenizer: Tokenizer corresponding to the model
        texts: List of input texts to evaluate
        device: Device to run evaluation on
        batch_size: Batch size for processing
        max_length: Maximum sequence length (truncate if longer)
        
    Returns:
        Dictionary containing aggregated perplexity metrics
    """
    model.eval()
    
    all_perplexities = []
    all_losses = []
    total_tokens = 0
    total_log_likelihood = 0.0
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', None)
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            
            # Calculate perplexity for each sequence in batch
            logits = outputs.logits
            
            for j in range(input_ids.size(0)):
                # Extract single sequence
                seq_logits = logits[j:j+1]
                seq_input_ids = input_ids[j:j+1]
                
                # Calculate perplexity
                results = calculate_perplexity_from_logits(seq_logits, seq_input_ids)
                
                all_perplexities.append(results['perplexity'])
                all_losses.append(results['loss'])
                total_tokens += results['num_tokens']
                total_log_likelihood += results['log_likelihood']
    
    # Calculate aggregate statistics
    mean_perplexity = np.mean(all_perplexities)
    median_perplexity = np.median(all_perplexities)
    std_perplexity = np.std(all_perplexities)
    min_perplexity = np.min(all_perplexities)
    max_perplexity = np.max(all_perplexities)
    
    mean_loss = np.mean(all_losses)
    avg_log_likelihood = total_log_likelihood / total_tokens if total_tokens > 0 else float('-inf')
    
    return {
        'mean_perplexity': mean_perplexity,
        'median_perplexity': median_perplexity,
        'std_perplexity': std_perplexity,
        'min_perplexity': min_perplexity,
        'max_perplexity': max_perplexity,
        'mean_loss': mean_loss,
        'avg_log_likelihood': avg_log_likelihood,
        'total_tokens': total_tokens,
        'num_sequences': len(all_perplexities),
        'individual_perplexities': all_perplexities,
        'individual_losses': all_losses
    }


def compare_model_perplexities(
    results1: Dict,
    results2: Dict,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Union[float, str]]:
    """
    Compare perplexity results between two models.
    
    Args:
        results1: Perplexity results from first model
        results2: Perplexity results from second model
        model1_name: Name/description of first model
        model2_name: Name/description of second model
        
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_perplexity': results1.get('mean_perplexity', results1.get('perplexity')),
        'model2_perplexity': results2.get('mean_perplexity', results2.get('perplexity')),
    }
    
    ppl1 = comparison['model1_perplexity']
    ppl2 = comparison['model2_perplexity']
    
    if ppl1 and ppl2:
        comparison['perplexity_ratio'] = ppl2 / ppl1
        comparison['perplexity_difference'] = ppl2 - ppl1
        comparison['improvement_pct'] = ((ppl1 - ppl2) / ppl1) * 100
        
        if ppl1 < ppl2:
            comparison['better_model'] = model1_name
            comparison['improvement'] = f"{model1_name} is {comparison['improvement_pct']:.2f}% better"
        elif ppl2 < ppl1:
            comparison['better_model'] = model2_name
            comparison['improvement'] = f"{model2_name} is {-comparison['improvement_pct']:.2f}% better"
        else:
            comparison['better_model'] = "Tie"
            comparison['improvement'] = "Both models perform equally"
    
    return comparison


def print_perplexity_summary(results: Dict, title: str = "Perplexity Evaluation Results"):
    """
    Print a formatted summary of perplexity evaluation results.
    
    Args:
        results: Dictionary containing perplexity metrics
        title: Title for the summary
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Single sequence results
    if 'perplexity' in results and isinstance(results['perplexity'], (int, float)):
        print(f"Perplexity:           {results['perplexity']:.4f}")
        print(f"Loss:                 {results.get('loss', 'N/A'):.4f}")
        print(f"Log Likelihood:       {results.get('log_likelihood', 'N/A'):.4f}")
        print(f"Number of Tokens:     {results.get('num_tokens', 'N/A')}")
        print(f"Sequence Length:      {results.get('sequence_length', 'N/A')}")
    
    # Batch/dataset results
    elif 'mean_perplexity' in results:
        print(f"Mean Perplexity:      {results['mean_perplexity']:.4f}")
        print(f"Median Perplexity:    {results['median_perplexity']:.4f}")
        print(f"Std Perplexity:       {results['std_perplexity']:.4f}")
        print(f"Min Perplexity:       {results['min_perplexity']:.4f}")
        print(f"Max Perplexity:       {results['max_perplexity']:.4f}")
        print(f"Mean Loss:            {results['mean_loss']:.4f}")
        print(f"Avg Log Likelihood:   {results['avg_log_likelihood']:.4f}")
        print(f"Total Tokens:         {results['total_tokens']}")
        print(f"Number of Sequences:  {results['num_sequences']}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage and tests
    print("Loss to Perplexity Conversion Examples:")
    print("-" * 40)
    
    # Test basic conversions
    loss_vals = [0.5, 1.0, 2.0, 3.0]
    for loss in loss_vals:
        ppl = loss_to_perplexity(loss)
        recovered_loss = perplexity_to_loss(ppl)
        print(f"Loss: {loss:.2f} -> Perplexity: {ppl:.4f} -> Recovered Loss: {recovered_loss:.4f}")
    
    print("\nTensor Operations:")
    print("-" * 20)
    
    # Test tensor operations
    loss_tensor = torch.tensor([0.5, 1.0, 2.0, 3.0])
    ppl_tensor = loss_to_perplexity(loss_tensor)
    recovered_tensor = perplexity_to_loss(ppl_tensor)
    
    print(f"Loss Tensor:      {loss_tensor}")
    print(f"Perplexity:       {ppl_tensor}")
    print(f"Recovered Loss:   {recovered_tensor}")
    
    print("\nLog-likelihood to Perplexity:")
    print("-" * 30)
    
    # Test log-likelihood conversion
    log_likelihood = -100.0
    num_tokens = 50
    ppl = log_likelihood_to_perplexity(log_likelihood, num_tokens)
    print(f"Log-likelihood: {log_likelihood}, Tokens: {num_tokens} -> Perplexity: {ppl:.4f}")
