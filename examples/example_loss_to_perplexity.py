#!/usr/bin/env python3
"""
Example script demonstrating how to use the loss_to_perplexity module
with the existing perplexity evaluation infrastructure.
"""

import sys
import os
import torch
import numpy as np

# Add the root directory to the path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.utils.loss_to_perplexity import (
    loss_to_perplexity,
    perplexity_to_loss,
    log_likelihood_to_perplexity,
    calculate_perplexity_from_logits,
    evaluate_sequence_perplexity,
    batch_evaluate_perplexity,
    compare_model_perplexities,
    print_perplexity_summary
)


def example_basic_conversions():
    """Demonstrate basic loss to perplexity conversions."""
    print("="*60)
    print("EXAMPLE 1: Basic Loss to Perplexity Conversions")
    print("="*60)
    
    # Example losses from different scenarios
    scenarios = [
        ("Very low loss (excellent model)", 0.1),
        ("Low loss (good model)", 0.5),
        ("Moderate loss (average model)", 1.0),
        ("High loss (poor model)", 2.0),
        ("Very high loss (very poor model)", 4.0),
    ]
    
    for description, loss in scenarios:
        perplexity = loss_to_perplexity(loss)
        print(f"{description:35} | Loss: {loss:4.1f} | Perplexity: {perplexity:7.2f}")
    
    print("\nNote: Lower perplexity indicates better model performance.")
    print()


def example_tensor_operations():
    """Demonstrate tensor operations with loss and perplexity."""
    print("="*60)
    print("EXAMPLE 2: Tensor Operations")
    print("="*60)
    
    # Create a batch of losses
    batch_losses = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
    batch_perplexities = loss_to_perplexity(batch_losses)
    
    print("Batch Loss to Perplexity Conversion:")
    print(f"Losses:      {batch_losses}")
    print(f"Perplexities: {batch_perplexities}")
    
    # Convert back to verify
    recovered_losses = perplexity_to_loss(batch_perplexities)
    print(f"Recovered:   {recovered_losses}")
    print(f"Match original: {torch.allclose(batch_losses, recovered_losses)}")
    print()


def example_log_likelihood_conversion():
    """Demonstrate log-likelihood to perplexity conversion."""
    print("="*60)
    print("EXAMPLE 3: Log-likelihood to Perplexity")
    print("="*60)
    
    # Simulate evaluation results from different models
    scenarios = [
        ("Small model, short sequence", -50.0, 25),
        ("Large model, short sequence", -40.0, 25),
        ("Small model, long sequence", -200.0, 100),
        ("Large model, long sequence", -150.0, 100),
    ]
    
    for description, log_likelihood, num_tokens in scenarios:
        perplexity = log_likelihood_to_perplexity(log_likelihood, num_tokens)
        avg_loss = -log_likelihood / num_tokens
        print(f"{description:30} | Log-likelihood: {log_likelihood:6.1f} | "
              f"Tokens: {num_tokens:3d} | Avg Loss: {avg_loss:.2f} | Perplexity: {perplexity:.2f}")
    print()


def example_logits_evaluation():
    """Demonstrate perplexity calculation from model logits."""
    print("="*60)
    print("EXAMPLE 4: Perplexity from Model Logits")
    print("="*60)
    
    # Simulate model outputs
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    # Create mock logits (higher values for correct tokens)
    torch.manual_seed(42)  # For reproducibility
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create target token IDs
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Make the logits slightly favor the target tokens (simulate a decent model)
    for i in range(batch_size):
        for j in range(seq_len):
            logits[i, j, target_ids[i, j]] += 2.0  # Boost correct token probability
    
    # Calculate perplexity
    results = calculate_perplexity_from_logits(logits, target_ids)
    
    print("Mock Model Evaluation Results:")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Log-likelihood: {results['log_likelihood']:.4f}")
    print(f"Number of tokens: {results['num_tokens']}")
    print()


def example_model_comparison():
    """Demonstrate comparing perplexity results between models."""
    print("="*60)
    print("EXAMPLE 5: Model Comparison")
    print("="*60)
    
    # Simulate results from two different models
    model1_results = {
        'mean_perplexity': 15.2,
        'median_perplexity': 14.8,
        'std_perplexity': 3.2,
        'total_tokens': 10000,
        'num_samples': 100
    }
    
    model2_results = {
        'mean_perplexity': 12.8,
        'median_perplexity': 12.1,
        'std_perplexity': 2.8,
        'total_tokens': 10000,
        'num_samples': 100
    }
    
    # Compare the two models
    comparison = compare_model_perplexities(
        model1_results, 
        model2_results,
        "Baseline Model",
        "Compressed Model"
    )
    
    print("Model Comparison Results:")
    print(f"Baseline Model Perplexity: {comparison['model1_perplexity']:.2f}")
    print(f"Compressed Model Perplexity: {comparison['model2_perplexity']:.2f}")
    print(f"Better Model: {comparison['better_model']}")
    print(f"Improvement: {comparison['improvement']}")
    print(f"Perplexity Ratio: {comparison['perplexity_ratio']:.3f}")
    print()


def example_summary_formatting():
    """Demonstrate formatted summary output."""
    print("="*60)
    print("EXAMPLE 6: Formatted Summary")
    print("="*60)
    
    # Simulate comprehensive evaluation results
    results = {
        'mean_perplexity': 18.45,
        'median_perplexity': 17.2,
        'std_perplexity': 4.1,
        'min_perplexity': 8.9,
        'max_perplexity': 45.7,
        'mean_loss': 2.91,
        'avg_log_likelihood': -2.91,
        'total_tokens': 25000,
        'num_sequences': 500
    }
    
    print_perplexity_summary(results, "Dataset Evaluation Results")


def example_integration_with_existing_code():
    """Show how to integrate with existing perplexity evaluation."""
    print("="*60)
    print("EXAMPLE 7: Integration with Existing Code")
    print("="*60)
    
    # This example shows how to use the loss_to_perplexity module
    # with results from the existing perplexity evaluation scripts
    
    print("Integration examples:")
    print("1. Convert losses from training to perplexity for monitoring")
    print("2. Compare perplexity between different model configurations")
    print("3. Analyze perplexity distributions across datasets")
    print("4. Convert between different evaluation metrics")
    print()
    
    # Example: Converting training losses to perplexity for monitoring
    training_losses = [3.2, 2.8, 2.5, 2.2, 2.0, 1.9, 1.8, 1.75]
    epochs = list(range(1, len(training_losses) + 1))
    
    print("Training Progress (Loss -> Perplexity):")
    print("Epoch | Loss  | Perplexity")
    print("------|-------|----------")
    for epoch, loss in zip(epochs, training_losses):
        ppl = loss_to_perplexity(loss)
        print(f"{epoch:5d} | {loss:5.2f} | {ppl:8.2f}")
    print()


def main():
    """Run all examples."""
    print("Loss to Perplexity Evaluation - Examples")
    print("="*60)
    print("This script demonstrates various uses of the loss_to_perplexity module.")
    print()
    
    # Run all examples
    example_basic_conversions()
    example_tensor_operations()
    example_log_likelihood_conversion()
    example_logits_evaluation()
    example_model_comparison()
    example_summary_formatting()
    example_integration_with_existing_code()
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main() 