#!/usr/bin/env python3
"""
Training script for VAE Convolutional KV Compressor.

This script trains a VAE convolutional compressor that combines the efficiency
of convolutional compression with probabilistic modeling via VAE.
"""

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import VAEConvolutionalCompressorConfig
from .kv_lightning_module import KVCompressorLightningModule
from .kv_dataset import KVDataModule


def main():
    parser = argparse.ArgumentParser(description="Train VAE Convolutional KV Compressor")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="microsoft/Phi-3.5-mini-instruct",
                       help="Hugging Face model name")
    parser.add_argument("--num_layers", type=int, default=32, 
                       help="Number of transformer layers")
    parser.add_argument("--hidden_dim", type=int, default=3072, 
                       help="Hidden dimension of the model")
    parser.add_argument("--kv_dim", type=int, default=1024,
                       help="Actual K/V projection dimension (for GQA models)")
    
    # Compression arguments
    parser.add_argument("--layer_stride", type=int, default=2,
                       help="Stride for layer dimension compression")
    parser.add_argument("--seq_stride", type=int, default=2,
                       help="Stride for sequence dimension compression")
    parser.add_argument("--kernel_size", type=int, default=3,
                       help="Kernel size for convolution")
    parser.add_argument("--hidden_channels", type=int, default=256,
                       help="Number of hidden channels in conv layers")
    parser.add_argument("--latent_channels", type=int, default=128,
                       help="Number of channels in VAE latent space")
    parser.add_argument("--compression_depth", type=int, default=3,
                       help="Number of compression stages")
    
    # VAE arguments
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Beta parameter for KL divergence weight")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                       help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10, 
                       help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, default=10000, 
                       help="Maximum number of training steps")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, 
                       help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, 
                       help="Number of batches to accumulate gradients")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, 
                       default="wikitext", 
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, 
                       default="wikitext-103-raw-v1", 
                       help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loading workers")
    
    # Logging and checkpointing
    parser.add_argument("--project_name", type=str, 
                       default="kv-compression-vae-conv", 
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, 
                       default=None, 
                       help="W&B run name")
    parser.add_argument("--log_every_n_steps", type=int, default=50, 
                       help="Log every N steps")
    parser.add_argument("--save_top_k", type=int, default=3, 
                       help="Save top K checkpoints")
    parser.add_argument("--patience", type=int, default=5, 
                       help="Early stopping patience")
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="gpu", 
                       help="Accelerator type")
    parser.add_argument("--devices", type=int, default=1, 
                       help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", 
                       help="Training precision")
    
    args = parser.parse_args()
    
    # Create compressor config
    config = VAEConvolutionalCompressorConfig(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        kv_dim=args.kv_dim,
        layer_stride=args.layer_stride,
        seq_stride=args.seq_stride,
        kernel_size=args.kernel_size,
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        compression_depth=args.compression_depth,
        beta=args.beta,
        activation="gelu",
        dropout_rate=0.1,
        use_residual=True,
        use_attention=True,
        use_layer_norm=True
    )
    
    # Calculate theoretical compression ratio
    # Input: batch_size × num_layers × seq_len × (2 * kv_dim)
    # After layer/seq compression: batch_size × (num_layers//layer_stride) × (seq_len//seq_stride) × latent_channels
    seq_compression = args.seq_stride ** min(2, args.compression_depth)
    layer_compression = args.layer_stride
    channel_compression = (2 * args.kv_dim) / args.latent_channels
    theoretical_compression = seq_compression * layer_compression * channel_compression
    
    print(f"Theoretical compression ratio: {theoretical_compression:.1f}x")
    print(f"  - Sequence compression: {seq_compression}x")
    print(f"  - Layer compression: {layer_compression}x") 
    print(f"  - Channel compression: {channel_compression:.1f}x")
    
    # Load base model for KV cache generation
    print(f"Loading model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create lightning module
    model = KVCompressorLightningModule(
        config=config,
        base_model=base_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        compressor_type="vae_convolutional"
    )
    
    # Create data module
    data_module = KVDataModule(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Set up logging
    if args.run_name is None:
        args.run_name = f"vae-conv-{args.hidden_channels}ch-{args.latent_channels}lat"
    
    logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        log_model=False  # Don't upload model to W&B to save space
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=args.save_top_k,
            filename='vae-conv-{epoch:02d}-{val_loss:.4f}',
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=args.patience,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,  # Set to True for reproducible results (but slower)
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Print model summary
    print("\n" + "="*50)
    print("VAE CONVOLUTIONAL COMPRESSOR SUMMARY")
    print("="*50)
    print(f"Input channels: {2 * args.kv_dim}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Latent channels: {args.latent_channels}")
    print(f"Compression depth: {args.compression_depth}")
    print(f"Layer stride: {args.layer_stride}")
    print(f"Sequence stride: {args.seq_stride}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Beta (KL weight): {args.beta}")
    print("="*50)
    
    # Start training
    try:
        trainer.fit(model, data_module)
        print("\nTraining completed successfully!")
        
        # Print final metrics
        if trainer.callback_metrics:
            print("\nFinal metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
                else:
                    print(f"  {key}: {value}")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 