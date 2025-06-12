#!/usr/bin/env python3
"""
Training script for Dual1D Convolutional KV Cache Compressor using PyTorch Lightning.
Integrates with existing Lightning infrastructure for training and evaluation.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse
import os

from .kv_lightning_module import KVCompressorLightningModule
from .kv_dataset import KVCacheDataset
from .model import Dual1DConvolutionalCompressorConfig
from .utils import load_model_and_tokenizer, print_model_info

torch.set_float32_matmul_precision('medium')


def create_dual1d_config(args):
    """Create Dual1D compressor configuration from arguments."""
    # Note: For the KV cache collector, kv_dim should be the full hidden_dim since
    # it collects from k_proj/v_proj which output the full dimension before reshaping
    actual_kv_dim = args.hidden_dim if args.kv_dim is None else args.kv_dim
    
    return Dual1DConvolutionalCompressorConfig(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        kv_dim=actual_kv_dim,  # Use full hidden_dim for KV cache collector
        layer_stride=args.layer_stride,
        seq_stride=args.seq_stride,
        kernel_size=args.kernel_size,
        hidden_channels=args.hidden_channels,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        use_activation=args.use_activation,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm
    )


def run_activation_ablation(args, model, tokenizer):
    """Run activation function ablation study."""
    print("\n🧪 Running Activation Function Ablation Study")
    print("=" * 60)
    
    results = {}
    
    # Test with and without activation functions
    for use_activation in [True, False]:
        activation_name = "with_activation" if use_activation else "without_activation"
        print(f"\n📊 Training model {activation_name}...")
        
        # Update args for this run
        args.use_activation = use_activation
        config = create_dual1d_config(args)
        
        # Calculate theoretical compression ratio
        input_size = args.num_layers * args.max_length * args.kv_dim * 2  # K + V
        layer_compressed = args.num_layers // args.layer_stride
        seq_compressed = args.max_length // args.seq_stride
        compressed_size = layer_compressed * seq_compressed * (args.hidden_channels // 2)
        theoretical_compression_ratio = input_size / compressed_size
        
        print(f"  - Theoretical compression: {theoretical_compression_ratio:.1f}x")
        try:
            from .model import Dual1DConvolutionalCompressor
        except ImportError:
            from model import Dual1DConvolutionalCompressor
        print(f"  - Model parameters: {sum(p.numel() for p in Dual1DConvolutionalCompressor(config).parameters()):,}")
        
        # Create dataset
        dataset = KVCacheDataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            num_samples=args.num_samples,
            max_length=args.max_length
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create Lightning module
        lightning_module = KVCompressorLightningModule(
            config=config,
            base_model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            compressor_type="dual1d_convolutional"
        )
        
        # Set up logging
        logger = WandbLogger(
            project=args.project_name,
            name=f"dual1d_{activation_name}",
            save_dir="./logs"
        ) if args.use_wandb else None
        
        # Set up callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/dual1d_{activation_name}",
            filename='{epoch}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitor
        if args.use_wandb:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        
        # Create trainer
        trainer_kwargs = {
            'max_epochs': args.max_epochs,
            'precision': 'bf16-mixed' if args.use_mixed_precision else 32,
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': args.log_every_n_steps,
            'gradient_clip_val': args.gradient_clip_val,
            'enable_progress_bar': True,
            'devices': 1,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu'
        }
        
        # Only add max_steps if it's positive
        if args.max_steps > 0:
            trainer_kwargs['max_steps'] = args.max_steps
            
        trainer = pl.Trainer(**trainer_kwargs)
        
        # Train model
        trainer.fit(lightning_module, train_loader, val_loader)
        
        # Get best validation loss
        best_val_loss = checkpoint_callback.best_model_score.item()
        
        # Test compression on validation set
        lightning_module.eval()
        sample_batch = next(iter(val_loader))
        with torch.no_grad():
            input_ids = sample_batch['input_ids']
            attention_mask = sample_batch['attention_mask']
            
            # Generate KV cache
            k_cache, v_cache = lightning_module._generate_kv_cache(input_ids, attention_mask)
            
            # Get compression stats
            stats = lightning_module.get_compression_stats(k_cache, v_cache)
        
        results[activation_name] = {
            'best_val_loss': best_val_loss,
            'compression_ratio': stats['compression_ratio'],
            'k_mse': stats['k_mse'],
            'v_mse': stats['v_mse'],
            'total_mse': stats['total_mse']
        }
        
        print(f"  ✅ {activation_name} completed:")
        print(f"     - Best val loss: {best_val_loss:.6f}")
        print(f"     - Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"     - Reconstruction MSE: {stats['total_mse']:.6f}")
    
    # Compare results
    print(f"\n📈 Ablation Results Comparison")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name:>15}: Val Loss = {result['best_val_loss']:.6f}, "
              f"Compression = {result['compression_ratio']:.2f}x, "
              f"MSE = {result['total_mse']:.6f}")
    
    if 'with_activation' in results and 'without_activation' in results:
        improvement = (results['without_activation']['best_val_loss'] - 
                      results['with_activation']['best_val_loss']) / results['without_activation']['best_val_loss'] * 100
        print(f"\n🎯 Activation functions improve validation loss by {improvement:.2f}%")
        
        # Save ablation results
        import json
        ablation_file = os.path.join("./experiments", "dual1d_ablation_results.json")
        os.makedirs(os.path.dirname(ablation_file), exist_ok=True)
        with open(ablation_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {ablation_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Dual1D Convolutional KV Cache Compressor with Lightning")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                       help="Base model name")
    parser.add_argument("--num_layers", type=int, default=16,
                       help="Number of transformer layers")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                       help="Hidden dimension")
    parser.add_argument("--kv_dim", type=int, default=512,
                       help="KV dimension (typically hidden_dim//4 for GQA)")
    
    # Dual1D specific arguments
    parser.add_argument("--layer_stride", type=int, default=2,
                       help="Stride for layer dimension compression")
    parser.add_argument("--seq_stride", type=int, default=2,
                       help="Stride for sequence dimension compression")
    parser.add_argument("--kernel_size", type=int, default=3,
                       help="Convolution kernel size")
    parser.add_argument("--hidden_channels", type=int, default=256,
                       help="Hidden channels in conv layers")
    parser.add_argument("--activation", type=str, default="gelu", 
                       choices=["gelu", "swish", "mish"],
                       help="Activation function")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--use_activation", action="store_true", default=True,
                       help="Use activation functions (for ablation)")
    parser.add_argument("--use_residual", action="store_true", default=True,
                       help="Use residual connections")
    parser.add_argument("--use_layer_norm", action="store_true", default=True,
                       help="Use layer normalization")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum number of steps (-1 for no limit)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to use")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Experiment arguments
    parser.add_argument("--project_name", type=str, default="kv-cache-compression",
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="W&B run name")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--use_mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                       help="Log every n steps")
    parser.add_argument("--run_ablation", action="store_true",
                       help="Run activation function ablation study")
    
    args = parser.parse_args()
    
    # Load base model and tokenizer
    print(f"🤗 Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print_model_info(model, "Base Model")
    
    if args.run_ablation:
        # Run ablation study
        run_activation_ablation(args, model, tokenizer)
    else:
        # Single training run
        print(f"\n🚀 Starting Dual1D Compressor Training")
        print("=" * 60)
        
        # Create configuration
        config = create_dual1d_config(args)
        
        # Calculate theoretical compression ratio
        input_size = args.num_layers * args.max_length * args.kv_dim * 2  # K + V
        layer_compressed = args.num_layers // args.layer_stride
        seq_compressed = args.max_length // args.seq_stride
        compressed_size = layer_compressed * seq_compressed * (args.hidden_channels // 2)
        theoretical_compression_ratio = input_size / compressed_size
        
        print(f"  - Input size: {input_size:,}")
        print(f"  - Compressed size: {compressed_size:,}")
        print(f"  - Theoretical compression: {theoretical_compression_ratio:.1f}x")
        
        # Create dataset
        print(f"\n📊 Creating dataset...")
        dataset = KVCacheDataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            num_samples=args.num_samples,
            max_length=args.max_length
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create Lightning module
        print(f"\n🧠 Creating compressor...")
        lightning_module = KVCompressorLightningModule(
            config=config,
            base_model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            compressor_type="dual1d_convolutional"
        )
        
        print_model_info(lightning_module.compressor, "Dual1D Compressor")
        
        # Set up logging
        if args.run_name is None:
            args.run_name = f"dual1d-{args.layer_stride}x{args.seq_stride}-{args.hidden_channels}ch"
        
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            save_dir="./logs"
        ) if args.use_wandb else None
        
        if logger:
            logger.log_hyperparams(vars(args))
        
        # Set up callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/dual1d_{args.run_name}",
            filename='{epoch}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=2,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitor
        if args.use_wandb:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        
        # Create trainer
        trainer_kwargs = {
            'max_epochs': args.max_epochs,
            'precision': 'bf16-mixed' if args.use_mixed_precision else 32,
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': args.log_every_n_steps,
            'gradient_clip_val': args.gradient_clip_val,
            'enable_progress_bar': True,
            'devices': 1,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu'
        }
        
        # Only add max_steps if it's positive
        if args.max_steps > 0:
            trainer_kwargs['max_steps'] = args.max_steps
            
        trainer = pl.Trainer(**trainer_kwargs)
        
        print(f"\n🏋️ Starting training...")
        print(f"  - Max epochs: {args.max_epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Device: {trainer.strategy.root_device}")
        
        # Start training
        trainer.fit(lightning_module, train_loader, val_loader)
        
        print(f"\n✅ Training completed!")
        
        # Test the trained model
        print(f"\n🧪 Testing compression...")
        lightning_module.eval()
        
        # Get a sample batch for testing
        sample_batch = next(iter(val_loader))
        with torch.no_grad():
            input_ids = sample_batch['input_ids']
            attention_mask = sample_batch['attention_mask']
            
            # Generate KV cache
            k_cache, v_cache = lightning_module._generate_kv_cache(input_ids, attention_mask)
            
            # Get compression stats
            stats = lightning_module.get_compression_stats(k_cache, v_cache)
            
            print(f"  - Actual compression ratio: {stats['compression_ratio']:.2f}x")
            print(f"  - Reconstruction MSE (K): {stats['k_mse']:.6f}")
            print(f"  - Reconstruction MSE (V): {stats['v_mse']:.6f}")
            print(f"  - Total MSE: {stats['total_mse']:.6f}")


if __name__ == "__main__":
    main() 