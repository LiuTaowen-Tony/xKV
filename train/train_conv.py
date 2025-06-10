#!/usr/bin/env python3
"""
Training script for ConvolutionalCompressor.
Uses strided convolutions for memory-efficient compression.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse

# Import our modules using relative imports
from .kv_cache_collector import KVCacheCollector
from .model import ConvolutionalCompressor, ConvolutionalCompressorConfig
from .utils import load_model_and_tokenizer, print_model_info
from .kv_lightning_module import KVCompressorLightningModule
from .kv_dataset import KVCacheDataset
torch.set_float32_matmul_precision('medium')


def main():
    # Simple argument parser
    parser = argparse.ArgumentParser(description='Train Convolutional KV Cache Compressor')
    parser.add_argument('--model_name', type=str, default="unsloth/Llama-3.2-1B-Instruct",
                        help='Base model to use')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for convolution')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--layer_stride', type=int, default=2,
                        help='Stride for layer dimension compression')
    parser.add_argument('--seq_stride', type=int, default=2,
                        help='Stride for sequence dimension compression')
    parser.add_argument('--hidden_channels', type=int, default=2048,
                        help='Number of hidden channels in conv layers')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Convolutional KV Cache Compressor Training")
    print("=" * 60)
    
    # Load base model
    print(f"\n📥 Loading base model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print_model_info(model, "Base Model")
    
    # Get model configuration
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    
    print(f"\n🔧 Model Configuration:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Layer stride: {args.layer_stride}")
    print(f"  - Sequence stride: {args.seq_stride}")
    print(f"  - Hidden channels: {args.hidden_channels}")
    
    # Create compressor configuration
    compressor_config = ConvolutionalCompressorConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        kv_dim=512,  # For Llama-3.2-1B: K/V projection output is 512 each
        layer_stride=args.layer_stride,
        seq_stride=args.seq_stride,
        kernel_size=3,
        hidden_channels=args.hidden_channels,
        activation="gelu",
        dropout_rate=0.1
    )
    
    # Calculate compression statistics
    # For conv2d with stride, output size = (input_size + 2*padding - kernel_size) / stride + 1
    compressed_layers = max(1, (num_layers + 2 * 1 - 3) // args.layer_stride + 1)
    compressed_seq = max(1, (args.max_length + 2 * 1 - 3) // args.seq_stride + 1)
    compressed_seq = max(1, (compressed_seq + 2 * 1 - 3) // 2 + 1)  # Second conv layer
    
    input_size = num_layers * args.max_length * 512 * 2  # K+V = 512+512 = 1024 total
    compressed_size = compressed_layers * compressed_seq * (args.hidden_channels // 2)
    theoretical_compression_ratio = input_size / compressed_size
    
    print(f"  - Input size: {input_size}")
    print(f"  - Compressed size: {compressed_size}")
    print(f"  - Theoretical compression ratio: {theoretical_compression_ratio:.1f}x")
    
    # Create dataset
    print(f"\n📊 Creating dataset...")
    dataset = KVCacheDataset(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
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
        num_workers=32,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    
    # Create Lightning module
    print(f"\n🧠 Creating compressor...")
    lightning_module = KVCompressorLightningModule(
        config=compressor_config,
        base_model=model,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        use_vae=False,
        compressor_type="convolutional"
    )
    
    print_model_info(lightning_module.compressor, "Compressor Model")
    
    # Set up logging
    logger = WandbLogger(
        project="kv-cache-compression",
        name=f"conv_compressor_s{args.layer_stride}-{args.seq_stride}",
        save_dir="./logs"
    )
    logger.log_hyperparams(vars(args))
    
    # Set up callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=f"./checkpoints/conv_s{args.layer_stride}-{args.seq_stride}",
    #     filename='{epoch}-{val_loss:.4f}',
    #     monitor='val_loss',
    #     mode='min',
    #     save_top_k=2,
    #     save_last=True
    # )
    
    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     mode='min',
    #     verbose=True
    # )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision='bf16-mixed',
        logger=logger,
        # callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )
    
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
        
        print(f"  - Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  - Reconstruction MSE (K): {stats['k_mse']:.6f}")
        print(f"  - Reconstruction MSE (V): {stats['v_mse']:.6f}")
        print(f"  - Total MSE: {stats['total_mse']:.6f}")


if __name__ == "__main__":
    main() 