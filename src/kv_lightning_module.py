import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union
import torchmetrics

from .compressor import (
    ConvolutionalCompressor,
    ConvolutionalCompressorConfig,
    EnhancedConvolutionalCompressor,
    EnhancedConvolutionalCompressorConfig,
    VAEConvolutionalCompressor,
    VAEConvolutionalCompressorConfig,
    Dual1DConvolutionalCompressor,
    Dual1DConvolutionalCompressorConfig
)
from .kv_cache_collector import KVCacheCollector


class KVCompressorLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training KV cache compressor.
    Supports basic, improved, and VAE modes.
    """

    def __init__(
        self,
        config: Union[ConvolutionalCompressorConfig, EnhancedConvolutionalCompressorConfig, VAEConvolutionalCompressorConfig, Dual1DConvolutionalCompressorConfig],
        base_model,  # Frozen base model for KV cache generation
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = None,
        loss_weights: Dict[str, float] = None,
        use_vae: bool = False,
        compressor_type: str = "convolutional"
    ):
        super().__init__()

        # Save hyperparameters (exclude base_model as it's not serializable)
        self.save_hyperparameters(ignore=['base_model'])

        # Model configuration
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_vae = use_vae
        self.compressor_type = compressor_type

        # Set up frozen base model and KV cache collector
        self.base_model = base_model
        self.base_model.eval()  # Ensure base model is in eval mode

        # Enable gradient checkpointing for base model to save memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Set up KV cache collector for on-the-fly generation
        self.kv_collector = KVCacheCollector(self.base_model)

        # Loss weights for different components
        if use_vae:
            self.loss_weights = loss_weights or {
                'reconstruction': 1.0,
                'kl_divergence': config.beta if hasattr(config, 'beta') else 1.0
            }
        else:
            self.loss_weights = loss_weights or {
                'reconstruction': 1.0
            }

        # Create the compressor model based on type
        if compressor_type == "convolutional":
            self.compressor = ConvolutionalCompressor(config)
        elif compressor_type == "enhanced_convolutional":
            self.compressor = EnhancedConvolutionalCompressor(config)
        elif compressor_type == "vae_convolutional":
            self.compressor = VAEConvolutionalCompressor(config)
        elif compressor_type == "dual1d_convolutional":
            self.compressor = Dual1DConvolutionalCompressor(config)
        else:
            raise ValueError(f"Unknown compressor type: {compressor_type}")

        # Metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        # Track best validation loss
        self.best_val_loss = float('inf')

    def forward(self, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the compressor.
        
        Args:
            k_tensor: Key tensor [batch_size, num_layers, seq_len, hidden_dim]
            v_tensor: Value tensor [batch_size, num_layers, seq_len, hidden_dim]
            
        Returns:
            Tuple of outputs depending on mode:
            - Basic mode: (k_recon, v_recon, compressed)
            - Improved mode: (k_recon, v_recon, compressed)
            - VAE mode: (k_recon, v_recon, z, mu, logvar)
        """
        if self.use_vae or self.compressor_type == "vae_convolutional":
            z, mu, logvar, shape_info = self.compressor.compress(k_tensor, v_tensor)
            k_recon, v_recon = self.compressor.decompress(z, shape_info)
            return k_recon, v_recon, z, mu, logvar
        elif self.compressor_type in ["basic", "convolutional", "enhanced_convolutional"]:
            compressed = self.compressor.compress(k_tensor, v_tensor)
            batch_size, seq_len = k_tensor.shape[0], k_tensor.shape[2]
            k_recon, v_recon = self.compressor.decompress(compressed, batch_size, seq_len)
            return k_recon, v_recon, compressed
        elif self.compressor_type == "dual1d_convolutional":
            compressed, shape_info = self.compressor.compress(k_tensor, v_tensor)
            k_recon, v_recon = self.compressor.decompress(compressed, shape_info)
            return k_recon, v_recon, compressed
        else:  # improved
            compressed, shape_info = self.compressor.compress(k_tensor, v_tensor)
            k_recon, v_recon = self.compressor.decompress(compressed, shape_info)
            return k_recon, v_recon, compressed

    def compute_reconstruction_loss(self, k_orig: torch.Tensor, v_orig: torch.Tensor,
                                  k_recon: torch.Tensor, v_recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss between original and reconstructed tensors."""
        k_loss = F.mse_loss(k_recon, k_orig)
        v_loss = F.mse_loss(v_recon, v_orig)
        return k_loss + v_loss

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for VAE."""
        return self.compressor.compute_kl_loss(mu, logvar)

    def compute_total_loss(self, k_orig: torch.Tensor, v_orig: torch.Tensor,
                          k_recon: torch.Tensor, v_recon: torch.Tensor,
                          mu: torch.Tensor = None, logvar: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute total loss with different components."""

        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(k_orig, v_orig, k_recon, v_recon)

        # Initialize total loss
        total_loss = self.loss_weights['reconstruction'] * recon_loss

        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss
        }

        # Add KL loss for VAE mode
        if self.use_vae and mu is not None and logvar is not None:
            kl_loss = self.compute_kl_loss(mu, logvar)
            total_loss = total_loss + self.loss_weights['kl_divergence'] * kl_loss
            loss_dict['total_loss'] = total_loss
            loss_dict['kl_loss'] = kl_loss

        return loss_dict

    def _generate_kv_cache(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate KV cache using the frozen base model."""
        # Clear previous cache
        self.kv_collector.clear_cache()

        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate KV cache with frozen base model
        with torch.no_grad():
            self.kv_collector.forward(input_ids, attention_mask=attention_mask)

        # Get KV cache
        k_cache, v_cache = self.kv_collector.get_kv_cache()

        if k_cache is None or v_cache is None:
            raise RuntimeError("Failed to generate KV cache")

        # Slice to only use the first num_layers layers as specified in config
        num_layers = self.config.num_layers
        k_cache = k_cache[:, :num_layers, :, :]  # [batch_size, num_layers, seq_len, kv_dim]
        v_cache = v_cache[:, :num_layers, :, :]  # [batch_size, num_layers, seq_len, kv_dim]

        # Debug: Print actual tensor shapes (remove in production)
        # print(f"DEBUG: KV cache shapes after slicing - K: {k_cache.shape}, V: {v_cache.shape}")

        # Move to GPU only when needed for computation
        k_cache = k_cache.to(self.device)
        v_cache = v_cache.to(self.device)

        return k_cache, v_cache

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""

        # Generate KV cache on-the-fly from input tokens
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        k_tensor, v_tensor = self._generate_kv_cache(input_ids, attention_mask)

        # Forward pass through compressor
        outputs = self(k_tensor, v_tensor)

        if self.use_vae or self.compressor_type == "vae_convolutional":
            k_recon, v_recon, z, mu, logvar = outputs

            # Compute losses
            losses = self.compute_total_loss(k_tensor, v_tensor, k_recon, v_recon, mu, logvar)

            # Log VAE-specific metrics
            self.log('train_kl_loss', losses['kl_loss'], on_step=True, on_epoch=True)

            # Log compression statistics
            original_size = k_tensor.numel() + v_tensor.numel()
            compressed_size = z.numel()
            compression_ratio = original_size / compressed_size

        else:
            k_recon, v_recon, compressed = outputs

            # Compute losses
            losses = self.compute_total_loss(k_tensor, v_tensor, k_recon, v_recon)

            # Log compression statistics
            original_size = k_tensor.numel() + v_tensor.numel()
            compressed_size = compressed.numel()
            compression_ratio = original_size / compressed_size

        # Update metrics
        self.train_mse(torch.cat([k_recon, v_recon], dim=-1).contiguous(), 
                       torch.cat([k_tensor, v_tensor], dim=-1).contiguous())
        self.train_mae(torch.cat([k_recon, v_recon], dim=-1).contiguous(), 
                       torch.cat([k_tensor, v_tensor], dim=-1).contiguous())

        # Log metrics
        self.log('train_loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_reconstruction_loss', losses['reconstruction_loss'], on_step=True, on_epoch=True)
        self.log('train_mse', self.train_mse, on_step=True, on_epoch=True)
        self.log('train_mae', self.train_mae, on_step=True, on_epoch=True)
        self.log('compression_ratio', compression_ratio, on_step=True, on_epoch=True)

        # Clear GPU cache after training step to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return losses['total_loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""

        # Generate KV cache on-the-fly from input tokens
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        k_tensor, v_tensor = self._generate_kv_cache(input_ids, attention_mask)

        # Forward pass through compressor
        outputs = self(k_tensor, v_tensor)

        if self.use_vae or self.compressor_type == "vae_convolutional":
            k_recon, v_recon, z, mu, logvar = outputs

            # Compute losses
            losses = self.compute_total_loss(k_tensor, v_tensor, k_recon, v_recon, mu, logvar)

            # Log VAE-specific metrics
            self.log('val_kl_loss', losses['kl_loss'], on_step=False, on_epoch=True, sync_dist=True)

        else:
            k_recon, v_recon, compressed = outputs

            # Compute losses
            losses = self.compute_total_loss(k_tensor, v_tensor, k_recon, v_recon)

        # Update metrics
        self.val_mse(torch.cat([k_recon, v_recon], dim=-1).contiguous(), 
                     torch.cat([k_tensor, v_tensor], dim=-1).contiguous())
        self.val_mae(torch.cat([k_recon, v_recon], dim=-1).contiguous(), 
                     torch.cat([k_tensor, v_tensor], dim=-1).contiguous())

        # Log metrics
        self.log('val_loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_reconstruction_loss', losses['reconstruction_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', self.val_mae, on_step=False, on_epoch=True, sync_dist=True)

        # Clear GPU cache after validation step to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return losses['total_loss']

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.log('best_val_loss', self.best_val_loss)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""

        # Separate parameters for different components
        compressor_params = list(self.compressor.parameters())

        # Create optimizer
        optimizer = AdamW(
            compressor_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Create learning rate scheduler
        if self.max_steps is not None:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps,
                eta_min=self.learning_rate * 0.01
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            return optimizer

    def get_compression_stats(self, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> Dict[str, float]:
        """Get compression statistics for analysis."""
        with torch.no_grad():
            outputs = self(k_tensor, v_tensor)

            if self.use_vae:
                k_recon, v_recon, z, mu, logvar = outputs

                # Compression ratio
                original_size = k_tensor.numel() + v_tensor.numel()
                compressed_size = z.numel()
                compression_ratio = original_size / compressed_size

                # Reconstruction error
                k_mse = F.mse_loss(k_recon, k_tensor).item()
                v_mse = F.mse_loss(v_recon, v_tensor).item()

                # KL divergence
                kl_loss = self.compute_kl_loss(mu, logvar).item()

                # Latent statistics
                latent_mean = z.mean().item()
                latent_std = z.std().item()

                return {
                    'compression_ratio': compression_ratio,
                    'k_mse': k_mse,
                    'v_mse': v_mse,
                    'total_mse': k_mse + v_mse,
                    'kl_loss': kl_loss,
                    'latent_mean': latent_mean,
                    'latent_std': latent_std,
                    'mu_mean': mu.mean().item(),
                    'mu_std': mu.std().item(),
                    'logvar_mean': logvar.mean().item(),
                    'logvar_std': logvar.std().item()
                }
            else:
                k_recon, v_recon, compressed = outputs

                # Compression ratio
                original_size = k_tensor.numel() + v_tensor.numel()
                compressed_size = compressed.numel()
                compression_ratio = original_size / compressed_size

                # Reconstruction error
                k_mse = F.mse_loss(k_recon, k_tensor).item()
                v_mse = F.mse_loss(v_recon, v_tensor).item()

                # Sparsity
                sparsity = (compressed == 0).float().mean().item()

                return {
                    "compression_ratio": compression_ratio,
                    "k_mse": k_mse,
                    "v_mse": v_mse,
                    "total_mse": k_mse + v_mse,
                    "sparsity": sparsity,
                    "compressed_std": compressed.std().item(),
                    "compressed_mean": compressed.mean().item(),
                }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Override state_dict to only save the compressor model, not the base model.
        This significantly reduces checkpoint size since the base model is frozen.
        """
        # Get the full state dict first
        full_state_dict = super().state_dict(destination, prefix, keep_vars)

        # Filter to only include compressor parameters
        compressor_state_dict = {}
        for key, value in full_state_dict.items():
            # Only save compressor parameters and hyperparameters
            if key.startswith("compressor.") or key in [
                "_hparams_name",
                "hyper_parameters",
            ]:
                compressor_state_dict[key] = value

        return compressor_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle loading only compressor parameters.
        The base model should be provided separately during initialization.
        """
        # Filter out any base_model keys that might be in the state dict
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Skip base_model parameters if they exist in the state dict
            if not key.startswith("base_model.") and not key.startswith(
                "kv_collector."
            ):
                filtered_state_dict[key] = value

        # Load with strict=False to allow missing base_model parameters
        return super().load_state_dict(filtered_state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint):
        """
        Called when saving a checkpoint. Customize what gets saved.
        Only save compressor-related components.
        """
        # Remove base_model and kv_collector from the checkpoint to save space
        if "state_dict" in checkpoint:
            filtered_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                # Only save compressor parameters and hyperparameters
                if (
                    key.startswith("compressor.")
                    or key in ["_hparams_name", "hyper_parameters"]
                    or key.startswith("train_")
                    or key.startswith("val_")
                    or key.startswith("best_")
                ):
                    filtered_state_dict[key] = value

            checkpoint["state_dict"] = filtered_state_dict

        # Add metadata about what was excluded
        checkpoint["excluded_components"] = [
            "base_model",
            "kv_collector",
            "Note: Base model must be provided during loading",
        ]

        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """
        Called when loading a checkpoint. Handle missing base_model gracefully.
        """
        # The base_model and kv_collector should be initialized separately
        # This method ensures we don't try to load them from the checkpoint
        if "excluded_components" in checkpoint:
            print(
                f"Loading checkpoint with excluded components: {checkpoint['excluded_components']}"
            )

        return checkpoint

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        base_model=None,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        """
        Override load_from_checkpoint to require base_model parameter.
        Since we don't save the base_model, it must be provided during loading.
        """
        if base_model is None:
            raise ValueError(
                "base_model must be provided when loading from checkpoint since "
                "the base model is not saved in the checkpoint to reduce size."
            )

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Get hyperparameters
        if hparams_file is not None:
            hparams = torch.load(hparams_file, map_location=map_location)
        else:
            hparams = checkpoint.get("hyper_parameters", {})

        # Create instance with the provided base_model
        model = cls(base_model=base_model, **hparams, **kwargs)

        # Load the state dict (only compressor parameters)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        return model
