import abc
import einops
import torch
from pydantic import BaseModel
from typing import Literal, Tuple
from typing_extensions import TypeAlias

# import typealias


# type alias for uncompressed per layer cache [b, s, h]
UncompressedPerLayerCache: TypeAlias = torch.Tensor

# type alias for uncompressed all layer cache list[tensor[b, s, h]]
UncompressedAllLayerCache: TypeAlias = list[torch.Tensor]

# [b, l, s, h]
UncompressedK: TypeAlias = torch.Tensor
# [b, l, s, h]
UncompressedV: TypeAlias = torch.Tensor
# [b, l, s, h]
UncompressedKVConcat: TypeAlias = torch.Tensor

# [b, l, s, h]
CompressedKVConcat: TypeAlias = torch.Tensor

# [b, l, s, h]
DecompressedK: TypeAlias = torch.Tensor
DecompressedV: TypeAlias = torch.Tensor
DecompressedKVConcat: TypeAlias = torch.Tensor

KVUncompressedPair: TypeAlias = tuple[UncompressedPerLayerCache, UncompressedAllLayerCache]


COMPRESSED_KV_LAYER_DIM = 1
COMPRESSED_KV_SEQ_DIM = 2
COMPRESSED_KV_HIDDEN_DIM = 3

UNCOMPRESSED_PER_LAYER_SEQ_DIM = 1
UNCOMPRESSED_PER_LAYER_HIDDEN_DIM = 2

UNCOMPRESSED_ALL_LAYER_LAYER_DIM = 0
UNCOMPRESSED_ALL_LAYER_SEQ_DIM = 2
UNCOMPRESSED_ALL_LAYER_HIDDEN_DIM = 3


class KVCompressor(abc.ABC, torch.nn.Module):

    def __init__(self, compression_seq_window: int):
        super().__init__()
        self.compression_seq_window = compression_seq_window
    
    def compress(self, k_tensor, v_tensor):
        kv_tensor = self.prepare_kv_tensor(k_tensor, v_tensor)
        compressed_kv_tensor = self._compress(kv_tensor)
        return compressed_kv_tensor

    def decompress(self, compressed_kv_tensor):
        decompressed_kv_tensor = self._decompress(compressed_kv_tensor)
        return self.split_decompressed_kv_tensor(decompressed_kv_tensor)

    @abc.abstractmethod
    def _compress(self, kv_tensor: UncompressedKVConcat) -> CompressedKVConcat:
        pass

    @abc.abstractmethod
    def _decompress(self, compressed_kv_tensor: CompressedKVConcat) -> UncompressedKVConcat:
        pass

    def prepare_kv_tensor(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> torch.Tensor:
        # input shape list[tensor[batch_size, seq_len, hidden_dim]]
        # output shape [batch_size, num_layers, seq_len, hidden_dim]
        # first concat on layer dimension
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        # then concat on hidden dimension
        return torch.cat([keys, values], dim=-1)

    def split_decompressed_kv_tensor(self, compressed_kv_tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # input shape [batch_size, num_layers, seq_len, hidden_dim]
        # output shape list[tensor[batch_size, seq_len, hidden_dim]]
        # split on layer dimension
        keys = compressed_kv_tensor[:, :, :, :self.hidden_dim]
        values = compressed_kv_tensor[:, :, :, self.hidden_dim:]

        # split on layer dimension
        keys = torch.split(keys, 1, dim=1)
        values = torch.split(values, 1, dim=1)
        return keys, values


class Mish(torch.nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))



        



class ConvolutionalCompressorConfig(BaseModel):
    num_layers: int
    hidden_dim: int
    kv_dim: int = None  # Actual K/V projection dimension (if None, will use hidden_dim//4 for GQA models)
    layer_stride: int = 2  # Stride for layer dimension compression
    seq_stride: int = 2    # Stride for sequence dimension compression
    kernel_size: int = 3   # Kernel size for convolution
    hidden_channels: int = 256  # Number of hidden channels in conv layers
    activation: Literal["gelu", "swish", "mish"] = "gelu"
    dropout_rate: float = 0.1

class ConvolutionalCompressor(KVCompressor):
    """
    Convolutional compressor that uses strided convolutions on layer and sequence dimensions.
    Much more memory-efficient than linear layers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input channels = 2 * kv_dim (K + V concatenated)
        # For GQA models like Llama-3.2-1B, kv_dim is typically hidden_dim//4
        if config.kv_dim is not None:
            input_channels = config.kv_dim * 2
        else:
            # Default assumption for GQA models  
            input_channels = (config.hidden_dim // 4) * 2
        
        # Encoder: compress layer and sequence dimensions
        self.encoder = torch.nn.Sequential(
            # First conv: reduce spatial dimensions
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                stride=(config.layer_stride, config.seq_stride),
                padding=config.kernel_size // 2
            ),
            self._get_activation_function(config.activation),
            torch.nn.Dropout2d(config.dropout_rate),
            
            # Second conv: further compression
            torch.nn.Conv2d(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels // 2,
                kernel_size=config.kernel_size,
                stride=(1, 2),  # Only compress sequence dimension more
                padding=config.kernel_size // 2
            ),
            self._get_activation_function(config.activation),
            torch.nn.Dropout2d(config.dropout_rate),
        )
        
        # Decoder: upsample back to original dimensions
        self.decoder = torch.nn.Sequential(
            # First transpose conv
            torch.nn.ConvTranspose2d(
                in_channels=config.hidden_channels // 2,
                out_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                stride=(1, 2),
                padding=config.kernel_size // 2,
                output_padding=(0, 1)
            ),
            self._get_activation_function(config.activation),
            torch.nn.Dropout2d(config.dropout_rate),
            
            # Second transpose conv: restore original dimensions
            torch.nn.ConvTranspose2d(
                in_channels=config.hidden_channels,
                out_channels=input_channels,
                kernel_size=config.kernel_size,
                stride=(config.layer_stride, config.seq_stride),
                padding=config.kernel_size // 2,
                output_padding=(config.layer_stride - 1, config.seq_stride - 1)
            ),
        )
        
        # Activation function
        self.activation = self._get_activation_function(config.activation)
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            "gelu": torch.nn.GELU(),
            "swish": torch.nn.SiLU(),
            "mish": Mish(),
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}. Choose from {list(activations.keys())}")
        
        return activations[activation_name]
    
    def compress(self, k_tensor, v_tensor):
        # k_tensor, v_tensor: [batch_size, num_layers, seq_len, hidden_dim]
        batch_size, num_layers, seq_len, hidden_dim = k_tensor.shape
        
        # Concatenate K and V
        kv_tensor = torch.cat([k_tensor, v_tensor], dim=-1)  # [batch_size, num_layers, seq_len, 2*hidden_dim]
        
        # Reshape for convolution: [batch_size, channels, height, width]
        # We treat layers as height and sequence as width
        kv_tensor = kv_tensor.permute(0, 3, 1, 2)  # [batch_size, 2*hidden_dim, num_layers, seq_len]
        
        # Apply encoder
        compressed = self.encoder(kv_tensor)
        
        return compressed
    
    def decompress(self, compressed_tensor, batch_size: int, seq_len: int):
        # Apply decoder
        decompressed = self.decoder(compressed_tensor)
        
        # Reshape back: [batch_size, 2*hidden_dim, num_layers, seq_len] -> [batch_size, num_layers, seq_len, 2*hidden_dim]
        decompressed = decompressed.permute(0, 2, 3, 1)
        
        # Split K and V  
        kv_dim = self.config.kv_dim if self.config.kv_dim is not None else (self.config.hidden_dim // 4)
        k_tensor, v_tensor = torch.split(decompressed, kv_dim, dim=-1)
        
        return k_tensor, v_tensor


class EnhancedConvolutionalCompressorConfig(BaseModel):
    num_layers: int
    hidden_dim: int
    kv_dim: int = None  # Actual K/V projection dimension (if None, will use hidden_dim//4 for GQA models)
    layer_stride: int = 2  # Stride for layer dimension compression
    seq_stride: int = 2    # Stride for sequence dimension compression
    kernel_size: int = 3   # Kernel size for convolution
    hidden_channels: int = 256  # Number of hidden channels in conv layers
    activation: Literal["gelu", "swish", "mish"] = "gelu"
    dropout_rate: float = 0.1
    use_residual: bool = True  # Use residual connections
    use_attention: bool = True  # Use channel attention
    use_layer_norm: bool = True  # Use layer normalization
    compression_depth: int = 3  # Number of compression stages


class EnhancedConvolutionalCompressor(KVCompressor):
    """
    Enhanced convolutional compressor with:
    - Residual connections for better gradient flow
    - Channel attention for adaptive feature weighting
    - Multiple compression stages for better compression ratio
    - Layer normalization for training stability
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input channels = 2 * kv_dim (K + V concatenated)
        # For GQA models like Llama-3.2-1B, kv_dim is typically hidden_dim//4
        if config.kv_dim is not None:
            input_channels = config.kv_dim * 2
        else:
            # Default assumption for GQA models  
            input_channels = (config.hidden_dim // 4) * 2
        
        # Build encoder with multiple stages
        encoder_layers = []
        in_channels = input_channels
        
        for i in range(config.compression_depth):
            out_channels = config.hidden_channels // (2 ** i)
            stride = (config.layer_stride if i == 0 else 1, 
                     config.seq_stride if i < 2 else 1)
            
            # Convolutional block
            encoder_layers.extend([
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=config.kernel_size // 2
                ),
                torch.nn.BatchNorm2d(out_channels) if config.use_layer_norm else torch.nn.Identity(),
                self._get_activation_function(config.activation),
                torch.nn.Dropout2d(config.dropout_rate),
            ])
            
            # Channel attention
            if config.use_attention:
                encoder_layers.append(ChannelAttention(out_channels))
            
            in_channels = out_channels
        
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        in_channels = config.hidden_channels // (2 ** (config.compression_depth - 1))
        
        for i in range(config.compression_depth):
            if i == config.compression_depth - 1:
                out_channels = input_channels
            else:
                out_channels = config.hidden_channels // (2 ** (config.compression_depth - 2 - i))
            
            stride = (config.layer_stride if i == config.compression_depth - 1 else 1, 
                     config.seq_stride if i >= config.compression_depth - 2 else 1)
            output_padding = (stride[0] - 1, stride[1] - 1)
            
            # Transpose convolutional block
            decoder_layers.extend([
                torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=config.kernel_size // 2,
                    output_padding=output_padding
                ),
                torch.nn.BatchNorm2d(out_channels) if config.use_layer_norm and i < config.compression_depth - 1 else torch.nn.Identity(),
                self._get_activation_function(config.activation) if i < config.compression_depth - 1 else torch.nn.Identity(),
                torch.nn.Dropout2d(config.dropout_rate) if i < config.compression_depth - 1 else torch.nn.Identity(),
            ])
            
            # Channel attention (except for the last layer)
            if config.use_attention and i < config.compression_depth - 1:
                decoder_layers.append(ChannelAttention(out_channels))
            
            in_channels = out_channels
        
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
        # Residual projection if needed
        if config.use_residual:
            self.residual_proj = torch.nn.Conv2d(
                input_channels, 
                config.hidden_channels // (2 ** (config.compression_depth - 1)), 
                kernel_size=1
            )
        
        # Activation function
        self.activation = self._get_activation_function(config.activation)
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            "gelu": torch.nn.GELU(),
            "swish": torch.nn.SiLU(),
            "mish": Mish(),
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}. Choose from {list(activations.keys())}")
        
        return activations[activation_name]
    
    def compress(self, k_tensor, v_tensor):
        # k_tensor, v_tensor: [batch_size, num_layers, seq_len, hidden_dim]
        batch_size, num_layers, seq_len, hidden_dim = k_tensor.shape
        
        # Concatenate K and V
        kv_tensor = torch.cat([k_tensor, v_tensor], dim=-1)  # [batch_size, num_layers, seq_len, 2*hidden_dim]
        
        # Reshape for convolution: [batch_size, channels, height, width]
        # We treat layers as height and sequence as width
        kv_tensor = kv_tensor.permute(0, 3, 1, 2)  # [batch_size, 2*hidden_dim, num_layers, seq_len]
        
        # Store for residual connection
        residual = kv_tensor
        
        # Apply encoder
        compressed = self.encoder(kv_tensor)
        
        # Add residual connection
        if self.config.use_residual:
            # Project residual to match compressed dimensions
            residual_proj = self.residual_proj(residual)
            # Adaptive pooling to match spatial dimensions
            residual_proj = torch.nn.functional.adaptive_avg_pool2d(
                residual_proj, compressed.shape[-2:]
            )
            compressed = compressed + residual_proj
        
        return compressed
    
    def decompress(self, compressed_tensor, batch_size: int, seq_len: int):
        # Apply decoder
        decompressed = self.decoder(compressed_tensor)
        
        # Reshape back: [batch_size, 2*hidden_dim, num_layers, seq_len] -> [batch_size, num_layers, seq_len, 2*hidden_dim]
        decompressed = decompressed.permute(0, 2, 3, 1)
        
        # Split K and V
        kv_dim = self.config.kv_dim if self.config.kv_dim is not None else (self.config.hidden_dim // 4)
        k_tensor, v_tensor = torch.split(decompressed, kv_dim, dim=-1)
        
        return k_tensor, v_tensor


class ChannelAttention(torch.nn.Module):
    """Channel attention module for convolutional layers."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels // reduction, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class VAEConvolutionalCompressorConfig(BaseModel):
    num_layers: int
    hidden_dim: int
    kv_dim: int = None  # Actual K/V projection dimension (if None, will use hidden_dim//4 for GQA models)
    layer_stride: int = 2  # Stride for layer dimension compression
    seq_stride: int = 2    # Stride for sequence dimension compression
    kernel_size: int = 3   # Kernel size for convolution
    hidden_channels: int = 256  # Number of hidden channels in conv layers
    latent_channels: int = 128  # Number of channels in latent space
    activation: Literal["gelu", "swish", "mish"] = "gelu"
    dropout_rate: float = 0.1
    use_residual: bool = True  # Use residual connections
    use_attention: bool = True  # Use channel attention
    use_layer_norm: bool = True  # Use layer normalization
    compression_depth: int = 3  # Number of compression stages
    beta: float = 1.0  # Beta parameter for KL divergence weight


class VAEConvolutionalCompressor(KVCompressor):
    """
    VAE Convolutional compressor that combines the efficiency of convolutional
    compression with probabilistic modeling via VAE.
    
    Features:
    - Convolutional encoder/decoder for spatial compression
    - VAE latent space for probabilistic modeling
    - Residual connections and attention mechanisms
    - Much higher compression ratio than linear methods
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input channels = 2 * kv_dim (K + V concatenated)
        # For GQA models like Llama-3.2-1B, kv_dim is typically hidden_dim//4
        if config.kv_dim is not None:
            input_channels = config.kv_dim * 2
        else:
            # Default assumption for GQA models  
            input_channels = (config.hidden_dim // 4) * 2
        
        # Build encoder with multiple stages
        encoder_layers = []
        in_channels = input_channels
        
        for i in range(config.compression_depth):
            out_channels = config.hidden_channels // (2 ** i)
            stride = (config.layer_stride if i == 0 else 1, 
                     config.seq_stride if i < 2 else 1)
            
            # Convolutional block
            encoder_layers.extend([
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=config.kernel_size // 2
                ),
                torch.nn.BatchNorm2d(out_channels) if config.use_layer_norm else torch.nn.Identity(),
                self._get_activation_function(config.activation),
                torch.nn.Dropout2d(config.dropout_rate),
            ])
            
            # Channel attention
            if config.use_attention:
                encoder_layers.append(ChannelAttention(out_channels))
            
            in_channels = out_channels
        
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # VAE components: mu and logvar heads
        feature_channels = config.hidden_channels // (2 ** (config.compression_depth - 1))
        self.conv_mu = torch.nn.Conv2d(feature_channels, config.latent_channels, kernel_size=1)
        self.conv_logvar = torch.nn.Conv2d(feature_channels, config.latent_channels, kernel_size=1)
        
        # Decoder input projection
        self.latent_to_features = torch.nn.Conv2d(config.latent_channels, feature_channels, kernel_size=1)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        in_channels = feature_channels
        
        for i in range(config.compression_depth):
            if i == config.compression_depth - 1:
                out_channels = input_channels
            else:
                out_channels = config.hidden_channels // (2 ** (config.compression_depth - 2 - i))
            
            stride = (config.layer_stride if i == config.compression_depth - 1 else 1, 
                     config.seq_stride if i >= config.compression_depth - 2 else 1)
            output_padding = (stride[0] - 1, stride[1] - 1)
            
            # Transpose convolutional block
            decoder_layers.extend([
                torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=config.kernel_size // 2,
                    output_padding=output_padding
                ),
                torch.nn.BatchNorm2d(out_channels) if config.use_layer_norm and i < config.compression_depth - 1 else torch.nn.Identity(),
                self._get_activation_function(config.activation) if i < config.compression_depth - 1 else torch.nn.Identity(),
                torch.nn.Dropout2d(config.dropout_rate) if i < config.compression_depth - 1 else torch.nn.Identity(),
            ])
            
            # Channel attention (except for the last layer)
            if config.use_attention and i < config.compression_depth - 1:
                decoder_layers.append(ChannelAttention(out_channels))
            
            in_channels = out_channels
        
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
        # Residual projection if needed
        if config.use_residual:
            self.residual_proj = torch.nn.Conv2d(
                input_channels, 
                config.latent_channels, 
                kernel_size=1
            )
        
        # Activation function
        self.activation = self._get_activation_function(config.activation)
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            "gelu": torch.nn.GELU(),
            "swish": torch.nn.SiLU(),
            "mish": Mish(),
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}. Choose from {list(activations.keys())}")
        
        return activations[activation_name]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compress(self, k_tensor, v_tensor):
        # k_tensor, v_tensor: [batch_size, num_layers, seq_len, hidden_dim]
        batch_size, num_layers, seq_len, hidden_dim = k_tensor.shape
        
        # Concatenate K and V
        kv_tensor = torch.cat([k_tensor, v_tensor], dim=-1)  # [batch_size, num_layers, seq_len, 2*hidden_dim]
        
        # Reshape for convolution: [batch_size, channels, height, width]
        # We treat layers as height and sequence as width
        kv_tensor = kv_tensor.permute(0, 3, 1, 2)  # [batch_size, 2*hidden_dim, num_layers, seq_len]
        
        # Store for residual connection
        residual = kv_tensor
        
        # Apply encoder
        features = self.encoder(kv_tensor)
        
        # Get VAE parameters
        mu = self.conv_mu(features)
        logvar = self.conv_logvar(features)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Add residual connection
        if self.config.use_residual:
            # Project residual to match latent dimensions and spatial size
            residual_proj = self.residual_proj(residual)
            residual_proj = torch.nn.functional.adaptive_avg_pool2d(
                residual_proj, z.shape[-2:]
            )
            z = z + residual_proj
        
        return z, mu, logvar, (batch_size, seq_len)
    
    def decompress(self, z, shape_info):
        batch_size, seq_len = shape_info
        
        # Project latent back to feature space
        features = self.latent_to_features(z)
        
        # Apply decoder
        decompressed = self.decoder(features)
        
        # Reshape back: [batch_size, 2*hidden_dim, num_layers, seq_len] -> [batch_size, num_layers, seq_len, 2*hidden_dim]
        decompressed = decompressed.permute(0, 2, 3, 1)
        
        # Split K and V
        kv_dim = self.config.kv_dim if self.config.kv_dim is not None else (self.config.hidden_dim // 4)
        k_tensor, v_tensor = torch.split(decompressed, kv_dim, dim=-1)
        
        return k_tensor, v_tensor
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for VAE."""
        # KL(q(z|x) || p(z)) where p(z) is standard normal
        # Sum over spatial and channel dimensions, then average over batch
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return torch.mean(kl_loss)


class Dual1DConvolutionalCompressorConfig(BaseModel):
    num_layers: int
    hidden_dim: int
    kv_dim: int = None  # Actual K/V projection dimension (if None, will use hidden_dim//4 for GQA models)
    layer_stride: int = 2  # Stride for layer dimension compression
    seq_stride: int = 2    # Stride for sequence dimension compression
    kernel_size: int = 3   # Kernel size for convolution
    hidden_channels: int = 256  # Number of hidden channels in conv layers
    activation: Literal["gelu", "swish", "mish"] = "gelu"
    dropout_rate: float = 0.1
    use_activation: bool = True  # Ablation parameter for activation functions
    use_residual: bool = True  # Use residual connections
    use_layer_norm: bool = True  # Use layer normalization


class Dual1DConvolutionalCompressor(KVCompressor):
    """
    Dual 1D Convolutional compressor that applies:
    1. First 1D conv along the layer dimension
    2. Second 1D conv along the sequence dimension
    
    This approach is more parameter-efficient than 2D convolutions and allows
    for better control over compression in each dimension.
    
    Features:
    - Separate 1D convolutions for layer and sequence dimensions
    - Ablation support for activation functions
    - Residual connections and layer normalization
    - Memory-efficient compared to 2D approaches
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input channels = 2 * kv_dim (K + V concatenated)
        # For GQA models like Llama-3.2-1B, kv_dim is typically hidden_dim//4
        if config.kv_dim is not None:
            input_channels = config.kv_dim * 2
        else:
            # Default assumption for GQA models  
            input_channels = (config.hidden_dim // 4) * 2
        
        # First stage: 1D convolution along layer dimension
        # Input shape: [batch_size, seq_len, input_channels, num_layers]
        self.layer_conv_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_channels,
                out_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                stride=config.layer_stride,
                padding=config.kernel_size // 2
            ),
            torch.nn.BatchNorm1d(config.hidden_channels) if config.use_layer_norm else torch.nn.Identity(),
            self._get_activation_function(config.activation) if config.use_activation else torch.nn.Identity(),
            torch.nn.Dropout(config.dropout_rate),
        )
        
        self.layer_conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=config.hidden_channels,
                out_channels=input_channels,
                kernel_size=config.kernel_size,
                stride=config.layer_stride,
                padding=config.kernel_size // 2,
                output_padding=config.layer_stride - 1
            ),
        )
        
        # Second stage: 1D convolution along sequence dimension
        # Input shape: [batch_size, compressed_layers, config.hidden_channels, seq_len]
        self.seq_conv_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels // 2,
                kernel_size=config.kernel_size,
                stride=config.seq_stride,
                padding=config.kernel_size // 2
            ),
            torch.nn.BatchNorm1d(config.hidden_channels // 2) if config.use_layer_norm else torch.nn.Identity(),
            self._get_activation_function(config.activation) if config.use_activation else torch.nn.Identity(),
            torch.nn.Dropout(config.dropout_rate),
        )
        
        self.seq_conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=config.hidden_channels // 2,
                out_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                stride=config.seq_stride,
                padding=config.kernel_size // 2,
                output_padding=config.seq_stride - 1
            ),
        )
        
        # Residual connections if enabled
        if config.use_residual:
            # For layer dimension residual
            self.layer_residual_proj = torch.nn.Conv1d(
                input_channels, 
                config.hidden_channels, 
                kernel_size=1
            ) if config.layer_stride != 1 else torch.nn.Identity()
            
            # For sequence dimension residual
            self.seq_residual_proj = torch.nn.Conv1d(
                config.hidden_channels, 
                config.hidden_channels // 2, 
                kernel_size=1
            ) if config.seq_stride != 1 else torch.nn.Identity()
        
        # Activation function
        self.activation = self._get_activation_function(config.activation) if config.use_activation else torch.nn.Identity()
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            "gelu": torch.nn.GELU(),
            "swish": torch.nn.SiLU(),
            "mish": Mish(),
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}. Choose from {list(activations.keys())}")
        
        return activations[activation_name]
    
    def compress(self, k_tensor, v_tensor):
        # k_tensor, v_tensor: [batch_size, num_layers, seq_len, hidden_dim]
        batch_size, num_layers, seq_len, hidden_dim = k_tensor.shape
        
        # Concatenate K and V
        kv_tensor = torch.cat([k_tensor, v_tensor], dim=-1)  # [batch_size, num_layers, seq_len, 2*hidden_dim]
        
        # Reshape for layer-wise 1D convolution: [batch_size * seq_len, 2*hidden_dim, num_layers]
        kv_tensor_layer = kv_tensor.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, 2*hidden_dim, num_layers]
        kv_tensor_layer = kv_tensor_layer.view(batch_size * seq_len, -1, num_layers)  # [batch_size * seq_len, 2*hidden_dim, num_layers]
        
        # Apply layer-wise compression
        layer_compressed = self.layer_conv_encoder(kv_tensor_layer)  # [batch_size * seq_len, hidden_channels, compressed_layers]
        
        # Add residual connection for layer dimension
        if self.config.use_residual and hasattr(self, 'layer_residual_proj'):
            if isinstance(self.layer_residual_proj, torch.nn.Conv1d):
                layer_residual = self.layer_residual_proj(kv_tensor_layer)
                # Adaptive pooling to match compressed size
                layer_residual = torch.nn.functional.adaptive_avg_pool1d(layer_residual, layer_compressed.shape[-1])
                layer_compressed = layer_compressed + layer_residual
        
        # Reshape for sequence-wise 1D convolution
        compressed_layers = layer_compressed.shape[-1]
        layer_compressed = layer_compressed.view(batch_size, seq_len, self.config.hidden_channels, compressed_layers)
        layer_compressed = layer_compressed.permute(0, 3, 2, 1).contiguous()  # [batch_size, compressed_layers, hidden_channels, seq_len]
        layer_compressed = layer_compressed.view(batch_size * compressed_layers, self.config.hidden_channels, seq_len)
        
        # Apply sequence-wise compression
        seq_compressed = self.seq_conv_encoder(layer_compressed)  # [batch_size * compressed_layers, hidden_channels//2, compressed_seq]
        
        # Add residual connection for sequence dimension
        if self.config.use_residual and hasattr(self, 'seq_residual_proj'):
            if isinstance(self.seq_residual_proj, torch.nn.Conv1d):
                seq_residual = self.seq_residual_proj(layer_compressed)
                # Adaptive pooling to match compressed size
                seq_residual = torch.nn.functional.adaptive_avg_pool1d(seq_residual, seq_compressed.shape[-1])
                seq_compressed = seq_compressed + seq_residual
        
        return seq_compressed, (batch_size, seq_len, compressed_layers)
    
    def decompress(self, compressed_tensor, shape_info):
        batch_size, seq_len, compressed_layers = shape_info
        
        # compressed_tensor: [batch_size * compressed_layers, hidden_channels//2, compressed_seq]
        compressed_seq = compressed_tensor.shape[-1]
        
        # Apply sequence-wise decompression
        seq_decompressed = self.seq_conv_decoder(compressed_tensor)  # [batch_size * compressed_layers, hidden_channels, seq_len]
        
        # Reshape for layer-wise decompression
        seq_decompressed = seq_decompressed.view(batch_size, compressed_layers, self.config.hidden_channels, seq_len)
        seq_decompressed = seq_decompressed.permute(0, 3, 2, 1).contiguous()  # [batch_size, seq_len, hidden_channels, compressed_layers]
        seq_decompressed = seq_decompressed.view(batch_size * seq_len, self.config.hidden_channels, compressed_layers)
        
        # Apply layer-wise decompression
        layer_decompressed = self.layer_conv_decoder(seq_decompressed)  # [batch_size * seq_len, 2*kv_dim, num_layers]
        
        # Reshape back to original format
        kv_dim = self.config.kv_dim if self.config.kv_dim is not None else (self.config.hidden_dim // 4)
        # layer_decompressed is [batch_size * seq_len, 2*kv_dim, num_layers]
        # We need to reshape it to [batch_size, seq_len, 2*kv_dim, num_layers] first
        layer_decompressed = layer_decompressed.view(batch_size, seq_len, kv_dim * 2, self.config.num_layers)
        layer_decompressed = layer_decompressed.permute(0, 3, 1, 2)  # [batch_size, num_layers, seq_len, 2*kv_dim]
        
        # Split K and V
        k_tensor, v_tensor = torch.split(layer_decompressed, kv_dim, dim=-1)
        
        return k_tensor, v_tensor


if __name__ == "__main__":
    # Test Dual 1D Convolutional implementation
    dual1d_config = Dual1DConvolutionalCompressorConfig(
        num_layers=12,
        hidden_dim=768,
        kv_dim=192,  # For GQA models: hidden_dim // 4
        layer_stride=2,
        seq_stride=2,
        kernel_size=3,
        hidden_channels=256,
        activation="gelu",
        dropout_rate=0.1,
        use_activation=True,
        use_residual=True,
        use_layer_norm=True
    )
    
    print("=== Testing Dual 1D Convolutional Compressor ===")
    dual1d_compressor = Dual1DConvolutionalCompressor(dual1d_config)
    
    # Use smaller test tensor for faster testing
    k_tensor = torch.randn(2, 12, 128, 192)  # [batch, layers, seq_len, kv_dim]
    v_tensor = torch.randn(2, 12, 128, 192)
    
    compressed, shape_info = dual1d_compressor.compress(k_tensor, v_tensor)
    print(f"Compressed shape: {compressed.shape}")
    
    k_recon, v_recon = dual1d_compressor.decompress(compressed, shape_info)
    print(f"Reconstructed shapes: K={k_recon.shape}, V={v_recon.shape}")
    
    # Calculate compression ratio and reconstruction error
    original_size = k_tensor.numel() + v_tensor.numel()
    compressed_size = compressed.numel()
    compression_ratio = original_size / compressed_size
    
    mse_k = torch.nn.functional.mse_loss(k_recon, k_tensor)
    mse_v = torch.nn.functional.mse_loss(v_recon, v_tensor)
    
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Reconstruction MSE - K: {mse_k:.6f}, V: {mse_v:.6f}")
    
    # Test without activation functions (ablation)
    print("\n=== Testing Dual 1D Compressor WITHOUT Activation ===")
    dual1d_config_no_act = Dual1DConvolutionalCompressorConfig(
        num_layers=12,
        hidden_dim=768,
        kv_dim=192,
        layer_stride=2,
        seq_stride=2,
        kernel_size=3,
        hidden_channels=256,
        activation="gelu",
        dropout_rate=0.1,
        use_activation=False,  # Ablation: no activation
        use_residual=True,
        use_layer_norm=True
    )
    
    dual1d_compressor_no_act = Dual1DConvolutionalCompressor(dual1d_config_no_act)
    
    compressed_no_act, shape_info = dual1d_compressor_no_act.compress(k_tensor, v_tensor)
    print(f"Compressed shape (no activation): {compressed_no_act.shape}")
    
    k_recon_no_act, v_recon_no_act = dual1d_compressor_no_act.decompress(compressed_no_act, shape_info)
    print(f"Reconstructed shapes (no activation): K={k_recon_no_act.shape}, V={v_recon_no_act.shape}")
    
    # Calculate compression ratio and reconstruction error
    original_size = k_tensor.numel() + v_tensor.numel()
    compressed_size_no_act = compressed_no_act.numel()
    compression_ratio_no_act = original_size / compressed_size_no_act
    
    mse_k_no_act = torch.nn.functional.mse_loss(k_recon_no_act, k_tensor)
    mse_v_no_act = torch.nn.functional.mse_loss(v_recon_no_act, v_tensor)
    
    print(f"Compression ratio (no activation): {compression_ratio_no_act:.2f}x")
    print(f"Reconstruction MSE (no activation) - K: {mse_k_no_act:.6f}, V: {mse_v_no_act:.6f}")
    
    print(f"\n=== Activation Ablation Comparison ===")
    print(f"With activation - MSE K: {mse_k:.6f}, MSE V: {mse_v:.6f}")
    print(f"Without activation - MSE K: {mse_k_no_act:.6f}, MSE V: {mse_v_no_act:.6f}")
    print(f"Activation improves K by: {(mse_k_no_act - mse_k) / mse_k_no_act * 100:.2f}%")
    print(f"Activation improves V by: {(mse_v_no_act - mse_v) / mse_v_no_act * 100:.2f}%")




    
