import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


class KVVisualizer:
    """
    A class to visualize KV cache data from transformer models.
    """
    
    def __init__(self, output_dir="kv_cache_plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots (default: "kv_cache_plots")
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_kv_cache_3d(self, k_outputs, v_outputs, batch_idx=0):
        """
        Create 3D plots of KV cache for all layers.
        
        Args:
            k_outputs: K cache tensor [batch_size, num_layers, seq_len, hidden_dim]
            v_outputs: V cache tensor [batch_size, num_layers, seq_len, hidden_dim]
            batch_idx: Which batch sample to visualize (default: 0)
        """
        if k_outputs is None or v_outputs is None:
            print("No KV cache data to plot")
            return
        
        batch_size, num_layers, seq_len, hidden_dim = k_outputs.shape
        print(f"Plotting KV cache: {num_layers} layers, {seq_len} tokens, {hidden_dim} hidden dim")
        
        # Take specified sample from batch for visualization
        k_sample = k_outputs[batch_idx]  # [num_layers, seq_len, hidden_dim]
        v_sample = v_outputs[batch_idx]  # [num_layers, seq_len, hidden_dim]
        
        # Plot K cache for each layer
        for layer_idx in range(num_layers):
            self._plot_single_kv_layer(
                k_sample[layer_idx].float().numpy(),
                layer_idx,
                cache_type='K',
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                colormap='viridis'
            )
        
        # Plot V cache for each layer
        for layer_idx in range(num_layers):
            self._plot_single_kv_layer(
                v_sample[layer_idx].float().numpy(),
                layer_idx,
                cache_type='V',
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                colormap='plasma'
            )
        
        print(f"All {num_layers * 2} plots saved in '{self.output_dir}' directory")
    
    def _plot_single_kv_layer(self, cache_data, layer_idx, cache_type, seq_len, hidden_dim, colormap):
        """
        Plot a single KV cache layer.
        
        Args:
            cache_data: Cache data for one layer [seq_len, hidden_dim]
            layer_idx: Layer index
            cache_type: 'K' or 'V'
            seq_len: Sequence length
            hidden_dim: Hidden dimension size
            colormap: Matplotlib colormap name
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        tokens = np.arange(seq_len)
        hidden_dims = np.arange(hidden_dim)
        T, H = np.meshgrid(tokens, hidden_dims, indexing='ij')
        
        # Use the actual cache values as the Z coordinate
        Z = cache_data
        
        # Create surface plot
        surf = ax.plot_surface(T, H, Z, cmap=colormap, alpha=0.7)
        
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Hidden Dimension')
        ax.set_zlabel(f'{cache_type} Cache Value')
        ax.set_title(f'{cache_type} Cache - Layer {layer_idx}')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Save plot
        filename = f"{cache_type.lower()}_cache_layer_{layer_idx:02d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {cache_type} cache plot for layer {layer_idx}")
    
    def plot_kv_heatmap(self, k_outputs, v_outputs, batch_idx=0, layer_idx=0):
        """
        Create 2D heatmap plots of KV cache for a specific layer.
        
        Args:
            k_outputs: K cache tensor [batch_size, num_layers, seq_len, hidden_dim]
            v_outputs: V cache tensor [batch_size, num_layers, seq_len, hidden_dim]
            batch_idx: Which batch sample to visualize (default: 0)
            layer_idx: Which layer to visualize (default: 0)
        """
        if k_outputs is None or v_outputs is None:
            print("No KV cache data to plot")
            return
        
        k_data = k_outputs[batch_idx, layer_idx].float().numpy()
        v_data = v_outputs[batch_idx, layer_idx].float().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # K cache heatmap
        im1 = ax1.imshow(k_data.T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Hidden Dimension')
        ax1.set_title(f'K Cache Heatmap - Layer {layer_idx}')
        plt.colorbar(im1, ax=ax1)
        
        # V cache heatmap
        im2 = ax2.imshow(v_data.T, aspect='auto', cmap='plasma', origin='lower')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Hidden Dimension')
        ax2.set_title(f'V Cache Heatmap - Layer {layer_idx}')
        plt.colorbar(im2, ax=ax2)
        
        # Save plot
        filename = f"kv_heatmap_layer_{layer_idx:02d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved KV heatmap for layer {layer_idx}")
    
    def plot_kv_statistics(self, k_outputs, v_outputs):
        """
        Plot statistics across layers (mean, std, etc.).
        
        Args:
            k_outputs: K cache tensor [batch_size, num_layers, seq_len, hidden_dim]
            v_outputs: V cache tensor [batch_size, num_layers, seq_len, hidden_dim]
        """
        if k_outputs is None or v_outputs is None:
            print("No KV cache data to plot")
            return
        
        # Calculate statistics across layers
        k_means = k_outputs.float().mean(dim=(0, 2, 3)).numpy()  # [num_layers]
        k_stds = k_outputs.float().std(dim=(0, 2, 3)).numpy()    # [num_layers]
        v_means = v_outputs.float().mean(dim=(0, 2, 3)).numpy()  # [num_layers]
        v_stds = v_outputs.float().std(dim=(0, 2, 3)).numpy()    # [num_layers]
        
        layers = np.arange(len(k_means))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # K cache mean
        ax1.plot(layers, k_means, 'b-o', label='K Cache Mean')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('K Cache Mean Across Layers')
        ax1.grid(True)
        
        # K cache std
        ax2.plot(layers, k_stds, 'b-s', label='K Cache Std')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('K Cache Std Across Layers')
        ax2.grid(True)
        
        # V cache mean
        ax3.plot(layers, v_means, 'r-o', label='V Cache Mean')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Mean Value')
        ax3.set_title('V Cache Mean Across Layers')
        ax3.grid(True)
        
        # V cache std
        ax4.plot(layers, v_stds, 'r-s', label='V Cache Std')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('V Cache Std Across Layers')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, "kv_statistics.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved KV cache statistics plot") 