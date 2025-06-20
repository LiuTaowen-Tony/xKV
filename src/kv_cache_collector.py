import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from .utils.utils import load_model_and_tokenizer
from .utils.kv_visualizer import KVVisualizer


class KVCacheCollector:
    """
    A class to manage persistent KV cache collection from transformer models.
    Hooks remain attached to the model until explicitly removed.
    """
    
    def __init__(self, model):
        """
        Initialize the KV cache collector with a model.
        
        Args:
            model: The transformer model to collect KV cache from
        """
        self.model = model
        self.collected_k_outputs = []
        self.collected_v_outputs = []
        self.hooks_k = []
        self.hooks_v = []
        self._setup_hooks()
    
    def _k_proj_hook(self, module, input, output):
        """
        Hook function to collect K projection outputs.
        
        Args:
            module: The layer that produced this output (k_proj)
            input: The input to k_proj
            output: The output from k_proj (shape [batch_size, seq_len, hidden_dim])
        """
        # Detach and move to CPU immediately to save GPU memory
        # Use half precision to save memory
        self.collected_k_outputs.append(output.detach().half().cpu())

    def _v_proj_hook(self, module, input, output):
        """
        Hook function to collect V projection outputs.
        
        Args:
            module: The layer that produced this output (v_proj)
            input: The input to v_proj
            output: The output from v_proj (shape [batch_size, seq_len, hidden_dim])
        """
        # Detach and move to CPU immediately to save GPU memory
        # Use half precision to save memory
        self.collected_v_outputs.append(output.detach().half().cpu())
    
    def _setup_hooks(self):
        """Register forward hooks for all transformer layers."""
        num_layers = len(self.model.model.layers)
        
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx].self_attn
            
            hook_k = layer.k_proj.register_forward_hook(self._k_proj_hook)
            hook_v = layer.v_proj.register_forward_hook(self._v_proj_hook)
            
            self.hooks_k.append(hook_k)
            self.hooks_v.append(hook_v)
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Run a forward pass and collect KV cache.
        
        Args:
            input_ids: Input token IDs (torch.Tensor)
            attention_mask: Attention mask (torch.Tensor, optional)
            **kwargs: Additional arguments to pass to the model
        
        Returns:
            Model output
        """
        self.model.eval()
        if attention_mask is not None:
            output = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        else:
            output = self.model(input_ids, **kwargs)
        return output
    
    def get_kv_cache(self):
        """
        Get the currently collected KV cache.
        
        Returns:
            tuple: (collected_k_outputs, collected_v_outputs)
                - Each is a tensor of shape [num_layers, batch_size, seq_len, hidden_dim]
        """
        if not self.collected_k_outputs or not self.collected_v_outputs:
            return None, None
        
        # Stack tensors from different layers and convert back to float32
        k_tensor = torch.stack(self.collected_k_outputs, dim=0).float()  # [num_layers, batch_size, seq_len, hidden_dim]
        v_tensor = torch.stack(self.collected_v_outputs, dim=0).float()  # [num_layers, batch_size, seq_len, hidden_dim]

        k_tensor = k_tensor.transpose(0, 1) # [batch_size, num_layers, seq_len, hidden_dim]
        v_tensor = v_tensor.transpose(0, 1) # [batch_size, num_layers, seq_len, hidden_dim]
        
        return k_tensor, v_tensor
    
    def clear_cache(self):
        """Clear the collected KV cache without removing hooks."""
        self.collected_k_outputs.clear()
        self.collected_v_outputs.clear()
    
    def get_cache_info(self):
        """Get information about the collected cache."""
        return {
            'num_k_outputs': len(self.collected_k_outputs),
            'num_v_outputs': len(self.collected_v_outputs),
            'num_layers': len(self.hooks_k)
        }
    
    def remove_hooks(self):
        """Remove all hooks from the model. Call this when done collecting."""
        for hook in self.hooks_k:
            hook.remove()
        for hook in self.hooks_v:
            hook.remove()
        self.hooks_k.clear()
        self.hooks_v.clear()
    
    def __del__(self):
        """Cleanup hooks when the object is destroyed."""
        self.remove_hooks()


# Example usage:
if __name__ == "__main__":
    model_name = "unsloth/Llama-3.2-1B-Instruct"  # Or any other Llama checkpoint available
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Set up persistent KV cache collection
    kv_collector = KVCacheCollector(model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.select(range(100))
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        print(batch.keys())
        print(batch["text"])
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        kv_collector.forward(input_ids, attention_mask=attention_mask)
        break
    
    k_outputs, v_outputs = kv_collector.get_kv_cache()
    print(k_outputs.shape, v_outputs.shape)

    # Create visualizer and generate plots
    visualizer = KVVisualizer()
    
    # Generate 3D plots for all layers
    visualizer.plot_kv_cache_3d(k_outputs, v_outputs)
    
    # Generate additional visualizations
    visualizer.plot_kv_heatmap(k_outputs, v_outputs, layer_idx=0)  # First layer heatmap
    visualizer.plot_kv_statistics(k_outputs, v_outputs)  # Statistics across layers
    
    # Clean up
    kv_collector.remove_hooks()





