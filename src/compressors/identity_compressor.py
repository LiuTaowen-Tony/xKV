import torch
from src.compressor import KVCompressor

class IdentityCompressor(KVCompressor):
        
    def _compress(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        return kv_tensor
    
    def _decompress(self, compressed_kv_tensor: torch.Tensor) -> torch.Tensor:
        return compressed_kv_tensor
        