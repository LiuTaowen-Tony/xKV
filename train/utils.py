import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


def load_model_and_tokenizer(model_name: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on ("auto", "cpu", "cuda", etc.)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device != "cuda":
        model = model.to(device)
    
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    return model, tokenizer


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get the size of a model in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model, name="Model"):
    """Print information about a model."""
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"\n{name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Model size: {size_mb:.2f} MB")
    print(f"  Device: {next(model.parameters()).device}") 