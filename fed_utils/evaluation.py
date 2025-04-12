"""
Evaluation utilities for federated learning models.
"""

from typing import Any, Callable, Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def evaluate_global_model(
    model: Any, 
    data_files: str, 
    preprocessing_fn: Callable, 
    batch_size: int, 
    device: str
) -> float:
    """
    Evaluate the global model on a validation dataset.
    
    Args:
        model: The model to evaluate
        data_files: Path to validation data files
        preprocessing_fn: Function to preprocess and tokenize the data
        batch_size: Batch size for evaluation
        device: Device to run evaluation on ('cuda' or 'cpu')
        
    Returns:
        Average loss on the validation dataset
    """
    # Load and preprocess validation data
    data = load_dataset("json", data_files=data_files)
    val_data = data["train"].shuffle().map(preprocessing_fn)
    val_data = val_data.with_format('torch')
    data_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Evaluate model on all batches
    loss_values = []
    for inputs in tqdm(data_loader, desc="Evaluating"):
        with torch.no_grad():
            # Move inputs to the appropriate device
            batch = {
                k: v.to(device) 
                for k, v in inputs.items() 
                if k in ['input_ids', 'attention_mask', 'labels']
            }
            
            # Forward pass
            output = model(**batch)
            loss_values.append(output[0].cpu())
    
    # Calculate average loss
    avg_loss = sum(loss_values) / len(loss_values)
    return avg_loss