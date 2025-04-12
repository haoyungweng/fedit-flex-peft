"""
Model aggregation methods for federated learning.
Contains implementations of FedAvg and HetLoRA aggregation methods.
"""

from typing import Dict, Set, Any
import torch
import os
import json
from torch.nn.functional import normalize
from peft import (
    set_peft_model_state_dict,
    get_peft_model,
    LoraConfig
)


def fedavg(model: Any, 
          selected_clients: Set[int], 
          output_dir: str, 
          local_dataset_lens: Dict[int, int], 
          epoch: int) -> Any:
    """
    Standard Federated Averaging for LoRA adapters with same rank.
    
    Args:
        model: The global model to update
        selected_clients: Set of client IDs selected for aggregation
        output_dir: Directory containing client model weights
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes
        epoch: Current communication round
        
    Returns:
        Updated global model with aggregated weights
    """
    # Normalize weights based on dataset sizes
    weights_array = normalize(
        torch.tensor([local_dataset_lens[client_id] for client_id in selected_clients],
                     dtype=torch.float32),
        p=1, dim=0)

    # Weighted aggregation of client models
    for k, client_id in enumerate(selected_clients):
        model_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        client_weights = torch.load(model_path)
        
        if k == 0:
            # Initialize weighted sum with first client's weights
            weighted_sum = {key: client_weights[key] * weights_array[k] for key in client_weights.keys()}
        else:
            # Add weighted client weights to aggregated weights
            weighted_sum = {
                key: weighted_sum[key] + client_weights[key] * weights_array[k] 
                for key in client_weights.keys()
            }

    # Update global model with aggregated weights
    set_peft_model_state_dict(model, weighted_sum, "default")
    return model


def load_hetlora_weights(client_model: Any, global_model: Any, client_rank: int) -> Any:
    """
    Load weights from the global model to a client model with a specific rank.
    Handles the case where ranks differ between models.
    
    Args:
        client_model: Target client model
        global_model: Source global model
        client_rank: LoRA rank for the client model
        
    Returns:
        Client model with weights loaded from global model
    """
    global_state = global_model.state_dict()
    client_state = client_model.state_dict()
    
    # Copy weights where possible
    for key in client_state.keys():
        if key in global_state:
            if "lora_A" in key:
                # For lora_A matrices: [rank, in_features]
                # Take only the first client_rank rows if shapes differ
                if client_state[key].shape[0] != global_state[key].shape[0]:
                    client_state[key] = global_state[key][:client_state[key].shape[0]].clone()
                else:
                    client_state[key] = global_state[key].clone()
            elif "lora_B" in key:
                # For lora_B matrices: [out_features, rank]
                # Take only the first client_rank columns if shapes differ
                if client_state[key].shape[1] != global_state[key].shape[1]:
                    client_state[key] = global_state[key][:, :client_state[key].shape[1]].clone()
                else:
                    client_state[key] = global_state[key].clone()
            else:
                # For other weights, copy directly
                client_state[key] = global_state[key].clone()
    
    # Load the adjusted state into the client model
    client_model.load_state_dict(client_state)
    return client_model


def hetlora(model: Any, 
           selected_clients: Set[int], 
           output_dir: str, 
           local_dataset_lens: Dict[int, int], 
           epoch: int, 
           client_lora_ranks: Dict[int, int]) -> Any:
    """
    Heterogeneous LoRA aggregation that handles different ranks across clients.
    
    Args:
        model: The global model to update
        selected_clients: Set of client IDs selected for aggregation
        output_dir: Directory containing client model weights
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes
        epoch: Current communication round
        client_lora_ranks: Dictionary mapping client IDs to their LoRA ranks
        
    Returns:
        Updated global model with aggregated weights
    """
    # Get the maximum rank used by any client
    max_rank = max(client_lora_ranks.values())
    
    # Normalize weights based on dataset sizes
    weights_array = normalize(
        torch.tensor([local_dataset_lens[client_id] for client_id in selected_clients],
                     dtype=torch.float32),
        p=1, dim=0)
    
    # Dictionary to store aggregated weights
    aggregated_weights = {}
    
    # Initialize aggregated weights structure based on the first client
    for client_id in selected_clients:
        client_weights_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        client_weights = torch.load(client_weights_path)
        
        if not aggregated_weights:
            for key, weight in client_weights.items():
                if "lora_A" in key:
                    # For lora_A matrices: [rank, in_features] -> [max_rank, in_features]
                    in_features = weight.shape[1]
                    aggregated_weights[key] = torch.zeros((max_rank, in_features), 
                                                        dtype=weight.dtype,
                                                        device=weight.device)
                elif "lora_B" in key:
                    # For lora_B matrices: [out_features, rank] -> [out_features, max_rank]
                    out_features = weight.shape[0]
                    aggregated_weights[key] = torch.zeros((out_features, max_rank), 
                                                        dtype=weight.dtype,
                                                        device=weight.device)
                else:
                    # For other weights, initialize with zeros of the same shape
                    aggregated_weights[key] = torch.zeros_like(weight)
        break  # Only need one client to initialize the structure
    
    # Perform the aggregation
    for k, client_id in enumerate(selected_clients):
        client_weights_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        client_weights = torch.load(client_weights_path)
        client_rank = client_lora_ranks[client_id]
        
        for key, weight in client_weights.items():
            if "lora_A" in key:
                # For lora_A, place client's weights in the first client_rank rows
                aggregated_weights[key][:client_rank] += weight * weights_array[k]
            elif "lora_B" in key:
                # For lora_B, place client's weights in the first client_rank columns
                aggregated_weights[key][:, :client_rank] += weight * weights_array[k]
            else:
                # For other weights, perform weighted averaging
                aggregated_weights[key] += weight * weights_array[k]
    
    # Create a fresh model with the maximum rank
    base_model = model.base_model
    config = LoraConfig(
        r=max_rank,
        lora_alpha=model.peft_config['default'].lora_alpha,
        target_modules=model.peft_config['default'].target_modules,
        lora_dropout=model.peft_config['default'].lora_dropout,
        bias=model.peft_config['default'].bias,
        task_type=model.peft_config['default'].task_type,
    )
    
    # Create and update new model with the maximum rank
    new_model = get_peft_model(base_model, config)
    set_peft_model_state_dict(new_model, aggregated_weights, "default")
    
    return new_model