"""
Model aggregation methods for federated learning.
Contains implementations of FedAvg and HetLoRA aggregation methods.
"""

from typing import Dict, Set, Any
import torch
import os
from torch.nn.functional import normalize
from peft import (
    set_peft_model_state_dict,
    get_peft_model,
    LoraConfig,
    PeftModel
)


def fedavg(
    global_params: Dict,
    selected_clients: Set[int],
    output_dir: str,
    local_dataset_lens: Dict[int, int],
    epoch: int
) -> Dict:
    """
    Standard Federated Averaging for LoRA adapters with same rank.
    
    Args:
        global_params: Global PEFT parameters to update
        selected_clients: Set of client IDs selected for aggregation
        output_dir: Directory containing client model weights
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes
        epoch: Current communication round
        
    Returns:
        Updated global PEFT parameters
    """
    # Normalize weights based on dataset sizes
    weights_array = normalize(
        torch.tensor([local_dataset_lens[client_id] for client_id in selected_clients],
                     dtype=torch.float32),
        p=1, dim=0)
    
    # Weighted aggregation of client models
    for k, client_id in enumerate(selected_clients):
        model_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        client_weights = torch.load(model_path, map_location="cpu")
        
        if k == 0:
            # Initialize weighted sum with first client's weights
            weighted_sum = {key: client_weights[key] * weights_array[k] for key in client_weights.keys()}
        else:
            # Add weighted client weights to aggregated weights
            weighted_sum = {
                key: weighted_sum[key] + client_weights[key] * weights_array[k] 
                for key in client_weights.keys()
            }
    
    # Return the updated global parameters
    return weighted_sum


def load_hetlora_weights(
    client_config: LoraConfig,
    global_params: Dict,
    client_rank: int
) -> Dict:
    """
    Extract weights from global parameters for a client with specific rank.
    Implements the truncation distribution method described in the HetLoRA paper.

    Args:
        client_config: LoRA configuration for the client
        global_params: Global PEFT parameters
        client_rank: LoRA rank for the client model

    Returns:
        Truncated weights for the client model
    """
    client_weights = {}
    
    # Get global rank from the parameters
    if any("lora_A." in key for key in global_params.keys()):
        global_rank = max(param.shape[0] for key, param in global_params.items() 
                         if "lora_A." in key)
        print(f"load_hetlora: Global rank = {global_rank}, Client rank = {client_rank}")
    else:
        print(f"load_hetlora: No global parameters with LoRA structure found")
        return client_weights
    
    # Group parameters by module for more efficient processing
    module_names = set()
    for key in global_params.keys():
        if "lora_A." in key:
            module_name = key.split('.lora_A.')[0]
            module_names.add(module_name)
    
    # Process each module's A and B matrices together
    for module_name in module_names:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"
        
        if a_key in global_params and b_key in global_params:
            # For lora_A matrices: [rank, in_features]
            if client_rank <= global_rank:
                client_weights[a_key] = global_params[a_key][:client_rank, :].clone()
            
            # For lora_B matrices: [out_features, rank]
            if client_rank <= global_rank:
                client_weights[b_key] = global_params[b_key][:, :client_rank].clone()
    
    # Process other parameters
    for key, global_param in global_params.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            client_weights[key] = global_param.clone()
    
    print(f"load_hetlora: Truncation complete with {len(client_weights)} parameters")
    return client_weights


def hetlora(
    global_params: Dict,
    selected_clients: Set[int],
    output_dir: str,
    local_dataset_lens: Dict[int, int],
    epoch: int,
    client_lora_ranks: Dict[int, int]
) -> Dict:
    """
    Heterogeneous LoRA aggregation using Sparsity-Weighted Aggregation.
    Follows the HetLoRA algorithm from the paper.

    Args:
        global_params: Global PEFT parameters to update
        selected_clients: Set of client IDs selected for aggregation
        output_dir: Directory containing client model weights
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes
        epoch: Current communication round
        client_lora_ranks: Dictionary mapping client IDs to their LoRA ranks

    Returns:
        Updated global PEFT parameters with maximum rank
    """
    # Determine the maximum rank among all clients (not just selected ones)
    max_rank = max(client_lora_ranks.values())
    print(f"HetLoRA: Maximum rank among all clients: {max_rank}")
    
    # Calculate sparsity scores for each client based on ||B_k @ A_k||_F
    client_sparsity_scores = {}
    client_weights = {}
    
    print(f"HetLoRA: Loading client weights and calculating sparsity scores...")
    
    # Load client weights and calculate sparsity scores
    for client_id in selected_clients:
        client_rank = client_lora_ranks[client_id]
        model_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        client_state_dict = torch.load(model_path, map_location="cpu")
        client_weights[client_id] = client_state_dict
        
        # Optimized Frobenius norm calculation using trace trick
        total_norm_squared = 0.0
        processed_modules = set()
        
        # Group parameters by module for more efficient processing
        module_to_keys = {}
        for key in client_state_dict.keys():
            if ".lora_A." in key:
                module_name = key.split('.lora_A.')[0]
                if module_name not in module_to_keys:
                    module_to_keys[module_name] = {"A": None, "B": None}
                module_to_keys[module_name]["A"] = key
            elif ".lora_B." in key:
                module_name = key.split('.lora_B.')[0]
                if module_name not in module_to_keys:
                    module_to_keys[module_name] = {"A": None, "B": None}
                module_to_keys[module_name]["B"] = key
        
        # Process each module more efficiently
        for module_name, keys in module_to_keys.items():
            if keys["A"] is not None and keys["B"] is not None:
                lora_A = client_state_dict[keys["A"]]  # [r, in_dim]
                lora_B = client_state_dict[keys["B"]]  # [out_dim, r]
                
                # Verify shapes match the expected client rank
                if lora_A.shape[0] == client_rank and lora_B.shape[1] == client_rank:
                    # Optimized computation of ||B × A||_F^2 using trace trick
                    # ||BA||_F^2 = Tr((BA)^T(BA)) = Tr(A^TB^TBA)
                    # This avoids materializing the full delta_W matrix
                    
                    # Method 1: Using matrix trace trick (more memory efficient)
                    BTB = torch.matmul(lora_B.T, lora_B)  # [r, r]
                    BTB_A = torch.matmul(BTB, lora_A)  # [r, in_dim]
                    module_norm_squared = torch.sum(lora_A * BTB_A).item()
                    
                    # Alternative method (less memory efficient but sometimes faster):
                    # delta_W = torch.matmul(lora_B, lora_A)  # [out_dim, in_dim]
                    # module_norm_squared = torch.sum(delta_W**2).item()
                    
                    total_norm_squared += module_norm_squared
                    processed_modules.add(module_name)
        
        # Calculate the Frobenius norm as the sparsity score
        client_sparsity_scores[client_id] = total_norm_squared ** 0.5
        print(f"  Client {client_id} (rank {client_rank}): Sparsity score = {client_sparsity_scores[client_id]:.6f}")
    
    # Calculate aggregation weights based on sparsity scores
    Z = sum(client_sparsity_scores.values())
    client_weights_array = {cid: client_sparsity_scores[cid] / Z for cid in selected_clients}
    
    print(f"HetLoRA: Aggregation weights based on sparsity scores:")
    for client_id in selected_clients:
        print(f"  Client {client_id}: Weight = {client_weights_array[client_id]:.6f}")
    
    # Check if we have global_params already, otherwise initialize
    if not global_params:
        print(f"HetLoRA: Initializing global parameters structure")
        # Use the first client to determine structure
        first_client_id = next(iter(selected_clients))
        first_client_state = client_weights[first_client_id]
        
        # Initialize structure with zeros of appropriate shape
        global_params = {}
        for key, param in first_client_state.items():
            if "lora_A." in key:
                # For A matrices: [max_rank, in_dim]
                in_dim = param.shape[1]
                global_params[key] = torch.zeros((max_rank, in_dim), device="cpu")
            elif "lora_B." in key:
                # For B matrices: [out_dim, max_rank]
                out_dim = param.shape[0]
                global_params[key] = torch.zeros((out_dim, max_rank), device="cpu")
            else:
                # For other parameters, copy directly
                global_params[key] = torch.zeros_like(param, device="cpu")
    
    # Ensure global params have right shape for current max_rank
    if any("lora_A." in key for key in global_params.keys()):
        current_rank = max(param.shape[0] for key, param in global_params.items() 
                          if "lora_A." in key)
        
        print(f"HetLoRA: Current global rank: {current_rank}, Target max rank: {max_rank}")
        
        if current_rank != max_rank:
            print(f"HetLoRA: Resizing global parameters from rank {current_rank} to {max_rank}")
            # Resize global parameters to new max_rank
            new_global_params = {}
            for key, param in global_params.items():
                if "lora_A." in key:
                    # For A matrices: [current_rank, in_dim] -> [max_rank, in_dim]
                    in_dim = param.shape[1]
                    new_param = torch.zeros((max_rank, in_dim), device="cpu")
                    # Copy existing weights
                    min_rank = min(current_rank, max_rank)
                    new_param[:min_rank, :] = param[:min_rank, :]
                    new_global_params[key] = new_param
                elif "lora_B." in key:
                    # For B matrices: [out_dim, current_rank] -> [out_dim, max_rank]
                    out_dim = param.shape[0]
                    new_param = torch.zeros((out_dim, max_rank), device="cpu")
                    # Copy existing weights
                    min_rank = min(current_rank, max_rank)
                    new_param[:, :min_rank] = param[:, :min_rank]
                    new_global_params[key] = new_param
                else:
                    new_global_params[key] = param.clone()
            
            global_params = new_global_params
    
    # Initialize aggregated weights with zeros
    aggregated_params = {}
    for key, param in global_params.items():
        aggregated_params[key] = torch.zeros_like(param, device="cpu")
    
    print(f"HetLoRA: Aggregating client weights with zero-padding...")
    # Aggregate client weights with sparsity-based weighting
    for client_id in selected_clients:
        client_rank = client_lora_ranks[client_id]
        client_state_dict = client_weights[client_id]
        weight = client_weights_array[client_id]
        
        # Group parameters by module for more efficient processing
        module_names = set()
        for key in client_state_dict.keys():
            if "lora_A." in key:
                module_name = key.split('.lora_A.')[0]
                module_names.add(module_name)
        
        # Process each module's parameters together
        for module_name in module_names:
            a_key = f"{module_name}.lora_A.weight"
            b_key = f"{module_name}.lora_B.weight"
            
            if a_key in client_state_dict and b_key in client_state_dict and a_key in aggregated_params and b_key in aggregated_params:
                # Zero-padding: place client weights in first client_rank rows/columns
                aggregated_params[a_key][:client_rank, :] += client_state_dict[a_key] * weight
                aggregated_params[b_key][:, :client_rank] += client_state_dict[b_key] * weight
        
        # Process other parameters
        for key, param in client_state_dict.items():
            if ".lora_A." not in key and ".lora_B." not in key:
                if key in aggregated_params and aggregated_params[key].shape == param.shape:
                    aggregated_params[key] += param * weight
    
    print(f"HetLoRA: Aggregation complete")
    return aggregated_params