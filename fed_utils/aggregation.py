"""
Model aggregation methods for federated learning.
Contains implementations of FedAvg and HetLoRA aggregation methods.
"""

from typing import Dict, Set, Tuple
import torch
import os
from torch.nn.functional import normalize
from peft import (
    LoraConfig,
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
        global_params: Global PEFT parameters to update (can be None initially).
        selected_clients: Set of client IDs selected for aggregation.
        output_dir: Directory containing client model weights for the current epoch.
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes.
        epoch: Current communication round number.

    Returns:
        Updated global PEFT parameters.
    """
    if not selected_clients:
        print("FedAvg: No clients selected. Returning existing global_params.")
        return global_params

    selected_clients_list = list(selected_clients)

    # Normalize weights based on dataset sizes
    weights_array = normalize(
        torch.tensor([local_dataset_lens.get(client_id, 0) for client_id in selected_clients_list],
                     dtype=torch.float32),
        p=1, dim=0)

    aggregated_params = None

    # Weighted aggregation of client models
    for k, client_id in enumerate(selected_clients_list):
        model_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")
        try:
            client_weights = torch.load(model_path, map_location="cpu")
        except FileNotFoundError:
            print(f"FedAvg Warning: Model file not found for client {client_id} at {model_path}. Skipping.")
            continue

        weight = weights_array[k]

        if aggregated_params is None:
            # Initialize aggregated_params structure and values from the first valid client
            aggregated_params = {key: param.clone() * weight for key, param in client_weights.items()}
        else:
            # Add weighted client weights to the aggregation
            for key in aggregated_params.keys():
                if key in client_weights:
                    aggregated_params[key] += client_weights[key] * weight
                else:
                    print(f"FedAvg Warning: Key '{key}' not found in client {client_id}'s weights. Skipping key.")

    return aggregated_params if aggregated_params is not None else global_params


def load_hetlora_weights(
    client_config: LoraConfig,
    global_params: Dict,
    client_rank: int
) -> Dict:
    """
    Extract weights from global parameters for a client with specific rank using truncation.

    Args:
        client_config: LoRA configuration for the client (used to get target modules).
        global_params: Global PEFT parameters state dict.
        client_rank: LoRA rank for the client model.

    Returns:
        Truncated weights state dict for the client model.
    """
    client_weights = {}
    global_rank = 0

    # Determine global rank from the provided global_params state dict
    for key, param in global_params.items():
        if "lora_A." in key:
            global_rank = max(global_rank, param.shape[0])
        elif "lora_B." in key:
            global_rank = max(global_rank, param.shape[1])

    if global_rank == 0:
        print(f"load_hetlora: No LoRA parameters found in global_params. Returning empty dict.")
        return client_weights

    print(f"load_hetlora: Global rank = {global_rank}, Client target rank = {client_rank}")

    if client_rank > global_rank:
        print(f"load_hetlora: Warning - Client rank {client_rank} > Global rank {global_rank}. Cannot truncate.")
        return global_params.copy() # Return full global params if client rank is larger

    # Process parameters based on target modules in client_config
    for key, global_param in global_params.items():
        is_lora_A = ".lora_A." in key
        is_lora_B = ".lora_B." in key

        if is_lora_A:
            # Truncate lora_A: [global_rank, in_features] -> [client_rank, in_features]
            client_weights[key] = global_param[:client_rank, :].clone()
        elif is_lora_B:
            # Truncate lora_B: [out_features, global_rank] -> [out_features, client_rank]
            client_weights[key] = global_param[:, :client_rank].clone()
        else:
            # Copy non-LoRA parameters directly
            client_weights[key] = global_param.clone()

    print(f"load_hetlora: Truncation complete with {len(client_weights)} parameters for rank {client_rank}")
    return client_weights


def hetlora(
    global_params: Dict,
    selected_clients: Set[int],
    output_dir: str,
    local_dataset_lens: Dict[int, int],
    epoch: int,
    client_lora_ranks: Dict[int, int],
    target_global_rank: int
) -> Tuple[Dict, Dict[int, float], Dict[int, float]]:
    """
    Heterogeneous LoRA aggregation using Sparsity-Weighted Aggregation.

    Args:
        global_params: Current global PEFT parameters state dict (can be None initially).
        selected_clients: Set of client IDs selected for aggregation.
        output_dir: Directory containing client model weights for the current epoch.
        local_dataset_lens: Dictionary mapping client IDs to their dataset sizes.
        epoch: Current communication round number.
        client_lora_ranks: Dictionary mapping client IDs to their LoRA ranks (for all clients).
        target_global_rank: The target rank for the aggregated global model.

    Returns:
        A tuple containing:
            - Updated global PEFT parameters state dict with target_global_rank.
            - Dictionary mapping participating client IDs to their sparsity scores.
            - Dictionary mapping participating client IDs to their aggregation weights.
    """
    if not selected_clients:
        print("HetLoRA: No clients selected. Returning existing global_params.")
        return global_params, {}, {}

    selected_clients_list = list(selected_clients)
    print(f"HetLoRA: Aggregating {len(selected_clients_list)} clients. Target global rank: {target_global_rank}")

    client_sparsity_scores = {}
    client_loaded_weights = {}

    print(f"HetLoRA: Loading client weights and calculating sparsity scores...")
    for client_id in selected_clients_list:
        client_rank = client_lora_ranks.get(client_id, 0)
        model_path = os.path.join(output_dir, str(epoch), f"client_{client_id}", "adapter_model.bin")

        try:
            client_state_dict = torch.load(model_path, map_location="cpu")
            client_loaded_weights[client_id] = client_state_dict
        except FileNotFoundError:
            print(f"HetLoRA Warning: Model file not found for client {client_id} at {model_path}. Skipping.")
            client_sparsity_scores[client_id] = 0.0
            continue

        if client_rank == 0:
            client_sparsity_scores[client_id] = 0.0
            continue

        # Calculate sparsity score ||B_k @ A_k||_F
        total_norm_squared = 0.0
        module_to_keys = {}
        for key in client_state_dict.keys():
            if ".lora_A." in key:
                module_name = key.split('.lora_A.')[0]
                if module_name not in module_to_keys: module_to_keys[module_name] = {"A": None, "B": None}
                module_to_keys[module_name]["A"] = key
            elif ".lora_B." in key:
                module_name = key.split('.lora_B.')[0]
                if module_name not in module_to_keys: module_to_keys[module_name] = {"A": None, "B": None}
                module_to_keys[module_name]["B"] = key

        for module_name, keys in module_to_keys.items():
            if keys["A"] is not None and keys["B"] is not None:
                lora_A = client_state_dict[keys["A"]].float()
                lora_B = client_state_dict[keys["B"]].float()

                if lora_A.shape[0] == client_rank and lora_B.shape[1] == client_rank:
                    BTB = torch.matmul(lora_B.T, lora_B)
                    BTB_A = torch.matmul(BTB, lora_A)
                    module_norm_squared = torch.sum(lora_A * BTB_A).item()
                    total_norm_squared += module_norm_squared
                else:
                    print(f"  HetLoRA Warning: Mismatched shapes for client {client_id}, module {module_name}.")

        client_sparsity_scores[client_id] = total_norm_squared ** 0.5
        print(f"  Client {client_id} (rank {client_rank}): Sparsity score = {client_sparsity_scores[client_id]:.6f}")

    # Calculate aggregation weights
    valid_scores = [score for cid, score in client_sparsity_scores.items() if cid in client_loaded_weights]
    Z = sum(valid_scores)
    if Z == 0:
        num_valid_clients = len(valid_scores)
        client_aggregation_weights = {cid: 1.0 / num_valid_clients for cid in client_loaded_weights} if num_valid_clients > 0 else {}
        print(f"HetLoRA: Warning - Sum of sparsity scores is zero. Assigning equal weights.")
    else:
        client_aggregation_weights = {cid: client_sparsity_scores.get(cid, 0.0) / Z for cid in client_loaded_weights}

    print(f"HetLoRA: Aggregation weights:")
    for client_id in client_loaded_weights:
        print(f"  Client {client_id}: Weight = {client_aggregation_weights.get(client_id, 0.0):.6f}")

    # Initialize or resize aggregated_params structure
    aggregated_params = {}
    current_global_rank = 0
    if global_params:
        # Determine current rank from existing global_params
        for key, param in global_params.items():
            if ".lora_A." in key: current_global_rank = max(current_global_rank, param.shape[0])
            elif ".lora_B." in key: current_global_rank = max(current_global_rank, param.shape[1])
        print(f"HetLoRA: Current global rank: {current_global_rank}, Target global rank: {target_global_rank}")

        # Initialize based on existing structure, resizing if necessary
        for key, param in global_params.items():
            if ".lora_A." in key:
                in_dim = param.shape[1]
                new_param = torch.zeros((target_global_rank, in_dim), device="cpu")
                copy_rank = min(current_global_rank, target_global_rank)
                if copy_rank > 0: new_param[:copy_rank, :] = param[:copy_rank, :].to("cpu")
                aggregated_params[key] = new_param
            elif ".lora_B." in key:
                out_dim = param.shape[0]
                new_param = torch.zeros((out_dim, target_global_rank), device="cpu")
                copy_rank = min(current_global_rank, target_global_rank)
                if copy_rank > 0: new_param[:, :copy_rank] = param[:, :copy_rank].to("cpu")
                aggregated_params[key] = new_param
            else:
                aggregated_params[key] = torch.zeros_like(param, device="cpu")
    else:
        # Initialize structure from the first valid client if global_params is None
        print(f"HetLoRA: Initializing global parameters structure with target rank {target_global_rank}")
        first_valid_client_id = next((cid for cid in selected_clients_list if cid in client_loaded_weights), None)
        if first_valid_client_id is None:
            print("HetLoRA Error: Cannot initialize global params, no valid client models found.")
            return {}, {}, {}

        first_client_state = client_loaded_weights[first_valid_client_id]
        for key, param in first_client_state.items():
            if ".lora_A." in key:
                in_dim = param.shape[1]
                aggregated_params[key] = torch.zeros((target_global_rank, in_dim), device="cpu")
            elif ".lora_B." in key:
                out_dim = param.shape[0]
                aggregated_params[key] = torch.zeros((out_dim, target_global_rank), device="cpu")
            else:
                aggregated_params[key] = torch.zeros_like(param, device="cpu")

    # Aggregate client weights with zero-padding
    print(f"HetLoRA: Aggregating client weights with zero-padding...")
    for client_id in client_loaded_weights:
        client_rank = client_lora_ranks.get(client_id, 0)
        if client_rank == 0: continue

        client_state_dict = client_loaded_weights[client_id]
        weight = client_aggregation_weights.get(client_id, 0.0)
        if weight == 0.0: continue

        for key, client_param in client_state_dict.items():
            if key not in aggregated_params:
                continue

            client_param_cpu = client_param.to("cpu")

            if ".lora_A." in key:
                # Add client's A weights to the top-left corner
                if client_rank <= target_global_rank:
                     aggregated_params[key][:client_rank, :] += client_param_cpu * weight
                else: # Should not happen if target_global_rank is max rank
                     aggregated_params[key][:, :] += client_param_cpu[:target_global_rank, :] * weight
            elif ".lora_B." in key:
                # Add client's B weights to the top-left corner
                if client_rank <= target_global_rank:
                     aggregated_params[key][:, :client_rank] += client_param_cpu * weight
                else:
                     aggregated_params[key][:, :] += client_param_cpu[:, :target_global_rank] * weight
            else:
                # Aggregate non-LoRA parameters
                if aggregated_params[key].shape == client_param_cpu.shape:
                    aggregated_params[key] += client_param_cpu * weight
                else:
                     print(f"  HetLoRA Warning: Shape mismatch for non-LoRA key '{key}'. Skipping.")

    print(f"HetLoRA: Aggregation complete")
    final_scores = {cid: score for cid, score in client_sparsity_scores.items() if cid in client_loaded_weights}
    final_weights = {cid: weight for cid, weight in client_aggregation_weights.items() if cid in client_loaded_weights}
    return aggregated_params, final_scores, final_weights