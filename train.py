"""
Federated learning training script for fine-tuning language models.
"""

import os
import random
from typing import List, Dict
import fire
import torch
import gc  # Import garbage collector
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
import datasets
import wandb # Ensure wandb is imported
from fed_utils import (
    fedavg,
    hetlora,
    load_hetlora_weights,
    select_clients,
    evaluate_global_model,
    FederatedClient,
    Prompter
)

# Reduce verbosity of datasets library
datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        exp_name: str = 'fedavg-1B',
        # Model/data params
        base_model: str = 'meta-llama/Llama-3.2-1B',
        data_path: str = './data',
        model_dir: str = './models',
        # FL hyperparams
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 20,
        num_clients: int = 8,
        # Federation params
        use_federation: bool = True, # If True, aggregate models; otherwise, train clients independently
        use_hetero: bool = False,   # If True, use heterogeneous ranks (and HetLoRA aggregation if use_federation=True)
        # Local training hyperparams
        local_batch_size: int = 128,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "",
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 4, # Base rank, used if use_hetero=False
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # LLM hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False
):
    """
    Train a language model using federated or isolated client learning.

    Args:
        exp_name: Experiment name
        base_model: Base model to fine-tune
        data_path: Path to training data
        model_dir: Directory to save outputs
        client_selection_strategy: Strategy for selecting clients
        client_selection_frac: Fraction of clients to select
        num_communication_rounds: Number of communication rounds
        num_clients: Total number of clients
        use_federation: If True, perform federated aggregation (FedAvg or HetLoRA).
                        If False, clients train independently and save their own models.
        use_hetero: If True, clients use potentially different LoRA ranks (hardcoded).
                    If use_federation is also True, HetLoRA aggregation is used.
                    If False, all clients use the base 'lora_r'.
        local_batch_size: Batch size for local training
        local_micro_batch_size: Micro batch size for gradient accumulation
        local_num_epochs: Number of epochs for local training
        local_learning_rate: Learning rate for local training
        local_val_set_size: Size of validation set (0 for no validation)
        val_data_path: Path to validation data
        cutoff_len: Maximum sequence length
        lora_r: Base LoRA rank (used when use_hetero=False)
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        train_on_inputs: Whether to train on inputs
        group_by_length: Whether to group sequences by length
    """
    # --- Mode Description ---
    mode_description = ""
    if use_federation:
        if use_hetero:
            mode_description = "Federated Heterogeneous (HetLoRA)"
        else:
            mode_description = "Federated Homogeneous (FedAvg)"
    else:
        if use_hetero:
            mode_description = "Isolated Clients (Heterogeneous Ranks)"
        else:
            mode_description = "Isolated Clients (Homogeneous Ranks)"
    print(f"Running in mode: {mode_description}")


    # Create experiment output directory
    model_output_dir = os.path.join(model_dir, exp_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Initialize wandb only on the main process (rank 0)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.init(project='10623 Project', name=exp_name) # Initialize wandb
        # Define a custom step metric for communication rounds
        wandb.define_metric("communication_round")
        # Define metrics that use the custom step metric (only if federating)
        if use_federation:
            wandb.define_metric("val_loss", step_metric="communication_round")
            if use_hetero:
                wandb.define_metric("sparsity_score/*", step_metric="communication_round")
                wandb.define_metric("aggregation_weight/*", step_metric="communication_round")
        print(
            f"Fine-tuning with:\n"
            f"Experiment: {exp_name}\n"
            f"Model: {base_model}\n"
            f"Data: {data_path}\n"
            f"Output: {model_output_dir}\n"
            f"Mode: {mode_description}\n"
            f"Clients: {num_clients} (selection: {client_selection_frac:.2f} using {client_selection_strategy})\n"
            f"Communication rounds: {num_communication_rounds}\n"
            f"Base LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
            f"Heterogeneous Ranks Active: {use_hetero}\n"
        )

    # Verify model and data paths
    assert base_model, "Please specify a base_model"
    client_data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(client_data_path), f"Data directory {client_data_path} not found"

    # Set up DDP if needed
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('ddp!')
        # For DDP, device_map needs to be set per rank
        # We will handle this inside the client loop when loading the model
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Initialize tokenizer and prompter (can be done once)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    prompter = Prompter()

    # Tokenization function for data preprocessing
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result


    # Data preprocessing function
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if 'input' in data_point.keys() else None,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt


    # Prepare client LoRA ranks and determine effective global rank
    client_lora_ranks = {}
    effective_global_rank = lora_r # Default global rank

    if use_hetero:
        # Assign potentially heterogeneous ranks (hardcoded example)
        if num_clients == 8:
            final_rank_assignments = [6, 8, 4, 4, 6, 8, 6, 6]
        else:
            # Fallback: assign base rank if not 8 clients (or implement other logic)
            print(f"Warning: Using base rank {lora_r} for all clients as num_clients is not 8 and use_hetero=True.")
            final_rank_assignments = [lora_r] * num_clients

        client_lora_ranks = dict(enumerate(final_rank_assignments))
        print(f"Using heterogeneous ranks: {client_lora_ranks}")
        rank_distribution = {}
        for rank in client_lora_ranks.values():
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        print("Rank distribution:", rank_distribution)
        # Determine the max rank needed for global model structure (if federating)
        effective_global_rank = max(client_lora_ranks.values()) if client_lora_ranks else lora_r
        print(f"Effective global rank set to {effective_global_rank} (max of client ranks)")
    else:
        # Assign homogeneous ranks
        effective_global_rank = lora_r # Use the provided base lora_r
        for client_id in range(num_clients):
            client_lora_ranks[client_id] = lora_r
        print(f"Using homogeneous LoRA with rank {lora_r} for all clients")

    # Initialize global parameters state dict (only if federating)
    global_params = None
    if use_federation:
        print("Initializing structure for global adapter state dict...")
        # Use effective_global_rank for the structure
        temp_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        temp_base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=temp_bnb_config,
            torch_dtype=torch.float16,
            device_map="cpu", # Load to CPU for structure init
        )
        temp_base_model = prepare_model_for_kbit_training(temp_base_model)

        temp_global_config = LoraConfig(
            r=effective_global_rank, # Use the determined global rank
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        temp_peft_model = get_peft_model(temp_base_model, temp_global_config)
        global_params = get_peft_model_state_dict(temp_peft_model, adapter_name="default")
        global_params = {k: v.to('cpu') for k, v in global_params.items()} # Keep on CPU

        del temp_peft_model
        del temp_base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Global adapter state dict structure initialized with rank {effective_global_rank}.")

    # Start training loop
    print(f"Starting training ({mode_description})...")

    previously_selected_clients = set()
    last_client_id = None
    local_dataset_len_dict = {}

    for epoch in tqdm(range(num_communication_rounds), desc="Communication Rounds"):
        print(f"\nRound {epoch+1}/{num_communication_rounds}")

        selected_clients = select_clients(
            num_clients,
            client_selection_frac,
            client_selection_strategy,
            seed=epoch # Use epoch for reproducibility
        )

        for client_id in selected_clients:
            # --- Client Setup ---
            client_rank = client_lora_ranks[client_id] # Get assigned rank
            print(f"\n--- Processing Client {client_id} (Rank: {client_rank}) ---")

            client_config = LoraConfig(
                r=client_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Load base model and apply PEFT config
            temp_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=temp_bnb_config,
                torch_dtype=torch.float16,
                device_map="auto" if not ddp else {"": int(os.environ.get("LOCAL_RANK") or 0)},
            )
            base_model_obj = prepare_model_for_kbit_training(base_model_obj)

            client_model = get_peft_model(
                base_model_obj,
                client_config,
                adapter_name="default"
            )

            # --- Load Previous State ---
            if epoch > 0:
                if use_federation:
                    # Load from global model (potentially truncated)
                    if use_hetero: # Check the flag for heterogeneous mode
                        print(f"Client {client_id}: Loading weights from global parameters (heterogeneous)")
                        if global_params is None:
                             raise RuntimeError("Global parameters are not initialized for HetLoRA loading.")
                        client_weights = load_hetlora_weights(
                            client_config, # Target config for truncation
                            global_params, # Current global state
                            client_rank    # Target rank
                        )
                        set_peft_model_state_dict(client_model, client_weights, "default")
                        del client_weights # Free memory
                    else: # Homogeneous federation
                        print(f"Client {client_id}: Loading weights from global parameters (homogeneous)")
                        if global_params is None:
                            raise RuntimeError("Global parameters are not initialized for FedAvg loading.")
                        set_peft_model_state_dict(client_model, global_params, "default")
                else: # Not federating, load client's own previous state
                    print(f"Client {client_id}: Attempting to load saved state from round {epoch - 1} (Federation OFF)")
                    prev_round_client_dir = os.path.join(model_output_dir, str(epoch - 1), f"client_{client_id}")
                    weights_path = os.path.join(prev_round_client_dir, "adapter_model.bin")

                    if os.path.exists(weights_path):
                        try:
                            saved_weights = torch.load(weights_path, map_location="cpu") # Load to CPU first
                            set_peft_model_state_dict(client_model, saved_weights, "default")
                            print(f"Client {client_id}: Successfully loaded weights from {weights_path}")
                            del saved_weights # Free memory
                        except Exception as e:
                            print(f"Client {client_id}: Error loading weights from {weights_path}: {e}. Starting fresh.")
                    else:
                        print(f"Client {client_id}: No saved weights found at {weights_path}. Starting fresh for this client.")

            # --- Client Training ---
            client_obj = FederatedClient(client_id, client_model, client_data_path, model_output_dir)

            print(f"Preparing client {client_id} dataset...")
            client_obj.prepare_dataset(generate_and_tokenize_prompt, local_val_set_size)
            print(f"Building trainer for client {client_id}...")
            client_obj.build_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp # Pass DDP flag
            )

            print(f"Starting training for client {client_id}...")
            client_obj.train()

            # --- Save Client State ---
            print(f"Saving model/state for client {client_id} for round {epoch}")
            # save_client_state now saves the model and config in the round-specific client dir
            local_dataset_len_dict, previously_selected_clients, last_client_id = client_obj.save_client_state(
                epoch, local_dataset_len_dict, previously_selected_clients
            )

            # --- Cleanup ---
            del client_obj
            del client_model
            del base_model_obj # Explicitly delete base model too
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"--- Finished Processing Client {client_id} ---")

        # --- Aggregation Step (only if use_federation is True) ---
        if use_federation:
            print("\n======= Aggregating Client Models =======")
            client_sparsity_scores = None # Initialize for logging
            client_aggregation_weights = None # Initialize for logging

            # Check if any clients were selected
            if not selected_clients: # Use the selected_clients list directly
                print("Warning: No clients were selected. Skipping aggregation for this round.")
            elif use_hetero: # Check the flag for heterogeneous mode
                print("Using HetLoRA sparsity-weighted aggregation (loading from disk)")
                # Pass info needed for disk loading
                global_params, client_sparsity_scores, client_aggregation_weights = hetlora(
                    global_params,             # Current global state
                    set(selected_clients),     # Pass the set of selected client IDs
                    model_output_dir,          # Base output directory
                    local_dataset_len_dict,    # Data lengths for weighting
                    epoch,                     # Current epoch/round number
                    client_lora_ranks,         # Ranks of ALL clients
                    effective_global_rank      # The target rank for the aggregated model
                )
            else: # Homogeneous FedAvg
                print("Using FedAvg homogeneous aggregation (loading from disk)")
                # Pass info needed for disk loading
                global_params = fedavg(
                    global_params,             # Current global state
                    set(selected_clients),     # Pass the set of selected client IDs
                    model_output_dir,          # Base output directory
                    local_dataset_len_dict,    # Data lengths for weighting
                    epoch                      # Current epoch/round number
                )

            # Log scores and weights if available (only on rank 0)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                log_dict = {"communication_round": epoch} # Use epoch (0-based) for step

                # Log sparsity scores if HetLoRA was used
                if use_hetero and client_sparsity_scores:
                    for cid, score in client_sparsity_scores.items():
                        log_dict[f"sparsity_score/client_{cid}"] = score
                    print(f"Prepared sparsity scores for round {epoch} for wandb.")

                # Log aggregation weights if HetLoRA was used
                if use_hetero and client_aggregation_weights:
                    for cid, weight in client_aggregation_weights.items():
                        log_dict[f"aggregation_weight/client_{cid}"] = weight
                    print(f"Prepared aggregation weights for round {epoch} for wandb.")

                # Log collected metrics for the current round if any exist besides the round number
                if len(log_dict) > 1:
                    wandb.log(log_dict)
                    print(f"Logged aggregation metrics for round {epoch} to wandb.")

            if selected_clients and global_params is not None: # Save if clients were selected and aggregation produced a result
                round_dir = os.path.join(model_output_dir, str(epoch))
                os.makedirs(round_dir, exist_ok=True) # Ensure round dir exists

                global_params_path = os.path.join(round_dir, "global_adapter_model.bin")
                torch.save(global_params, global_params_path)
                print(f"Saved aggregated global parameters to {global_params_path}")

                # Save the config corresponding to the aggregated global model
                global_config = LoraConfig(
                    r=effective_global_rank, # Rank of the saved global model
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                # Save config in the round directory
                global_config.save_pretrained(round_dir)
                print(f"Saved global config with rank {effective_global_rank} to {round_dir}")

        else: # Not using federation
            print(f"\nNo federation: Client models saved individually in round {epoch} directories.")

        # --- Optional: Evaluate Global Model (only if federating and val data exists) ---
        if use_federation and val_data_path and global_params is not None: # Check global_params exists
            if int(os.environ.get("LOCAL_RANK", 0)) == 0: # Evaluate only on rank 0
                print("\nEvaluating aggregated global model on validation data...")
                try:
                    # Load the base model for evaluation
                    eval_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    eval_base_model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        quantization_config=eval_bnb_config,
                        torch_dtype=torch.float16,
                        device_map="auto", # Use available device for evaluation
                    )
                    eval_base_model = prepare_model_for_kbit_training(eval_base_model)

                    # Create PEFT model with the global config
                    # Load config from the saved directory for this round
                    eval_config = LoraConfig.from_pretrained(os.path.join(model_output_dir, str(epoch)))
                    eval_model = get_peft_model(eval_base_model, eval_config)

                    # Load the aggregated global parameters
                    set_peft_model_state_dict(eval_model, global_params, "default")
                    eval_model.eval() # Set to evaluation mode

                    eval_loss = evaluate_global_model(
                        eval_model,
                        val_data_path,
                        generate_and_tokenize_prompt,
                        batch_size=local_micro_batch_size, # Use micro batch size for eval
                        device=eval_model.device # Use the model's device
                    )
                    print(f"Round {epoch + 1} validation loss: {eval_loss}")
                    wandb.log({"val_loss": eval_loss, "communication_round": epoch})

                    # Cleanup evaluation model
                    del eval_model
                    del eval_base_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Evaluation error during round {epoch + 1}: {e}")
                    # Cleanup in case of error during evaluation
                    if 'eval_model' in locals(): del eval_model
                    if 'eval_base_model' in locals(): del eval_base_model
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                # If not rank 0 in DDP, wait for rank 0 to finish evaluation potentially
                if ddp: torch.distributed.barrier()


    print(f"\nTraining completed. Models saved to {model_output_dir}")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.finish() # Finish wandb run


if __name__ == "__main__":
    fire.Fire(fl_finetune)