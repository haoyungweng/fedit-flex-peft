"""
Federated learning training script for fine-tuning language models.
"""

import os
import random
from typing import List, Dict
import fire
import torch
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
        # Federation mode
        federation_mode: str = "homo",  # "none", "homo", or "hetero"
        # Local training hyperparams
        local_batch_size: int = 128,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "",
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 4,
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
    Train a language model using federated learning.
    
    Args:
        exp_name: Experiment name
        base_model: Base model to fine-tune
        data_path: Path to training data
        model_dir: Directory to save outputs
        client_selection_strategy: Strategy for selecting clients
        client_selection_frac: Fraction of clients to select
        num_communication_rounds: Number of communication rounds
        num_clients: Total number of clients
        federation_mode: Federation strategy ("none", "homo", or "hetero")
        local_batch_size: Batch size for local training
        local_micro_batch_size: Micro batch size for gradient accumulation
        local_num_epochs: Number of epochs for local training
        local_learning_rate: Learning rate for local training
        local_val_set_size: Size of validation set (0 for no validation)
        val_data_path: Path to validation data
        cutoff_len: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        train_on_inputs: Whether to train on inputs
        group_by_length: Whether to group sequences by length
    """
    # Validate federation mode
    assert federation_mode in ["none", "homo", "hetero"], \
        "federation_mode must be one of 'none', 'homo', or 'hetero'"

    use_hetlora = (federation_mode == "hetero")
    use_federation = (federation_mode in ["homo", "hetero"])

    # Create experiment output directory
    model_output_dir = os.path.join(model_dir, exp_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Log experiment parameters
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Fine-tuning with:\n"
            f"Experiment: {exp_name}\n"
            f"Model: {base_model}\n"
            f"Data: {data_path}\n"
            f"Output: {model_output_dir}\n"
            f"Federation mode: {federation_mode}\n"
            f"Clients: {num_clients} (selection: {client_selection_frac:.2f} using {client_selection_strategy})\n"
            f"Communication rounds: {num_communication_rounds}\n"
            f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
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
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = "auto"

    # Initialize tokenizer and prompter
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    prompter = Prompter()

    # Load the base model with quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # Prepare model for training
    base_model_obj = prepare_model_for_kbit_training(base_model_obj)

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

    # Prepare client LoRA ranks for HetLoRA
    client_lora_ranks = {}
    if use_hetlora:
        ranks = [4, 8, 16]
        num_rank_categories = len(ranks)
        base_count = num_clients // num_rank_categories
        remainder = num_clients % num_rank_categories

        counts = [base_count + 1 if i < remainder else base_count
                for i in range(num_rank_categories)]
        rank_assignments = [rank for i, rank in enumerate(ranks)
                             for _ in range(counts[i])]
        # rank_assignments = [4, 8, 8, 8, 8, 8, 8, 8]
        
        random.seed(309)  # For reproducibility
        random.shuffle(rank_assignments)
        
        client_lora_ranks = dict(enumerate(rank_assignments))
        print("Using HetLoRA with client ranks:", client_lora_ranks)
        rank_distribution = {}
        for rank in client_lora_ranks.values():
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        print("Rank distribution:", rank_distribution)
        
        # Global rank will be the maximum rank
        global_rank = max(client_lora_ranks.values())
        print(f"Global rank set to {global_rank} (max of all client ranks)")
    else:
        # Use the same rank for all clients when using FedAvg or no federation
        global_rank = lora_r
        for client_id in range(num_clients):
            client_lora_ranks[client_id] = lora_r
        print(f"Using homogeneous LoRA with rank {lora_r} for all clients")

    # Initialize global parameters
    # Create a global config for reference
    global_config = LoraConfig(
        r=global_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Initialize global model and get initial parameters
    global_model = get_peft_model(base_model_obj, global_config)
    global_params = get_peft_model_state_dict(global_model)
    
    # Now remove the adapter from the base model
    base_model_obj = global_model.get_base_model()
    print(type(global_model), type(base_model_obj))
    del global_model  # Free up memory

    # Start federated training
    print(f"Starting {'federated' if use_federation else 'isolated'} training...")
    
    # Initialize tracking variables
    previously_selected_clients = set()
    last_client_id = None
    local_dataset_len_dict = {}

    # Main training loop
    for epoch in tqdm(range(num_communication_rounds), desc="Communication Rounds"):
        print(f"\nRound {epoch+1}/{num_communication_rounds}")
        
        # Select clients for this round
        selected_clients = select_clients(
            num_clients, 
            client_selection_frac, 
            client_selection_strategy,
            seed=epoch
        )

        # Train each selected client
        for client_id in selected_clients:
            # Get client-specific rank
            client_rank = client_lora_ranks[client_id]
            print(f"\nClient {client_id}: Using LoRA rank {client_rank}")
            
            # Create client-specific configuration with the correct rank
            client_config = LoraConfig(
                r=client_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Create a fresh client model
            # Ensure base_model_obj has no PEFT adapters
            if hasattr(base_model_obj, "get_base_model"):
                print('\n\n\nGET BASE MODEL\n\n\n')
                base_model_obj = base_model_obj.get_base_model()
                
            client_model = get_peft_model(
                base_model_obj, 
                client_config,
                adapter_name="default"
            )
            
            # Load weights from global parameters
            if use_federation:
                if use_hetlora:
                    print(f"Client {client_id}: Loading weights from global parameters (heterogeneous)")
                    # For HetLoRA, we need to handle different ranks
                    client_weights = load_hetlora_weights(
                        client_config,
                        global_params,
                        client_rank
                    )
                    # Apply the weights to the client model
                    set_peft_model_state_dict(client_model, client_weights, "default")
                else:
                    print(f"Client {client_id}: Loading weights from global parameters (homogeneous)")
                    # For homogeneous LoRA, we can directly load the weights
                    set_peft_model_state_dict(client_model, global_params, "default")
                
            # Initialize client
            client = FederatedClient(client_id, client_model, client_data_path, model_output_dir)
            
            # Prepare client for training
            print(f"Preparing client {client_id} for training")
            client.prepare_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp
            )

            # Train client
            print(f"Training client {client_id}")
            client.train()

            # Save client model weights
            print(f"Saving model for client {client_id}")
            local_dataset_len_dict, previously_selected_clients, last_client_id = client.save_client_state(
                epoch, local_dataset_len_dict, previously_selected_clients
            )
            
            # Get the base model back
            base_model_obj = client_model.get_base_model()
            
            # Clean up
            del client
            del client_model

        # Perform model aggregation if using federation
        if use_federation:
            print("\n======= Aggregating client models =======")
            if use_hetlora:
                print("Using HetLoRA sparsity-weighted aggregation")
                # Use HetLoRA aggregation
                global_params = hetlora(
                    global_params,
                    selected_clients,
                    model_output_dir,
                    local_dataset_len_dict,
                    epoch,
                    client_lora_ranks
                )
            else:
                print("Using FedAvg homogeneous aggregation")
                # Use standard FedAvg
                global_params = fedavg(
                    global_params,
                    selected_clients,
                    model_output_dir,
                    local_dataset_len_dict,
                    epoch
                )
            
            # Save global model parameters
            round_dir = os.path.join(model_output_dir, str(epoch))
            os.makedirs(round_dir, exist_ok=True)
            global_params_path = os.path.join(round_dir, "global_adapter_model.bin")
            torch.save(global_params, global_params_path)
            print(f"Saved global parameters to {global_params_path}")
            
            # Save global config
            global_config = LoraConfig(
                r=global_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            config_path = os.path.join(round_dir)
            global_config.save_pretrained(config_path)
            print(f"Saved global config with rank {global_rank} to {config_path}")
        else:
            print("\nNo federation: client models saved individually")

        # Evaluate global model if validation data is provided
        if val_data_path:
            try:
                print("\nEvaluating global model on validation data...")
                # Ensure base_model_obj has no PEFT adapters
                if hasattr(base_model_obj, "get_base_model"):
                    base_model_obj = base_model_obj.get_base_model()
                
                # Create a temporary model with global parameters for evaluation
                eval_config = LoraConfig(
                    r=global_rank,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                eval_model = get_peft_model(base_model_obj, eval_config)
                
                # Load the global parameters
                set_peft_model_state_dict(eval_model, global_params, "default")
                
                # Evaluate
                eval_loss = evaluate_global_model(
                    eval_model, 
                    val_data_path, 
                    generate_and_tokenize_prompt, 
                    batch_size=1, 
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"Round {epoch + 1} validation loss: {eval_loss}")
                
                # Get base model back
                base_model_obj = eval_model.get_base_model()
                
                # Clean up
                del eval_model
            except Exception as e:
                print(f"Evaluation error: {e}")

    print(f"Training completed. Models saved to {model_output_dir}")


if __name__ == "__main__":
    fire.Fire(fl_finetune)