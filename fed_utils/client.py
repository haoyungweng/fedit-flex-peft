"""
Client implementation for federated learning.
"""

from typing import Any, Dict, Set, Callable, Optional, Tuple
import os
import copy
from collections import OrderedDict

import torch
import transformers
from datasets import load_dataset
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class FederatedClient:
    """
    Client implementation for federated learning with LoRA adapters.
    """
    
    def __init__(self, client_id: int, model: Any, data_path: str, output_dir: str):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            model: The model to be trained (with LoRA adapters)
            data_path: Path to the data directory
            output_dir: Directory to save outputs
        """
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        # Use a temporary directory for trainer checkpoints
        self.trainer_checkpoint_dir = os.path.join(self.output_dir, "_trainer_temp", f"client_{self.client_id}")
        
        # Will be set later
        self.local_train_dataset = None
        self.local_eval_dataset = None
        self.local_val_set_size = 0
        self.train_args = None
        self.local_trainer = None

    def prepare_dataset(self, preprocessing_fn: Callable, val_set_size: int = 0):
        """
        Prepare the local dataset for training and evaluation.
        
        Args:
            preprocessing_fn: Function to tokenize and format the data
            val_set_size: Size of validation set (0 for no validation)
        """
        if val_set_size > 0:
            # Split into train and validation sets
            local_train_val = self.local_data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=309
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(preprocessing_fn)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(preprocessing_fn)
            )
        else:
            # Use all data for training
            self.local_train_dataset = self.local_data["train"].shuffle().map(preprocessing_fn)
            self.local_eval_dataset = None
            
        self.local_val_set_size = val_set_size

    def build_trainer(self,
                      tokenizer: Any,
                      micro_batch_size: int,
                      gradient_accumulation_steps: int,
                      num_epochs: int,
                      learning_rate: float,
                      group_by_length: bool = False,
                      use_ddp: bool = False):
        """
        Build the local trainer for this client.
        
        Args:
            tokenizer: Tokenizer for the model
            micro_batch_size: Batch size per device
            gradient_accumulation_steps: Number of steps to accumulate gradients
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            group_by_length: Whether to group sequences by length
            use_ddp: Whether to use DistributedDataParallel
        """
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="no",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            output_dir=self.trainer_checkpoint_dir,
            save_total_limit=1,  # Keep this just in case internal checkpoints are created
            load_best_model_at_end=self.local_val_set_size > 0,  # Load best model if doing evaluation
            ddp_find_unused_parameters=False if use_ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

    def train(self):
        """
        Train the model on the local dataset.
        """
        # Disable caching during training
        self.model.config.use_cache = False
        
        # Train the model
        self.local_trainer.train()

    def save_client_state(self, 
                         epoch: int, 
                         dataset_len_dict: Dict[int, int], 
                         participating_clients: Set[int]) -> Tuple[Dict[int, int], Set[int], int]:
        """
        Save the client model state after training.
        
        Args:
            epoch: Current communication round
            dataset_len_dict: Dictionary mapping client IDs to dataset lengths
            participating_clients: Set of client IDs that have participated
            
        Returns:
            Tuple of (updated dataset_len_dict, updated participating_clients, client_id)
        """
        # Store dataset length
        dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        
        # Create client directory
        client_dir = os.path.join(self.output_dir, str(epoch), f"client_{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Save adapter weights
        adapter_weights = get_peft_model_state_dict(self.model)
        torch.save(adapter_weights, os.path.join(client_dir, "adapter_model.bin"))
        
        # Save client-specific config
        if hasattr(self.model, 'peft_config') and 'default' in self.model.peft_config:
            self.model.peft_config['default'].save_pretrained(client_dir)
        
        # Update metadata
        participating_clients = participating_clients | {self.client_id}

        return dataset_len_dict, participating_clients, self.client_id