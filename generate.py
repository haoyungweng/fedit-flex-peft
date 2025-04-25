"""
Generation script for federated learning models.
This script loads trained models and generates responses for evaluation.
"""

import os
import json
import torch
import fire
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    set_seed,
)
# Make sure load_hetlora_weights is available in fed_utils
from fed_utils import Prompter, load_hetlora_weights
from typing import List # Added for type hinting

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


class EvalDataset(Dataset):
    """
    Dataset for generating predictions on test data.
    """
    
    def __init__(self, file_path: str, prompter: Prompter, tokenizer):
        """
        Initialize the evaluation dataset.
        
        Args:
            file_path: Path to the test data file
            prompter: Prompter object for formatting prompts
            tokenizer: Tokenizer for the model
        """
        self.prompter = prompter
        self.tokenizer = tokenizer
        
        # Load test data
        with open(file_path, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (prompt, sample_text)
        """
        line = self.data[idx].strip()
        ques = json.loads(line)
        sample = ques['instruction']
        
        # Generate prompt
        prompt = self.prompter.generate_prompt(
            ques['instruction'],
            ques["input"] if 'input' in ques.keys() else None,
        )
        
        return prompt, sample


def generate(
    exp_name: str = 'fedavg-1B',
    base_model: str = "",
    model_dir: str = './models',
    is_global_model: bool = False,
    hetlora: bool = False, # Added hetlora flag
    client_id: int = 0,
    # num_clients: int = 8, # No longer needed for rank reconstruction
    communication_rounds: int = 50,
    test_file_path: str = "",
    prediction_dir: str = "./predictions",
    batch_size: int = 2,
    # Added LoRA params needed for config reconstruction (only for fallback/global)
    lora_r: int = 4, # Default rank for homo/client mode
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
):
    """
    Generate responses using trained models.
    
    Args:
        exp_name: Experiment name
        base_model: Base model name
        model_dir: Directory containing trained models
        is_global_model: Whether to use the global model (if hetlora=False)
        hetlora: If True, load global weights but use client rank/config for generation
        client_id: Client ID to use (if hetlora=True or is_global_model=False)
        communication_rounds: Number of communication rounds completed
        test_file_path: Path to test data
        prediction_dir: Directory to save results
        batch_size: Batch size for generation
        lora_r: Default LoRA rank (used if not HetLoRA or for fallback)
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
    """
    # Set random seed for reproducibility
    set_seed(309)

    # Validate flags
    if hetlora and is_global_model:
        print("Warning: --hetlora is True, ignoring --is_global_model=True. Will generate using client rank from global weights.")
        is_global_model = False # Ensure consistency
    if hetlora and client_id is None:
        raise ValueError("Must provide --client_id when --hetlora is True.")

    # Ensure base model is provided
    if not base_model:
        base_model = os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model"

    # Set up paths
    experiment_model_dir = os.path.join(model_dir, exp_name)
    round_idx = communication_rounds - 1  # 0-based indexing
    round_dir = os.path.join(experiment_model_dir, str(round_idx))

    # Create prompter and count available GPUs
    prompter = Prompter()
    gpu_count = torch.cuda.device_count()

    print(f"Loading base model: {base_model}")
    print(f"Using device: {device} (GPUs: {gpu_count})")

    # --- No Rank Reconstruction Needed Here ---

    # Load base model with 8-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # --- Load LoRA weights and Config based on mode ---
    loaded_weights = None
    peft_config = None
    config_path = None # Define config_path variable

    if hetlora:
        print(f"Mode: HetLoRA Generation for Client {client_id}")

        # Path to the CLIENT's saved config to get the correct rank
        client_config_dir = os.path.join(round_dir, f"client_{client_id}")
        config_path = client_config_dir # Use client config dir
        print(f"Loading client config from: {config_path}")
        if not os.path.exists(os.path.join(config_path, "adapter_config.json")):
             raise FileNotFoundError(f"Client config file not found at {config_path}")

        peft_config = LoraConfig.from_pretrained(config_path)
        client_rank = peft_config.r
        print(f"Loaded Client Rank from config: {client_rank}")

        # Path to GLOBAL weights
        weights_path = os.path.join(round_dir, "global_adapter_model.bin")
        print(f"Loading global weights from: {weights_path}")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Global weights not found at {weights_path}")

        global_weights = torch.load(weights_path, map_location='cpu') # Load to CPU first

        # Truncate global weights for the client using the loaded client config
        print("Applying HetLoRA weight truncation...")
        loaded_weights = load_hetlora_weights(peft_config, global_weights, client_rank)
        del global_weights # Free memory

    elif is_global_model:
        print("Mode: Global Model Generation")
        # Path to global config and weights
        config_path = round_dir # Global config is saved in the round directory
        weights_path = os.path.join(round_dir, "global_adapter_model.bin")
        print(f"Loading global weights from: {weights_path}")
        print(f"Loading global config from: {config_path}")

        if not os.path.exists(os.path.join(config_path, "adapter_config.json")):
            # Try finding config in the base experiment dir as a fallback?
            # Or just error out if not found in round dir.
            raise FileNotFoundError(f"Global config file not found at {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Global weights not found at {weights_path}")

        peft_config = LoraConfig.from_pretrained(config_path)
        print(f"Loaded Global Rank from config: {peft_config.r}")
        loaded_weights = torch.load(weights_path, map_location='cpu') # Load to CPU

    else: # Client-specific model (non-HetLoRA)
        print(f"Mode: Client-Specific Model Generation for Client {client_id}")
        # Path to client config and weights
        client_model_dir = os.path.join(round_dir, f"client_{client_id}")
        config_path = client_model_dir
        weights_path = os.path.join(client_model_dir, "adapter_model.bin")
        print(f"Loading client weights from: {weights_path}")
        print(f"Loading client config from: {config_path}")

        if not os.path.exists(os.path.join(config_path, "adapter_config.json")):
             raise FileNotFoundError(f"Client config file not found at {config_path}")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Client weights not found at {weights_path}")

        peft_config = LoraConfig.from_pretrained(config_path)
        print(f"Loaded Client Rank from config: {peft_config.r}")
        loaded_weights = torch.load(weights_path, map_location='cpu') # Load to CPU

    # Apply LoRA weights to model
    # Ensure peft_config is loaded before this point
    if peft_config is None:
        raise RuntimeError("PEFT config was not loaded correctly.")

    model = PeftModel(model, peft_config, adapter_name="default") # Pass config here
    print(f"Applying loaded weights with config: r={peft_config.r}")
    set_peft_model_state_dict(model, loaded_weights, "default")
    del loaded_weights  # Free up memory

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set model to evaluation mode
    model.eval()

    # Define generation function
    def evaluate(
        instruction=None,
        input_text=None,
        temperature=0.1,
        top_p=0.75,
        top_k=50,
        num_beams=1,
        max_new_tokens=128,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        """Generate text from instruction or input_ids."""
        if input_ids is not None:
            # Use provided input_ids and attention_mask
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            # Generate input_ids from instruction and input
            prompt = prompter.generate_prompt(instruction, input_text)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

        # Configure generation parameters
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            **kwargs,
        )

        # Generate output
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        # Process the output
        if input_ids.shape[0] == 1: # Check if batch size is 1
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            ans = prompter.get_response(output).split(tokenizer.eos_token)[0]
        else:
            s = generation_output.sequences.cpu()
            # Decode without skipping special tokens
            outputs = tokenizer.batch_decode(s)
            ans = [prompter.get_response(t).split(tokenizer.eos_token)[0] for t in outputs]
            
        return ans

    # Create evaluation dataset and dataloader
    eval_dataset = EvalDataset(test_file_path, prompter, tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Generate responses for all test samples
    all_responses = []
    for prompts, texts in tqdm(dataloader, desc="Generating responses"):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        responses = evaluate(None, input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle both single and batch responses
        if isinstance(responses, list):
            all_responses.extend(responses)
        else:
            # This case might not happen with batch dataloader, but good practice
            all_responses.append(responses)

    # Create output directories
    round_prediction_dir = os.path.join(prediction_dir, exp_name, str(communication_rounds))
    os.makedirs(round_prediction_dir, exist_ok=True)
    
    # Determine output filename based on mode
    if hetlora:
        output_file = os.path.join(round_prediction_dir, f"client_{client_id}_hetlora_output.jsonl")
    elif is_global_model:
        output_file = os.path.join(round_prediction_dir, "global_output.jsonl")
    else: # Client-specific model
        output_file = os.path.join(round_prediction_dir, f"client_{client_id}_output.jsonl")

    # Write results to file
    with open(test_file_path, 'r') as f:
        test_data = [json.loads(line.strip()) for line in f]

    if os.path.exists(output_file):
        print(f"Removing existing output file: {output_file}")
        os.remove(output_file)  # Remove existing file to avoid appending

    print(f"Writing {len(all_responses)} results to {output_file}")
    for i, (test_sample, response) in enumerate(zip(test_data, all_responses)):
        # Ensure category exists, provide default if not
        category = test_sample.get('category', 'unknown')

        result = {
            'text': test_sample['instruction'],
            'answer': response,
            'category': category
        }

        # Write to output file
        with open(output_file, 'a+', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

        # Print progress
        if i < 5 or i % 50 == 0 or i == len(test_data) - 1: # Print first few and then periodically
            print(f'Sample {i+1}/{len(test_data)}')
            print(f"Instruction: {result['text']}")
            print(f"Response: {result['answer']}")
            print("="*50)

    print(f"Generation completed. Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(generate)