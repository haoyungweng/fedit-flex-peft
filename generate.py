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
from fed_utils import Prompter

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
    is_global_model: bool = True,
    client_id: int = 0,
    communication_rounds: int = 50,
    test_file_path: str = "",
    prediction_dir: str = "./predictions",
    batch_size: int = 2,
):
    """
    Generate responses using trained models.
    
    Args:
        exp_name: Experiment name
        base_model: Base model name
        output_dir: Directory containing trained models
        is_global_model: Whether to use the global model or a client model
        client_id: Client ID to use for client-specific model
        communication_rounds: Number of communication rounds completed
        prompt_template_name: Prompt template name
        test_file_path: Path to test data
        results_dir: Directory to save results
        batch_size: Batch size for generation
    """
    # Set random seed for reproducibility
    set_seed(309)
    
    # Ensure base model is provided
    if not base_model:
        base_model = os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model"

    # Set up paths
    experiment_model_dir = os.path.join(model_dir, exp_name)
    round_idx = communication_rounds - 1  # 0-based indexing
    
    # Create prompter and count available GPUs
    prompter = Prompter()
    gpu_count = torch.cuda.device_count()
    
    print(f"Loading model from {experiment_model_dir}")
    print(f"Using device: {device} (GPUs: {gpu_count})")
    
    # Load base model with 8-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Load LoRA weights based on whether using global or client model
    if is_global_model:
        # Load global model
        config_path = os.path.join(experiment_model_dir, str(round_idx))
        weights_path = os.path.join(experiment_model_dir, str(round_idx), "global_adapter_model.bin")
        print(f"Loading global model weights from {weights_path}")
        print(f"Using global config from {config_path}")
    else:
        # Load client-specific model
        client_model_dir = os.path.join(experiment_model_dir, str(round_idx), f"client_{client_id}")
        config_path = os.path.join(client_model_dir)
        weights_path = os.path.join(client_model_dir, "adapter_model.bin")
        print(f"Loading client {client_id} weights from {weights_path}")
        print(f"Using client config from {config_path}")
    
    # Verify paths exist
    if not os.path.exists(os.path.join(config_path, "adapter_config.json")):
        raise FileNotFoundError(f"Config file not found at {config_path}. Make sure the path is correct.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Make sure the path is correct.")
    
    # Load LoRA configuration
    config = LoraConfig.from_pretrained(config_path)
    
    # Load weights with appropriate device handling
    if gpu_count < 3:
        lora_weights = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0))
    else:
        lora_weights = torch.load(weights_path)
    
    # Apply LoRA weights to model
    model = PeftModel(model, config)
    set_peft_model_state_dict(model, lora_weights, "default")
    del lora_weights  # Free up memory

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
        if len(generation_output.sequences) == 1:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            ans = prompter.get_response(output).split(tokenizer.eos_token)[0]
        else:
            s = generation_output.sequences.cpu()
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
            all_responses.append(responses)
    
    # Create output directories
    round_prediction_dir = os.path.join(prediction_dir, exp_name, str(communication_rounds))
    os.makedirs(round_prediction_dir, exist_ok=True)
    
    # Determine output filename
    if is_global_model:
        output_file = os.path.join(round_prediction_dir, "global_output.jsonl")
    else:
        output_file = os.path.join(round_prediction_dir, f"client_{client_id}_output.jsonl")
    
    # Write results to file
    with open(test_file_path, 'r') as f:
        test_data = [json.loads(line.strip()) for line in f]
    
    if os.path.exists(output_file):
        os.remove(output_file)  # Remove existing file to avoid appending
        
    for i, (test_sample, response) in enumerate(zip(test_data, all_responses)):
        result = {
            'text': test_sample['instruction'],
            'answer': response,
            'category': test_sample['category']
        }
        
        # Write to output file
        with open(output_file, 'a+', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
            
        # Print progress
        if i % 10 == 0 or i == len(test_data) - 1:
            print(f'Sample {i+1}/{len(test_data)}')
            print(f"Instruction: {result['text']}")
            print(f"Response: {result['answer']}")
            print("="*50)
    
    print(f"Generation completed. Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(generate)