"""
Prompt handling utilities for model inputs and outputs.
"""

import json
import os.path as osp
from typing import Dict, Optional, Union


class Prompter:
    """
    Helper class to manage templates and prompt building.
    """
    
    # Default Alpaca template for simplicity
    DEFAULT_TEMPLATE = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }
    
    def __init__(self, template_name: str = "", verbose: bool = False):
        """
        Initialize the prompter with a template.
        
        Args:
            template_name: Name of the template file (without extension)
            verbose: Whether to print verbose information
        """
        self._verbose = verbose
        self.template = self.DEFAULT_TEMPLATE  # Default to built-in template
        
        # If a template name is provided, try to load it
        if template_name:
            try:
                file_name = osp.join("templates", f"{template_name}.json")
                if osp.exists(file_name):
                    with open(file_name) as fp:
                        self.template = json.load(fp)
                    if self._verbose:
                        print(f"Using prompt template {template_name}: {self.template['description']}")
            except Exception as e:
                print(f"Error loading template {template_name}, using default: {e}")

    def generate_prompt(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """
        Generate a prompt from instruction and optional input.
        
        Args:
            instruction: The instruction for the model
            input_text: Optional input context for the instruction
            label: Optional response/output to append
            
        Returns:
            Formatted prompt string
        """
        if input_text:
            # Use the template with input
            prompt = self.template["prompt_input"].format(
                instruction=instruction, input=input_text
            )
        else:
            # Use the template without input
            prompt = self.template["prompt_no_input"].format(
                instruction=instruction
            )
            
        # Append the label if provided
        if label:
            prompt = f"{prompt}{label}"
            
        if self._verbose:
            print(prompt)
            
        return prompt

    def get_response(self, output: str) -> str:
        """
        Extract the response from model output using the template's split point.
        
        Args:
            output: Full model output string
            
        Returns:
            Extracted response part
        """
        # Split on the response separator and take the second part
        parts = output.split(self.template["response_split"])
        if len(parts) < 2:
            return ""
        return parts[1].strip()