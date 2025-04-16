"""
Evaluation script for federated learning models.
Computes ROUGE scores for model-generated outputs compared to reference outputs.
"""

from typing import Dict, List
import os
import json
import fire
import evaluate
from tqdm import tqdm

TASK_METRICS = {
    "entailment": "accuracy",           # Classification task
    "paraphrase": "accuracy",           # Classification task
    "text_formatting": "rouge",         # Generation task
    "structure_to_text": "rouge",       # Generation task
    "linguistic_acceptability": "accuracy", # Classification task
    "word_disambiguation": "accuracy",  # Classification task
    "coreference": "accuracy",          # Classification task
    "question_classification": "accuracy" # Classification task
}


def load_data(file_path: str, key: str) -> Dict[str, List[Dict]]:
    """
    Load and organize data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        key: Key to extract from each JSON object
        
    Returns:
        Dictionary mapping categories to lists of samples
    """
    result = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Initialize category if not already present
            category = data['category']
            if category not in result:
                result[category] = []
            
            # Extract the specified key value
            value = data[key]
            if value.endswith('</s>'):
                value = value.split('</s>')[0]
                
            # Get the correct instruction key name
            instruction_key = "instruction" if "instruction" in data else "text"
            
            # Add to results
            result[category].append({
                "instruction": data[instruction_key],
                "output": value
            })
    
    return result


def compute_rouge_scores(targets: List[str], predictions: List[str]) -> Dict:
    """
    Compute ROUGE scores between target and prediction texts.
    
    Args:
        targets: List of reference texts
        predictions: List of generated texts
        
    Returns:
        Dictionary of ROUGE metrics
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=targets)
    return results

def compute_scores(targets: List[str], predictions: List[str], metric_type: str) -> Dict:
    """
    Compute scores between target and prediction texts based on the appropriate metric.
    
    Args:
        targets: List of reference texts
        predictions: List of generated texts
        metric_type: Type of metric to use ('rouge', 'accuracy', etc.)
        
    Returns:
        Dictionary of metric results
    """
    if metric_type == 'rouge':
        # For summarization and generation tasks
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=targets)
        return results
    
    elif metric_type == 'accuracy':
        # For classification tasks
        # Normalize text to account for whitespace/case differences
        clean_preds = [p.strip().lower() for p in predictions]
        clean_targets = [t.strip().lower() for t in targets]
        
        # Calculate accuracy manually
        correct = sum(1 for p, t in zip(clean_preds, clean_targets) if p == t)
        accuracy = correct / len(targets) if len(targets) > 0 else 0
        
        return {"accuracy": accuracy}
    
    else:
        # Default to ROUGE if unknown metric type
        print(f"Warning: Unknown metric type '{metric_type}', using ROUGE")
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=targets)
        return results
    
def evaluate_results(
    targets: Dict[str, List[Dict]], 
    predictions: Dict[str, List[Dict]],
    output_path: str
):
    """
    Evaluate predictions against targets and save results.
    
    Args:
        targets: Dictionary of reference outputs by category
        predictions: Dictionary of model outputs by category
        output_path: Path to save evaluation results
    """
    results = {}
    # all_targets = []
    # all_predictions = []
    
    # Evaluate each category
    for category in targets.keys():
        target_outputs = []
        prediction_outputs = []
        
        # Ensure predictions exist for this category
        if category not in predictions:
            print(f"Warning: No predictions found for category {category}")
            continue
            
        # Match targets and predictions
        for i, target in enumerate(targets[category]):
            if i >= len(predictions[category]):
                print(f"Warning: Missing prediction for sample {i} in category {category}")
                continue
                
            prediction = predictions[category][i]
            
            # Verify that instructions match
            assert target['instruction'] == prediction['instruction'], \
                f"Instruction mismatch in category {category}, sample {i}"
                
            # Add to lists for evaluation
            target_outputs.append(target['output'])
            prediction_outputs.append(prediction['output'])
        
        # Get the appropriate metric for this category
        metric_type = TASK_METRICS.get(category, "rouge")  # Default to rouge if category not found
        
        # Compute scores for this category
        category_results = compute_scores(target_outputs, prediction_outputs, metric_type)
        results[category] = category_results
        
        # Add to overall evaluation lists
        # all_targets.extend(target_outputs)
        # all_predictions.extend(prediction_outputs)
    
    # Compute overall scores
    # results['total'] = compute_rouge_scores(all_targets, all_predictions)
    
    # Print results
    print("\nEvaluation Results:")
    for category, scores in results.items():
        metric_used = TASK_METRICS.get(category, "rouge")
        print(f"{category} (evaluated with {metric_used}):")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main(
    exp_name: str = 'fedavg-1B',
    target_file: str = 'data/test.jsonl',
    target_key: str = 'output',
    prediction_dir: str = './predictions',
    prediction_key: str = 'answer',
    evaluation_dir: str = './evaluations',
    communication_rounds: int = 50,
    client_id: int = None,  # None for global model, otherwise specific client
):
    """
    Evaluate model-generated outputs against reference outputs.
    
    Args:
        exp_name: Experiment name
        target_file: Path to the file containing reference outputs
        target_key: Key in target file containing reference text
        prediction_dir: Directory containing prediction files
        prediction_key: Key in prediction file containing predicted text
        evaluation_dir: Directory to save evaluation results
        communication_rounds: Number of communication rounds
        client_id: Client ID for client-specific evaluation (None for global model)
    """
    # Construct the prediction file path
    if client_id is None:
        prediction_filename = "global_output.jsonl"
    else:
        prediction_filename = f"client_{client_id}_output.jsonl"
        
    prediction_file = os.path.join(
        prediction_dir, 
        exp_name, 
        str(communication_rounds), 
        prediction_filename
    )
    
    print(f"Evaluating predictions from {prediction_file}")
    print(f"Against targets from {target_file}")
    
    # Load target and prediction data
    targets = load_data(file_path=target_file, key=target_key)
    predictions = load_data(file_path=prediction_file, key=prediction_key)
    
    # Create output directory
    prediction_filename = os.path.basename(prediction_file)
    output_path = os.path.join(
        evaluation_dir, 
        exp_name, 
        str(communication_rounds), 
        f"{prediction_filename.replace('.jsonl', '_metrics.json')}"
    )
    
    # Evaluate and save results
    evaluate_results(targets, predictions, output_path)


if __name__ == "__main__":
    fire.Fire(main)