"""
Evaluation script for federated learning models.
Computes appropriate metrics for model-generated outputs compared to reference outputs.
"""

from typing import Dict, List
import os
import json
import fire
import evaluate
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef

# Updated metrics for each task type
TASK_METRICS = {
    "entailment": "accuracy",                 # Classification task
    "paraphrase": "f1",                       # Classification task
    "text_formatting": "rouge",               # Generation task
    "structure_to_text": "rouge",             # Generation task
    "linguistic_acceptability": "accuracy",   # Classification task
    "word_disambiguation": "f1",              # Classification task
    "coreference": "accuracy",                # Classification task
    "question_classification": "accuracy"     # Classification task
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
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"]
    }


def compute_scores(targets: List[str], predictions: List[str], metric_type: str) -> Dict:
    """
    Compute scores between target and prediction texts based on the appropriate metric.
    Always includes Rouge metrics for all tasks.
    
    Args:
        targets: List of reference texts
        predictions: List of generated texts
        metric_type: Type of metric to use ('rouge', 'accuracy', 'f1', 'mcc', etc.)
        
    Returns:
        Dictionary of metric results including Rouge metrics
    """
    # Normalize text to account for whitespace/case differences
    clean_preds = [p.strip().lower() for p in predictions]
    clean_targets = [t.strip().lower() for t in targets]
    
    # Always compute Rouge metrics for all tasks
    rouge_results = compute_rouge_scores(targets, predictions)
    
    # For tasks that already use Rouge, just return the Rouge metrics
    if metric_type == 'rouge':
        return rouge_results
    
    # For other metrics, compute both the original metric and Rouge
    results = {}
    
    if metric_type == 'accuracy':
        # For classification tasks with balanced classes (entailment, question_classification)
        correct = sum(1 for p, t in zip(clean_preds, clean_targets) if p == t)
        accuracy = correct / len(targets) if len(targets) > 0 else 0
        
        results["accuracy"] = accuracy
        
    elif metric_type == 'f1':
        # First, identify the valid options from the target outputs
        unique_targets = sorted(list(set(clean_targets)))
        
        # Verify that we have a binary classification task
        if len(unique_targets) != 2:
            print(f"Warning: Expected binary classification (2 classes), but found {len(unique_targets)} classes in targets.")
            # Fall back to accuracy if it's not binary classification
            correct = sum(1 for p, t in zip(clean_preds, clean_targets) if p == t)
            accuracy = correct / len(targets) if len(targets) > 0 else 0
            results["accuracy"] = accuracy
        else:
            # Valid options are the two unique values from the target outputs
            valid_options = unique_targets
            
            # Process predictions - if a prediction is not in valid options, treat it as incorrect
            processed_preds = []
            for pred, target in zip(clean_preds, clean_targets):
                # If prediction is not a valid option, replace it with a value that ensures it will be wrong
                # by using the "other" valid option (different from the target)
                if pred not in valid_options:
                    # Find the option that's different from the target to ensure it's counted as wrong
                    wrong_option = valid_options[0] if target == valid_options[1] else valid_options[1]
                    processed_preds.append(wrong_option)
                else:
                    processed_preds.append(pred)
            
            # Calculate accuracy (percentage of correct predictions)
            correct = sum(1 for p, t in zip(processed_preds, clean_targets) if p == t)
            accuracy = correct / len(clean_targets) if len(clean_targets) > 0 else 0
            
            # Use the first valid option as the positive class for binary F1 calculation
            positive_class = valid_options[0]
            
            binary_targets = [1 if t == positive_class else 0 for t in clean_targets]
            binary_preds = [1 if p == positive_class else 0 for p in processed_preds]
            
            # Calculate F1 score
            f1 = f1_score(binary_targets, binary_preds)
            
            results["f1_score"] = f1
            results["accuracy"] = accuracy
        
    elif metric_type == 'mcc':
        # First, identify the valid options from the target outputs
        unique_targets = sorted(list(set(clean_targets)))
        
        # Verify that we have a binary classification task
        if len(unique_targets) != 2:
            print(f"Warning: Expected binary classification (2 classes), but found {len(unique_targets)} classes in targets.")
            # Fall back to accuracy if it's not binary classification
            correct = sum(1 for p, t in zip(clean_preds, clean_targets) if p == t)
            accuracy = correct / len(targets) if len(targets) > 0 else 0
            results["accuracy"] = accuracy
        else:
            # Valid options are the two unique values from the target outputs
            valid_options = unique_targets
            
            # Process predictions - if a prediction is not in valid options, treat it as incorrect
            processed_preds = []
            for pred, target in zip(clean_preds, clean_targets):
                # If prediction is not a valid option, replace it with a value that ensures it will be wrong
                # by using the "other" valid option (different from the target)
                if pred not in valid_options:
                    # Find the option that's different from the target to ensure it's counted as wrong
                    wrong_option = valid_options[0] if target == valid_options[1] else valid_options[1]
                    processed_preds.append(wrong_option)
                else:
                    processed_preds.append(pred)
            
            # Create mapping from text classes to numerical indices
            class_to_idx = {c: i for i, c in enumerate(valid_options)}
            
            # Convert text classes to indices for MCC calculation
            target_indices = [class_to_idx[t] for t in clean_targets]
            pred_indices = [class_to_idx[p] for p in processed_preds]
            
            # Calculate MCC
            mcc = matthews_corrcoef(target_indices, pred_indices)
            
            # Also include accuracy for comparison
            correct = sum(1 for p, t in zip(processed_preds, clean_targets) if p == t)
            accuracy = correct / len(targets) if len(targets) > 0 else 0
            
            results["mcc"] = mcc
            results["accuracy"] = accuracy
    
    else:
        # Unknown metric type - use original implementation (default to ROUGE)
        print(f"Warning: Unknown metric type '{metric_type}', using ROUGE")
        return rouge_results
    
    # Add Rouge metrics to the results
    results.update(rouge_results)
    
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
    overall_results = {}
    
    # Track metrics for calculating overall scores
    all_correct = 0
    all_total = 0
    all_metrics = {}
    
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
        
        # Track results for overall calculation
        if "accuracy" in category_results:
            # For classification tasks
            category_correct = category_results["accuracy"] * len(target_outputs)
            all_correct += category_correct
            all_total += len(target_outputs)
            
        # Track metrics by type for overall averages
        for metric_name, value in category_results.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {"total": 0, "count": 0}
            all_metrics[metric_name]["total"] += value
            all_metrics[metric_name]["count"] += 1
    
    # Calculate overall metrics
    if all_total > 0:
        overall_results["overall_accuracy"] = all_correct / all_total
        
    # Calculate average for each metric type
    for metric_name, data in all_metrics.items():
        if data["count"] > 0:
            overall_results[f"avg_{metric_name}"] = data["total"] / data["count"]
    
    # Add overall results
    results["overall"] = overall_results
    
    # Print results
    print("\nEvaluation Results:")
    for category, scores in results.items():
        if category != "overall":
            metric_used = TASK_METRICS.get(category, "rouge")
            # Modified to show that other metrics are included alongside the primary metric
            print(f"{category} (primary metric: {metric_used}, with additional Rouge metrics):")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.4f}")
    
    print("\nOverall Results:")
    for metric, value in results["overall"].items():
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