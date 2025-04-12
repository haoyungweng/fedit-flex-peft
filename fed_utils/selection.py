"""
Client selection strategies for federated learning.
Currently implements random selection, but can be extended with other strategies.
"""

from typing import Set, Optional, Any
import numpy as np


def select_clients(
    num_clients: int, 
    selection_fraction: float, 
    selection_strategy: str = "random", 
    seed: Optional[Any] = None
) -> Set[int]:
    """
    Select a subset of clients to participate in the current communication round.
    
    Args:
        num_clients: Total number of available clients
        selection_fraction: Fraction of clients to select (between 0 and 1)
        selection_strategy: Strategy for client selection ("random" or custom strategies)
        seed: Random seed for reproducibility
        
    Returns:
        Set of selected client IDs
    """
    np.random.seed(seed)
    
    if selection_strategy == "random":
        # Select at least one client, but not more than available
        num_selected = max(int(selection_fraction * num_clients), 1)
        selected_clients = set(np.random.choice(
            np.arange(num_clients), 
            num_selected, 
            replace=False
        ))
        return selected_clients
    
    # Add more selection strategies here in the future
    
    # Default to random selection if strategy not recognized
    return select_clients(num_clients, selection_fraction, "random", seed)