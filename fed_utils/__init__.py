"""
Federated Learning Utilities for PyTorch models.
"""

# Import main components to expose in the public API
from .aggregation import fedavg, hetlora, load_hetlora_weights
from .selection import select_clients
from .client import FederatedClient
from .evaluation import evaluate_global_model
from .prompter import Prompter

# Make main components available in the top-level package
__all__ = [
    'fedavg',
    'hetlora', 
    'load_hetlora_weights',
    'select_clients',
    'FederatedClient',
    'evaluate_global_model',
    'Prompter',
]