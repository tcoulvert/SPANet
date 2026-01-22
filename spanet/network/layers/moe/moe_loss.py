"""MoE auxiliary loss computation.

Computes load balancing loss to encourage uniform expert usage.
"""

import torch
from torch import Tensor


def compute_moe_loss(gate_logits: Tensor, num_experts: int) -> Tensor:
    """Compute MoE auxiliary loss for load balancing.
    
    This loss encourages uniform distribution of tokens across experts.
    It computes the coefficient of variation of expert usage.
    
    Parameters
    ----------
    gate_logits: [T*B, num_experts]
        Router gate logits for all tokens
    num_experts: int
        Total number of experts
        
    Returns
    -------
    loss: scalar Tensor
        MoE auxiliary loss (load balancing loss)
    """
    if gate_logits is None:
        return torch.tensor(0.0)
    
    # Compute gate probabilities (softmax over all experts)
    gate_probs = torch.softmax(gate_logits, dim=1)  # [T*B, num_experts]
    
    # Compute average gate probability per expert (load per expert)
    load_per_expert = gate_probs.mean(dim=0)  # [num_experts]
    
    # Compute coefficient of variation: std / mean
    # Higher CV means more imbalanced expert usage
    mean_load = load_per_expert.mean()
    std_load = load_per_expert.std()
    
    # Avoid division by zero
    if mean_load > 0:
        cv_squared = (std_load / mean_load) ** 2
    else:
        cv_squared = torch.tensor(0.0, device=gate_logits.device)
    
    return cv_squared
