"""Mixture of Experts (MoE) layer implementation.

Implements a MoE layer with top-k expert selection for transformer feedforward networks.
"""

import torch
from torch import Tensor, nn
from typing import Tuple

from spanet.options import Options
from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.network.layers.linear_block.gru_block import GRUGate


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k expert selection.
    
    This layer replaces standard feedforward layers in transformers with a MoE architecture.
    Each token is routed to the top-k experts, and their outputs are combined.
    """
    
    def __init__(
        self,
        options: Options,
        input_dim: int,
        output_dim: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        skip_connection: bool = False
    ):
        super(MoELayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.skip_connection = skip_connection
        self.hidden_dim = int(round(options.transformer_dim_scale * input_dim))
        
        # Router network: maps input to expert logits
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Create expert networks
        self.experts = nn.ModuleList([
            self._create_expert(options, input_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Normalization layer
        self.normalization = create_normalization(options.normalization, input_dim)
        
        # GRU gate for skip connection (if enabled)
        if skip_connection:
            self.gru = GRUGate(output_dim)
            self.residual = create_residual_connection(skip_connection, input_dim, output_dim)
        else:
            self.gru = None
            self.residual = None
        
        # Masking layer
        self.masking = create_masking(options.masking)
        
        # Store gate logits for loss computation
        self.gate_logits = None
        
    def _create_expert(self, options: Options, input_dim: int, output_dim: int) -> nn.Module:
        """Create a single expert network (feedforward MLP)."""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            create_activation(options.linear_activation, self.hidden_dim),
            create_dropout(options.dropout),
            nn.Linear(self.hidden_dim, output_dim),
            create_activation(options.linear_activation, output_dim),
            create_dropout(options.dropout)
        )
    
    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through MoE layer.
        
        Parameters
        ----------
        x: [T, B, D]
            Input tensor (sequence_length, batch_size, hidden_dim)
        sequence_mask: [T, B, 1]
            Positive mask indicating valid tokens
            
        Returns
        -------
        output: [T, B, D]
            Output tensor
        gate_logits: [T*B, num_experts]
            Gate logits for computing MoE loss
        """
        timesteps, batch_size, input_dim = x.shape
        total_tokens = timesteps * batch_size
        
        # Normalize input
        x_normalized = self.normalization(x, sequence_mask)
        
        # Reshape to [T*B, D] for processing
        x_flat = x_normalized.reshape(total_tokens, input_dim)
        
        # Compute router logits: [T*B, num_experts]
        gate_logits = self.router(x_flat)
        
        # Store gate logits for loss computation
        self.gate_logits = gate_logits
        
        # Top-k selection: get top-k experts for each token
        top_k_gate_logits, top_k_indices = torch.topk(
            gate_logits, 
            k=self.num_experts_per_tok, 
            dim=1
        )
        
        # Compute gate probabilities (softmax over top-k)
        gate_probs = torch.softmax(top_k_gate_logits, dim=1)  # [T*B, k]
        
        # Initialize output tensor
        output = torch.zeros(total_tokens, self.output_dim, device=x.device, dtype=x.dtype)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=1)  # [T*B]
            
            if expert_mask.any():
                # Get tokens that use this expert
                expert_tokens = x_flat[expert_mask]  # [num_tokens, D]
                
                # Compute expert output
                expert_output = self.experts[expert_idx](expert_tokens)  # [num_tokens, D]
                
                # Find which position in top-k this expert is for each token
                expert_positions = (top_k_indices[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]
                
                # Get corresponding gate probabilities
                batch_indices = expert_mask.nonzero(as_tuple=True)[0]
                expert_gates = gate_probs[batch_indices, expert_positions]  # [num_tokens]
                
                # Weight and accumulate expert outputs
                output[expert_mask] += expert_gates.unsqueeze(1) * expert_output
        
        # Reshape back to [T, B, D]
        output = output.reshape(timesteps, batch_size, self.output_dim)
        
        # Apply GRU gate and skip connection if enabled
        if self.skip_connection and self.gru is not None:
            output = self.gru(output, self.residual(x))
        
        # Apply masking
        output = self.masking(output, sequence_mask)
        
        return output, gate_logits
