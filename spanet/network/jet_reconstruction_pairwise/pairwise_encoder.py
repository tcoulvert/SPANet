"""Encoder layers with pairwise interaction support.

These classes extend the standard SPANet encoder layers to propagate
pairwise attention bias through the transformer stack.
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.network.layers.linear_stack import create_linear_stack
from .pairwise_transformer import create_transformer_with_pairwise


class StackedEncoderWithPairwise(nn.Module):
    """Stacked encoder with pairwise attention bias support."""

    def __init__(self, options: Options, num_linear_layers: int, num_encoder_layers: int):
        super().__init__()

        self.hidden_dim = options.hidden_dim
        self.num_heads = options.num_attention_heads
        self.particle_vector = nn.Parameter(torch.randn(1, 1, options.hidden_dim))

        self.encoder = create_transformer_with_pairwise(options, num_encoder_layers)
        self.embedding = create_linear_stack(options, num_linear_layers, options.hidden_dim, options.skip_connections)

    def forward(
        self,
        encoded_vectors: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[list]]:
        num_vectors, batch_size, hidden_dim = encoded_vectors.shape

        encoded_vectors = self.embedding(encoded_vectors, sequence_mask)

        particle_vector = self.particle_vector.expand(1, batch_size, hidden_dim)
        combined_vectors = torch.cat((particle_vector, encoded_vectors), dim=0)

        particle_padding_mask = padding_mask.new_zeros(batch_size, 1)
        combined_padding_mask = torch.cat((particle_padding_mask, padding_mask), dim=1)

        particle_sequence_mask = sequence_mask.new_ones(1, batch_size, 1, dtype=torch.bool)
        combined_sequence_mask = torch.cat((particle_sequence_mask, sequence_mask), dim=0)

        combined_pairwise_bias = None
        if pairwise_bias is not None:
            batch_heads, t, _ = pairwise_bias.shape
            combined_pairwise_bias = pairwise_bias.new_zeros(batch_heads, t + 1, t + 1)
            combined_pairwise_bias[:, 1:, 1:] = pairwise_bias

        combined_vectors, gate_logits_list = self.encoder(
            combined_vectors,
            combined_padding_mask,
            combined_sequence_mask,
            combined_pairwise_bias,
        )
        particle_vector, encoded_vectors = combined_vectors[0], combined_vectors[1:]
        return encoded_vectors, particle_vector, gate_logits_list


class JetEncoderWithPairwise(StackedEncoderWithPairwise):
    """Central jet encoder with pairwise attention bias support."""

    def __init__(self, options: Options):
        super().__init__(options, 0, options.num_encoder_layers)
