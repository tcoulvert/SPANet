"""Encoder layers with pairwise interaction support.

These classes extend the standard SPANet encoder layers to propagate
pairwise attention bias through the transformer stack.
"""
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.network.layers.linear_stack import create_linear_stack
from spanet.network.jet_reconstruction_pairwise.pairwise_transformer import (
    create_transformer_with_pairwise
)


class StackedEncoderWithPairwise(nn.Module):
    """Stacked encoder with pairwise attention bias support.

    Extends StackedEncoder to:
    1. Accept pairwise_bias parameter
    2. Expand bias from [B*H, T, T] to [B*H, T+1, T+1] for particle vector
    3. Pass bias to transformer layers

    Parameters
    ----------
    options : Options
        SPANet configuration options
    num_linear_layers : int
        Number of linear embedding layers
    num_encoder_layers : int
        Number of transformer encoder layers
    """

    def __init__(
        self,
        options: Options,
        num_linear_layers: int,
        num_encoder_layers: int
    ):
        super().__init__()

        self.hidden_dim = options.hidden_dim
        self.num_heads = options.num_attention_heads
        self.particle_vector = nn.Parameter(torch.randn(1, 1, options.hidden_dim))

        self.encoder = create_transformer_with_pairwise(options, num_encoder_layers)
        self.embedding = create_linear_stack(
            options, num_linear_layers, options.hidden_dim, options.skip_connections
        )

    def forward(
        self,
        encoded_vectors: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with optional pairwise attention bias.

        Parameters
        ----------
        encoded_vectors : Tensor [T, B, D]
            Input sequence
        padding_mask : Tensor [B, T]
            Padding mask for transformer
        sequence_mask : Tensor [T, B, 1]
            Sequence mask for linear layers
        pairwise_bias : Tensor [B*H, T, T], optional
            Pairwise attention bias

        Returns
        -------
        encoded_vectors : Tensor [T, B, D]
            Encoded output vectors
        particle_vector : Tensor [B, D]
            Aggregated particle-level representation
        """
        num_vectors, batch_size, hidden_dim = encoded_vectors.shape

        # -----------------------------------------------------------------------------
        # Embed vectors again into particle space
        # vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        encoded_vectors = self.embedding(encoded_vectors, sequence_mask)

        # -----------------------------------------------------------------------------
        # Add a "particle vector" which will store particle level data.
        # particle_vector: [1, B, D]
        # combined_vectors: [T + 1, B, D]
        # -----------------------------------------------------------------------------
        particle_vector = self.particle_vector.expand(1, batch_size, hidden_dim)
        combined_vectors = torch.cat((particle_vector, encoded_vectors), dim=0)

        # -----------------------------------------------------------------------------
        # Also modify the padding mask to indicate that the particle vector is real.
        # particle_padding_mask: [B, 1]
        # combined_padding_mask: [B, T + 1]
        # -----------------------------------------------------------------------------
        particle_padding_mask = padding_mask.new_zeros(batch_size, 1)
        combined_padding_mask = torch.cat((particle_padding_mask, padding_mask), dim=1)

        # -----------------------------------------------------------------------------
        # Also modify the sequence mask to indicate that the particle vector is real.
        # particle_sequence_mask: [1, B, 1]
        # combined_sequence_mask: [T + 1, B, 1]
        # -----------------------------------------------------------------------------
        particle_sequence_mask = sequence_mask.new_ones(1, batch_size, 1, dtype=torch.bool)
        combined_sequence_mask = torch.cat((particle_sequence_mask, sequence_mask), dim=0)

        # -----------------------------------------------------------------------------
        # Expand pairwise bias to include particle vector position
        # pairwise_bias: [B*H, T, T] -> [B*H, T+1, T+1]
        # The particle vector position (index 0) has no pairwise bias (zeros)
        # -----------------------------------------------------------------------------
        combined_pairwise_bias = None
        if pairwise_bias is not None:
            batch_heads, t, _ = pairwise_bias.shape
            combined_pairwise_bias = pairwise_bias.new_zeros(batch_heads, t + 1, t + 1)
            combined_pairwise_bias[:, 1:, 1:] = pairwise_bias

        # -----------------------------------------------------------------------------
        # Run all of the vectors through transformer encoder
        # combined_vectors: [T + 1, B, D]
        # particle_vector: [B, D]
        # encoded_vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        combined_vectors = self.encoder(
            combined_vectors,
            combined_padding_mask,
            combined_sequence_mask,
            combined_pairwise_bias
        )
        particle_vector, encoded_vectors = combined_vectors[0], combined_vectors[1:]

        return encoded_vectors, particle_vector


class JetEncoderWithPairwise(StackedEncoderWithPairwise):
    """Central jet encoder with pairwise attention bias support.

    This is the main encoder used in JetReconstructionNetwork.
    Uses num_encoder_layers transformer layers with no linear embedding.

    Parameters
    ----------
    options : Options
        SPANet configuration options
    """

    def __init__(self, options: Options):
        super().__init__(options, 0, options.num_encoder_layers)
