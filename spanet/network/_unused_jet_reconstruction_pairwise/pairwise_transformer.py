"""Transformer layers with pairwise interaction support.

These classes extend the standard SPANet transformer layers to accept
pairwise attention bias, which is added to the attention scores before softmax.
"""
from typing import Optional

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.network.layers.linear_block.gru_block import GRUGate, GRUBlock
from spanet.network.layers.transformer.transformer_base import TransformerBase


class GTrXLWithPairwise(nn.Module):
    """Gated Transformer-XL layer with pairwise attention bias support.

    Extends the standard GTrXL to pass attn_mask (pairwise bias) to
    the MultiheadAttention layer.

    Parameters
    ----------
    options : Options
        SPANet configuration options
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout rate
    """

    def __init__(self, options: Options, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = GRUBlock(options, hidden_dim, hidden_dim, skip_connection=True)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with optional pairwise attention bias.

        Parameters
        ----------
        x : Tensor [T, B, D]
            Input sequence
        padding_mask : Tensor [B, T]
            Padding mask for attention (True = ignore)
        sequence_mask : Tensor [T, B, 1]
            Sequence mask for feed-forward
        pairwise_bias : Tensor [B*H, T, T], optional
            Pairwise attention bias to add to attention scores

        Returns
        -------
        Tensor [T, B, D]
            Output sequence
        """
        output = self.attention_norm(x)
        output, _ = self.attention(
            output, output, output,
            key_padding_mask=padding_mask,
            attn_mask=pairwise_bias,
            need_weights=False
        )

        output = self.attention_gate(output, x)

        return self.feed_forward(output, sequence_mask)


class GatedTransformerWithPairwise(TransformerBase):
    """Gated Transformer with pairwise attention bias support.

    Extends GatedTransformer to propagate pairwise_bias through all layers.

    Parameters
    ----------
    options : Options
        SPANet configuration options
    num_layers : int
        Number of transformer layers
    """

    def __init__(self, options: Options, num_layers: int):
        super().__init__(options, num_layers)

        self.layers = nn.ModuleList([
            GTrXLWithPairwise(options, self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with optional pairwise attention bias.

        Parameters
        ----------
        x : Tensor [T, B, D]
            Input sequence
        padding_mask : Tensor [B, T]
            Padding mask for attention
        sequence_mask : Tensor [T, B, 1]
            Sequence mask for feed-forward
        pairwise_bias : Tensor [B*H, T, T], optional
            Pairwise attention bias

        Returns
        -------
        Tensor [T, B, D]
            Output sequence
        """
        output = x

        for layer in self.layers:
            output = layer(output, padding_mask, sequence_mask, pairwise_bias)

        return output


def create_transformer_with_pairwise(options: Options, num_layers: int) -> nn.Module:
    """Factory function to create a transformer with pairwise support.

    Currently only supports Gated transformer type. For other types,
    falls back to standard transformer (without pairwise support).

    Parameters
    ----------
    options : Options
        SPANet configuration options
    num_layers : int
        Number of transformer layers

    Returns
    -------
    nn.Module
        Transformer module with pairwise support
    """
    if num_layers == 0:
        return nn.Identity()

    if options.transformer_type == "Gated":
        return GatedTransformerWithPairwise(options, num_layers)
    else:
        # Fall back to standard transformer for other types
        # Note: These won't have pairwise support
        from spanet.network.layers.transformer import create_transformer
        return create_transformer(options, num_layers)
