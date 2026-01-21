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
    """Gated Transformer-XL layer with pairwise attention bias support."""

    def __init__(self, options: Options, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = GRUBlock(options, hidden_dim, hidden_dim, skip_connection=True)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.attention_norm(x)
        # Convert padding_mask to float to match pairwise_bias dtype (PyTorch requirement)
        # bool True (padding) -> float 1.0 (ignore), bool False (real) -> float 0.0 (attend)
        padding_mask_float = padding_mask.float() if padding_mask.dtype == torch.bool else padding_mask
        output, _ = self.attention(
            output,
            output,
            output,
            key_padding_mask=padding_mask_float,
            attn_mask=pairwise_bias,
            need_weights=False,
        )

        output = self.attention_gate(output, x)
        return self.feed_forward(output, sequence_mask)


class GatedTransformerWithPairwise(TransformerBase):
    """Gated Transformer that propagates pairwise_bias through all layers."""

    def __init__(self, options: Options, num_layers: int):
        super().__init__(options, num_layers)

        self.layers = nn.ModuleList(
            [GTrXLWithPairwise(options, self.hidden_dim, self.num_heads, self.dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        sequence_mask: Tensor,
        pairwise_bias: Optional[Tensor] = None,
    ) -> Tensor:
        output = x
        for layer in self.layers:
            output = layer(output, padding_mask, sequence_mask, pairwise_bias)
        return output


def create_transformer_with_pairwise(options: Options, num_layers: int) -> nn.Module:
    """Factory function to create a transformer with pairwise support."""
    if num_layers == 0:
        return nn.Identity()

    if options.transformer_type == "Gated":
        return GatedTransformerWithPairwise(options, num_layers)

    # Fall back to standard transformer for other types (no pairwise support).
    from spanet.network.layers.transformer import create_transformer

    return create_transformer(options, num_layers)
