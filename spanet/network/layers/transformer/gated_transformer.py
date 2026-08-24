from torch import Tensor, nn, jit
from typing import Tuple, Optional

from spanet.options import Options
from spanet.network.layers.linear_block.gru_block import GRUGate, GRUBlock
from spanet.network.layers.transformer.transformer_base import TransformerBase


class GTrXL(nn.Module):
    def __init__(self, options, hidden_dim: int, num_heads: int, dropout: float):
        super(GTrXL, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Conditionally use MoE or standard feedforward
        self.use_moe = getattr(options, 'use_moe', False)
        if self.use_moe:
            from spanet.network.layers.moe import MoELayer
            self.feed_forward = MoELayer(
                options,
                hidden_dim,
                hidden_dim,
                num_experts=getattr(options, 'num_experts', 8),
                num_experts_per_tok=getattr(options, 'num_experts_per_tok', 2),
                skip_connection=True
            )
        else:
            self.feed_forward = GRUBlock(options, hidden_dim, hidden_dim, skip_connection=True)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        output = self.attention_norm(x)
        output, _ = self.attention(
            output, output, output,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        output = self.attention_gate(output, x)

        if self.use_moe:
            output, gate_logits = self.feed_forward(output, sequence_mask)
            return output, gate_logits
        else:
            output = self.feed_forward(output, sequence_mask)
            return output, None


class GatedTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(GatedTransformer, self).__init__(options, num_layers)

        self.use_moe = getattr(options, 'use_moe', False)
        self.layers = nn.ModuleList([
            GTrXL(options, self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tuple[Tensor, Optional[list]]:
        output = x
        gate_logits_list = [] if self.use_moe else None

        for layer in self.layers:
            if self.use_moe:
                output, gate_logits = layer(output, padding_mask, sequence_mask)
                if gate_logits is not None:
                    gate_logits_list.append(gate_logits)
            else:
                output, _ = layer(output, padding_mask, sequence_mask)

        return output, gate_logits_list
