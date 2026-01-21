"""Pairwise interaction support for SPANet jet reconstruction.

This package provides pairwise-aware embedding and encoder components that are
optionally enabled via Options.use_pairwise_interactions.

The main jet reconstruction model remains `JetReconstructionModel`; pairwise
functionality is activated through hooks in
`spanet.network.jet_reconstruction.jet_reconstruction_network.JetReconstructionNetwork`.
"""

from .pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    sincos_to_phi,
)
from .pairwise_transformer import create_transformer_with_pairwise
from .pairwise_encoder import JetEncoderWithPairwise, StackedEncoderWithPairwise
from .pairwise_embedding import MultiInputVectorEmbeddingWithPairwise

__all__ = [
    "PairwiseFeatureComputer",
    "PairwiseEmbedding",
    "auto_detect_kinematic_features",
    "sincos_to_phi",
    "create_transformer_with_pairwise",
    "JetEncoderWithPairwise",
    "StackedEncoderWithPairwise",
    "MultiInputVectorEmbeddingWithPairwise",
]
