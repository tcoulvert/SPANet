"""Jet reconstruction with Particle Transformer-style pairwise interactions.

This module provides pairwise-aware versions of the SPANet components.
Use JetReconstructionModelWithPairwise as a drop-in replacement for
JetReconstructionModel when pairwise interactions are desired.

Usage:
    In options.json, set "use_pairwise_interactions": true
    The factory function in spanet/__init__.py will automatically select
    the appropriate model class.
"""
from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    sincos_to_phi,
    delta_phi
)

from spanet.network.jet_reconstruction_pairwise.pairwise_transformer import (
    GTrXLWithPairwise,
    GatedTransformerWithPairwise,
    create_transformer_with_pairwise
)

from spanet.network.jet_reconstruction_pairwise.pairwise_encoder import (
    StackedEncoderWithPairwise,
    JetEncoderWithPairwise
)

from spanet.network.jet_reconstruction_pairwise.pairwise_embedding import (
    MultiInputVectorEmbeddingWithPairwise
)

from spanet.network.jet_reconstruction_pairwise.jet_reconstruction_pairwise import (
    JetReconstructionNetworkWithPairwise
)

from spanet.network.jet_reconstruction_pairwise.jet_reconstruction_model_pairwise import (
    JetReconstructionModelWithPairwise
)

__all__ = [
    # Feature computation
    'PairwiseFeatureComputer',
    'PairwiseEmbedding',
    'auto_detect_kinematic_features',
    'sincos_to_phi',
    'delta_phi',

    # Transformer
    'GTrXLWithPairwise',
    'GatedTransformerWithPairwise',
    'create_transformer_with_pairwise',

    # Encoder
    'StackedEncoderWithPairwise',
    'JetEncoderWithPairwise',

    # Embedding
    'MultiInputVectorEmbeddingWithPairwise',

    # Network and Model
    'JetReconstructionNetworkWithPairwise',
    'JetReconstructionModelWithPairwise',
]
