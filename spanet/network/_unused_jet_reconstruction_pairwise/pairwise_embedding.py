"""Embedding layers with pairwise interaction support.

Extends MultiInputVectorEmbedding to compute and return pairwise attention bias.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.dataset.types import InputType
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset

from spanet.network.layers.linear_block import create_linear_block
from spanet.network.layers.embedding.combined_vector_embedding import CombinedVectorEmbedding

from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    sincos_to_phi
)


class MultiInputVectorEmbeddingWithPairwise(nn.Module):
    """Multi-input embedding with pairwise feature computation.

    Extends MultiInputVectorEmbedding to:
    1. Auto-detect kinematic features (pt, eta, phi, mass) from event info
    2. Compute pairwise features between particles
    3. Embed pairwise features into attention bias format

    Parameters
    ----------
    options : Options
        SPANet configuration options
    training_dataset : JetReconstructionDataset
        Training dataset for normalization statistics
    """

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super().__init__()

        self.options = options

        # Primary embedding blocks (same as original)
        self.vector_embedding_layers = nn.ModuleList([
            CombinedVectorEmbedding(options, training_dataset, input_name, input_type)
            for input_name, input_type in training_dataset.event_info.input_types.items()
        ])

        self.final_embedding_layer = create_linear_block(
            options,
            options.position_embedding_dim + options.hidden_dim,
            options.hidden_dim,
            options.skip_connections
        )

        # Pairwise feature components
        self._setup_pairwise_features(options, training_dataset)

    def _setup_pairwise_features(
        self,
        options: Options,
        training_dataset: JetReconstructionDataset
    ):
        """Set up pairwise feature computation."""
        event_info = training_dataset.event_info

        # Find the sequential input to use for pairwise features
        self.pairwise_source_idx = -1
        self.pairwise_source_name = ""
        self.kinematic_indices = None

        # Find first SEQUENTIAL input or use specified source
        for idx, (input_name, input_type) in enumerate(event_info.input_types.items()):
            if input_type == InputType.Sequential:
                if options.pairwise_input_source == "" or options.pairwise_input_source == input_name:
                    self.pairwise_source_idx = idx
                    self.pairwise_source_name = input_name
                    break

        if self.pairwise_source_idx == -1:
            raise ValueError(
                f"Could not find SEQUENTIAL input for pairwise features. "
                f"Specified: '{options.pairwise_input_source}', "
                f"Available: {list(event_info.input_types.keys())}"
            )

        # Get feature names and auto-detect kinematic indices
        feature_infos = event_info.input_features[self.pairwise_source_name]
        feature_names = [f.name for f in feature_infos]
        self.kinematic_indices = auto_detect_kinematic_features(
            feature_names, self.pairwise_source_name
        )

        # Create pairwise feature modules
        self.pairwise_computer = PairwiseFeatureComputer(
            num_features=options.num_pairwise_features
        )
        self.pairwise_embedding = PairwiseEmbedding(
            num_features=options.num_pairwise_features,
            num_heads=options.num_attention_heads,
            embed_dim=options.pairwise_embedding_dim
        )

    def _extract_kinematics(
        self,
        source_data: Tensor,
        source_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Extract pt, eta, phi, mass from source data.

        Parameters
        ----------
        source_data : Tensor [B, N, D]
            Raw input features
        source_mask : Tensor [B, N]
            Boolean mask for valid particles

        Returns
        -------
        pt, eta, phi, mass, mask : Tensors
        """
        idx = self.kinematic_indices

        pt = source_data[:, :, idx['pt_idx']]
        eta = source_data[:, :, idx['eta_idx']]

        if idx['use_sincos_phi']:
            sinphi = source_data[:, :, idx['sinphi_idx']]
            cosphi = source_data[:, :, idx['cosphi_idx']]
            phi = sincos_to_phi(sinphi, cosphi)
        else:
            phi = source_data[:, :, idx['phi_idx']]

        if idx['mass_idx'] >= 0:
            mass = source_data[:, :, idx['mass_idx']]
        else:
            mass = None

        return pt, eta, phi, mass, source_mask

    def forward(
        self,
        sources: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Forward pass with pairwise feature computation.

        Parameters
        ----------
        sources : List[Tuple[Tensor, Tensor]]
            List of (source_data, source_mask) tuples for each input type
            source_data: [B, N, D] batch-first input features
            source_mask: [B, N] boolean mask

        Returns
        -------
        embeddings : Tensor [T, B, D]
            Combined embeddings in sequence-first format
        padding_mask : Tensor [B, T]
            Negative mask for transformer (True = padding)
        sequence_mask : Tensor [T, B, 1]
            Positive mask (True = real token)
        global_mask : Tensor [T]
            Mask for global vs sequential features
        pairwise_bias : Tensor [B*H, T, T] or None
            Pairwise attention bias
        """
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []

        # Standard embedding for all input types
        for input_index, vector_embedding_layer in enumerate(self.vector_embedding_layers):
            source_data, source_mask = sources[input_index]

            current_embeddings = vector_embedding_layer(source_data, source_mask)

            embeddings.append(current_embeddings[0])
            padding_masks.append(current_embeddings[1])
            sequence_masks.append(current_embeddings[2])
            global_masks.append(current_embeddings[3])

        embeddings = torch.cat(embeddings, dim=0)
        padding_masks = torch.cat(padding_masks, dim=1)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        global_masks = torch.cat(global_masks, dim=0)

        embeddings = self.final_embedding_layer(embeddings, sequence_masks)

        # Compute pairwise features from the designated sequential input
        source_data, source_mask = sources[self.pairwise_source_idx]
        pt, eta, phi, mass, mask = self._extract_kinematics(source_data, source_mask)

        # Compute pairwise features: [B, N, N, num_features]
        pairwise_features = self.pairwise_computer(pt, eta, phi, mass, mask)

        # Embed into attention bias format: [B*H, N, N]
        pairwise_bias = self.pairwise_embedding(pairwise_features)

        # Expand pairwise bias to full sequence length if there are multiple inputs
        # Currently only supports pairwise features for the first sequential input
        # The bias for other positions (global inputs) will be zero
        total_seq_len = embeddings.shape[0]
        source_seq_len = source_data.shape[1]

        if total_seq_len != source_seq_len:
            batch_heads = pairwise_bias.shape[0]
            full_bias = pairwise_bias.new_zeros(batch_heads, total_seq_len, total_seq_len)
            # Place the pairwise bias at the beginning (sequential input comes first)
            full_bias[:, :source_seq_len, :source_seq_len] = pairwise_bias
            pairwise_bias = full_bias

        return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias
