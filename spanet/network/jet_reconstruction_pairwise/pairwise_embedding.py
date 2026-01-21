"""Embedding layers with pairwise interaction support.

Extends the standard multi-input embedding to compute and return pairwise
attention bias.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.dataset.types import InputType
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset

from spanet.network.layers.linear_block import create_linear_block
from spanet.network.layers.embedding.combined_vector_embedding import CombinedVectorEmbedding

from .pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    sincos_to_phi,
)


class MultiInputVectorEmbeddingWithPairwise(nn.Module):
    """Multi-input embedding with pairwise feature computation."""

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super().__init__()

        self.options = options

        self.vector_embedding_layers = nn.ModuleList(
            [
                CombinedVectorEmbedding(options, training_dataset, input_name, input_type)
                for input_name, input_type in training_dataset.event_info.input_types.items()
            ]
        )

        self.final_embedding_layer = create_linear_block(
            options,
            options.position_embedding_dim + options.hidden_dim,
            options.hidden_dim,
            options.skip_connections,
        )

        self._setup_pairwise_features(options, training_dataset)

    def _setup_pairwise_features(self, options: Options, training_dataset: JetReconstructionDataset):
        event_info = training_dataset.event_info

        self.pairwise_source_idx = -1
        self.pairwise_source_name = ""
        self.kinematic_indices = None
        self._pt_is_log = False
        self._eta_is_normalized = False
        self._phi_is_normalized = False
        self._mass_is_normalized = False
        self._denorm_mean = None
        self._denorm_std = None

        for idx, (input_name, input_type) in enumerate(event_info.input_types.items()):
            if input_type == InputType.Sequential:
                if options.pairwise_input_source == "" or options.pairwise_input_source == input_name:
                    self.pairwise_source_idx = idx
                    self.pairwise_source_name = input_name
                    break

        if self.pairwise_source_idx == -1:
            raise ValueError(
                "Could not find SEQUENTIAL input for pairwise features. "
                f"Specified: '{options.pairwise_input_source}', "
                f"Available: {list(event_info.input_types.keys())}"
            )

        feature_infos = event_info.input_features[self.pairwise_source_name]
        feature_names = [f.name for f in feature_infos]
        kinematic_indices = auto_detect_kinematic_features(feature_names, self.pairwise_source_name)
        self.kinematic_indices = kinematic_indices
        
        # Check which transforms were applied to kinematics in the dataset
        pt_idx = kinematic_indices["pt_idx"]
        eta_idx = kinematic_indices["eta_idx"]
        mass_idx = kinematic_indices["mass_idx"]
        
        self._pt_is_log = bool(feature_infos[pt_idx].log_scale)
        self._eta_is_normalized = bool(feature_infos[eta_idx].normalize)
        # Check if phi is normalized (only relevant if using direct phi, not sinphi/cosphi)
        if not kinematic_indices["use_sincos_phi"]:
            phi_idx = kinematic_indices["phi_idx"]
            self._phi_is_normalized = bool(feature_infos[phi_idx].normalize)
        else:
            # sinphi/cosphi are typically not normalized (they're in [-1, 1] range)
            self._phi_is_normalized = False
        self._mass_is_normalized = bool(mass_idx >= 0 and feature_infos[mass_idx].normalize)

        # Get normalization stats for denormalizing eta and mass back to physical values
        # The embedding layer computes these stats, so we get them from the normalizer
        if options.normalize_features:
            # Ensure stats are computed (same as CombinedVectorEmbedding does)
            if training_dataset.mean is None:
                training_dataset.compute_source_statistics()
            mean = training_dataset.mean[self.pairwise_source_name]
            std = training_dataset.std[self.pairwise_source_name]
            # Store only the stats we need for denormalization
            self._denorm_mean = nn.Parameter(mean, requires_grad=False)
            self._denorm_std = nn.Parameter(std, requires_grad=False)
        else:
            # No normalization was applied, so no need to denormalize
            self._denorm_mean = None
            self._denorm_std = None

        self.pairwise_computer = PairwiseFeatureComputer(num_features=options.num_pairwise_features)
        self.pairwise_embedding = PairwiseEmbedding(
            num_features=options.num_pairwise_features,
            num_heads=options.num_attention_heads,
            embed_dim=options.pairwise_embedding_dim,
        )

    def _extract_kinematics(
        self, source_data: Tensor, source_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        idx = self.kinematic_indices

        # Extract pt and undo log transform if applied
        pt = source_data[:, :, idx["pt_idx"]]
        if self._pt_is_log:
            # Dataset transform is log(pt + 1). Recover pt.
            pt = torch.expm1(pt).clamp(min=0)

        # Extract eta and denormalize if z-score normalization was applied
        eta = source_data[:, :, idx["eta_idx"]]
        if self._eta_is_normalized and self._denorm_mean is not None:
            # Denormalize: x_physical = x_normalized * std + mean
            eta = eta * self._denorm_std[idx["eta_idx"]] + self._denorm_mean[idx["eta_idx"]]

        # Extract phi
        if idx["use_sincos_phi"]:
            # sinphi/cosphi are typically not normalized (they're in [-1, 1] range)
            sinphi = source_data[:, :, idx["sinphi_idx"]]
            cosphi = source_data[:, :, idx["cosphi_idx"]]
            phi = sincos_to_phi(sinphi, cosphi)
        else:
            # Direct phi: check if it needs denormalization
            phi = source_data[:, :, idx["phi_idx"]]
            if self._phi_is_normalized and self._denorm_mean is not None:
                # Denormalize: x_physical = x_normalized * std + mean
                phi = phi * self._denorm_std[idx["phi_idx"]] + self._denorm_mean[idx["phi_idx"]]

        # Extract mass and denormalize if z-score normalization was applied
        mass = None
        if idx["mass_idx"] >= 0:
            mass = source_data[:, :, idx["mass_idx"]]
            if self._mass_is_normalized and self._denorm_mean is not None:
                # Denormalize: x_physical = x_normalized * std + mean
                mass = mass * self._denorm_std[idx["mass_idx"]] + self._denorm_mean[idx["mass_idx"]]
                # Ensure mass is non-negative after denormalization
                mass = mass.clamp(min=0)

        return pt, eta, phi, mass, source_mask

    def forward(
        self, sources: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []

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

        source_data, source_mask = sources[self.pairwise_source_idx]
        pt, eta, phi, mass, mask = self._extract_kinematics(source_data, source_mask)

        pairwise_features = self.pairwise_computer(pt, eta, phi, mass, mask)
        pairwise_bias = self.pairwise_embedding(pairwise_features)

        total_seq_len = embeddings.shape[0]
        source_seq_len = source_data.shape[1]

        if total_seq_len != source_seq_len:
            batch_heads = pairwise_bias.shape[0]
            full_bias = pairwise_bias.new_zeros(batch_heads, total_seq_len, total_seq_len)
            full_bias[:, :source_seq_len, :source_seq_len] = pairwise_bias
            pairwise_bias = full_bias

        return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias

