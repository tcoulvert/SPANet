"""Embedding layers with pairwise interaction support.

Extends the standard multi-input embedding to compute and return a pairwise
attention bias over the *full* concatenated particle sequence.

The bias is assembled block-by-block over the sequential inputs:

    * Same-type blocks (a collection with itself, e.g. Jets-Jets,
      BoostedJets-BoostedJets) are each embedded by their own MLP.
    * Cross-type blocks (two different collections, e.g. Jets-BoostedJets)
      are embedded by an *independent* MLP, and are only created when a
      resonance in the event file is built from more than one kind of jet
      (a "cross-type resonance", such as SRqqt1 = b: Jets + qq: BoostedJets).

This lets resonances made of mixed jet types receive proper pairwise features
instead of a zeroed-out attention bias.
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
    """Multi-input embedding with per-type and cross-type pairwise features."""

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super().__init__()

        self.options = options
        self.num_heads = options.num_attention_heads

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

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _cross_key(name_a: str, name_b: str) -> str:
        """Order-independent module key for a cross-type block."""
        first, second = sorted((name_a, name_b))
        return f"{first}___{second}"

    def _setup_pairwise_features(self, options: Options, training_dataset: JetReconstructionDataset):
        event_info = training_dataset.event_info
        input_names = list(event_info.input_types.keys())
        self.input_names = input_names

        # -----------------------------------------------------------------
        # Determine which SEQUENTIAL inputs participate in pairwise features.
        # An explicit pairwise_input_source restricts to a single collection
        # (legacy single-block behaviour); the default ("") uses all of them.
        # -----------------------------------------------------------------
        restrict = options.pairwise_input_source
        self.sequential_indices: List[int] = []
        for idx, name in enumerate(input_names):
            if event_info.input_types[name] == InputType.Sequential:
                if restrict == "" or restrict == name:
                    self.sequential_indices.append(idx)

        if len(self.sequential_indices) == 0:
            raise ValueError(
                "Could not find SEQUENTIAL input for pairwise features. "
                f"Specified: '{options.pairwise_input_source}', "
                f"Available: {list(event_info.input_types.items())}"
            )

        # Make sure normalization statistics exist before we snapshot them.
        if options.normalize_features and training_dataset.mean is None:
            training_dataset.compute_source_statistics()

        # -----------------------------------------------------------------
        # Per-input kinematic indices, transform flags, and de-norm stats.
        # -----------------------------------------------------------------
        self.kinematics = {}
        for idx in self.sequential_indices:
            name = input_names[idx]
            feature_infos = event_info.input_features[name]
            feature_names = [f.name for f in feature_infos]
            kin = auto_detect_kinematic_features(feature_names, name)

            pt_idx = kin["pt_idx"]
            eta_idx = kin["eta_idx"]
            mass_idx = kin["mass_idx"]

            kin["pt_is_log"] = bool(feature_infos[pt_idx].log_scale)
            kin["eta_is_normalized"] = bool(feature_infos[eta_idx].normalize)
            if not kin["use_sincos_phi"]:
                kin["phi_is_normalized"] = bool(feature_infos[kin["phi_idx"]].normalize)
            else:
                # sin/cos phi live in [-1, 1] and are not normalized.
                kin["phi_is_normalized"] = False
            kin["mass_is_normalized"] = bool(mass_idx >= 0 and feature_infos[mass_idx].normalize)

            self.kinematics[idx] = kin

            # Store de-normalization stats (mean/std) as buffers so eta/phi/mass
            # can be recovered to physical units. Registered per-input.
            if options.normalize_features:
                mean = training_dataset.mean[name].clone().detach()
                std = training_dataset.std[name].clone().detach()
            else:
                mean = torch.zeros(len(feature_infos))
                std = torch.ones(len(feature_infos))
            self.register_buffer(f"denorm_mean_{idx}", mean)
            self.register_buffer(f"denorm_std_{idx}", std)

        # -----------------------------------------------------------------
        # Detect cross-type resonances: an event particle whose daughters are
        # drawn from more than one participating SEQUENTIAL collection.
        # -----------------------------------------------------------------
        self.cross_pairs: List[Tuple[int, int]] = []
        if getattr(options, "pairwise_cross_type", True):
            seq_set = set(self.sequential_indices)
            coupled = set()
            for _particle, products in event_info.product_particles.items():
                sources = sorted({s for s in products.sources if s in seq_set})
                for a_pos in range(len(sources)):
                    for b_pos in range(a_pos + 1, len(sources)):
                        coupled.add((sources[a_pos], sources[b_pos]))
            self.cross_pairs = sorted(coupled)

        # -----------------------------------------------------------------
        # Feature computer (shared) and the per-block embedding MLPs.
        # -----------------------------------------------------------------
        num_features = options.num_pairwise_features
        num_heads = options.num_attention_heads
        embed_dim = options.pairwise_embedding_dim

        self.pairwise_computer = PairwiseFeatureComputer(num_features=num_features)

        # One independent MLP per collection (Jets-only, BoostedJets-only, ...).
        self.same_type_embeddings = nn.ModuleDict({
            input_names[idx]: PairwiseEmbedding(num_features, num_heads, embed_dim)
            for idx in self.sequential_indices
        })

        # One independent MLP per coupled cross-type collection pair.
        self.cross_type_embeddings = nn.ModuleDict({
            self._cross_key(input_names[a], input_names[b]): PairwiseEmbedding(num_features, num_heads, embed_dim)
            for (a, b) in self.cross_pairs
        })

    # -------------------------------------------------------------- kinematics
    def _extract_kinematics(
        self, idx: int, source_data: Tensor, source_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Recover physical (pt, eta, phi, mass) for a single sequential input."""
        kin = self.kinematics[idx]
        mean = getattr(self, f"denorm_mean_{idx}", None)
        std = getattr(self, f"denorm_std_{idx}", None)

        # pt: undo log(1 + pt) transform if applied.
        pt = source_data[:, :, kin["pt_idx"]]
        if kin["pt_is_log"]:
            pt = torch.expm1(pt).clamp(min=0)

        # eta: undo z-score normalization if applied.
        eta = source_data[:, :, kin["eta_idx"]]
        if kin["eta_is_normalized"] and mean is not None:
            eta = eta * std[kin["eta_idx"]] + mean[kin["eta_idx"]]

        # phi: from sin/cos or directly (denormalized if needed).
        if kin["use_sincos_phi"]:
            sinphi = source_data[:, :, kin["sinphi_idx"]]
            cosphi = source_data[:, :, kin["cosphi_idx"]]
            phi = sincos_to_phi(sinphi, cosphi)
        else:
            phi = source_data[:, :, kin["phi_idx"]]
            if kin["phi_is_normalized"] and mean is not None:
                phi = phi * std[kin["phi_idx"]] + mean[kin["phi_idx"]]

        # mass: undo z-score normalization if applied.
        mass = None
        if kin["mass_idx"] >= 0:
            mass = source_data[:, :, kin["mass_idx"]]
            if kin["mass_is_normalized"] and mean is not None:
                mass = mass * std[kin["mass_idx"]] + mean[kin["mass_idx"]]
                mass = mass.clamp(min=0)

        return pt, eta, phi, mass, source_mask

    # ------------------------------------------------------------------ forward
    def forward(
        self, sources: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []
        seq_lengths = []

        for input_index, vector_embedding_layer in enumerate(self.vector_embedding_layers):
            source_data, source_mask = sources[input_index]
            current_embeddings = vector_embedding_layer(source_data, source_mask)

            embeddings.append(current_embeddings[0])
            padding_masks.append(current_embeddings[1])
            sequence_masks.append(current_embeddings[2])
            global_masks.append(current_embeddings[3])
            # Sequence length this input occupies in the concatenated sequence.
            seq_lengths.append(current_embeddings[0].shape[0])

        embeddings = torch.cat(embeddings, dim=0)
        padding_masks = torch.cat(padding_masks, dim=1)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        global_masks = torch.cat(global_masks, dim=0)

        embeddings = self.final_embedding_layer(embeddings, sequence_masks)

        total_seq_len = embeddings.shape[0]
        batch_size = embeddings.shape[1]

        # Offset of every input inside the concatenated particle sequence.
        offsets = [0]
        for length in seq_lengths:
            offsets.append(offsets[-1] + length)

        # Extract physical kinematics for each participating sequential input.
        kinematics = {}
        for idx in self.sequential_indices:
            source_data, source_mask = sources[idx]
            kinematics[idx] = self._extract_kinematics(idx, source_data, source_mask)

        # Full attention bias over the concatenated sequence; zero-filled so any
        # non-sequential positions or uncoupled cross blocks contribute nothing.
        pairwise_bias = embeddings.new_zeros(
            batch_size * self.num_heads, total_seq_len, total_seq_len
        )

        # --- Same-type blocks: each collection with itself ---
        for idx in self.sequential_indices:
            pt, eta, phi, mass, mask = kinematics[idx]
            features = self.pairwise_computer(
                (pt, eta, phi, mass), (pt, eta, phi, mass),
                mask, mask, remove_self_pair=True,
            )
            block_bias = self.same_type_embeddings[self.input_names[idx]](features)
            offset = offsets[idx]
            length = pt.shape[1]
            pairwise_bias[:, offset:offset + length, offset:offset + length] = block_bias

        # --- Cross-type blocks: coupled collections (mixed-type resonances) ---
        for (a, b) in self.cross_pairs:
            pt_a, eta_a, phi_a, mass_a, mask_a = kinematics[a]
            pt_b, eta_b, phi_b, mass_b, mask_b = kinematics[b]
            features = self.pairwise_computer(
                (pt_a, eta_a, phi_a, mass_a), (pt_b, eta_b, phi_b, mass_b),
                mask_a, mask_b, remove_self_pair=False,
            )
            block_bias = self.cross_type_embeddings[
                self._cross_key(self.input_names[a], self.input_names[b])
            ](features)

            offset_a, offset_b = offsets[a], offsets[b]
            len_a, len_b = pt_a.shape[1], pt_b.shape[1]
            pairwise_bias[:, offset_a:offset_a + len_a, offset_b:offset_b + len_b] = block_bias
            # The transposed block: features are symmetric under i<->j, so the
            # bias for (b, a) is the transpose of the bias for (a, b).
            pairwise_bias[:, offset_b:offset_b + len_b, offset_a:offset_a + len_a] = block_bias.transpose(1, 2)

        return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias
