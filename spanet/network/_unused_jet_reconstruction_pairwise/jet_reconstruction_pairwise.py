"""Jet reconstruction network with Particle Transformer-style pairwise interactions.

Extends JetReconstructionNetwork to use pairwise attention bias in the central encoder.
"""
import numpy as np
import torch
from torch import nn

from spanet.options import Options
from spanet.dataset.types import Tuple, Outputs, Source, Predictions

from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder

from spanet.network.prediction_selection import extract_predictions
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase

from spanet.network.jet_reconstruction_pairwise.pairwise_embedding import (
    MultiInputVectorEmbeddingWithPairwise
)
from spanet.network.jet_reconstruction_pairwise.pairwise_encoder import (
    JetEncoderWithPairwise
)

TArray = np.ndarray


def default_assignment_fn(outputs: Outputs):
    return extract_predictions([
        np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
        for assignment in outputs.assignments
    ])


class JetReconstructionNetworkWithPairwise(JetReconstructionBase):
    """SPANet architecture with Particle Transformer-style pairwise attention.

    Extends JetReconstructionNetwork to:
    1. Use MultiInputVectorEmbeddingWithPairwise for embedding
    2. Use JetEncoderWithPairwise for central encoder
    3. Pass pairwise attention bias from embedding to encoder

    Parameters
    ----------
    options : Options
        Global options for the network
    torch_script : bool
        Whether to compile modules with TorchScript
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim

        # Use pairwise-aware embedding
        self.embedding = compile_module(MultiInputVectorEmbeddingWithPairwise(
            options,
            self.training_dataset
        ))

        # Use pairwise-aware encoder
        self.encoder = compile_module(JetEncoderWithPairwise(
            options,
        ))

        # Branch decoders (same as original - no pairwise in branch decoders)
        self.branch_decoders = nn.ModuleList([
            BranchDecoder(
                options,
                event_particle_name,
                self.event_info.product_particles[event_particle_name].names,
                product_symmetry,
                self.enable_softmax
            )
            for event_particle_name, product_symmetry
            in self.event_info.product_symmetries.items()
        ])

        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        self.classification_decoder = compile_module(ClassificationDecoder(
            options,
            self.training_dataset
        ))

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Source, ...]) -> Outputs:
        """Forward pass with pairwise attention.

        Parameters
        ----------
        sources : Tuple[Source, ...]
            Input sources (data, mask) for each input type

        Returns
        -------
        Outputs
            Model outputs including assignments, detections, regressions, classifications
        """
        # Embed inputs and compute pairwise features
        embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias = self.embedding(sources)

        # Extract features using transformer with pairwise attention
        hidden, event_vector = self.encoder(
            embeddings, padding_masks, sequence_masks, pairwise_bias
        )

        # Create output lists for each particle in event
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }

        # Pass the shared hidden state to every decoder branch
        for decoder in self.branch_decoders:
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarising vectors to their correct structure
            encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
            for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict regressions for any real values associated with the event
        regressions = self.regression_decoder(encoded_vectors)

        # Predict additional classification targets
        classifications = self.classification_decoder(encoded_vectors)

        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications
        )

    def predict(self, sources: Tuple[Source, ...], assignment_fn=default_assignment_fn) -> Predictions:
        with torch.no_grad():
            outputs = self.forward(sources)

            assignments = assignment_fn(outputs)

            detections = np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in outputs.detections
            ])

            regressions = {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in outputs.classifications.items()
            }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in self.forward(sources).assignments
            ]

        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5
