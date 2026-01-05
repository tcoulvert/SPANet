"""
Network architecture for Resonance Regression model.

Architecture: Jets -> Embedding -> Transformer Encoder -> Regression/Classification Decoder
"""
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor, nn

from spanet.options import Options
from spanet.dataset.types import Tuple, Source

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.branch_linear import BranchLinear

from spanet.network.resonance_regression.resonance_regression_base import ResonanceRegressionBase


class RegressionOutputs(NamedTuple):
    """Output structure for resonance regression model."""
    event_vector: Tensor                          # [B, D] - encoded event representation
    regressions: Dict[str, Tensor]                # {name: [B]} - regression predictions
    classifications: Optional[Dict[str, Tensor]] = None  # {name: [B, num_classes]} - classification logits


class ResonanceRegressionNetwork(ResonanceRegressionBase):
    """Resonance regression network: embedding -> transformer -> regressor.

    This model takes jet inputs, embeds them, processes through a transformer encoder,
    and outputs regression predictions. No jet assignment is performed.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim
        self.use_mass_classification = options.use_mass_classification
        self.mass_classes = options.mass_classes

        # Embedding layer: converts jet features to hidden dimension
        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))

        # Transformer encoder: processes embedded jets
        self.encoder = compile_module(JetEncoder(options))

        # Regression decoder: predicts regression targets from event vector
        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        # Classification decoder: predicts mass class from event vector (if enabled)
        self.classification_decoder = None
        if self.use_mass_classification and self.mass_classes:
            num_classes = len(self.mass_classes)
            # Create a classification head for each regression target
            from collections import OrderedDict
            networks = OrderedDict()
            for name in self.training_dataset.regressions.keys():
                networks[name] = BranchLinear(
                    options,
                    options.num_classification_layers,
                    num_classes
                )
            self.classification_decoder = nn.ModuleDict(networks)

    @property
    def event_info(self):
        return self.training_dataset.event_info

    def forward(self, sources: Tuple[Source, ...]) -> RegressionOutputs:
        """Forward pass through the network.

        Parameters
        ----------
        sources : Tuple[Source, ...]
            Input sources, each containing (data, mask) tensors.

        Returns
        -------
        RegressionOutputs
            Named tuple containing event_vector, regression predictions, and optional classifications.
        """
        # Embed all input sources into the same latent space
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features using transformer encoder
        # hidden: [T, B, D] - per-jet representations
        # event_vector: [B, D] - global event representation
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Build vector dictionary for regression decoder
        # The regression decoder uses keys like "EVENT/gen_mass" and looks up "EVENT" vector
        encoded_vectors = {"EVENT": event_vector}

        # Predict regression targets
        regressions = self.regression_decoder(encoded_vectors)

        # Predict classification targets (if enabled)
        classifications = None
        if self.classification_decoder is not None:
            classifications = {
                key: network(event_vector)
                for key, network in self.classification_decoder.items()
            }

        return RegressionOutputs(
            event_vector=event_vector,
            regressions=regressions,
            classifications=classifications
        )

    def predict(self, sources: Tuple[Source, ...]) -> Dict[str, np.ndarray]:
        """Run inference and return numpy arrays.

        Parameters
        ----------
        sources : Tuple[Source, ...]
            Input sources.

        Returns
        -------
        Dict[str, np.ndarray]
            Regression predictions as numpy arrays. If classification mode is enabled,
            also includes classification logits under keys with '_logits' suffix.
        """
        with torch.no_grad():
            outputs = self.forward(sources)
            results = {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }
            if outputs.classifications is not None:
                for key, value in outputs.classifications.items():
                    results[f"{key}_logits"] = value.cpu().numpy()
                    results[f"{key}_pred_class"] = value.argmax(dim=-1).cpu().numpy()
            return results
