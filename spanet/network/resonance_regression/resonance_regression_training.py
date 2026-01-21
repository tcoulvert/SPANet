"""
Training logic for Resonance Regression model.

Implements training_step with regression loss and optional classification loss.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from spanet.options import Options
from spanet.dataset.types import Batch
from spanet.dataset.regressions import regression_loss

from spanet.network.resonance_regression.resonance_regression_network import ResonanceRegressionNetwork


class ResonanceRegressionTraining(ResonanceRegressionNetwork):
    """Training mixin for resonance regression model.

    Implements training_step with regression loss computation.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

    def compute_regression_loss(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Tensor:
        """Compute regression loss for all targets.

        Parameters
        ----------
        predictions : Dict[str, Tensor]
            Model predictions for each regression target.
        targets : Dict[str, Tensor]
            Ground truth values for each regression target.

        Returns
        -------
        Tensor
            Total regression loss.
        """
        loss_terms = []

        for key in targets:
            # Get regression type (gaussian, laplacian, log_gaussian)
            regression_type = self.training_dataset.regression_types[key]

            prediction = predictions[key]
            target = targets[key]
            # Handle 2D targets/predictions (like gen_mass_logits) - convert to 1D class indices
            if target.ndim > 1:
                target = target.argmax(dim=-1)
            if prediction.ndim > 1:
                prediction = prediction.argmax(dim=-1)

            # Get normalization statistics from decoder
            mean = self.regression_decoder.networks[key].mean
            std = self.regression_decoder.networks[key].std

            # Handle NaN targets (missing values)
            valid_mask = ~torch.isnan(target)

            if valid_mask.sum() == 0:
                continue

            # Compute loss for valid entries
            loss = regression_loss(regression_type)(
                prediction[valid_mask],
                target[valid_mask],
                mean,
                std
            )
            loss = torch.mean(loss)

            # Log individual regression losses
            self.log(f"loss/regression/{key}", loss, sync_dist=True)

            loss_terms.append(self.options.regression_loss_scale * loss)

        if not loss_terms:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(loss_terms).sum()

    def compute_classification_loss(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Tensor:
        """Compute classification loss for classification targets.

        Parameters
        ----------
        predictions : Dict[str, Tensor]
            Model classification predictions (logits) for each target. Shape: [B, num_classes]
        targets : Dict[str, Tensor]
            Ground truth classification labels (integer class indices). Shape: [B]

        Returns
        -------
        Tensor
            Total classification loss.
        """
        loss_terms = []

        for key in predictions:
            if key not in targets:
                continue

            prediction = predictions[key]  # [B, num_classes]
            target = targets[key]  # [B] - integer class labels

            # Handle missing values: -1 indicates no class for this event
            valid_mask = target >= 0
            if valid_mask.sum() == 0:
                continue

            # Compute cross-entropy loss with ignore_index=-1 for missing values
            loss = F.cross_entropy(
                prediction[valid_mask],
                target[valid_mask],
                ignore_index=-1
            )

            # Log individual classification losses
            self.log(f"loss/classification/{key}", loss, sync_dist=True)

            loss_terms.append(self.options.classification_loss_scale * loss)

        if not loss_terms:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(loss_terms).sum()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : Batch
            Input batch containing sources and regression targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tensor
            Total loss for this batch.
        """
        # Forward pass
        outputs = self.forward(batch.sources)

        # Compute regression loss
        total_loss = self.compute_regression_loss(
            outputs.regressions,
            batch.regression_targets
        )

        # Compute classification loss (if enabled)
        if outputs.classifications is not None and self.options.classification_loss_scale > 0:
            classification_loss = self.compute_classification_loss(
                outputs.classifications,
                batch.classification_targets
            )
            total_loss = total_loss + classification_loss

        # Log total loss
        self.log("loss/total_loss", total_loss, sync_dist=True, prog_bar=True)
        # Also log a slash-free alias for convenience in filenames/monitors.
        self.log("loss_total_loss", total_loss, sync_dist=True, prog_bar=False)

        return total_loss
