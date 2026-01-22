"""Complete jet reconstruction model with pairwise interactions.

This module creates the full model by combining:
- JetReconstructionNetworkWithPairwise (forward pass with pairwise attention)
- Training methods from JetReconstructionTraining
- Validation methods from JetReconstructionValidation

The implementation reuses the original training/validation code since they
only depend on the forward() method, which is provided by our pairwise network.
"""
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.types import Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.utilities.divergence_losses import (
    assignment_cross_entropy_loss,
    jensen_shannon_divergence
)

from spanet.network.jet_reconstruction_pairwise.jet_reconstruction_pairwise import (
    JetReconstructionNetworkWithPairwise
)


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list
    return output


class JetReconstructionTrainingWithPairwise(JetReconstructionNetworkWithPairwise):
    """Training mixin for pairwise jet reconstruction network.

    Provides training_step and loss computation methods.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

    def particle_symmetric_loss(
        self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor, weight: Tensor
    ) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(
            assignment, target, mask, weight, self.options.focal_gamma
        )
        detection_loss = F.binary_cross_entropy_with_logits(
            detection, mask.float(), weight=weight, reduction='none'
        )

        return torch.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(
        self, assignments: List[Tensor], detections: List[Tensor], targets
    ):
        symmetric_losses = []

        for permutation in self.event_permutation_tensor.cpu().numpy():
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask, weight)
                for assignment, detection, (target, mask, weight)
                in zip(assignments, detections, targets[permutation])
            )

            symmetric_losses.append(torch.stack(current_permutation_loss))

        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        total_symmetric_loss = symmetric_losses.sum((1, 2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        assignments = [
            prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
            for prediction, decoder in zip(assignments, self.branch_decoders)
        ]

        targets = numpy_tensor_array(targets)
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            div = jensen_shannon_divergence(predictions[i], predictions[j])
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)
            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)

    def add_kl_loss(
        self,
        total_loss: List[Tensor],
        assignments: List[Tensor],
        masks: Tensor,
        weights: Tensor
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
        self,
        total_loss: List[Tensor],
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> List[Tensor]:
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                self.log(f"loss/regression/{key}", current_loss, sync_dist=True)

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
        self,
        total_loss: List[Tensor],
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight
            )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                self.log(f"loss/classification/{key}", current_loss, sync_dist=True)

        return total_loss + classification_terms

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        outputs = self.forward(batch.sources)

        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets,
        )

        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        weights = torch.ones_like(symmetric_losses)

        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)

        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets)

        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets)

        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

        return total_loss.mean()


class JetReconstructionValidationWithPairwise(JetReconstructionNetworkWithPairwise):
    """Validation mixin for pairwise jet reconstruction network.

    Provides validation_step and validation metrics.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

    def validation_step(self, batch: Batch, batch_nb: int) -> None:
        # Run the prediction step to get both assignments and detections
        assignments, detections, regressions, classifications = self.predict(batch.sources)

        # Force all detections to be true if we don't train on detection
        if self.options.detection_loss_scale == 0:
            detections += 1

        # Compute validation metrics
        self._log_assignment_accuracy(assignments, batch)
        self._log_detection_accuracy(detections, batch)
        self._log_regression_accuracy(regressions, batch)
        self._log_classification_accuracy(classifications, batch)

    def _log_assignment_accuracy(self, assignments: np.ndarray, batch: Batch) -> None:
        """Log assignment accuracy metrics."""
        event_info = self.training_dataset.event_info
        total_weighted_correct = 0.0
        total_weight = 0.0

        for particle_index, event_particle in enumerate(event_info.event_particles):
            product_particles = event_info.product_particles[event_particle]
            target = batch.assignment_targets[particle_index]
            mask = target.mask.cpu().numpy()
            weights = target.weight.cpu().numpy()
            particle_correct = np.ones_like(mask, dtype=bool)

            for product_index, product_particle in enumerate(product_particles):
                # Get predictions and targets for this particle
                predicted = assignments[particle_index][:, product_index]
                actual = target.indices[:, product_index].cpu().numpy()
                current_mask = mask

                # Compute accuracy only on valid events
                correct = (predicted == actual) & current_mask
                total = current_mask.sum()

                if total > 0:
                    accuracy = correct.sum() / total
                    self.log(
                        f"validation/{event_particle}/{product_particle}/accuracy",
                        accuracy,
                        sync_dist=True
                    )

                # Track per-event correctness across all daughters of this particle
                particle_correct &= np.logical_or(~current_mask, predicted == actual)

            # Accumulate weighted event-level accuracy for the monitored metric
            effective_weights = weights * mask
            total_weighted_correct += (particle_correct.astype(float) * effective_weights).sum()
            total_weight += effective_weights.sum()

        if total_weight > 0:
            average_accuracy = total_weighted_correct / total_weight
            self.log("validation_average_jet_accuracy", average_accuracy, sync_dist=True, on_epoch=True)

    def _log_detection_accuracy(self, detections: np.ndarray, batch: Batch) -> None:
        """Log detection accuracy metrics."""
        event_info = self.training_dataset.event_info

        for particle_index, event_particle in enumerate(event_info.event_particles):
            target = batch.assignment_targets[particle_index]

            predicted = detections[particle_index] >= 0.5
            actual = target.mask.cpu().numpy()

            accuracy = (predicted == actual).mean()
            self.log(
                f"validation/{event_particle}/detection_accuracy",
                accuracy,
                sync_dist=True
            )

    def _log_regression_accuracy(self, regressions: Dict[str, np.ndarray], batch: Batch) -> None:
        """Log regression MAE metrics."""
        for key, prediction in regressions.items():
            if key not in batch.regression_targets:
                continue

            target = batch.regression_targets[key].cpu().numpy()
            mask = ~np.isnan(target)

            if mask.sum() > 0:
                mae = np.abs(prediction[mask] - target[mask]).mean()
                self.log(f"validation/{key}/mae", mae, sync_dist=True)

    def _log_classification_accuracy(self, classifications: Dict[str, np.ndarray], batch: Batch) -> None:
        """Log classification accuracy metrics."""
        for key, prediction in classifications.items():
            if key not in batch.classification_targets:
                continue

            target = batch.classification_targets[key].cpu().numpy()
            mask = target >= 0

            if mask.sum() > 0:
                accuracy = (prediction[mask] == target[mask]).mean()
                self.log(f"validation/{key}/accuracy", accuracy, sync_dist=True)


class JetReconstructionModelWithPairwise(
    JetReconstructionValidationWithPairwise,
    JetReconstructionTrainingWithPairwise
):
    """Complete jet reconstruction model with pairwise attention.

    This is the main model class that combines:
    - Pairwise attention in the central encoder
    - Training logic (training_step, loss computation)
    - Validation logic (validation_step, metrics)

    Use this as a drop-in replacement for JetReconstructionModel when
    pairwise interactions are desired.

    Parameters
    ----------
    options : Options
        SPANet configuration options. Set use_pairwise_interactions=True.
    torch_script : bool
        Whether to compile with TorchScript
    """
    pass
