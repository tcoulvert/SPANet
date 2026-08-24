"""
Validation logic for Resonance Regression model.

Implements validation_step with regression metrics and optional classification metrics.
"""
from typing import Dict

import numpy as np
import torch

from spanet.options import Options
from spanet.dataset.types import Batch

from spanet.network.resonance_regression.resonance_regression_network import ResonanceRegressionNetwork


class ResonanceRegressionValidation(ResonanceRegressionNetwork):
    """Validation mixin for resonance regression model.

    Implements validation_step with regression metrics computation.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """Perform a single validation step.

        Parameters
        ----------
        batch : Batch
            Input batch containing sources and regression targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict[str, float]
            Dictionary of validation metrics.
        """
        sources, num_jets, assignment_targets, regression_targets, classification_targets = batch

        # Compute validation loss (same as training) for overfitting monitoring
        with torch.no_grad():
            outputs = self.forward(batch.sources)
            val_loss = self.compute_regression_loss(
                outputs.regressions, batch.regression_targets, log_metrics=False
            )
            if outputs.classifications is not None and self.options.classification_loss_scale > 0:
                val_loss = val_loss + self.compute_classification_loss(
                    outputs.classifications, batch.classification_targets, log_metrics=False
                )
        self.log("loss/val_total_loss", val_loss, sync_dist=True, on_epoch=True)

        # Get predictions
        predictions = self.predict(sources)

        # Convert targets to numpy
        regression_targets_np = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        metrics = {}

        # Compute metrics for each regression target
        for key in predictions:
            if key not in regression_targets_np:
                continue
            pred = predictions[key]
            target = regression_targets_np[key]
            # Handle 2D targets/predictions (like gen_mass_logits) - convert to 1D class indices
            if hasattr(target, "ndim") and target.ndim > 1:
                target = target.argmax(axis=-1)
            if hasattr(pred, "ndim") and pred.ndim > 1:
                pred = pred.argmax(axis=-1)

            # Handle NaN targets
            valid_mask = ~np.isnan(target)
            if valid_mask.sum() == 0:
                continue

            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]

            # Compute errors
            delta = pred_valid - target_valid

            # Mean Absolute Error
            mae = np.abs(delta).mean()
            self.log(f"REGRESSION/{key}_mae", mae, sync_dist=True)
            metrics[f"{key}_mae"] = mae

            # Mean Squared Error
            mse = (delta ** 2).mean()
            self.log(f"REGRESSION/{key}_mse", mse, sync_dist=True)
            metrics[f"{key}_mse"] = mse

            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            self.log(f"REGRESSION/{key}_rmse", rmse, sync_dist=True)
            metrics[f"{key}_rmse"] = rmse

            # Mean Absolute Percent Error (avoid division by zero)
            nonzero_mask = target_valid != 0
            if nonzero_mask.sum() > 0:
                mape = np.abs(delta[nonzero_mask] / target_valid[nonzero_mask]).mean()
                self.log(f"REGRESSION/{key}_mape", mape, sync_dist=True)
                metrics[f"{key}_mape"] = mape

            # Log histograms for detailed analysis
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_histogram'):
                self.logger.experiment.add_histogram(
                    f"REGRESSION/{key}_residuals",
                    delta,
                    self.global_step
                )

                if nonzero_mask.sum() > 0:
                    percent_error = delta[nonzero_mask] / target_valid[nonzero_mask]
                    self.logger.experiment.add_histogram(
                        f"REGRESSION/{key}_percent_error",
                        percent_error,
                        self.global_step
                    )

        # Compute classification metrics if classification targets exist
        classification_targets_np = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        # Check for classification predictions (they have _pred_class suffix)
        for cls_key in classification_targets_np:
            pred_class_key = f"{cls_key}_pred_class"
            logits_key = f"{cls_key}_logits"
            if pred_class_key not in predictions:
                continue

            pred_class = predictions[pred_class_key]
            # Handle case where pred_class is 2D logits instead of 1D class indices
            if pred_class.ndim > 1:
                pred_class = pred_class.argmax(axis=-1)
            else:
                pred_class = pred_class if isinstance(pred_class, np.ndarray) else pred_class.cpu().numpy()
            
            target_class = classification_targets_np[cls_key]  # [B] - integer class labels

            # Handle missing values: -1 indicates no class for this event
            valid_mask = target_class >= 0
            if valid_mask.sum() == 0:
                continue

            pred_class_valid = pred_class[valid_mask]
            target_class_valid = target_class[valid_mask]

            # Accuracy
            accuracy = (pred_class_valid == target_class_valid).mean()
            self.log(f"CLASSIFICATION/{cls_key}_accuracy", accuracy, sync_dist=True)
            metrics[f"{cls_key}_accuracy"] = accuracy

            # Top-3 accuracy: fraction of samples where the true class is among the top-3 predicted
            if logits_key in predictions:
                logits = predictions[logits_key]  # [B, num_classes]
                logits_valid = logits[valid_mask]
                top3_indices = np.argsort(-logits_valid, axis=-1)[:, :3]  # [N, 3]
                target_in_top3 = (target_class_valid[:, None] == top3_indices).any(axis=1)
                top3_accuracy = target_in_top3.mean()
                self.log(f"CLASSIFICATION/{cls_key}_top3_accuracy", top3_accuracy, sync_dist=True)
                metrics[f"{cls_key}_top3_accuracy"] = top3_accuracy

            # Per-class accuracy (for debugging)
            unique_classes = np.unique(target_class_valid)
            for cls_idx in unique_classes:
                cls_mask = target_class_valid == cls_idx
                if cls_mask.sum() > 0:
                    cls_acc = (pred_class_valid[cls_mask] == cls_idx).mean()
                    self.log(f"CLASSIFICATION/{cls_key}_acc_class_{int(cls_idx)}", cls_acc, sync_dist=True)
                    metrics[f"{cls_key}_acc_class_{int(cls_idx)}"] = cls_acc

        # Use MAE as validation metric (lower is better) for regression mode
        # Use accuracy for classification mode
        if metrics:
            if regression_targets_np:
                first_reg_key = list(regression_targets_np.keys())[0]
                if f"{first_reg_key}_mae" in metrics:
                    self.log("validation_mae", metrics[f"{first_reg_key}_mae"], sync_dist=True)
                    self.log("validation_mse", metrics[f"{first_reg_key}_mse"], sync_dist=True)
                    if f"{first_reg_key}_mape" in metrics:
                        self.log("validation_mape", metrics[f"{first_reg_key}_mape"], sync_dist=True)
            
            if classification_targets_np:
                first_cls_key = list(classification_targets_np.keys())[0]
                if f"{first_cls_key}_accuracy" in metrics:
                    self.log("validation_accuracy", metrics[f"{first_cls_key}_accuracy"], sync_dist=True)
                if f"{first_cls_key}_top3_accuracy" in metrics:
                    self.log("validation_top3_accuracy", metrics[f"{first_cls_key}_top3_accuracy"], sync_dist=True)

        return metrics

    def test_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """Perform a single test step (same as validation)."""
        return self.validation_step(batch, batch_idx)
