from typing import Dict, Callable
import warnings
from collections import defaultdict

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.dataset.eff_at_purity import candidate_table, summarize, topology_name
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        if self.balance_particles:
            self.particle_index_tensor_np = self.particle_index_tensor.cpu().detach().numpy()
            self.particle_weights_tensor_np = self.particle_weights_tensor.cpu().detach().numpy()
        # self.validation_step_metrics_outputs = []

        self.eff_at_purity_target = options.checkpoint_eff_at_purity
        if self.eff_at_purity_target > 0:
            self.slot_topologies = [topology_name(name) for name in self.event_info.event_particles.names]
            self.reset_eff_at_purity_buffers()

    def reset_eff_at_purity_buffers(self):
        self.eff_scores, self.eff_correct, self.eff_slot = [], [], []
        self.eff_num_truth = np.zeros(len(self.slot_topologies), dtype=np.int64)

    def accumulate_eff_at_purity(self, jet_predictions, particle_scores, stacked_targets, stacked_masks):
        correct, candidate = candidate_table(
            jet_predictions, particle_scores, stacked_targets, stacked_masks,
            self.event_permutation_tensor.cpu().numpy(),
        )
        slot = np.broadcast_to(np.arange(candidate.shape[0])[:, np.newaxis], candidate.shape)

        self.eff_scores.append(particle_scores[candidate])
        self.eff_correct.append(correct[candidate])
        self.eff_slot.append(slot[candidate])
        self.eff_num_truth += stacked_masks.sum(1)

    def log_eff_at_purity(self):
        if self.eff_at_purity_target <= 0:
            return

        scores = np.concatenate(self.eff_scores) if self.eff_scores else np.zeros(0)
        correct = np.concatenate(self.eff_correct) if self.eff_correct else np.zeros(0, dtype=bool)
        slot = np.concatenate(self.eff_slot) if self.eff_slot else np.zeros(0, dtype=np.int64)
        num_truth = self.eff_num_truth
        self.reset_eff_at_purity_buffers()

        if self.trainer.world_size > 1:
            parts = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(parts, (scores, correct, slot, num_truth))
            scores = np.concatenate([part[0] for part in parts])
            correct = np.concatenate([part[1] for part in parts])
            slot = np.concatenate([part[2] for part in parts])
            num_truth = sum(part[3] for part in parts)

        metrics = summarize(scores, correct, slot, num_truth, self.slot_topologies, self.eff_at_purity_target)

        # The monitored key is logged even when NaN: ModelCheckpoint needs it
        # every epoch and treats NaN as the worst value.
        for name, value in metrics.items():
            if name == "validation_eff_at_purity" or not np.isnan(value):
                self.log(name, float(value), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log_eff_at_purity()

    def on_test_epoch_end(self):
        self.log_eff_at_purity()

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        # float, NOT bool: this array stores accuracy * weight. With a bool
        # dtype any non-zero per-target weight silently truncates to 1 while
        # the denominator (tot_target_weights) keeps the real weights, biasing
        # validation_average_jet_accuracy whenever weights != 1.
        weighted_jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.float64)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target, weight) in enumerate(zip(jet_predictions, stacked_targets[permutation], stacked_weights[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)
                weighted_jet_accuracies[i, j] = np.all(prediction == target, axis=1) * weight

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        weighted_jet_accuracies = weighted_jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        tot_target_weights = (stacked_masks * stacked_weights).sum(0)
        jet_accuracies = jet_accuracies.max(0)
        weighted_jet_accuracies = weighted_jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        # CAVEAT (multi-topology event files): this is the mean over events where
        # ALL num_targets particles are present. For event files whose topologies
        # are mutually exclusive (e.g. FR/SRqq/FB tops -- no event can have all
        # six targets), that selection is empty, the mean is NaN, and NaN metrics
        # are skipped by the logger -- so "validation_accuracy" is silently never
        # logged. Do not optimize sweeps or early stopping on it in that setup;
        # use validation_average_jet_accuracy instead.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        has_targets = tot_target_weights > 0
        weighted_avg_jet_accuracy = weighted_jet_accuracies[has_targets] / tot_target_weights[has_targets]
        metrics["validation_average_jet_accuracy"] = np.mean(weighted_avg_jet_accuracy)

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Compute validation loss (same as training loss) for overfitting monitoring
        val_loss = self.compute_validation_loss(batch)
        self.log("loss/val_total_loss", val_loss, sync_dist=True, on_epoch=True)

        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets = batch
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        stacked_weights = np.zeros((num_targets, batch_size), dtype=float)
        for i, (target, mask, weight) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()
            stacked_weights[i] = weight.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights))

        if self.eff_at_purity_target > 0:
            self.accumulate_eff_at_purity(jet_predictions, particle_scores, stacked_targets, stacked_masks)

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value, sync_dist=True, on_epoch=True)

        # self.validation_step_metrics_outputs.append(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

#    def on_validation_epoch_end(self):
#        # merge metrics from different mini batches into one dict
#        metrics_merged = defaultdict(list) 
#        for m in self.validation_step_metrics_outputs:
#            for key, value in m.items():
#                metrics_merged[key].append(value)
#
#        # average each metric over number of mini batches
#        metrics_averaged = {}
#        for key, values in metrics_merged.items():
#            metrics_averaged[f"mean_{key}"] = np.mean(values)
#
#        # log metrics
#        for name, value in metrics_averaged.items():
#            if not np.isnan(value):
#                self.log(name, value, sync_dist=True)
#
#        self.validation_step_metrics_outputs.clear()


