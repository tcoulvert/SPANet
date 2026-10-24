"""Efficiency at a fixed purity: the validation metric behind the optional
best-efficiency checkpoint (Options.checkpoint_eff_at_purity).

A candidate is a predicted particle whose detection score passes the SPANet
detection cut. Candidates are ranked by that score; the operating point is
the loosest cut whose purity (correct / selected) still reaches the target,
and the metric is the efficiency (correct / truth particles) at that cut.
NaN when no cut reaches the target.
"""
import re
from typing import List, Sequence, Tuple

import numpy as np

DETECTION_CUT = 0.5

_SLOT_SUFFIX = re.compile(r"t\d+$")


def topology_name(particle: str) -> str:
    """FRt1 -> FR, SRqqt2 -> SRqq; names without a slot suffix are unchanged."""
    return _SLOT_SUFFIX.sub("", particle) or particle


def efficiency_at_purity(scores: np.ndarray, correct: np.ndarray, num_truth: int, purity: float) -> float:
    if num_truth == 0 or len(scores) == 0:
        return np.nan

    order = np.argsort(-scores, kind="stable")
    found = np.cumsum(correct[order])
    selected = np.arange(1, len(order) + 1)

    reached = found / selected >= purity
    if not reached.any():
        return np.nan

    return found[reached].max() / num_truth


def candidate_table(
    jet_predictions: Sequence[np.ndarray],
    particle_scores: np.ndarray,
    stacked_targets: Sequence[np.ndarray],
    stacked_masks: np.ndarray,
    permutation_group: np.ndarray,
    detection_cut: float = DETECTION_CUT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-target (correct, candidate) boolean tables of shape (num_targets, batch_size).

    As in compute_metrics, each event is scored under the event permutation that
    maximises its number of correct targets. A target is correct when every
    predicted daughter matches the permuted truth and that truth is present.
    """
    num_targets, batch_size = stacked_masks.shape
    correct = np.zeros((num_targets, batch_size), dtype=bool)
    best_count = np.full(batch_size, -1)

    for permutation in permutation_group:
        hit = np.stack([
            np.all(jet_predictions[i] == stacked_targets[p], axis=1) & stacked_masks[p]
            for i, p in enumerate(permutation)
        ])
        count = hit.sum(0)
        better = count > best_count
        correct[:, better] = hit[:, better]
        best_count[better] = count[better]

    candidate = particle_scores >= detection_cut
    return correct, candidate


def summarize(
    scores: np.ndarray,
    correct: np.ndarray,
    slot: np.ndarray,
    num_truth: np.ndarray,
    slot_topologies: Sequence[str],
    purity: float,
) -> dict:
    """Metrics over one validation epoch's candidates.

    scores, correct, slot: one entry per candidate (slot = target index).
    num_truth: truth particles per target slot.
    """
    slot_topologies = np.asarray(slot_topologies)
    metrics = {
        "validation_purity": correct.mean() if len(correct) else np.nan,
        "validation_efficiency": correct.sum() / num_truth.sum() if num_truth.sum() else np.nan,
        "validation_eff_at_purity": efficiency_at_purity(scores, correct, num_truth.sum(), purity),
    }

    per_topology = []
    for topology in dict.fromkeys(slot_topologies):
        in_topology = slot_topologies[slot] == topology
        value = efficiency_at_purity(
            scores[in_topology], correct[in_topology],
            num_truth[slot_topologies == topology].sum(), purity,
        )
        metrics[f"validation_eff_at_purity/{topology}"] = value
        per_topology.append(value)

    with np.errstate(all="ignore"):
        metrics["validation_balanced_eff_at_purity"] = (
            np.nanmean(per_topology) if not np.all(np.isnan(per_topology)) else np.nan
        )

    return metrics
