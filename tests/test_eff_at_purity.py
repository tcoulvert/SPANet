"""Efficiency-at-fixed-purity metric (spanet/dataset/eff_at_purity.py)."""
import numpy as np
import pytest

from spanet.dataset.eff_at_purity import candidate_table, efficiency_at_purity, summarize, topology_name


def test_topology_name_strips_slot_suffix():
    assert topology_name("FRt1") == "FR"
    assert topology_name("SRqqt2") == "SRqq"
    assert topology_name("FBt1") == "FB"
    assert topology_name("bh1") == "bh1"
    assert topology_name("t1") == "t1"


def test_overall_purity_above_target_gives_plain_efficiency():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    assert efficiency_at_purity(scores, correct, num_truth=4, purity=0.5) == pytest.approx(0.5)


def test_tightening_the_cut_recovers_the_purity_target():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, False, False])
    # cuts keeping 1..4 candidates have purity 1, .5, .33, .25: deepest at .5 keeps 2
    assert efficiency_at_purity(scores, correct, num_truth=4, purity=0.5) == pytest.approx(0.25)
    assert efficiency_at_purity(scores, correct, num_truth=4, purity=0.6) == pytest.approx(0.25)


def test_ranking_is_by_score_not_input_order():
    scores = np.array([0.6, 0.9])
    correct = np.array([False, True])
    assert efficiency_at_purity(scores, correct, num_truth=2, purity=1.0) == pytest.approx(0.5)


def test_unreachable_purity_and_empty_inputs_are_nan():
    scores = np.array([0.9, 0.8])
    assert np.isnan(efficiency_at_purity(scores, np.array([False, False]), num_truth=2, purity=0.5))
    assert np.isnan(efficiency_at_purity(np.zeros(0), np.zeros(0, dtype=bool), num_truth=2, purity=0.5))
    assert np.isnan(efficiency_at_purity(scores, np.array([True, True]), num_truth=0, purity=0.5))


def _two_slot_batch():
    """Two FR slots, event permutation group {identity, swap}, two events.

    Event 0: truth only in slot 0, prediction slot 0 matches it.
    Event 1: truth only in slot 1, prediction slot 0 matches it (needs the swap).
    """
    jet_predictions = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[3, 4, 5], [0, 1, 2]]),
    ]
    stacked_targets = np.empty(2, dtype=object)
    stacked_targets[0] = np.array([[0, 1, 2], [-1, -1, -1]])
    stacked_targets[1] = np.array([[-1, -1, -1], [3, 4, 5]])
    stacked_masks = np.array([[True, False], [False, True]])
    permutation_group = np.array([[0, 1], [1, 0]])
    return jet_predictions, stacked_targets, stacked_masks, permutation_group


def test_candidate_table_picks_the_best_event_permutation():
    jet_predictions, stacked_targets, stacked_masks, permutation_group = _two_slot_batch()
    particle_scores = np.array([[0.9, 0.7], [0.4, 0.8]])

    correct, candidate = candidate_table(
        jet_predictions, particle_scores, stacked_targets, stacked_masks, permutation_group
    )

    assert correct.tolist() == [[True, True], [False, False]]
    assert candidate.tolist() == [[True, True], [False, True]]


def test_summarize_reports_overall_per_topology_and_balanced():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, True, False])
    slot = np.array([0, 0, 2, 3])
    num_truth = np.array([2, 0, 1, 1])
    topologies = ["FR", "FR", "SRqq", "SRqq"]

    metrics = summarize(scores, correct, slot, num_truth, topologies, purity=0.5)

    assert metrics["validation_purity"] == pytest.approx(0.5)
    assert metrics["validation_efficiency"] == pytest.approx(0.5)
    assert metrics["validation_eff_at_purity"] == pytest.approx(0.5)
    assert metrics["validation_eff_at_purity/FR"] == pytest.approx(0.5)
    assert metrics["validation_eff_at_purity/SRqq"] == pytest.approx(0.5)
    assert metrics["validation_balanced_eff_at_purity"] == pytest.approx(0.5)


def test_summarize_balanced_ignores_unreachable_topologies():
    scores = np.array([0.9, 0.8])
    correct = np.array([True, False])
    slot = np.array([0, 1])
    num_truth = np.array([1, 1])

    metrics = summarize(scores, correct, slot, num_truth, ["FR", "FB"], purity=0.5)

    assert metrics["validation_eff_at_purity/FR"] == pytest.approx(1.0)
    assert np.isnan(metrics["validation_eff_at_purity/FB"])
    assert metrics["validation_balanced_eff_at_purity"] == pytest.approx(1.0)
