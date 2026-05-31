"""Minimal regression checks for molecule-local loss aggregation."""

from __future__ import annotations

import math

import jax.numpy as jnp

from utils.losses import (
    crossSetMarginLoss,
    moleculeLocalWeightedMae,
    pairwiseRankLoss,
    topMassObjectiveLoss,
)


def assertClose(actual: float, expected: float, name: str) -> None:
    """Assert two scalar values are numerically close."""

    actual_value = float(actual)
    if not math.isclose(actual_value, expected, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise AssertionError(f"{name}: expected {expected}, got {actual_value}")


def testMoleculeLocalWeightedMaeAveragesPerMolecule() -> None:
    """Each molecule should contribute equally after local normalization."""

    prediction = jnp.asarray([1.0, 0.0, 0.0, 2.0], dtype=jnp.float32)
    target = jnp.zeros_like(prediction)
    counts = jnp.asarray([3, 1], dtype=jnp.int32)
    loss, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
    )
    assertClose(loss, 7.0 / 6.0, "moleculeLocalWeightedMae")


def testMoleculeLocalWeightedMaeSkipsMaskedMolecules() -> None:
    """Fully masked molecules should not dilute the batch average."""

    prediction = jnp.asarray([1.0, 9.0, 3.0, 4.0], dtype=jnp.float32)
    target = jnp.zeros_like(prediction)
    counts = jnp.asarray([2, 2], dtype=jnp.int32)
    sample_mask = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    loss, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        sample_mask=sample_mask,
    )
    assertClose(loss, 1.0, "moleculeLocalWeightedMae masked")


def testPairwiseRankLossAveragesPerMolecule() -> None:
    """Pairwise ranking should average molecule losses instead of raw pair totals."""

    prediction = jnp.asarray([3.0, 2.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    target = jnp.asarray([3.0, 2.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    counts = jnp.asarray([3, 2], dtype=jnp.int32)
    loss = pairwiseRankLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        margin=1.0,
        min_delta=0.0,
        pair_power=1.0,
    )
    assertClose(loss, 1.0, "pairwiseRankLoss")


def testCrossSetMarginLossAveragesPerMolecule() -> None:
    """Cross-set ranking should also average valid molecules equally."""

    prediction = jnp.asarray([3.0, 2.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    target = jnp.asarray([3.0, 2.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    counts = jnp.asarray([3, 2], dtype=jnp.int32)
    focus_mask = jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    tail_mask = 1.0 - focus_mask
    loss = crossSetMarginLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        left_mask=focus_mask,
        right_mask=tail_mask,
        margin=1.0,
        min_delta=0.0,
        pair_power=1.0,
    )
    assertClose(loss, 1.0, "crossSetMarginLoss")


def testTopMassObjectiveUsesMoleculeLocalComponents() -> None:
    """Top-mass objective should expose the molecule-local helper outputs unchanged."""

    prediction = jnp.asarray([3.0, 2.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    target = jnp.asarray([3.0, 2.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    focus_mask = jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    counts = jnp.asarray([3, 2], dtype=jnp.int32)

    total_loss, terms = topMassObjectiveLoss(
        prediction=prediction,
        target=target,
        top_mass_focus_mask=focus_mask,
        num_structures_per_molecule=counts,
        target_weight_power=0.0,
        target_weight_offset=0.0,
        top_mass_regression_weight=1.0,
        top_mass_ranking_weight=1.0,
        tail_suppression_weight=0.5,
        rank_loss_margin=1.0,
        rank_loss_min_delta=0.0,
        rank_pair_power=1.0,
    )

    expected_focus_regression, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        power=0.0,
        offset=0.0,
        sample_mask=focus_mask,
    )
    expected_tail_regression, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        power=0.0,
        offset=0.0,
        sample_mask=1.0 - focus_mask,
    )
    expected_focus_rank = pairwiseRankLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        margin=1.0,
        min_delta=0.0,
        pair_power=1.0,
        sample_mask=focus_mask,
    )
    expected_focus_tail_rank = crossSetMarginLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=counts,
        left_mask=focus_mask,
        right_mask=1.0 - focus_mask,
        margin=1.0,
        min_delta=0.0,
        pair_power=1.0,
    )
    expected_total = (
        expected_focus_regression
        + 0.5 * expected_tail_regression
        + expected_focus_rank
        + expected_focus_tail_rank
    )

    assertClose(terms["focus_regression_loss"], expected_focus_regression, "focus_regression_loss")
    assertClose(terms["tail_regression_loss"], expected_tail_regression, "tail_regression_loss")
    assertClose(terms["focus_rank_loss"], expected_focus_rank, "focus_rank_loss")
    assertClose(terms["focus_tail_rank_loss"], expected_focus_tail_rank, "focus_tail_rank_loss")
    assertClose(total_loss, expected_total, "topMassObjectiveLoss")


def main() -> None:
    """Run all molecule-local loss checks."""

    testMoleculeLocalWeightedMaeAveragesPerMolecule()
    testMoleculeLocalWeightedMaeSkipsMaskedMolecules()
    testPairwiseRankLossAveragesPerMolecule()
    testCrossSetMarginLossAveragesPerMolecule()
    testTopMassObjectiveUsesMoleculeLocalComponents()
    print("test_losses.py: all checks passed")


if __name__ == "__main__":
    main()
