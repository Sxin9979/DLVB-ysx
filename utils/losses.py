"""Loss utilities for end-to-end E3VB training."""

from __future__ import annotations

import jax.numpy as jnp


def moleculeIndexFromCounts(
    counts: jnp.ndarray,
    total_length: int,
) -> jnp.ndarray:
    """
    Expand per-molecule structure counts into one flat structure-to-molecule map.
    """

    counts = jnp.asarray(counts, dtype=jnp.int32)
    return jnp.repeat(
        jnp.arange(counts.shape[0], dtype=jnp.int32),
        counts,
        total_repeat_length=total_length,
    )


def segmentSumByMolecule(
    values: jnp.ndarray,
    molecule_index: jnp.ndarray,
    num_molecules: int,
) -> jnp.ndarray:
    """
    Sum one flat per-structure tensor into per-molecule totals.
    """

    return jnp.bincount(
        jnp.asarray(molecule_index, dtype=jnp.int32),
        weights=jnp.asarray(values, dtype=jnp.float32),
        length=int(num_molecules),
    )


def normalizedTargetWeights(
    target: jnp.ndarray,
    power: float,
    offset: float,
    sample_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Build non-negative per-sample regression weights from normalized targets.

    The weights are normalized to mean 1.0 so changing the weighting scheme does
    not unexpectedly rescale the total regression loss magnitude.
    """

    target = jnp.asarray(target, dtype=jnp.float32)
    power = jnp.asarray(power, dtype=jnp.float32)
    offset = jnp.asarray(offset, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)
    raw_weight = jnp.power(jnp.maximum(target, 0.0) + offset, power)
    masked_weight = raw_weight * sample_mask
    mean_weight = jnp.maximum(
        jnp.sum(masked_weight) / jnp.maximum(jnp.sum(sample_mask), 1.0e-12),
        1.0e-12,
    )
    return masked_weight / mean_weight


def weightedMae(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    power: float = 0.0,
    offset: float = 0.0,
    sample_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute weighted MAE and return both loss and normalized per-sample weights.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)
    weights = normalizedTargetWeights(
        target=target,
        power=power,
        offset=offset,
        sample_mask=sample_mask,
    )
    absolute_error = jnp.abs(prediction - target)
    total_weight = jnp.maximum(jnp.sum(weights), 1.0e-12)
    return jnp.sum(weights * absolute_error) / total_weight, weights


def moleculeLocalWeightedMae(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    num_structures_per_molecule: jnp.ndarray,
    power: float = 0.0,
    offset: float = 0.0,
    sample_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute weighted MAE per molecule, then average over valid molecules.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    counts = jnp.asarray(num_structures_per_molecule, dtype=jnp.int32)
    power = jnp.asarray(power, dtype=jnp.float32)
    offset = jnp.asarray(offset, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)

    num_molecules = counts.shape[0]
    molecule_index = moleculeIndexFromCounts(counts=counts, total_length=prediction.shape[0])
    raw_weight = jnp.power(jnp.maximum(target, 0.0) + offset, power) * sample_mask
    masked_count_per_molecule = segmentSumByMolecule(
        values=sample_mask,
        molecule_index=molecule_index,
        num_molecules=num_molecules,
    )
    raw_weight_sum_per_molecule = segmentSumByMolecule(
        values=raw_weight,
        molecule_index=molecule_index,
        num_molecules=num_molecules,
    )
    mean_weight_per_molecule = raw_weight_sum_per_molecule / jnp.maximum(
        masked_count_per_molecule,
        1.0e-12,
    )
    normalized_weight = raw_weight / jnp.maximum(mean_weight_per_molecule[molecule_index], 1.0e-12)
    absolute_error = jnp.abs(prediction - target)
    weighted_error_per_molecule = segmentSumByMolecule(
        values=normalized_weight * absolute_error,
        molecule_index=molecule_index,
        num_molecules=num_molecules,
    )
    total_weight_per_molecule = segmentSumByMolecule(
        values=normalized_weight,
        molecule_index=molecule_index,
        num_molecules=num_molecules,
    )
    per_molecule_loss = weighted_error_per_molecule / jnp.maximum(total_weight_per_molecule, 1.0e-12)
    valid_molecules = total_weight_per_molecule > 1.0e-12
    return (
        jnp.sum(jnp.where(valid_molecules, per_molecule_loss, 0.0))
        / jnp.maximum(jnp.sum(valid_molecules.astype(jnp.float32)), 1.0),
        normalized_weight,
    )


def pairwiseRankLoss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    num_structures_per_molecule: jnp.ndarray,
    margin: float = 0.0,
    min_delta: float = 0.0,
    pair_power: float = 1.0,
    sample_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Compute molecule-local pairwise hinge ranking loss.

    For each molecule, only ordered pairs with ``target_i > target_j + min_delta``
    contribute. Pair importance is proportional to ``(target_i - target_j)^pair_power``.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    counts = jnp.asarray(num_structures_per_molecule, dtype=jnp.int32)
    margin = jnp.asarray(margin, dtype=jnp.float32)
    min_delta = jnp.asarray(min_delta, dtype=jnp.float32)
    pair_power = jnp.asarray(pair_power, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)

    num_molecules = counts.shape[0]
    molecule_index = moleculeIndexFromCounts(counts=counts, total_length=prediction.shape[0])
    pred_i = prediction[:, None]
    pred_j = prediction[None, :]
    target_i = target[:, None]
    target_j = target[None, :]

    same_molecule = molecule_index[:, None] == molecule_index[None, :]
    target_delta = target_i - target_j
    valid_sample = sample_mask > 0.5
    valid_pair = (
        same_molecule
        & (target_delta > min_delta)
        & valid_sample[:, None]
        & valid_sample[None, :]
    )

    pair_weight = jnp.power(jnp.maximum(target_delta, 0.0), pair_power)
    hinge = jnp.maximum(margin - (pred_i - pred_j), 0.0)
    weighted_hinge = pair_weight * hinge
    pair_molecule_index = jnp.broadcast_to(molecule_index[:, None], valid_pair.shape)
    valid_pair_weight = jnp.where(valid_pair, pair_weight, 0.0)
    valid_pair_loss = jnp.where(valid_pair, weighted_hinge, 0.0)
    total_weight_per_molecule = jnp.bincount(
        pair_molecule_index.reshape(-1),
        weights=valid_pair_weight.reshape(-1),
        length=int(num_molecules),
    )
    total_loss_per_molecule = jnp.bincount(
        pair_molecule_index.reshape(-1),
        weights=valid_pair_loss.reshape(-1),
        length=int(num_molecules),
    )
    per_molecule_loss = total_loss_per_molecule / jnp.maximum(total_weight_per_molecule, 1.0e-12)
    valid_molecules = total_weight_per_molecule > 1.0e-12
    return (
        jnp.sum(jnp.where(valid_molecules, per_molecule_loss, 0.0))
        / jnp.maximum(jnp.sum(valid_molecules.astype(jnp.float32)), 1.0)
    )


def crossSetMarginLoss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    num_structures_per_molecule: jnp.ndarray,
    left_mask: jnp.ndarray,
    right_mask: jnp.ndarray,
    margin: float = 0.0,
    min_delta: float = 0.0,
    pair_power: float = 1.0,
    sample_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Compute a molecule-local hinge loss between two structure subsets.

    The loss only considers same-molecule pairs ``(i, j)`` where
    ``left_mask[i]`` and ``right_mask[j]`` are true and the true target gap is
    larger than ``min_delta``.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    counts = jnp.asarray(num_structures_per_molecule, dtype=jnp.int32)
    left_mask = jnp.asarray(left_mask, dtype=jnp.float32) > 0.5
    right_mask = jnp.asarray(right_mask, dtype=jnp.float32) > 0.5
    margin = jnp.asarray(margin, dtype=jnp.float32)
    min_delta = jnp.asarray(min_delta, dtype=jnp.float32)
    pair_power = jnp.asarray(pair_power, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)

    num_molecules = counts.shape[0]
    molecule_index = moleculeIndexFromCounts(counts=counts, total_length=prediction.shape[0])
    pred_i = prediction[:, None]
    pred_j = prediction[None, :]
    target_i = target[:, None]
    target_j = target[None, :]

    same_molecule = molecule_index[:, None] == molecule_index[None, :]
    valid_sample = sample_mask > 0.5
    target_delta = target_i - target_j
    valid_pair = (
        same_molecule
        & left_mask[:, None]
        & right_mask[None, :]
        & valid_sample[:, None]
        & valid_sample[None, :]
        & (target_delta > min_delta)
    )
    pair_weight = jnp.power(jnp.maximum(target_delta, 0.0), pair_power)
    hinge = jnp.maximum(margin - (pred_i - pred_j), 0.0)
    weighted_hinge = pair_weight * hinge
    pair_molecule_index = jnp.broadcast_to(molecule_index[:, None], valid_pair.shape)
    valid_pair_weight = jnp.where(valid_pair, pair_weight, 0.0)
    valid_pair_loss = jnp.where(valid_pair, weighted_hinge, 0.0)
    total_weight_per_molecule = jnp.bincount(
        pair_molecule_index.reshape(-1),
        weights=valid_pair_weight.reshape(-1),
        length=int(num_molecules),
    )
    total_loss_per_molecule = jnp.bincount(
        pair_molecule_index.reshape(-1),
        weights=valid_pair_loss.reshape(-1),
        length=int(num_molecules),
    )
    per_molecule_loss = total_loss_per_molecule / jnp.maximum(total_weight_per_molecule, 1.0e-12)
    valid_molecules = total_weight_per_molecule > 1.0e-12
    return (
        jnp.sum(jnp.where(valid_molecules, per_molecule_loss, 0.0))
        / jnp.maximum(jnp.sum(valid_molecules.astype(jnp.float32)), 1.0)
    )


def topMassObjectiveLoss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    top_mass_focus_mask: jnp.ndarray,
    num_structures_per_molecule: jnp.ndarray,
    target_weight_power: float,
    target_weight_offset: float,
    top_mass_regression_weight: float,
    top_mass_ranking_weight: float,
    tail_suppression_weight: float,
    rank_loss_margin: float,
    rank_loss_min_delta: float,
    rank_pair_power: float,
    sample_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute the top-mass objective for high near-degeneracy datasets.

    The focus-set structures receive the main regression objective. Ranking is
    enforced both within the focus set and across the focus-vs-tail boundary.
    Tail structures keep only a low-priority suppression loss so the model does
    not collapse important structures beneath the long near-zero tail.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    top_mass_focus_mask = jnp.asarray(top_mass_focus_mask, dtype=jnp.float32)
    if sample_mask is None:
        sample_mask = jnp.ones_like(target, dtype=jnp.float32)
    else:
        sample_mask = jnp.asarray(sample_mask, dtype=jnp.float32)

    focus_sample_mask = sample_mask * top_mass_focus_mask
    tail_sample_mask = sample_mask * (1.0 - top_mass_focus_mask)

    focus_regression_loss, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=num_structures_per_molecule,
        power=target_weight_power,
        offset=target_weight_offset,
        sample_mask=focus_sample_mask,
    )
    tail_regression_loss, _ = moleculeLocalWeightedMae(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=num_structures_per_molecule,
        power=0.0,
        offset=0.0,
        sample_mask=tail_sample_mask,
    )
    focus_rank_loss = pairwiseRankLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=num_structures_per_molecule,
        margin=rank_loss_margin,
        min_delta=rank_loss_min_delta,
        pair_power=rank_pair_power,
        sample_mask=focus_sample_mask,
    )
    focus_tail_rank_loss = crossSetMarginLoss(
        prediction=prediction,
        target=target,
        num_structures_per_molecule=num_structures_per_molecule,
        left_mask=top_mass_focus_mask,
        right_mask=1.0 - top_mass_focus_mask,
        margin=rank_loss_margin,
        min_delta=rank_loss_min_delta,
        pair_power=rank_pair_power,
        sample_mask=sample_mask,
    )

    total_regression_loss = (
        top_mass_regression_weight * focus_regression_loss
        + tail_suppression_weight * tail_regression_loss
    )
    total_ranking_loss = top_mass_ranking_weight * (focus_rank_loss + focus_tail_rank_loss)
    total_loss = total_regression_loss + total_ranking_loss
    return total_loss, {
        "focus_regression_loss": focus_regression_loss,
        "tail_regression_loss": tail_regression_loss,
        "focus_rank_loss": focus_rank_loss,
        "focus_tail_rank_loss": focus_tail_rank_loss,
        "regression_loss": total_regression_loss,
        "rank_loss": total_ranking_loss,
    }
