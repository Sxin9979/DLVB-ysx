"""Loss utilities for end-to-end E3VB training."""

from __future__ import annotations

import jax.numpy as jnp


def normalizedTargetWeights(
    target: jnp.ndarray,
    power: float,
    offset: float,
) -> jnp.ndarray:
    """
    Build non-negative per-sample regression weights from normalized targets.

    The weights are normalized to mean 1.0 so changing the weighting scheme does
    not unexpectedly rescale the total regression loss magnitude.
    """

    target = jnp.asarray(target, dtype=jnp.float32)
    power = jnp.asarray(power, dtype=jnp.float32)
    offset = jnp.asarray(offset, dtype=jnp.float32)
    raw_weight = jnp.power(jnp.maximum(target, 0.0) + offset, power)
    mean_weight = jnp.maximum(jnp.mean(raw_weight), 1.0e-12)
    return raw_weight / mean_weight


def weightedMae(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    power: float = 0.0,
    offset: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute weighted MAE and return both loss and normalized per-sample weights.
    """

    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    weights = normalizedTargetWeights(target=target, power=power, offset=offset)
    absolute_error = jnp.abs(prediction - target)
    return jnp.mean(weights * absolute_error), weights


def pairwiseRankLoss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    num_structures_per_molecule: jnp.ndarray,
    margin: float = 0.0,
    min_delta: float = 0.0,
    pair_power: float = 1.0,
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

    molecule_index = jnp.repeat(
        jnp.arange(counts.shape[0], dtype=jnp.int32),
        counts,
        total_repeat_length=prediction.shape[0],
    )
    pred_i = prediction[:, None]
    pred_j = prediction[None, :]
    target_i = target[:, None]
    target_j = target[None, :]

    same_molecule = molecule_index[:, None] == molecule_index[None, :]
    target_delta = target_i - target_j
    valid_pair = same_molecule & (target_delta > min_delta)

    pair_weight = jnp.power(jnp.maximum(target_delta, 0.0), pair_power)
    hinge = jnp.maximum(margin - (pred_i - pred_j), 0.0)
    weighted_hinge = pair_weight * hinge

    total_weight = jnp.maximum(jnp.sum(jnp.where(valid_pair, pair_weight, 0.0)), 1.0e-12)
    total_loss = jnp.sum(jnp.where(valid_pair, weighted_hinge, 0.0))
    return total_loss / total_weight
