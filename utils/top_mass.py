"""Top-mass objective helpers shared by training, metrics, and inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def computeTopMassFocusMask(
    target: np.ndarray,
    num_structures_per_molecule: Iterable[int],
    cumulative_mass: float,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mark the smallest per-molecule prefix whose cumulative target mass reaches
    ``cumulative_mass``.

    The input target values may be any non-negative scale that is consistent
    within each molecule. The existing normalized target ``y / y_max`` is valid
    because the top-mass prefix is invariant to a constant scale factor.
    """

    target = np.asarray(target, dtype=np.float32)
    counts = np.asarray(list(num_structures_per_molecule), dtype=np.int32)
    if sample_mask is None:
        sample_mask = np.ones_like(target, dtype=np.float32)
    else:
        sample_mask = np.asarray(sample_mask, dtype=np.float32)

    cumulative_mass = float(np.clip(cumulative_mass, 0.0, 1.0))
    focus_mask = np.zeros_like(target, dtype=np.float32)
    offset = 0
    for count in counts.tolist():
        count = int(count)
        if count <= 0:
            continue
        target_slice = target[offset : offset + count]
        sample_slice = sample_mask[offset : offset + count] > 0.5
        valid_index = np.flatnonzero(sample_slice)
        if valid_index.size == 0:
            offset += count
            continue
        valid_target = np.maximum(target_slice[valid_index], 0.0)
        order = np.argsort(valid_target)[::-1]
        ordered_index = valid_index[order]
        ordered_target = valid_target[order]
        total_mass = float(np.sum(ordered_target))
        if total_mass <= 1.0e-12:
            focus_mask[offset + ordered_index[:1]] = 1.0
            offset += count
            continue
        running = np.cumsum(ordered_target)
        focus_count = int(np.searchsorted(running, cumulative_mass * total_mass, side="left")) + 1
        focus_mask[offset + ordered_index[:focus_count]] = 1.0
        offset += count
    return focus_mask.astype(np.float32, copy=False)


def buildPredictedTopMassMask(
    prediction: np.ndarray,
    num_structures_per_molecule: Iterable[int],
    cumulative_mass: float,
    sample_mask: np.ndarray | None = None,
    fallback_focus_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a predicted top-mass set from non-negative predicted weights.

    When a molecule has no positive predicted mass, fall back to selecting the
    same number of structures as the true focus set if provided, otherwise pick
    the single top-scoring structure.
    """

    prediction = np.asarray(prediction, dtype=np.float32)
    counts = np.asarray(list(num_structures_per_molecule), dtype=np.int32)
    if sample_mask is None:
        sample_mask = np.ones_like(prediction, dtype=np.float32)
    else:
        sample_mask = np.asarray(sample_mask, dtype=np.float32)
    if fallback_focus_mask is not None:
        fallback_focus_mask = np.asarray(fallback_focus_mask, dtype=np.float32)

    cumulative_mass = float(np.clip(cumulative_mass, 0.0, 1.0))
    predicted_mask = np.zeros_like(prediction, dtype=np.float32)
    offset = 0
    for count in counts.tolist():
        count = int(count)
        if count <= 0:
            continue
        pred_slice = prediction[offset : offset + count]
        sample_slice = sample_mask[offset : offset + count] > 0.5
        valid_index = np.flatnonzero(sample_slice)
        if valid_index.size == 0:
            offset += count
            continue
        valid_prediction = pred_slice[valid_index]
        order = np.argsort(valid_prediction)[::-1]
        ordered_index = valid_index[order]
        ordered_prediction = np.maximum(valid_prediction[order], 0.0)
        total_mass = float(np.sum(ordered_prediction))
        if total_mass <= 1.0e-12:
            if fallback_focus_mask is not None:
                focus_count = int(
                    max(1, np.sum(fallback_focus_mask[offset : offset + count] > 0.5))
                )
            else:
                focus_count = 1
            predicted_mask[offset + ordered_index[:focus_count]] = 1.0
            offset += count
            continue
        running = np.cumsum(ordered_prediction)
        focus_count = int(np.searchsorted(running, cumulative_mass * total_mass, side="left")) + 1
        predicted_mask[offset + ordered_index[:focus_count]] = 1.0
        offset += count
    return predicted_mask.astype(np.float32, copy=False)


def selectTopMassStructureIndices(
    target: np.ndarray,
    focus_mask: np.ndarray,
    strategy: str,
    max_tail_samples: int | None,
    rng: np.random.Generator,
    mixed_top_fraction: float = 0.5,
) -> np.ndarray:
    """
    Select structure indices for one molecule under a top-mass sampling policy.
    """

    target = np.asarray(target, dtype=np.float32)
    focus_mask = np.asarray(focus_mask, dtype=np.float32) > 0.5
    focus_index = np.flatnonzero(focus_mask)
    tail_index = np.flatnonzero(~focus_mask)

    if strategy == "full_molecule":
        return np.arange(target.shape[0], dtype=np.int32)
    if strategy == "focus_only":
        return focus_index.astype(np.int32, copy=False)

    if (max_tail_samples is None) or (max_tail_samples < 0) or (tail_index.size <= max_tail_samples):
        selected_tail = tail_index
    elif strategy == "focus_plus_top_tail":
        order = np.argsort(target[tail_index])[::-1]
        selected_tail = tail_index[order[: int(max_tail_samples)]]
    elif strategy == "focus_plus_mixed_tail":
        order = np.argsort(target[tail_index])[::-1]
        clipped_fraction = float(np.clip(mixed_top_fraction, 0.0, 1.0))
        top_count = int(round(int(max_tail_samples) * clipped_fraction))
        top_count = min(max(top_count, 0), int(max_tail_samples), int(tail_index.size))
        selected_top = tail_index[order[:top_count]]
        remaining_candidates = tail_index[order[top_count:]]
        remaining_count = min(
            max(int(max_tail_samples) - top_count, 0),
            int(remaining_candidates.size),
        )
        if remaining_count > 0:
            selected_random = np.sort(
                rng.choice(remaining_candidates, size=int(remaining_count), replace=False)
            )
            selected_tail = np.concatenate([selected_top, selected_random], axis=0)
        else:
            selected_tail = selected_top
    else:
        selected_tail = np.sort(
            rng.choice(tail_index, size=int(max_tail_samples), replace=False)
        )

    selected = np.concatenate([focus_index, selected_tail], axis=0)
    if selected.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.sort(selected.astype(np.int32, copy=False))


def capTopMassTailSamplesByBudget(
    num_structures: int,
    num_focus_structures: int,
    structure_cost: float,
    max_tail_samples: int | None,
    max_structure_cost: int | None,
) -> int | None:
    """
    Derive one effective tail-sampling cap from a per-molecule complexity budget.

    The budget is expressed in the same heuristic cost units as the runtime
    batch planner. Focus structures are always preserved; only tail capacity is
    reduced when the estimated per-molecule cost would otherwise exceed the
    configured limit.
    """

    num_structures = max(int(num_structures), 0)
    num_focus_structures = min(max(int(num_focus_structures), 0), num_structures)
    tail_count = max(num_structures - num_focus_structures, 0)
    if tail_count == 0:
        return 0 if max_tail_samples is not None else None

    effective_tail_cap = tail_count if max_tail_samples is None else min(int(max_tail_samples), tail_count)
    if (max_structure_cost is None) or (int(max_structure_cost) <= 0):
        return effective_tail_cap

    structure_cost = max(float(structure_cost), 1.0)
    max_total_structures = max(
        num_focus_structures,
        int(np.floor(float(max_structure_cost) / structure_cost)),
    )
    budget_tail_cap = max(max_total_structures - num_focus_structures, 0)
    return min(effective_tail_cap, budget_tail_cap)
