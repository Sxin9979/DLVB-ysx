"""Diagnose NaNs on one real packed training batch without updating parameters."""

from __future__ import annotations

import argparse
import types
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from main import ExperimentRunner
from utils import topMassObjectiveLoss


def finite_summary(name: str, value: Any) -> None:
    """Print finite/count/min/max diagnostics for one array-like value."""

    array = jnp.asarray(value)
    finite = jnp.isfinite(array)
    finite_count = int(jnp.sum(finite))
    total_count = int(array.size)
    if total_count == 0:
        print(f"{name}: shape={array.shape} empty")
        return
    safe = jnp.where(finite, array, 0.0)
    print(
        f"{name}: shape={array.shape} finite={finite_count}/{total_count} "
        f"nan={int(jnp.sum(jnp.isnan(array)))} inf={int(jnp.sum(jnp.isinf(array)))} "
        f"min={float(jnp.min(safe)):.8g} max={float(jnp.max(safe)):.8g} "
        f"mean={float(jnp.sum(safe) / jnp.maximum(jnp.sum(finite), 1)):.8g}"
    )


def tree_finite_summary(name: str, tree: Any) -> None:
    """Print finite diagnostics over all floating leaves in a pytree."""

    leaves = jax.tree_util.tree_leaves(tree)
    arrays = []
    for leaf in leaves:
        try:
            array = jnp.asarray(leaf)
        except TypeError:
            continue
        if jnp.issubdtype(array.dtype, jnp.floating):
            arrays.append(array.reshape(-1))
    if not arrays:
        print(f"{name}: no floating leaves")
        return
    flat = jnp.concatenate(arrays, axis=0)
    finite_summary(name, flat)


def tree_nonfinite_paths(name: str, tree: Any, limit: int = 40) -> None:
    """Print parameter/state paths whose floating leaves contain NaN or Inf."""

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    rows = []
    for path, leaf in path_leaves:
        try:
            array = jnp.asarray(leaf)
        except TypeError:
            continue
        if not jnp.issubdtype(array.dtype, jnp.floating):
            continue
        nonfinite = int(jnp.sum(~jnp.isfinite(array)))
        if nonfinite <= 0:
            continue
        rows.append((nonfinite, int(array.size), path, array.shape))
    rows.sort(key=lambda item: item[0], reverse=True)
    print(f"{name}.nonfinite_leaf_count={len(rows)}")
    for nonfinite, size, path, shape in rows[:limit]:
        print(f"{name}.nonfinite path={path} shape={shape} nonfinite={nonfinite}/{size}")


def make_first_train_batch(runner: ExperimentRunner):
    """Build the first real training batch using the same iterator settings as training."""

    iterator = runner.pipeline.createMoleculeIterator(
        molecule_groups=runner.split_molecule_chunk_groups["train"],
        batch_size=runner.config.training.batch_size,
        shuffle=True,
        seed=runner.config.training.seed,
        drop_remainder=runner.config.training.train_drop_remainder,
        bucket_key=runner.trainBucketKey(),
        fixed_bucket_config=runner.fixedBucketConfig(),
        split_name="train",
        focus_cumulative_mass=runner.config.training.focus_cumulative_mass,
        top_mass_sample_strategy=(
            runner.config.training.top_mass_sample_strategy
            if runner.config.training.use_top_mass_objective
            else "full_molecule"
        ),
        mixed_tail_top_fraction=runner.config.training.mixed_tail_top_fraction,
        max_tail_samples_per_molecule=(
            runner.config.training.max_tail_samples_per_molecule
            if runner.config.training.use_top_mass_objective
            else None
        ),
        dataset_sampling_strategy=runner.config.training.dataset_sampling_strategy,
        max_batch_cost=runner.config.training.max_batch_cost,
        max_structure_cost_per_molecule=(
            runner.config.training.max_structure_cost_per_molecule
            if runner.config.training.use_top_mass_objective
            else None
        ),
    )
    return next(iter(iterator))


def projection_diagnostics(runner: ExperimentRunner, batch) -> dict[str, jnp.ndarray]:
    """Run atom encoder and projection directly so projection tensors are visible."""

    model = runner.model
    expanded_atom_numbers = batch.static_atom_numbers[batch.expanded_atom_to_static_atom_index]
    expanded_atom_positions = batch.static_atom_positions[batch.expanded_atom_to_static_atom_index]
    expanded_local_frame_e1 = batch.local_frame_e1[batch.expanded_atom_to_static_atom_index]
    expanded_local_frame_e2 = batch.local_frame_e2[batch.expanded_atom_to_static_atom_index]
    expanded_local_frame_e3 = batch.local_frame_e3[batch.expanded_atom_to_static_atom_index]

    atom_output = model.atom_encoder(
        batch.atom_graph,
        atom_number=expanded_atom_numbers,
        positions=expanded_atom_positions,
        lap_evecs=batch.lap_evecs,
        lap_evals=batch.lap_evals,
    )
    for key, value in atom_output.items():
        if key in ("scalar", "vector", "positions"):
            finite_summary(f"atom_output.{key}", value)

    projection_output = model.projection(
        scalar_feature=atom_output["scalar"],
        vector_feature=atom_output["vector"],
        positions=atom_output["positions"],
        num_atoms_per_graph=batch.num_atoms_per_graph,
        orbital_atom_index=batch.orbital_atom_index,
        orbital_role=batch.orbital_role,
        active_slot_index=batch.active_slot_index,
        active_rumer_graph=batch.active_rumer_graph,
        active_orbital_index=batch.active_orbital_index,
        local_frame_e1=expanded_local_frame_e1,
        local_frame_e2=expanded_local_frame_e2,
        local_frame_e3=expanded_local_frame_e3,
    )
    for key in (
        "orbital_feature",
        "orbital_feature_q3_flipped",
        "slot_alpha",
        "slot_direction",
        "slot_role_prior",
        "active_capacity",
    ):
        finite_summary(f"projection.{key}", projection_output[key])
    return projection_output


def loss_closure(runner: ExperimentRunner, batch):
    """Return the same top-mass training loss used by trainStepTopMass."""

    prediction, auxiliary = runner.model.forwardWithAux(batch)
    objective_loss, objective_terms = topMassObjectiveLoss(
        prediction=prediction,
        target=batch.targets,
        top_mass_focus_mask=batch.top_mass_focus_mask,
        num_structures_per_molecule=batch.num_structures_per_molecule,
        target_weight_power=runner.config.training.target_weight_power,
        target_weight_offset=runner.config.training.target_weight_offset,
        top_mass_regression_weight=runner.config.training.top_mass_regression_weight,
        top_mass_ranking_weight=runner.config.training.top_mass_ranking_weight,
        tail_suppression_weight=runner.config.training.tail_suppression_weight,
        rank_loss_margin=runner.config.training.rank_loss_margin,
        rank_loss_min_delta=runner.config.training.rank_loss_min_delta,
        rank_pair_power=runner.config.training.rank_pair_power,
        max_focus_rank_samples_per_molecule=runner.config.training.max_focus_rank_samples_per_molecule,
        max_tail_rank_samples_per_molecule=runner.config.training.max_tail_rank_samples_per_molecule,
        max_rank_pairs_per_molecule=runner.config.training.max_rank_pairs_per_molecule,
        sample_mask=batch.sample_mask,
    )
    slot_penalty = auxiliary["slot_diversity_penalty"]
    slot_weight = getattr(runner, "_diagnostic_slot_weight", runner.config.training.slot_diversity_weight)
    total_loss = objective_loss + slot_weight * slot_penalty
    return total_loss, {
        "prediction": prediction,
        "slot_penalty": slot_penalty,
        **objective_terms,
    }


def disable_directional_slots(runner: ExperimentRunner) -> None:
    """Replace the directional slot branch with zeros for isolation diagnostics."""

    matcher = runner.model.projection.slot_matcher

    def zero_directional_slots(
        self,
        scalar_feature,
        vector_feature,
        e1,
        e2,
        e3,
        query,
        slot_role_prior=None,
    ):
        del vector_feature, e1, e2, e3, query, slot_role_prior
        total_atoms = scalar_feature.shape[0]
        direction = jnp.zeros(
            (total_atoms, self.config.max_active_slots, 3),
            dtype=scalar_feature.dtype,
        )
        feature = jnp.zeros(
            (total_atoms, self.config.max_active_slots, self.config.slot_dim),
            dtype=scalar_feature.dtype,
        )
        return {"direction": direction, "feature": feature}

    matcher.buildDirectionalSlots = types.MethodType(zero_directional_slots, matcher)


def main() -> None:
    """Run one real-batch NaN diagnostic."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--skip_grad", action="store_true")
    parser.add_argument("--grad_path_limit", type=int, default=40)
    parser.add_argument("--slot_weight", type=float, default=None)
    parser.add_argument("--disable_directional_slots", action="store_true")
    args = parser.parse_args()

    print("jax_backend", jax.default_backend(), "devices", jax.devices())
    runner = ExperimentRunner(
        config_path=args.config,
        allow_cpu=args.allow_cpu,
        quiet_stdout=True,
    )
    runner.loadConfig()
    runner.setupLogging()
    runner.inspectTrainingDevice()
    runner.buildDataset()
    runner.buildModel()
    if args.slot_weight is not None:
        runner._diagnostic_slot_weight = float(args.slot_weight)
        print("diagnostic_slot_weight", runner._diagnostic_slot_weight)
    if args.disable_directional_slots:
        disable_directional_slots(runner)
        print("diagnostic_disable_directional_slots true")

    batch = make_first_train_batch(runner)
    print("batch targets", batch.targets.shape, "sample_mask_sum", float(jnp.sum(batch.sample_mask)))
    print("num_atoms", int(batch.atom_graph.nodes["features"].shape[0]))
    print("num_orbitals", int(batch.orbital_role.shape[0]))
    print("num_active_orbitals", int(batch.active_orbital_index.shape[0]))
    print("active_edges", int(batch.active_rumer_graph.senders.shape[0]))
    finite_summary("batch.targets", batch.targets)
    finite_summary("batch.sample_mask", batch.sample_mask)
    finite_summary("batch.top_mass_focus_mask", batch.top_mass_focus_mask)
    finite_summary("batch.local_frame_e1", batch.local_frame_e1)
    finite_summary("batch.local_frame_e2", batch.local_frame_e2)
    finite_summary("batch.local_frame_e3", batch.local_frame_e3)
    finite_summary("batch.active_orbital_index", batch.active_orbital_index)
    finite_summary("batch.active_rumer_edge_type", batch.active_rumer_graph.edges["edge_type"])
    print(
        "active sender max",
        int(jnp.max(batch.active_rumer_graph.senders)),
        "receiver max",
        int(jnp.max(batch.active_rumer_graph.receivers)),
    )

    projection_diagnostics(runner, batch)
    loss, terms = loss_closure(runner, batch)
    finite_summary("loss.total", loss)
    for key, value in terms.items():
        finite_summary(f"loss_terms.{key}", value)

    if args.skip_grad:
        return

    def closure(model):
        runner.model = model
        return loss_closure(runner, batch)

    (grad_loss, grad_terms), gradient = nnx.value_and_grad(closure, has_aux=True)(runner.model)
    finite_summary("grad_loss.total", grad_loss)
    for key, value in grad_terms.items():
        finite_summary(f"grad_terms.{key}", value)
    tree_finite_summary("gradient", gradient)
    tree_nonfinite_paths("gradient", gradient, limit=args.grad_path_limit)


if __name__ == "__main__":
    main()
