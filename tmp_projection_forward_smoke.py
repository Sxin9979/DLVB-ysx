"""Smoke test for the atom-to-active-orbital projection stack."""

import jax
import jax.numpy as jnp
import jraph
from flax import nnx

from model.orbital_projection import AtomToOrbitalProjectionStack, OrbitalProjectionConfig


def make_graph(num_nodes: int, senders: list[int], receivers: list[int]) -> jraph.GraphsTuple:
    """Build a minimal active Rumer graph with paired-edge labels."""

    return jraph.GraphsTuple(
        nodes=jnp.zeros((num_nodes, 1), dtype=jnp.float32),
        edges={"edge_type": jnp.full((len(senders),), 2, dtype=jnp.int32)},
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        n_node=jnp.asarray([num_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([len(senders)], dtype=jnp.int32),
        globals=None,
    )


def assert_finite(name: str, value: jnp.ndarray) -> None:
    """Raise if a projection output contains NaN/Inf."""

    if not bool(jnp.all(jnp.isfinite(value))):
        raise AssertionError(f"{name} contains non-finite values")


def run_case(
    name: str,
    stack: AtomToOrbitalProjectionStack,
    orbital_atom_index: list[list[int]],
    orbital_role: list[int],
    active_slot_index: list[int],
    active_orbital_index: list[int],
    active_rumer_graph: jraph.GraphsTuple,
) -> None:
    """Run one projection case and print compact diagnostics."""

    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
        dtype=jnp.float32,
    )
    scalar = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0
    vector = jnp.asarray(
        [
            [[0.2, 0.1, 0.3], [0.0, 0.4, 0.1]],
            [[0.1, 0.3, 0.2], [0.5, 0.0, 0.2]],
            [[0.3, 0.2, 0.1], [0.2, 0.1, 0.5]],
        ],
        dtype=jnp.float32,
    )

    output = stack(
        scalar_feature=scalar,
        vector_feature=vector,
        positions=positions,
        num_atoms_per_graph=jnp.asarray([3], dtype=jnp.int32),
        orbital_atom_index=jnp.asarray(orbital_atom_index, dtype=jnp.int32),
        orbital_role=jnp.asarray(orbital_role, dtype=jnp.int32),
        active_slot_index=jnp.asarray(active_slot_index, dtype=jnp.int32),
        active_rumer_graph=active_rumer_graph,
        active_orbital_index=jnp.asarray(active_orbital_index, dtype=jnp.int32),
    )

    for key in (
        "orbital_feature",
        "orbital_feature_q3_flipped",
        "slot_direction",
        "slot_role_prior",
        "slot_alpha",
    ):
        assert_finite(key, output[key])

    print(f"case={name}")
    print("orbital_feature", output["orbital_feature"].shape)
    print("slot_direction", output["slot_direction"].shape)
    print("active_capacity", output["active_capacity"])
    print("slot_role_prior", output["slot_role_prior"])
    print("slot_direction_norm", jnp.linalg.norm(output["slot_direction"], axis=-1))


def main() -> None:
    """Run bond-like and lone-pair-like active-slot prior smoke cases."""

    print("backend", jax.default_backend())
    config = OrbitalProjectionConfig(
        scalar_dim=4,
        vector_dim=2,
        slot_dim=5,
        slot_query_dim=6,
        slot_embedding_dim=3,
        max_active_slots=3,
        orbital_feature_dim=7,
    )
    stack = AtomToOrbitalProjectionStack(config=config, rngs=nnx.Rngs(0))

    run_case(
        name="cross_atom_bond_prior",
        stack=stack,
        orbital_atom_index=[[0, 0], [1, 1], [2, 2], [0, 1]],
        orbital_role=[2, 2, 0, 1],
        active_slot_index=[0, 0, -1, -1],
        active_orbital_index=[0, 1],
        active_rumer_graph=make_graph(2, [0, 1], [1, 0]),
    )
    run_case(
        name="same_atom_lone_pair_prior",
        stack=stack,
        orbital_atom_index=[[0, 0], [0, 0], [1, 1], [2, 2]],
        orbital_role=[2, 2, 0, 1],
        active_slot_index=[0, 1, -1, -1],
        active_orbital_index=[0, 1],
        active_rumer_graph=make_graph(2, [0, 1], [1, 0]),
    )
    print("tmp_projection_forward_smoke: ok")


if __name__ == "__main__":
    main()
