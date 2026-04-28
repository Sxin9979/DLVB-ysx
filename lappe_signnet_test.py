"""Minimal tests for LapPE + SignNet enhancement in the JAX atom encoder."""

from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.grain_pipeline import GraphPackingAdapter
from data.lap_pe import lapPeFromEdges
from data.schema import UnifiedSample
from model.atom_encoder import AtomE3Encoder, AtomEncoderConfig
from model.signnet import SignNet, SignNetConfig


def makeUnifiedSample() -> UnifiedSample:
    """
    Create one tiny unified sample carrying LapPE fields.
    """

    atom_numbers = np.asarray([6, 1, 1], dtype=np.int32)
    atom_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    atom_node_features = np.asarray(
        [
            [6.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    atom_senders = np.asarray([0, 1, 0, 2], dtype=np.int32)
    atom_receivers = np.asarray([1, 0, 2, 0], dtype=np.int32)
    atom_pair_features = np.zeros((4, 2), dtype=np.float32)
    lap_evals, lap_evecs = lapPeFromEdges(
        num_nodes=3,
        senders=atom_senders,
        receivers=atom_receivers,
        k=2,
        eps=1.0e-12,
        add_self_loops=False,
    )

    orbital_atom_index = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    orbital_role = np.asarray([0, 2, 2], dtype=np.int32)
    active_slot_index = np.asarray([-1, 0, 0], dtype=np.int32)
    rumer_senders = np.asarray([0, 1, 2, 1, 2], dtype=np.int32)
    rumer_receivers = np.asarray([0, 1, 2, 2, 1], dtype=np.int32)
    rumer_edge_type = np.asarray([0, 2, 2, 2, 2], dtype=np.int32)
    active_orbital_index = np.asarray([1, 2], dtype=np.int32)
    active_rumer_senders = np.asarray([0, 1], dtype=np.int32)
    active_rumer_receivers = np.asarray([1, 0], dtype=np.int32)
    active_rumer_edge_type = np.asarray([2, 2], dtype=np.int32)

    return UnifiedSample(
        molecule_id="toy",
        vb_index=0,
        atom_numbers=atom_numbers,
        atom_positions=atom_positions,
        atom_node_features=atom_node_features,
        atom_senders=atom_senders,
        atom_receivers=atom_receivers,
        atom_pair_features=atom_pair_features,
        lap_evals=lap_evals,
        lap_evecs=lap_evecs,
        orbital_atom_index=orbital_atom_index,
        orbital_role=orbital_role,
        active_slot_index=active_slot_index,
        rumer_senders=rumer_senders,
        rumer_receivers=rumer_receivers,
        rumer_edge_type=rumer_edge_type,
        active_orbital_index=active_orbital_index,
        active_rumer_senders=active_rumer_senders,
        active_rumer_receivers=active_rumer_receivers,
        active_rumer_edge_type=active_rumer_edge_type,
        target=1.0,
        target_max=1.0,
    )


def testLapPeProducesExpectedShapes() -> None:
    """
    LapPE should return padded non-trivial eigenpairs with expected shapes.
    """

    senders = np.asarray([0, 1, 1, 2], dtype=np.int32)
    receivers = np.asarray([1, 0, 2, 1], dtype=np.int32)
    evals, evecs = lapPeFromEdges(
        num_nodes=3,
        senders=senders,
        receivers=receivers,
        k=4,
        eps=1.0e-12,
        add_self_loops=False,
    )

    assert evals.shape == (4,)
    assert evecs.shape == (3, 4)
    assert float(evals[0]) > 0.0


def testSignNetIsSignInvariant() -> None:
    """
    SignNet output should be invariant to flipping Laplacian eigenvector signs.
    """

    signnet = SignNet(
        k=2,
        config=SignNetConfig(
            phi_hidden=8,
            phi_out=4,
            phi_layers=2,
            rho_hidden=8,
            out_dim=6,
            rho_layers=2,
        ),
        rngs=nnx.Rngs(0),
    )
    lap_evecs = jnp.asarray(
        [
            [0.5, -0.3],
            [0.1, 0.7],
            [-0.4, 0.2],
        ],
        dtype=jnp.float32,
    )
    lap_evals = jnp.asarray(
        [
            [0.2, 0.6],
            [0.2, 0.6],
            [0.2, 0.6],
        ],
        dtype=jnp.float32,
    )

    positive = signnet(lap_evecs=lap_evecs, lap_evals=lap_evals)
    negative = signnet(lap_evecs=-lap_evecs, lap_evals=lap_evals)
    assert jnp.allclose(positive, negative, atol=1e-6)


def testGraphPackingAdapterBroadcastsLapEvalsPerNode() -> None:
    """
    Batch packing should repeat graph-level eigenvalues to node-aligned tensors.
    """

    adapter = GraphPackingAdapter(lap_pe_k=2)
    sample = makeUnifiedSample()
    batch = adapter.packBatch(samples=[sample], orbital_feature_dim=8)

    assert batch.lap_evals.shape == (3, 2)
    assert batch.lap_evecs.shape == (3, 2)
    expected = np.repeat(sample.lap_evals[None, :], repeats=3, axis=0)
    assert np.allclose(np.asarray(batch.lap_evals), expected)


def testAtomEncoderForwardWithLapPeConcatRuns() -> None:
    """
    Atom encoder should accept LapPE/SignNet-enhanced input and produce outputs.
    """

    sample = makeUnifiedSample()
    atom_graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
            "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
            "positions": jnp.asarray(sample.atom_positions, dtype=jnp.float32),
        },
        edges={
            "pair": jnp.asarray(sample.atom_pair_features, dtype=jnp.float32),
        },
        senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
        receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
        n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    lap_evals = jnp.asarray(np.repeat(sample.lap_evals[None, :], repeats=3, axis=0), dtype=jnp.float32)
    lap_evecs = jnp.asarray(sample.lap_evecs, dtype=jnp.float32)

    encoder = AtomE3Encoder(
        config=AtomEncoderConfig(
            input_feature_dim=3,
            scalar_dim=16,
            vector_dim=4,
            radial_dim=8,
            layers=2,
            max_atomic_number=16,
            radial_min=0.0,
            radial_max=2.0,
            radial_basis=8,
            lmax=1,
            lap_pe_k=2,
            signnet=SignNetConfig(
                phi_hidden=8,
                phi_out=4,
                phi_layers=2,
                rho_hidden=8,
                out_dim=6,
                rho_layers=2,
            ),
        ),
        rngs=nnx.Rngs(0),
    )
    output = encoder(atom_graph=atom_graph, lap_evecs=lap_evecs, lap_evals=lap_evals)

    assert output["scalar"].shape == (3, 16)
    assert output["vector"].shape == (3, 4, 3)


def rotationMatrixZ(angle_radians: float) -> jnp.ndarray:
    """
    Build one deterministic rotation matrix around z axis.
    """

    angle = jnp.asarray(angle_radians, dtype=jnp.float32)
    cos_value = jnp.cos(angle)
    sin_value = jnp.sin(angle)
    return jnp.asarray(
        [
            [cos_value, -sin_value, 0.0],
            [sin_value, cos_value, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )


def testAtomEncoderEdgeContentForwardRuns() -> None:
    """
    Edge-content modulation path should run end-to-end in atom encoder.
    """

    sample = makeUnifiedSample()
    atom_pair_features = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    atom_graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
            "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
            "positions": jnp.asarray(sample.atom_positions, dtype=jnp.float32),
        },
        edges={
            "pair": jnp.asarray(atom_pair_features, dtype=jnp.float32),
        },
        senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
        receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
        n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    lap_evals = jnp.asarray(np.repeat(sample.lap_evals[None, :], repeats=3, axis=0), dtype=jnp.float32)
    lap_evecs = jnp.asarray(sample.lap_evecs, dtype=jnp.float32)

    encoder = AtomE3Encoder(
        config=AtomEncoderConfig(
            input_feature_dim=3,
            scalar_dim=16,
            vector_dim=4,
            radial_dim=8,
            layers=2,
            max_atomic_number=16,
            radial_min=0.0,
            radial_max=2.0,
            radial_basis=8,
            lmax=1,
            lap_pe_k=2,
            signnet=SignNetConfig(
                phi_hidden=8,
                phi_out=4,
                phi_layers=2,
                rho_hidden=8,
                out_dim=6,
                rho_layers=2,
            ),
            edge_content_modulation=True,
            edge_content_hidden_dim=8,
        ),
        rngs=nnx.Rngs(0),
    )
    output = encoder(atom_graph=atom_graph, lap_evecs=lap_evecs, lap_evals=lap_evals)

    assert output["scalar"].shape == (3, 16)
    assert output["vector"].shape == (3, 4, 3)


def testAtomEncoderEdgeContentPreservesEquivariance() -> None:
    """
    Edge-content modulation should preserve atom-level 0e/1o equivariance.
    """

    sample = makeUnifiedSample()
    atom_pair_features = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions = jnp.asarray(sample.atom_positions, dtype=jnp.float32)
    rotation = rotationMatrixZ(angle_radians=0.7)
    rotated_positions = positions @ rotation.T

    atom_graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
            "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
            "positions": positions,
        },
        edges={"pair": jnp.asarray(atom_pair_features, dtype=jnp.float32)},
        senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
        receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
        n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    rotated_atom_graph = atom_graph._replace(
        nodes={
            "features": atom_graph.nodes["features"],
            "numbers": atom_graph.nodes["numbers"],
            "positions": rotated_positions,
        }
    )
    lap_evals = jnp.asarray(np.repeat(sample.lap_evals[None, :], repeats=3, axis=0), dtype=jnp.float32)
    lap_evecs = jnp.asarray(sample.lap_evecs, dtype=jnp.float32)

    encoder = AtomE3Encoder(
        config=AtomEncoderConfig(
            input_feature_dim=3,
            scalar_dim=16,
            vector_dim=4,
            radial_dim=8,
            layers=2,
            max_atomic_number=16,
            radial_min=0.0,
            radial_max=2.0,
            radial_basis=8,
            lmax=1,
            lap_pe_k=2,
            signnet=SignNetConfig(
                phi_hidden=8,
                phi_out=4,
                phi_layers=2,
                rho_hidden=8,
                out_dim=6,
                rho_layers=2,
            ),
            edge_content_modulation=True,
            edge_content_hidden_dim=8,
        ),
        rngs=nnx.Rngs(0),
    )

    output = encoder(atom_graph=atom_graph, lap_evecs=lap_evecs, lap_evals=lap_evals)
    rotated_output = encoder(
        atom_graph=rotated_atom_graph,
        lap_evecs=lap_evecs,
        lap_evals=lap_evals,
    )

    expected_vector = jnp.einsum("ncd,df->ncf", output["vector"], rotation.T)
    assert jnp.allclose(output["scalar"], rotated_output["scalar"], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(expected_vector, rotated_output["vector"], atol=5e-5, rtol=5e-5)


def testAtomEncoderStableEdgeContentForwardRuns() -> None:
    """
    Stable residual edge-content modulation should run end-to-end in atom encoder.
    """

    sample = makeUnifiedSample()
    atom_pair_features = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    atom_graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
            "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
            "positions": jnp.asarray(sample.atom_positions, dtype=jnp.float32),
        },
        edges={"pair": jnp.asarray(atom_pair_features, dtype=jnp.float32)},
        senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
        receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
        n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    lap_evals = jnp.asarray(np.repeat(sample.lap_evals[None, :], repeats=3, axis=0), dtype=jnp.float32)
    lap_evecs = jnp.asarray(sample.lap_evecs, dtype=jnp.float32)

    encoder = AtomE3Encoder(
        config=AtomEncoderConfig(
            input_feature_dim=3,
            scalar_dim=16,
            vector_dim=4,
            radial_dim=8,
            layers=2,
            max_atomic_number=16,
            radial_min=0.0,
            radial_max=2.0,
            radial_basis=8,
            lmax=1,
            lap_pe_k=2,
            signnet=SignNetConfig(
                phi_hidden=8,
                phi_out=4,
                phi_layers=2,
                rho_hidden=8,
                out_dim=6,
                rho_layers=2,
            ),
            edge_content_modulation=True,
            edge_content_hidden_dim=8,
            edge_content_mode="stable_residual",
            edge_content_residual_scale=0.1,
        ),
        rngs=nnx.Rngs(0),
    )
    output = encoder(atom_graph=atom_graph, lap_evecs=lap_evecs, lap_evals=lap_evals)

    assert output["scalar"].shape == (3, 16)
    assert output["vector"].shape == (3, 4, 3)


def testAtomEncoderStableEdgeContentPreservesEquivariance() -> None:
    """
    Stable residual edge-content modulation should preserve 0e/1o equivariance.
    """

    sample = makeUnifiedSample()
    atom_pair_features = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions = jnp.asarray(sample.atom_positions, dtype=jnp.float32)
    rotation = rotationMatrixZ(angle_radians=0.7)
    rotated_positions = positions @ rotation.T

    atom_graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
            "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
            "positions": positions,
        },
        edges={"pair": jnp.asarray(atom_pair_features, dtype=jnp.float32)},
        senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
        receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
        n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    rotated_atom_graph = atom_graph._replace(
        nodes={
            "features": atom_graph.nodes["features"],
            "numbers": atom_graph.nodes["numbers"],
            "positions": rotated_positions,
        }
    )
    lap_evals = jnp.asarray(np.repeat(sample.lap_evals[None, :], repeats=3, axis=0), dtype=jnp.float32)
    lap_evecs = jnp.asarray(sample.lap_evecs, dtype=jnp.float32)

    encoder = AtomE3Encoder(
        config=AtomEncoderConfig(
            input_feature_dim=3,
            scalar_dim=16,
            vector_dim=4,
            radial_dim=8,
            layers=2,
            max_atomic_number=16,
            radial_min=0.0,
            radial_max=2.0,
            radial_basis=8,
            lmax=1,
            lap_pe_k=2,
            signnet=SignNetConfig(
                phi_hidden=8,
                phi_out=4,
                phi_layers=2,
                rho_hidden=8,
                out_dim=6,
                rho_layers=2,
            ),
            edge_content_modulation=True,
            edge_content_hidden_dim=8,
            edge_content_mode="stable_residual",
            edge_content_residual_scale=0.1,
        ),
        rngs=nnx.Rngs(0),
    )

    output = encoder(atom_graph=atom_graph, lap_evecs=lap_evecs, lap_evals=lap_evals)
    rotated_output = encoder(
        atom_graph=rotated_atom_graph,
        lap_evecs=lap_evecs,
        lap_evals=lap_evals,
    )

    expected_vector = jnp.einsum("ncd,df->ncf", output["vector"], rotation.T)
    assert jnp.allclose(output["scalar"], rotated_output["scalar"], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(expected_vector, rotated_output["vector"], atol=5e-5, rtol=5e-5)
