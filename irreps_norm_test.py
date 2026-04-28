"""Minimal tests for irreps-aware normalization in the JAX atom encoder."""

from pathlib import Path
import sys

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.norms import IrrepsLayerNorm


def testIrrepsLayerNormMatchesScalarLNAndVectorRMS() -> None:
    """
    Scalars should use LayerNorm-style normalization, vectors should use RMS only.
    """

    irreps = e3nn.Irreps("2x0e + 2x1o")
    norm = IrrepsLayerNorm(irreps=irreps, rngs=nnx.Rngs(0), affine=False)

    packed = jnp.asarray(
        [
            [1.0, 3.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0],
            [2.0, 6.0, 3.0, 0.0, 4.0, 1.0, 2.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    state = e3nn.IrrepsArray(irreps, packed)

    normalized = norm(state)
    scalar = normalized.filter(keep="0e").array
    vector = normalized.filter(keep="1o").array

    scalar_input = packed[:, :2]
    scalar_mean = jnp.mean(scalar_input, axis=1, keepdims=True)
    scalar_var = jnp.mean(jnp.square(scalar_input - scalar_mean), axis=1, keepdims=True)
    expected_scalar = (scalar_input - scalar_mean) / jnp.sqrt(scalar_var + 1e-5)

    vector_input = packed[:, 2:]
    vector_rms = jnp.sqrt(jnp.mean(jnp.square(vector_input), axis=1, keepdims=True) + 1e-5)
    expected_vector = vector_input / vector_rms

    assert jnp.allclose(scalar, expected_scalar, atol=1e-6)
    assert jnp.allclose(vector, expected_vector, atol=1e-6)


def testIrrepsLayerNormUsesWholeNonScalarBlockAcrossMultiplicity() -> None:
    """
    One l>0 block should be normalized over both multiplicity and irrep components.
    """

    irreps = e3nn.Irreps("2x1o")
    norm = IrrepsLayerNorm(irreps=irreps, rngs=nnx.Rngs(0), affine=False)

    packed = jnp.asarray([[1.0, 2.0, 2.0, 2.0, 0.0, 1.0]], dtype=jnp.float32)
    normalized = norm(e3nn.IrrepsArray(irreps, packed)).array

    expected_rms = jnp.sqrt(jnp.mean(jnp.square(packed), axis=1, keepdims=True) + 1e-5)
    expected = packed / expected_rms

    assert jnp.allclose(normalized, expected, atol=1e-6)
