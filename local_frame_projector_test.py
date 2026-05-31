"""Tests for chemistry-aware local frame construction."""

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.orbital_projection import LocalOrbitalProjector


def makePlanarPositions() -> jnp.ndarray:
    """Build one simple planar geometry with a clear sigma bond at atom 0."""

    return jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [-1.5, 0.2, 0.0],
        ],
        dtype=jnp.float32,
    )


def rotationAroundZ(angle_radians: float) -> np.ndarray:
    """Return one 3x3 rotation matrix around the z axis."""

    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def testChemistryAwareFrameMatchesPlanarSigmaPiSemantics() -> None:
    """Atom-0 frame should resolve to sigma, in-plane perpendicular, and plane normal."""

    projector = LocalOrbitalProjector()
    positions = makePlanarPositions()
    e1, e2, e3 = projector.buildLocalFrames(positions=positions, num_atoms_per_graph=jnp.asarray([4], dtype=jnp.int32))

    assert np.allclose(np.asarray(e1[0]), np.asarray([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(e2[0]), np.asarray([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(e3[0]), np.asarray([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)


def testChemistryAwareFrameFallsBackForSingleAtom() -> None:
    """Single-atom graphs should still receive the default orthonormal frame."""

    projector = LocalOrbitalProjector()
    positions = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    e1, e2, e3 = projector.buildLocalFrames(positions=positions, num_atoms_per_graph=jnp.asarray([1], dtype=jnp.int32))

    assert np.allclose(np.asarray(e1[0]), np.asarray([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(e2[0]), np.asarray([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(np.asarray(e3[0]), np.asarray([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)


def testChemistryAwareFrameCoRotatesWithGeometry() -> None:
    """Local sigma/in-plane/normal frames should co-rotate with the input geometry."""

    projector = LocalOrbitalProjector()
    positions = np.asarray(makePlanarPositions(), dtype=np.float32)
    rotation = rotationAroundZ(np.pi / 2.0)

    base_e1, base_e2, base_e3 = projector.buildLocalFrames(
        positions=jnp.asarray(positions, dtype=jnp.float32),
        num_atoms_per_graph=jnp.asarray([4], dtype=jnp.int32),
    )
    rotated_positions = positions @ rotation.T
    rotated_e1, rotated_e2, rotated_e3 = projector.buildLocalFrames(
        positions=jnp.asarray(rotated_positions, dtype=jnp.float32),
        num_atoms_per_graph=jnp.asarray([4], dtype=jnp.int32),
    )

    assert np.allclose(np.asarray(rotated_e1), np.asarray(base_e1) @ rotation.T, atol=1e-6)
    assert np.allclose(np.asarray(rotated_e2), np.asarray(base_e2) @ rotation.T, atol=1e-6)
    assert np.allclose(np.asarray(rotated_e3), np.asarray(base_e3) @ rotation.T, atol=1e-6)


if __name__ == "__main__":
    testChemistryAwareFrameMatchesPlanarSigmaPiSemantics()
    testChemistryAwareFrameFallsBackForSingleAtom()
    testChemistryAwareFrameCoRotatesWithGeometry()
    print("local_frame_projector_test: ok")
