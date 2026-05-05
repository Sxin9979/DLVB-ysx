"""Numerical E(3)/O(3) property checks for the end-to-end E3VB model."""

import argparse
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from data import GraphPackingAdapter, UnifiedSampleProcessor
from data.schema import UnifiedBatch
from model.end_to_end import EndToEndE3VBModel
from utils import ConfigFactory, NNXCheckpointManager


@dataclass
class TransformSpec:
    """
    Describe one geometric transform used in tests.

    Arguments:
    - name: Transform name.
    - matrix: 3x3 linear transform matrix.
    - translation: Translation vector, shape [3].
    """

    name: str
    matrix: jnp.ndarray
    translation: jnp.ndarray


class GeometryTransformSuite:
    """
    Build translation, rotation, and reflection transforms.
    """

    def translation(self) -> TransformSpec:
        """
        Build a pure translation transform.

        Returns:
        - TransformSpec with identity matrix and non-zero translation.
        """

        return TransformSpec(
            name="translation",
            matrix=jnp.eye(3, dtype=jnp.float32),
            translation=jnp.asarray([0.37, -0.52, 0.19], dtype=jnp.float32),
        )

    def rotation(self, seed: int) -> TransformSpec:
        """
        Build one deterministic random rotation transform.

        Arguments:
        - seed: Random seed for rotation axis sampling.

        Returns:
        - TransformSpec with rotation matrix and zero translation.
        """

        key = jax.random.PRNGKey(seed)
        axis = jax.random.normal(key, shape=(3,))
        axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-8)
        angle = jnp.asarray(1.113, dtype=jnp.float32)
        ax, ay, az = axis[0], axis[1], axis[2]
        skew = jnp.asarray(
            [
                [0.0, -az, ay],
                [az, 0.0, -ax],
                [-ay, ax, 0.0],
            ],
            dtype=jnp.float32,
        )
        identity = jnp.eye(3, dtype=jnp.float32)
        matrix = identity + jnp.sin(angle) * skew + (1.0 - jnp.cos(angle)) * (skew @ skew)
        return TransformSpec(
            name="rotation",
            matrix=matrix,
            translation=jnp.zeros((3,), dtype=jnp.float32),
        )

    def reflection(self) -> TransformSpec:
        """
        Build one reflection transform across yz-plane.

        Returns:
        - TransformSpec with reflection matrix and zero translation.
        """

        matrix = jnp.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        return TransformSpec(
            name="reflection",
            matrix=matrix,
            translation=jnp.zeros((3,), dtype=jnp.float32),
        )

    def apply(self, batch: UnifiedBatch, spec: TransformSpec) -> UnifiedBatch:
        """
        Apply transform to atom coordinates in one batch.

        Arguments:
        - batch: UnifiedBatch with one or more graphs.
        - spec: TransformSpec instance.

        Returns:
        - New UnifiedBatch with transformed atom positions.
        """

        return UnifiedBatch(
            atom_graph=batch.atom_graph,
            rumer_graph=batch.rumer_graph,
            active_rumer_graph=batch.active_rumer_graph,
            orbital_atom_index=batch.orbital_atom_index,
            orbital_role=batch.orbital_role,
            active_slot_index=batch.active_slot_index,
            active_orbital_index=batch.active_orbital_index,
            lap_evals=batch.lap_evals,
            lap_evecs=batch.lap_evecs,
            expanded_atom_to_static_atom_index=batch.expanded_atom_to_static_atom_index,
            static_atom_numbers=batch.static_atom_numbers,
            static_atom_positions=(
                batch.static_atom_positions @ spec.matrix.T + spec.translation[None, :]
            ),
            num_atoms_per_graph=batch.num_atoms_per_graph,
            num_orbitals_per_graph=batch.num_orbitals_per_graph,
            num_structures_per_molecule=batch.num_structures_per_molecule,
            dataset_index_per_molecule=batch.dataset_index_per_molecule,
            num_molecules_in_batch=batch.num_molecules_in_batch,
            local_frame_e1=batch.local_frame_e1 @ spec.matrix.T,
            local_frame_e2=batch.local_frame_e2 @ spec.matrix.T,
            local_frame_e3=batch.local_frame_e3 @ spec.matrix.T,
            top_mass_focus_mask=batch.top_mass_focus_mask,
            targets=batch.targets,
            sample_mask=batch.sample_mask,
        )


class EquivarianceError:
    """
    Compute absolute and relative errors for property checks.
    """

    def absolute(self, reference: jnp.ndarray, current: jnp.ndarray) -> float:
        """
        Compute max absolute error.

        Arguments:
        - reference: Reference tensor.
        - current: Current tensor.

        Returns:
        - Maximum absolute error.
        """

        return float(jnp.max(jnp.abs(reference - current)))

    def relative(self, reference: jnp.ndarray, current: jnp.ndarray) -> float:
        """
        Compute max relative error.

        Arguments:
        - reference: Reference tensor.
        - current: Current tensor.

        Returns:
        - Maximum relative error.
        """

        denominator = jnp.maximum(jnp.abs(reference), 1e-12)
        ratio = jnp.abs(reference - current) / denominator
        return float(jnp.max(ratio))


class InvarianceTestRunner:
    """
    Run E(3)/O(3) numerical checks for one sample.
    """

    def __init__(
        self,
        config_path: str,
        split: str,
        sample_index: int,
        seed: int,
        include_reflection: bool,
        checkpoint_path: str | None,
    ):
        """
        Initialize test runner.

        Arguments:
        - config_path: YAML config path.
        - split: Split name in {train, val, test}.
        - sample_index: Sample index within the selected split.
        - seed: Seed for deterministic rotation transform.
        - include_reflection: Whether to run reflection test.
        """

        self.config_path = config_path
        self.split = split
        self.sample_index = sample_index
        self.seed = seed
        self.include_reflection = include_reflection
        self.checkpoint_path = checkpoint_path
        self.error = EquivarianceError()
        self.transforms = GeometryTransformSuite()

    def loadConfig(self):
        """
        Load YAML config and build typed config object.

        Returns:
        - ExperimentConfig instance.
        """

        with open(self.config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return ConfigFactory(payload).createExperimentConfig()

    def loadSingleBatch(self, experiment_config) -> UnifiedBatch:
        """
        Build a single-sample batch from configured split.

        Arguments:
        - experiment_config: Parsed experiment config object.

        Returns:
        - UnifiedBatch with batch size 1.
        """

        processor = UnifiedSampleProcessor(experiment_config.data)
        split_samples = processor.createSplitSamples()
        sample = split_samples[self.split][self.sample_index]
        adapter = GraphPackingAdapter(lap_pe_k=experiment_config.model.atom.lap_pe_k)
        return adapter.packBatch(
            samples=[sample],
            orbital_feature_dim=experiment_config.model.orbital.orbital_feature_dim,
        )

    def buildModel(self, experiment_config) -> EndToEndE3VBModel:
        """
        Build end-to-end model with random initialization.

        Arguments:
        - experiment_config: Parsed experiment config object.

        Returns:
        - Initialized EndToEndE3VBModel instance.
        """

        rngs = nnx.Rngs(experiment_config.training.seed)
        model = EndToEndE3VBModel(config=experiment_config.model, rngs=rngs)
        if self.checkpoint_path is not None:
            NNXCheckpointManager().load(model=model, path=self.checkpoint_path)
        return model

    def expandedAtomInputs(self, batch: UnifiedBatch) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build expanded atom-number and position inputs exactly as main forward does.
        """

        expanded_atom_numbers = batch.static_atom_numbers[batch.expanded_atom_to_static_atom_index]
        expanded_atom_positions = batch.static_atom_positions[batch.expanded_atom_to_static_atom_index]
        return expanded_atom_numbers, expanded_atom_positions

    def transformVector(self, vector_feature: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Apply 3x3 transform matrix to vector channels.

        Arguments:
        - vector_feature: Tensor shape [num_nodes, vector_dim, 3].
        - matrix: Transform matrix shape [3, 3].

        Returns:
        - Transformed tensor shape [num_nodes, vector_dim, 3].
        """

        return jnp.einsum("ij,nvj->nvi", matrix, vector_feature)

    def reportTransform(
        self,
        model: EndToEndE3VBModel,
        original_batch: UnifiedBatch,
        spec: TransformSpec,
    ) -> None:
        """
        Evaluate one transform and print invariance/equivariance report.

        Arguments:
        - model: End-to-end model.
        - original_batch: Original single-sample batch.
        - spec: TransformSpec to evaluate.

        Returns:
        - None.
        """

        transformed_batch = self.transforms.apply(original_batch, spec)
        original_atom_numbers, original_positions = self.expandedAtomInputs(original_batch)
        transformed_atom_numbers, transformed_positions = self.expandedAtomInputs(transformed_batch)
        original_atom = model.atom_encoder(
            original_batch.atom_graph,
            atom_number=original_atom_numbers,
            positions=original_positions,
            lap_evecs=original_batch.lap_evecs,
            lap_evals=original_batch.lap_evals,
        )
        transformed_atom = model.atom_encoder(
            transformed_batch.atom_graph,
            atom_number=transformed_atom_numbers,
            positions=transformed_positions,
            lap_evecs=transformed_batch.lap_evecs,
            lap_evals=transformed_batch.lap_evals,
        )
        original_prediction = model(original_batch)
        transformed_prediction = model(transformed_batch)

        expected_scalar = original_atom["scalar"]
        expected_vector = self.transformVector(original_atom["vector"], spec.matrix)

        scalar_abs = self.error.absolute(expected_scalar, transformed_atom["scalar"])
        scalar_rel = self.error.relative(expected_scalar, transformed_atom["scalar"])
        vector_abs = self.error.absolute(expected_vector, transformed_atom["vector"])
        vector_rel = self.error.relative(expected_vector, transformed_atom["vector"])

        original_value = float(np.asarray(original_prediction)[0])
        transformed_value = float(np.asarray(transformed_prediction)[0])
        output_abs = abs(original_value - transformed_value)
        output_rel = output_abs / max(abs(original_value), 1e-12)

        print("=" * 78)
        print(f"Transform: {spec.name}")
        print("Intermediate Equivariance Checks (Atom Encoder)")
        print(f"  scalar_invariance_abs_error: {scalar_abs:.8e}")
        print(f"  scalar_invariance_rel_error: {scalar_rel:.8e}")
        print(f"  vector_equivariance_abs_error: {vector_abs:.8e}")
        print(f"  vector_equivariance_rel_error: {vector_rel:.8e}")
        print("Final Scalar Output Invariance Check")
        print(f"  prediction_original:    {original_value:.10f}")
        print(f"  prediction_transformed: {transformed_value:.10f}")
        print(f"  prediction_abs_error:   {output_abs:.8e}")
        print(f"  prediction_rel_error:   {output_rel:.8e}")

    def run(self) -> None:
        """
        Execute full E(3)/O(3) numerical checks.

        Returns:
        - None.
        """

        experiment_config = self.loadConfig()
        batch = self.loadSingleBatch(experiment_config)
        model = self.buildModel(experiment_config)

        test_specs = [
            self.transforms.translation(),
            self.transforms.rotation(seed=self.seed),
        ]
        if self.include_reflection:
            test_specs.append(self.transforms.reflection())

        print("E(3)/O(3) Numerical Property Test")
        print(f"Config: {self.config_path}")
        print(f"Split: {self.split}, Sample Index: {self.sample_index}")
        print(f"Reflection Enabled: {self.include_reflection}")
        print(f"Checkpoint: {self.checkpoint_path}")

        for spec in test_specs:
            self.reportTransform(model=model, original_batch=batch, spec=spec)


def main() -> None:
    """
    Parse CLI arguments and run E(3)/O(3) test suite.

    Returns:
    - None.
    """

    parser = argparse.ArgumentParser(description="E(3)/O(3) numerical checks for E3VB")
    parser.add_argument(
        "--config",
        type=str,
        default="config_e2e.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split for sample selection.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index in selected split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for rotation transform.",
    )
    parser.add_argument(
        "--disable_reflection",
        action="store_true",
        help="Disable reflection test.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional trained checkpoint path. When omitted, test a randomly initialized model.",
    )
    args = parser.parse_args()

    runner = InvarianceTestRunner(
        config_path=args.config,
        split=args.split,
        sample_index=args.sample_index,
        seed=args.seed,
        include_reflection=not args.disable_reflection,
        checkpoint_path=args.checkpoint,
    )
    runner.run()


if __name__ == "__main__":
    main()
