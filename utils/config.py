"""Configuration objects for the JAX end-to-end E3VB training pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from data.processor import UnifiedDataConfig
from model.atom_encoder import AtomEncoderConfig
from model.end_to_end import EndToEndModelConfig
from model.orbital_projection import OrbitalProjectionConfig
from model.rumer_encoder import RumerEncoderConfig
from model.signnet import SignNetConfig


@dataclass
class TrainingConfig:
    """
    Configure optimization and experiment runtime settings.

    Arguments:
    - seed: Global random seed.
    - batch_size: Number of molecules per batch.
    - epochs: Number of epochs.
    - learning_rate: Initial AdamW learning rate.
    - min_learning_rate: Final cosine-decay learning rate floor.
    - weight_decay: AdamW weight decay.
    - early_stopping_patience: Number of post-warmup epochs allowed without val improvement.
    - early_stopping_start_epoch: First epoch index where early stopping becomes active.
    - bucketed_batching: Whether to bucket training molecule groups by size.
    - bucket_key: Size key used for training molecule buckets.
    - train_drop_remainder: Whether to drop the final short training batch.
    - slot_diversity_weight: Weight applied to slot-collapse regularization during training.
    - checkpoint_path: File path for best checkpoint.
    - log_path: Optional file path for training log output.
    """

    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    early_stopping_start_epoch: int
    bucketed_batching: bool
    bucket_key: str
    train_drop_remainder: bool
    slot_diversity_weight: float
    target_weight_power: float
    target_weight_offset: float
    rank_loss_weight: float
    rank_loss_margin: float
    rank_loss_min_delta: float
    rank_pair_power: float
    checkpoint_path: str
    log_path: Optional[str]


@dataclass
class ExperimentConfig:
    """
    Bundle data, model, and training configs.

    Arguments:
    - data: UnifiedDataConfig.
    - model: EndToEndModelConfig.
    - training: TrainingConfig.
    """

    data: UnifiedDataConfig
    model: EndToEndModelConfig
    training: TrainingConfig


class ConfigFactory:
    """
    Build typed config objects from YAML dictionary payload.
    """

    def __init__(self, payload: Dict[str, Any]):
        """
        Initialize config factory from parsed YAML payload.

        Arguments:
        - payload: Parsed YAML dictionary.
        """

        self.payload = payload

    def createDataConfig(self) -> UnifiedDataConfig:
        """
        Build unified data config.

        Returns:
        - UnifiedDataConfig object.
        """

        data_payload = self.payload["data"]
        return UnifiedDataConfig(
            xmo_dir=str(data_payload["xmo_dir"]),
            seed=int(data_payload["seed"]),
            split=tuple(float(x) for x in data_payload["split"]),
            processed_cache_path=(
                None
                if data_payload.get("processed_cache_path") is None
                else str(data_payload["processed_cache_path"])
            ),
            packed_cache_path=(
                None
                if data_payload.get("packed_cache_path") is None
                else str(data_payload["packed_cache_path"])
            ),
            max_structures_per_chunk=(
                None
                if data_payload.get("max_structures_per_chunk") is None
                else int(data_payload["max_structures_per_chunk"])
            ),
            lap_pe_k=int(self.payload["model"]["atom"].get("lap_pe_k", 0)),
        )

    def createAtomConfig(self) -> AtomEncoderConfig:
        """
        Build atom encoder config.

        Returns:
        - AtomEncoderConfig object.
        """

        atom_payload = self.payload["model"]["atom"]
        signnet_payload = atom_payload.get("signnet")
        return AtomEncoderConfig(
            input_feature_dim=int(atom_payload["input_feature_dim"]),
            scalar_dim=int(atom_payload["scalar_dim"]),
            vector_dim=int(atom_payload["vector_dim"]),
            radial_dim=int(atom_payload["radial_dim"]),
            layers=int(atom_payload["layers"]),
            max_atomic_number=int(atom_payload["max_atomic_number"]),
            radial_min=float(atom_payload["radial_min"]),
            radial_max=float(atom_payload["radial_max"]),
            radial_basis=int(atom_payload["radial_basis"]),
            lmax=int(atom_payload.get("lmax", 2)),
            lap_pe_k=int(atom_payload.get("lap_pe_k", 0)),
            signnet=(
                None
                if signnet_payload is None
                else SignNetConfig(
                    phi_hidden=int(signnet_payload["phi_hidden"]),
                    phi_out=int(signnet_payload["phi_out"]),
                    phi_layers=int(signnet_payload["phi_layers"]),
                    rho_hidden=int(signnet_payload["rho_hidden"]),
                    out_dim=int(signnet_payload["out_dim"]),
                    rho_layers=int(signnet_payload["rho_layers"]),
                )
            ),
            edge_content_modulation=bool(atom_payload.get("edge_content_modulation", False)),
            edge_content_hidden_dim=int(atom_payload.get("edge_content_hidden_dim", 0)),
            edge_content_mode=str(atom_payload.get("edge_content_mode", "legacy")),
            edge_content_residual_scale=float(
                atom_payload.get("edge_content_residual_scale", 0.1)
            ),
        )

    def createOrbitalConfig(self) -> OrbitalProjectionConfig:
        """
        Build orbital projection config.

        Returns:
        - OrbitalProjectionConfig object.
        """

        orbital_payload = self.payload["model"]["orbital"]
        return OrbitalProjectionConfig(
            scalar_dim=int(orbital_payload["scalar_dim"]),
            vector_dim=int(orbital_payload["vector_dim"]),
            slot_dim=int(orbital_payload["slot_dim"]),
            slot_query_dim=int(orbital_payload["slot_query_dim"]),
            slot_embedding_dim=int(orbital_payload["slot_embedding_dim"]),
            max_active_slots=int(orbital_payload["max_active_slots"]),
            orbital_feature_dim=int(orbital_payload["orbital_feature_dim"]),
        )

    def createRumerConfig(self) -> RumerEncoderConfig:
        """
        Build Rumer encoder config.

        Returns:
        - RumerEncoderConfig object.
        """

        rumer_payload = self.payload["model"]["rumer"]
        mode = str(rumer_payload.get("mode", "full"))
        valid_modes = {
            "full",
            "active_only_low_risk",
        }
        if mode not in valid_modes:
            raise ValueError(
                "model.rumer.mode must be one of "
                f"{sorted(valid_modes)}, got {mode}."
            )
        return RumerEncoderConfig(
            orbital_feature_dim=int(rumer_payload["orbital_feature_dim"]),
            hidden_dim=int(rumer_payload["hidden_dim"]),
            edge_embedding_dim=int(rumer_payload["edge_embedding_dim"]),
            layers=int(rumer_payload["layers"]),
            mode=mode,
        )

    def createModelConfig(self) -> EndToEndModelConfig:
        """
        Build full end-to-end model config.

        Returns:
        - EndToEndModelConfig object.
        """

        atom_config = self.createAtomConfig()
        orbital_config = self.createOrbitalConfig()
        rumer_config = self.createRumerConfig()
        return EndToEndModelConfig(
            atom=atom_config,
            orbital=orbital_config,
            rumer=rumer_config,
        )

    def createTrainingConfig(self) -> TrainingConfig:
        """
        Build training config.

        Returns:
        - TrainingConfig object.
        """

        training_payload = self.payload["training"]
        bucket_key = str(training_payload.get("bucket_key", "num_atoms_num_orbitals"))
        valid_bucket_keys = {
            "num_atoms",
            "num_atoms_num_orbitals",
            "(num_atoms,num_orbitals)",
        }
        if bucket_key not in valid_bucket_keys:
            raise ValueError(
                "training.bucket_key must be one of "
                f"{sorted(valid_bucket_keys)}, got {bucket_key}."
            )
        batch_size = int(training_payload["batch_size"])
        if batch_size < 2:
            raise ValueError(
                "training.batch_size must be at least 2 so molecule mini-batching "
                "does not degenerate to single-molecule updates."
            )
        return TrainingConfig(
            seed=int(training_payload["seed"]),
            batch_size=batch_size,
            epochs=int(training_payload["epochs"]),
            learning_rate=float(training_payload["learning_rate"]),
            min_learning_rate=float(training_payload.get("min_learning_rate", 1.0e-6)),
            weight_decay=float(training_payload["weight_decay"]),
            early_stopping_patience=int(training_payload.get("early_stopping_patience", 0)),
            early_stopping_start_epoch=int(training_payload.get("early_stopping_start_epoch", 0)),
            bucketed_batching=bool(training_payload.get("bucketed_batching", True)),
            bucket_key=bucket_key,
            train_drop_remainder=bool(training_payload.get("train_drop_remainder", True)),
            slot_diversity_weight=float(training_payload.get("slot_diversity_weight", 0.0)),
            target_weight_power=float(training_payload.get("target_weight_power", 0.0)),
            target_weight_offset=float(training_payload.get("target_weight_offset", 0.0)),
            rank_loss_weight=float(training_payload.get("rank_loss_weight", 0.0)),
            rank_loss_margin=float(training_payload.get("rank_loss_margin", 0.0)),
            rank_loss_min_delta=float(training_payload.get("rank_loss_min_delta", 0.0)),
            rank_pair_power=float(training_payload.get("rank_pair_power", 1.0)),
            checkpoint_path=str(training_payload["checkpoint_path"]),
            log_path=(
                None
                if training_payload.get("log_path") is None
                else str(training_payload["log_path"])
            ),
        )

    def createExperimentConfig(self) -> ExperimentConfig:
        """
        Build full experiment config.

        Returns:
        - ExperimentConfig object.
        """

        return ExperimentConfig(
            data=self.createDataConfig(),
            model=self.createModelConfig(),
            training=self.createTrainingConfig(),
        )
