"""Configuration objects for the JAX end-to-end E3VB training pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from data.processor import UnifiedDataConfig
from data.schema import DatasetSourceConfig
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
    - fixed_bucket_batching: Whether to pad training batches to coarse fixed buckets.
    - fixed_bucket_graph_step: Graph-count bucket size.
    - fixed_bucket_static_atom_step: Static-atom bucket size.
    - fixed_bucket_atom_step: Expanded atom-node bucket size.
    - fixed_bucket_atom_edge_step: Atom-edge bucket size.
    - fixed_bucket_orbital_step: Full-orbital bucket size.
    - fixed_bucket_rumer_edge_step: Full-Rumer-edge bucket size.
    - fixed_bucket_active_orbital_step: Active-orbital bucket size.
    - fixed_bucket_active_edge_step: Active-Rumer-edge bucket size.
    - eval_interval_epochs: Run validation every N epochs.
    - test_on_best_only: Whether to run test only when validation improves.
    - batch_log_interval: Emit split-batch logs every N batches.
    - iterator_log_interval: Emit iterator-stage logs every N batches.
    - slot_diversity_weight: Weight applied to slot-collapse regularization during training.
    - use_top_mass_objective: Whether to optimize the high-near-degeneracy top-mass task.
    - focus_cumulative_mass: Per-molecule cumulative target mass used to define the focus set.
    - validation_monitor: Validation metric used for checkpointing / early stopping.
    - focus_monitor_pair_acc_weight: Weight for the focus-set pair-accuracy term in the
      composite top-mass monitor.
    - focus_monitor_spearman_weight: Weight for the focus-set Spearman term in the
      composite top-mass monitor.
    - focus_monitor_recall_weight: Weight for the focus-set recall term in the
      composite top-mass monitor.
    - focus_monitor_precision_weight: Weight for the focus-set precision term in the
      composite top-mass monitor.
    - focus_monitor_tail_fpr_weight: Weight for the tail false-positive term in the
      composite top-mass monitor.
    - top_mass_regression_weight: Weight applied to focus-set regression.
    - top_mass_ranking_weight: Weight applied to focus-set and focus-vs-tail ranking.
    - tail_suppression_weight: Weight applied to low-priority tail regression suppression.
    - top_mass_sample_strategy: Runtime per-molecule structure sampling policy for top-mass training.
    - mixed_tail_top_fraction: Fraction of sampled tail slots reserved for the
      hardest tail examples when using ``focus_plus_mixed_tail``.
    - max_tail_samples_per_molecule: Optional cap on sampled tail structures during top-mass training.
    - molecule_balanced_sampling: Whether to batch runtime data by molecule instead of packed chunk.
    - dataset_sampling_strategy: Dataset-level training sampler. ``natural`` keeps
      dataset frequency proportional to molecule count while ``balanced`` interleaves
      batches across datasets.
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
    fixed_bucket_batching: bool
    fixed_bucket_graph_step: int
    fixed_bucket_static_atom_step: int
    fixed_bucket_atom_step: int
    fixed_bucket_atom_edge_step: int
    fixed_bucket_orbital_step: int
    fixed_bucket_rumer_edge_step: int
    fixed_bucket_active_orbital_step: int
    fixed_bucket_active_edge_step: int
    eval_interval_epochs: int
    test_on_best_only: bool
    batch_log_interval: int
    iterator_log_interval: int
    slot_diversity_weight: float
    target_weight_power: float
    target_weight_offset: float
    rank_loss_weight: float
    rank_loss_margin: float
    rank_loss_min_delta: float
    rank_pair_power: float
    use_top_mass_objective: bool
    focus_cumulative_mass: float
    validation_monitor: str
    focus_monitor_pair_acc_weight: float
    focus_monitor_spearman_weight: float
    focus_monitor_recall_weight: float
    focus_monitor_precision_weight: float
    focus_monitor_tail_fpr_weight: float
    top_mass_regression_weight: float
    top_mass_ranking_weight: float
    tail_suppression_weight: float
    top_mass_sample_strategy: str
    mixed_tail_top_fraction: float
    max_tail_samples_per_molecule: Optional[int]
    molecule_balanced_sampling: bool
    dataset_sampling_strategy: str
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
        datasets_payload = data_payload.get("datasets")
        datasets = None
        xmo_dir = data_payload.get("xmo_dir")
        if datasets_payload is not None:
            datasets = tuple(
                DatasetSourceConfig(
                    dataset_id=str(entry["dataset_id"]),
                    xmo_dir=str(entry["xmo_dir"]),
                    split=(
                        None
                        if entry.get("split") is None
                        else tuple(float(x) for x in entry["split"])
                    ),
                )
                for entry in datasets_payload
            )
            if len(datasets) == 0:
                raise ValueError("data.datasets must contain at least one dataset entry.")
            dataset_ids = [dataset.dataset_id for dataset in datasets]
            if len(set(dataset_ids)) != len(dataset_ids):
                raise ValueError(f"data.datasets contains duplicate dataset_id values: {dataset_ids}")
        elif xmo_dir is None:
            raise ValueError("data must define either xmo_dir or datasets.")
        return UnifiedDataConfig(
            xmo_dir=None if xmo_dir is None else str(xmo_dir),
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
            datasets=datasets,
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
        use_top_mass_objective = bool(training_payload.get("use_top_mass_objective", False))
        validation_monitor = str(
            training_payload.get(
                "validation_monitor",
                "focus_weighted_mae" if use_top_mass_objective else "mae",
            )
        )
        valid_validation_monitors = {
            "mae",
            "macro_mae",
            "focus_weighted_mae",
            "focus_priority_score",
            "macro_focus_weighted_mae",
            "macro_focus_priority_score",
        }
        if validation_monitor not in valid_validation_monitors:
            raise ValueError(
                "training.validation_monitor must be one of "
                f"{sorted(valid_validation_monitors)}, got {validation_monitor}."
            )
        if (validation_monitor != "mae") and (not use_top_mass_objective):
            if validation_monitor != "macro_mae":
                raise ValueError(
                    "training.validation_monitor can only use focus-* metrics when "
                    "training.use_top_mass_objective is true."
                )
        dataset_sampling_strategy = str(training_payload.get("dataset_sampling_strategy", "natural"))
        valid_dataset_sampling_strategies = {
            "natural",
            "balanced",
        }
        if dataset_sampling_strategy not in valid_dataset_sampling_strategies:
            raise ValueError(
                "training.dataset_sampling_strategy must be one of "
                f"{sorted(valid_dataset_sampling_strategies)}, got {dataset_sampling_strategy}."
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
            fixed_bucket_batching=bool(training_payload.get("fixed_bucket_batching", False)),
            fixed_bucket_graph_step=int(training_payload.get("fixed_bucket_graph_step", 16)),
            fixed_bucket_static_atom_step=int(training_payload.get("fixed_bucket_static_atom_step", 32)),
            fixed_bucket_atom_step=int(training_payload.get("fixed_bucket_atom_step", 256)),
            fixed_bucket_atom_edge_step=int(training_payload.get("fixed_bucket_atom_edge_step", 2048)),
            fixed_bucket_orbital_step=int(training_payload.get("fixed_bucket_orbital_step", 512)),
            fixed_bucket_rumer_edge_step=int(training_payload.get("fixed_bucket_rumer_edge_step", 4096)),
            fixed_bucket_active_orbital_step=int(training_payload.get("fixed_bucket_active_orbital_step", 128)),
            fixed_bucket_active_edge_step=int(training_payload.get("fixed_bucket_active_edge_step", 1024)),
            eval_interval_epochs=max(1, int(training_payload.get("eval_interval_epochs", 1))),
            test_on_best_only=bool(training_payload.get("test_on_best_only", False)),
            batch_log_interval=max(1, int(training_payload.get("batch_log_interval", 1))),
            iterator_log_interval=max(1, int(training_payload.get("iterator_log_interval", 1))),
            slot_diversity_weight=float(training_payload.get("slot_diversity_weight", 0.0)),
            target_weight_power=float(training_payload.get("target_weight_power", 0.0)),
            target_weight_offset=float(training_payload.get("target_weight_offset", 0.0)),
            rank_loss_weight=float(training_payload.get("rank_loss_weight", 0.0)),
            rank_loss_margin=float(training_payload.get("rank_loss_margin", 0.0)),
            rank_loss_min_delta=float(training_payload.get("rank_loss_min_delta", 0.0)),
            rank_pair_power=float(training_payload.get("rank_pair_power", 1.0)),
            use_top_mass_objective=use_top_mass_objective,
            focus_cumulative_mass=float(training_payload.get("focus_cumulative_mass", 0.98)),
            validation_monitor=validation_monitor,
            focus_monitor_pair_acc_weight=float(
                training_payload.get("focus_monitor_pair_acc_weight", 0.0)
            ),
            focus_monitor_spearman_weight=float(
                training_payload.get("focus_monitor_spearman_weight", 0.0)
            ),
            focus_monitor_recall_weight=float(
                training_payload.get("focus_monitor_recall_weight", 0.0)
            ),
            focus_monitor_precision_weight=float(
                training_payload.get("focus_monitor_precision_weight", 0.0)
            ),
            focus_monitor_tail_fpr_weight=float(
                training_payload.get("focus_monitor_tail_fpr_weight", 0.0)
            ),
            top_mass_regression_weight=float(training_payload.get("top_mass_regression_weight", 1.0)),
            top_mass_ranking_weight=float(training_payload.get("top_mass_ranking_weight", 0.0)),
            tail_suppression_weight=float(training_payload.get("tail_suppression_weight", 0.0)),
            top_mass_sample_strategy=str(
                training_payload.get("top_mass_sample_strategy", "focus_plus_random_tail")
            ),
            mixed_tail_top_fraction=float(training_payload.get("mixed_tail_top_fraction", 0.5)),
            max_tail_samples_per_molecule=(
                None
                if training_payload.get("max_tail_samples_per_molecule") is None
                else int(training_payload["max_tail_samples_per_molecule"])
            ),
            molecule_balanced_sampling=bool(training_payload.get("molecule_balanced_sampling", False)),
            dataset_sampling_strategy=dataset_sampling_strategy,
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
