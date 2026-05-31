"""Main training entry for end-to-end JAX + Flax NNX E3VB."""

import argparse
from datetime import datetime
from functools import partial
import os
from pathlib import Path
from typing import Dict, List

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from data import GrainPipeline, GraphPackingAdapter, UnifiedSampleProcessor
from model.end_to_end import EndToEndE3VBModel
from utils import (
    ConfigFactory,
    NNXCheckpointManager,
    RegressionMetrics,
    pairwiseRankLoss,
    topMassObjectiveLoss,
    weightedMae,
)


def validStructureCounts(
    num_structures_per_molecule: np.ndarray,
    sample_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert padded per-molecule structure counts into real counts using sample_mask.
    """

    counts = np.asarray(num_structures_per_molecule, dtype=np.int32)
    valid_mask = np.asarray(sample_mask, dtype=np.float32) > 0.5
    valid_counts = []
    offset = 0
    for count in counts.tolist():
        count = int(count)
        if count <= 0:
            continue
        valid_count = int(np.sum(valid_mask[offset : offset + count]))
        if valid_count > 0:
            valid_counts.append(valid_count)
        offset += count
    return np.asarray(valid_counts, dtype=np.int32)


def validDatasetIndices(
    num_structures_per_molecule: np.ndarray,
    dataset_index_per_molecule: np.ndarray,
) -> np.ndarray:
    """
    Filter padded molecule slots from one dataset-index vector.
    """

    counts = np.asarray(num_structures_per_molecule, dtype=np.int32)
    dataset_index = np.asarray(dataset_index_per_molecule, dtype=np.int32)
    keep_mask = counts > 0
    return dataset_index[keep_mask]


@nnx.jit
def trainStep(
    model,
    optimizer,
    batch,
    slot_diversity_weight,
    target_weight_power,
    target_weight_offset,
    rank_loss_weight,
    rank_loss_margin,
    rank_loss_min_delta,
    rank_pair_power,
):
    """
    Jitted training step for one packed batch.
    """

    def lossClosure(model):
        prediction, auxiliary = model.forwardWithAux(batch)
        absolute_error = jnp.abs(prediction - batch.targets)
        mae = jnp.sum(batch.sample_mask * absolute_error) / jnp.maximum(jnp.sum(batch.sample_mask), 1.0)
        regression_loss, _ = weightedMae(
            prediction=prediction,
            target=batch.targets,
            power=target_weight_power,
            offset=target_weight_offset,
            sample_mask=batch.sample_mask,
        )
        rank_loss = pairwiseRankLoss(
            prediction=prediction,
            target=batch.targets,
            num_structures_per_molecule=batch.num_structures_per_molecule,
            margin=rank_loss_margin,
            min_delta=rank_loss_min_delta,
            pair_power=rank_pair_power,
            sample_mask=batch.sample_mask,
        )
        slot_diversity_penalty = auxiliary["slot_diversity_penalty"]
        loss = (
            regression_loss
            + rank_loss_weight * rank_loss
            + slot_diversity_weight * slot_diversity_penalty
        )
        return loss, {
            "prediction": prediction,
            "mae": mae,
            "regression_loss": regression_loss,
            "rank_loss": rank_loss,
            "slot_diversity_penalty": slot_diversity_penalty,
        }

    (loss, auxiliary), gradient = nnx.value_and_grad(lossClosure, has_aux=True)(model)
    optimizer.update(gradient)
    return (
        loss,
        auxiliary["prediction"],
        batch.targets,
        batch.sample_mask,
        batch.top_mass_focus_mask,
        batch.num_structures_per_molecule,
        auxiliary["mae"],
        auxiliary["regression_loss"],
        auxiliary["rank_loss"],
        auxiliary["slot_diversity_penalty"],
    )


@partial(nnx.jit, static_argnums=(12, 13, 14))
def trainStepTopMass(
    model,
    optimizer,
    batch,
    slot_diversity_weight,
    target_weight_power,
    target_weight_offset,
    rank_loss_margin,
    rank_loss_min_delta,
    rank_pair_power,
    top_mass_regression_weight,
    top_mass_ranking_weight,
    tail_suppression_weight,
    max_focus_rank_samples_per_molecule,
    max_tail_rank_samples_per_molecule,
    max_rank_pairs_per_molecule,
):
    """
    Jitted training step for the top-mass objective.
    """

    def lossClosure(model):
        prediction, auxiliary = model.forwardWithAux(batch)
        absolute_error = jnp.abs(prediction - batch.targets)
        mae = jnp.sum(batch.sample_mask * absolute_error) / jnp.maximum(jnp.sum(batch.sample_mask), 1.0)
        objective_loss, objective_terms = topMassObjectiveLoss(
            prediction=prediction,
            target=batch.targets,
            top_mass_focus_mask=batch.top_mass_focus_mask,
            num_structures_per_molecule=batch.num_structures_per_molecule,
            target_weight_power=target_weight_power,
            target_weight_offset=target_weight_offset,
            top_mass_regression_weight=top_mass_regression_weight,
            top_mass_ranking_weight=top_mass_ranking_weight,
            tail_suppression_weight=tail_suppression_weight,
            rank_loss_margin=rank_loss_margin,
            rank_loss_min_delta=rank_loss_min_delta,
            rank_pair_power=rank_pair_power,
            max_focus_rank_samples_per_molecule=max_focus_rank_samples_per_molecule,
            max_tail_rank_samples_per_molecule=max_tail_rank_samples_per_molecule,
            max_rank_pairs_per_molecule=max_rank_pairs_per_molecule,
            sample_mask=batch.sample_mask,
        )
        slot_diversity_penalty = auxiliary["slot_diversity_penalty"]
        loss = objective_loss + slot_diversity_weight * slot_diversity_penalty
        return loss, {
            "prediction": prediction,
            "mae": mae,
            "regression_loss": objective_terms["regression_loss"],
            "rank_loss": objective_terms["rank_loss"],
            "slot_diversity_penalty": slot_diversity_penalty,
        }

    (loss, auxiliary), gradient = nnx.value_and_grad(lossClosure, has_aux=True)(model)
    optimizer.update(gradient)
    return (
        loss,
        auxiliary["prediction"],
        batch.targets,
        batch.sample_mask,
        batch.top_mass_focus_mask,
        batch.num_structures_per_molecule,
        auxiliary["mae"],
        auxiliary["regression_loss"],
        auxiliary["rank_loss"],
        auxiliary["slot_diversity_penalty"],
    )


@nnx.jit
def evalStep(model, batch):
    """
    Jitted evaluation step for one packed batch.
    """

    prediction = model(batch)
    return (
        prediction,
        batch.targets,
        batch.sample_mask,
        batch.top_mass_focus_mask,
        batch.num_structures_per_molecule,
    )


class ConsoleFileLogger:
    """
    Write identical log lines to stdout and one on-disk log file.
    """

    def __init__(self, path: str, emit_stdout: bool = True):
        """
        Initialize logger and create parent directory.

        Arguments:
        - path: Log file path.
        - emit_stdout: Whether to mirror log lines to stdout.
        """

        self.path = path
        self.emit_stdout = emit_stdout
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "w", encoding="utf-8")

    def log(self, message: str) -> None:
        """
        Emit one log line to stdout and file.

        Arguments:
        - message: Log message.
        """

        if self.emit_stdout:
            print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        """
        Close backing file handle.
        """

        self.handle.close()


class PreprocessRunner:
    """
    Export offline processed molecules and packed chunk graphs.
    """

    def __init__(
        self,
        config_path: str,
        log_path_override: str | None = None,
        quiet_stdout: bool = False,
        force_rebuild: bool = False,
    ):
        self.config_path = config_path
        self.log_path_override = log_path_override
        self.quiet_stdout = quiet_stdout
        self.force_rebuild = force_rebuild
        self.config = None
        self.logger = None
        self.runtime_log_path = None

    def loadConfig(self) -> None:
        """
        Parse YAML file and construct typed config objects.
        """

        with open(self.config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        factory = ConfigFactory(payload)
        self.config = factory.createExperimentConfig()

    def resolveRuntimeLogPath(self) -> str:
        """
        Determine the runtime preprocess log path.
        """

        if self.log_path_override is not None:
            return self.log_path_override

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = Path(self.config_path).stem
        return str(Path(__file__).resolve().parent / "logs" / f"{config_stem}_{timestamp}.preprocess.log")

    def setupLogging(self) -> None:
        """
        Initialize file-backed preprocess logging.
        """

        self.runtime_log_path = self.resolveRuntimeLogPath()
        self.logger = ConsoleFileLogger(
            self.runtime_log_path,
            emit_stdout=not self.quiet_stdout,
        )

    def log(self, message: str) -> None:
        """
        Write one preprocess log line.
        """

        if self.logger is None:
            print(message, flush=True)
            return
        self.logger.log(message)

    def run(self) -> None:
        """
        Export processed and packed offline caches.
        """

        try:
            self.loadConfig()
            self.setupLogging()
            processor = UnifiedSampleProcessor(self.config.data)
            self.log(
                "PREPROCESS "
                f"config={self.config_path} "
                f"log_path={self.runtime_log_path} "
                f"processed_cache={self.config.data.processed_cache_path} "
                f"packed_cache={self.config.data.packed_cache_path} "
                f"force_rebuild={str(self.force_rebuild).lower()}"
            )
            processed_cache = processor.createProcessedDatasetCache(force_rebuild=self.force_rebuild)
            packed_cache = processor.createPackedDatasetCache(force_rebuild=self.force_rebuild)
            self.log(
                "PREPROCESS_DONE "
                f"train_molecules={len(processed_cache.split_ids['train'])} "
                f"train_chunks={len(packed_cache.splitChunks('train'))} "
                f"train_structures={packed_cache.splitStructureCounts()['train']} "
                f"val_molecules={len(processed_cache.split_ids['val'])} "
                f"val_chunks={len(packed_cache.splitChunks('val'))} "
                f"val_structures={packed_cache.splitStructureCounts()['val']} "
                f"test_molecules={len(processed_cache.split_ids['test'])} "
                f"test_chunks={len(packed_cache.splitChunks('test'))} "
                f"test_structures={packed_cache.splitStructureCounts()['test']}"
            )
        finally:
            if self.logger is not None:
                self.logger.close()


class ExperimentRunner:
    """
    Orchestrate end-to-end training, validation, and testing.
    """

    def __init__(
        self,
        config_path: str,
        smoke_mode: bool = False,
        smoke_epochs: int = 1,
        smoke_max_batches: int = 2,
        allow_cpu: bool = False,
        sample_limit_per_split: int | None = None,
        checkpoint_path_override: str | None = None,
        log_path_override: str | None = None,
        quiet_stdout: bool = False,
    ):
        """
        Initialize experiment runner.

        Arguments:
        - config_path: YAML config file path.
        - smoke_mode: Whether to run a low-cost pipeline smoke run.
        - smoke_epochs: Number of epochs in smoke mode.
        - smoke_max_batches: Maximum batches per split in smoke mode.
        - allow_cpu: Whether to bypass the default GPU requirement.
        - sample_limit_per_split: Optional cap applied to each split in molecule units.
        - checkpoint_path_override: Optional runtime checkpoint path override.
        - log_path_override: Optional runtime log path override.
        - quiet_stdout: Whether to suppress mirrored training logs on stdout.
        """

        self.config_path = config_path
        self.smoke_mode = smoke_mode
        self.smoke_epochs = smoke_epochs
        self.smoke_max_batches = smoke_max_batches
        self.allow_cpu = allow_cpu
        self.sample_limit_per_split = sample_limit_per_split
        self.checkpoint_path_override = checkpoint_path_override
        self.log_path_override = log_path_override
        self.quiet_stdout = quiet_stdout
        self.config = None
        self.dataset_ids = None
        self.split_structure_counts = None
        self.split_structure_counts_by_dataset = None
        self.split_molecule_ids_by_dataset = None
        self.split_molecule_ids = None
        self.split_molecule_chunks = None
        self.split_molecule_chunk_groups = None
        self.model = None
        self.optimizer = None
        self.pipeline = None
        self.logger = None
        self.runtime_checkpoint_path = None
        self.runtime_log_path = None
        self.device_info = None
        self.learning_rate_schedule = None
        self.total_train_steps = 0
        self.completed_train_steps = 0
        self.metrics = RegressionMetrics()
        self.checkpoint = NNXCheckpointManager()

    def loadConfig(self) -> None:
        """
        Parse YAML file and construct typed config objects.

        Returns:
        - None.
        """

        with open(self.config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        factory = ConfigFactory(payload)
        self.config = factory.createExperimentConfig()

    def buildDataset(self) -> None:
        """
        Load offline-packed split chunks and build Grain pipeline.

        Returns:
        - None.
        """

        processor = UnifiedSampleProcessor(self.config.data)
        packed_cache = processor.requirePackedDatasetCache()
        packed_view_name = self.config.training.packed_molecule_view
        self.dataset_ids = list(getattr(packed_cache, "dataset_ids", None) or ["default"])
        self.split_molecule_ids_by_dataset = getattr(packed_cache, "split_ids_by_dataset", None)
        self.split_structure_counts_by_dataset = packed_cache.splitStructureCountsByDataset(
            view_name=packed_view_name
        )
        self.split_molecule_ids = {
            split_name: list(packed_cache.split_ids[split_name])
            for split_name in ["train", "val", "test"]
        }
        self.split_molecule_chunks = {
            split_name: packed_cache.splitChunks(split_name, view_name=packed_view_name)
            for split_name in ["train", "val", "test"]
        }
        self.split_structure_counts = packed_cache.splitStructureCounts(view_name=packed_view_name)
        if self.sample_limit_per_split is not None:
            limit = int(self.sample_limit_per_split)
            for split_name in ["train", "val", "test"]:
                allowed_ids = set(self.split_molecule_ids[split_name][:limit])
                self.split_molecule_ids[split_name] = self.split_molecule_ids[split_name][:limit]
                if self.split_molecule_ids_by_dataset is not None:
                    self.split_molecule_ids_by_dataset[split_name] = {
                        dataset_id: [
                            molecule_id
                            for molecule_id in self.split_molecule_ids_by_dataset[split_name].get(dataset_id, [])
                            if molecule_id in allowed_ids
                        ]
                        for dataset_id in self.dataset_ids
                    }
                self.split_molecule_chunks[split_name] = [
                    chunk
                    for chunk in self.split_molecule_chunks[split_name]
                    if chunk.molecule_id in allowed_ids
                ]
                self.split_structure_counts[split_name] = int(
                    sum(self.chunkStructureCount(chunk) for chunk in self.split_molecule_chunks[split_name])
                )
                if self.split_structure_counts_by_dataset is not None:
                    self.split_structure_counts_by_dataset[split_name] = {
                        dataset_id: int(
                            sum(
                                self.chunkStructureCount(chunk)
                                for chunk in self.split_molecule_chunks[split_name]
                                if getattr(chunk, "dataset_id", "default") == dataset_id
                            )
                        )
                        for dataset_id in self.dataset_ids
                    }

        self.log(
            "CACHE "
            f"processed_cache={self.config.data.processed_cache_path} "
            f"packed_cache={self.config.data.packed_cache_path} "
            f"packed_view={packed_view_name}"
        )
        adapter = GraphPackingAdapter(lap_pe_k=self.config.model.atom.lap_pe_k)
        self.pipeline = GrainPipeline(
            adapter=adapter,
            orbital_feature_dim=self.config.model.orbital.orbital_feature_dim,
            dataset_ids=self.dataset_ids,
            iterator_log_interval=self.config.training.iterator_log_interval,
        )
        self.split_molecule_chunk_groups = {}
        for split_name in ["train", "val", "test"]:
            grouped_chunks: dict[str, list] = {molecule_id: [] for molecule_id in self.split_molecule_ids[split_name]}
            for chunk in self.split_molecule_chunks[split_name]:
                grouped_chunks.setdefault(chunk.molecule_id, []).append(chunk)
            self.split_molecule_chunk_groups[split_name] = [
                grouped_chunks[molecule_id]
                for molecule_id in self.split_molecule_ids[split_name]
                if molecule_id in grouped_chunks
            ]

    def buildModel(self) -> None:
        """
        Create end-to-end model and optimizer.

        Returns:
        - None.
        """

        rngs = nnx.Rngs(self.config.training.seed)
        self.model = EndToEndE3VBModel(config=self.config.model, rngs=rngs)
        total_epochs = self.smoke_epochs if self.smoke_mode else self.config.training.epochs
        train_batches_per_epoch = self.splitBatchCount(split_name="train", training=True)
        self.total_train_steps = max(1, int(total_epochs) * int(train_batches_per_epoch))
        self.completed_train_steps = 0
        cosine_alpha = (
            float(self.config.training.min_learning_rate) / float(self.config.training.learning_rate)
        )
        self.learning_rate_schedule = optax.cosine_decay_schedule(
            init_value=self.config.training.learning_rate,
            decay_steps=max(1, self.total_train_steps - 1),
            alpha=cosine_alpha,
        )
        adamw = optax.adamw(
            learning_rate=self.learning_rate_schedule,
            weight_decay=self.config.training.weight_decay,
        )
        if float(self.config.training.gradient_clip_norm) > 0.0:
            transformation = optax.chain(
                optax.clip_by_global_norm(float(self.config.training.gradient_clip_norm)),
                adamw,
            )
        else:
            transformation = adamw
        self.optimizer = nnx.Optimizer(self.model, transformation)

    def resolveRuntimeLogPath(self) -> str:
        """
        Determine the runtime log path.

        Returns:
        - Log file path.
        """

        if self.log_path_override is not None:
            return self.log_path_override
        if self.config.training.log_path is not None:
            return self.config.training.log_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = Path(self.config_path).stem
        return str(Path(__file__).resolve().parent / "logs" / f"{config_stem}_{timestamp}.log")

    def resolveRuntimeCheckpointPath(self) -> str:
        """
        Determine the runtime checkpoint path.

        Returns:
        - Checkpoint file path.
        """

        if self.checkpoint_path_override is not None:
            return self.checkpoint_path_override
        return self.config.training.checkpoint_path

    def setupLogging(self) -> None:
        """
        Initialize file-backed runtime logging.
        """

        self.runtime_log_path = self.resolveRuntimeLogPath()
        self.logger = ConsoleFileLogger(
            self.runtime_log_path,
            emit_stdout=not self.quiet_stdout,
        )

    def log(self, message: str) -> None:
        """
        Write one log line.

        Arguments:
        - message: Log message.
        """

        if self.logger is None:
            print(message, flush=True)
            return
        self.logger.log(message)

    def deviceSummary(self, device) -> str:
        """
        Format one JAX device into one readable token.

        Arguments:
        - device: JAX device object.

        Returns:
        - Compact device string.
        """

        device_kind = getattr(device, "device_kind", str(device))
        return f"{device.platform}:{device_kind}:id={device.id}"

    def inspectTrainingDevice(self) -> None:
        """
        Inspect JAX backend/devices and enforce GPU availability by default.
        """

        devices = jax.devices()
        backend = jax.default_backend()
        gpu_devices = [device for device in devices if device.platform == "gpu"]
        selected_device = gpu_devices[0] if gpu_devices else devices[0]
        self.device_info = {
            "backend": backend,
            "devices": [self.deviceSummary(device) for device in devices],
            "selected": self.deviceSummary(selected_device),
            "gpu_count": len(gpu_devices),
        }

        self.log(
            "DEVICE "
            f"backend={backend} "
            f"detected={self.device_info['devices']} "
            f"selected={self.device_info['selected']} "
            f"require_gpu={str(not self.allow_cpu).lower()}"
        )

        if (not self.allow_cpu) and (len(gpu_devices) == 0):
            raise RuntimeError(
                "GPU is required for training, but JAX did not detect any GPU device. "
                "Use --allow_cpu only for non-formal diagnostics."
            )

    def trainBucketKey(self) -> str | None:
        """
        Return the training bucket key when bucketing is enabled.

        Returns:
        - Bucket key string or None.
        """

        if not self.config.training.bucketed_batching:
            return None
        return self.config.training.bucket_key

    def fixedBucketConfig(self) -> dict[str, int] | None:
        """
        Return the fixed-bucket padding configuration when enabled.
        """

        if not self.config.training.fixed_bucket_batching:
            return None
        return {
            "graph_step": int(self.config.training.fixed_bucket_graph_step),
            "static_atom_step": int(self.config.training.fixed_bucket_static_atom_step),
            "atom_step": int(self.config.training.fixed_bucket_atom_step),
            "atom_edge_step": int(self.config.training.fixed_bucket_atom_edge_step),
            "orbital_step": int(self.config.training.fixed_bucket_orbital_step),
            "rumer_edge_step": int(self.config.training.fixed_bucket_rumer_edge_step),
            "active_orbital_step": int(self.config.training.fixed_bucket_active_orbital_step),
            "active_edge_step": int(self.config.training.fixed_bucket_active_edge_step),
        }

    def usesMoleculeGroupedBatches(self) -> bool:
        """
        Return whether training/eval should batch whole molecules instead of raw chunks.
        """

        return bool(
            self.config.training.use_top_mass_objective
            or self.config.training.molecule_balanced_sampling
        )

    def validationMonitorName(self) -> str:
        """
        Return the validation metric used for checkpointing and early stopping.
        """

        return str(self.config.training.validation_monitor)

    def focusPriorityScore(self, metrics: Dict[str, float]) -> float:
        """
        Compute one composite validation score for the top-mass objective.

        Lower is better. The score keeps focus-set weighted MAE as the anchor
        term and adds penalties when the ranking / retrieval quality degrades.
        """

        focus_weighted_mae = float(metrics.get("focus_weighted_mae", 0.0))
        pair_penalty = 1.0 - float(metrics.get("focus_pair_acc", 0.0))
        spearman_penalty = 1.0 - float(metrics.get("focus_spearman", 0.0))
        recall_penalty = 1.0 - float(metrics.get("focus_recall", 0.0))
        precision_penalty = 1.0 - float(metrics.get("focus_precision", 0.0))
        tail_false_positive = float(metrics.get("tail_false_positive_rate", 0.0))
        return (
            focus_weighted_mae
            + float(self.config.training.focus_monitor_pair_acc_weight) * pair_penalty
            + float(self.config.training.focus_monitor_spearman_weight) * spearman_penalty
            + float(self.config.training.focus_monitor_recall_weight) * recall_penalty
            + float(self.config.training.focus_monitor_precision_weight) * precision_penalty
            + float(self.config.training.focus_monitor_tail_fpr_weight) * tail_false_positive
        )

    def computeLoss(self, batch) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute MAE loss for one batch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - loss: Scalar MAE.
        - prediction: Predicted labels, shape [batch_size].
        """

        prediction = self.model(batch)
        absolute_error = jnp.abs(prediction - batch.targets)
        loss = jnp.sum(batch.sample_mask * absolute_error) / jnp.maximum(jnp.sum(batch.sample_mask), 1.0)
        return loss, prediction

    def runTrainBatch(self, batch) -> Dict[str, np.ndarray]:
        """
        Run one optimization step for one batch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - Dictionary with keys loss, prediction, target.
        """

        if self.config.training.use_top_mass_objective:
            (
                loss,
                prediction,
                target,
                sample_mask,
                top_mass_focus_mask,
                num_structures_per_molecule,
                mae,
                regression_loss,
                rank_loss,
                slot_diversity_penalty,
            ) = trainStepTopMass(
                self.model,
                self.optimizer,
                batch,
                self.config.training.slot_diversity_weight,
                self.config.training.target_weight_power,
                self.config.training.target_weight_offset,
                self.config.training.rank_loss_margin,
                self.config.training.rank_loss_min_delta,
                self.config.training.rank_pair_power,
                self.config.training.top_mass_regression_weight,
                self.config.training.top_mass_ranking_weight,
                self.config.training.tail_suppression_weight,
                self.config.training.max_focus_rank_samples_per_molecule,
                self.config.training.max_tail_rank_samples_per_molecule,
                self.config.training.max_rank_pairs_per_molecule,
            )
        else:
            (
                loss,
                prediction,
                target,
                sample_mask,
                top_mass_focus_mask,
                num_structures_per_molecule,
                mae,
                regression_loss,
                rank_loss,
                slot_diversity_penalty,
            ) = trainStep(
                self.model,
                self.optimizer,
                batch,
                self.config.training.slot_diversity_weight,
                self.config.training.target_weight_power,
                self.config.training.target_weight_offset,
                self.config.training.rank_loss_weight,
                self.config.training.rank_loss_margin,
                self.config.training.rank_loss_min_delta,
                self.config.training.rank_pair_power,
            )
        self.completed_train_steps += 1
        valid_mask = np.asarray(sample_mask) > 0.5
        result = {
            "loss": np.asarray(loss),
            "prediction": np.asarray(prediction)[valid_mask],
            "target": np.asarray(target)[valid_mask],
            "mae": np.asarray(mae),
            "regression_loss": np.asarray(regression_loss),
            "rank_loss": np.asarray(rank_loss),
            "slot_diversity_penalty": np.asarray(slot_diversity_penalty),
            "dataset_index_per_molecule": validDatasetIndices(
                num_structures_per_molecule=np.asarray(num_structures_per_molecule),
                dataset_index_per_molecule=np.asarray(batch.dataset_index_per_molecule),
            ),
            "num_structures_per_molecule": validStructureCounts(
                num_structures_per_molecule=np.asarray(num_structures_per_molecule),
                sample_mask=np.asarray(sample_mask),
            ),
        }
        if self.config.training.use_top_mass_objective:
            result["top_mass_focus_mask"] = np.asarray(top_mass_focus_mask)[valid_mask]
        return result

    def runEvalBatch(self, batch) -> Dict[str, np.ndarray]:
        """
        Run forward-only evaluation for one batch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - Dictionary with keys prediction and target.
        """

        prediction, target, sample_mask, top_mass_focus_mask, num_structures_per_molecule = evalStep(self.model, batch)
        valid_mask = np.asarray(sample_mask) > 0.5
        result = {
            "prediction": np.asarray(prediction)[valid_mask],
            "target": np.asarray(target)[valid_mask],
            "dataset_index_per_molecule": validDatasetIndices(
                num_structures_per_molecule=np.asarray(num_structures_per_molecule),
                dataset_index_per_molecule=np.asarray(batch.dataset_index_per_molecule),
            ),
            "num_structures_per_molecule": validStructureCounts(
                num_structures_per_molecule=np.asarray(num_structures_per_molecule),
                sample_mask=np.asarray(sample_mask),
            ),
        }
        if self.config.training.use_top_mass_objective:
            result["top_mass_focus_mask"] = np.asarray(top_mass_focus_mask)[valid_mask]
        return result

    def splitBatchCount(self, split_name: str, training: bool) -> int:
        """
        Estimate the number of batches for one split.
        """

        batch_size = (
            self.config.training.batch_size
            if training
            else self.config.training.eval_batch_size
        )
        if self.usesMoleculeGroupedBatches():
            grouped_chunks = self.split_molecule_chunk_groups[split_name]
            if training:
                bucket_key = self.trainBucketKey()
                if (
                    self.config.training.max_batch_cost is not None
                    and int(self.config.training.max_batch_cost) > 0
                ):
                    return len(
                        self.pipeline.buildBudgetedGroupedBatchIndices(
                            chunk_groups=grouped_chunks,
                            batch_size=batch_size,
                            shuffle=True,
                            seed=self.config.training.seed,
                            bucket_key=bucket_key,
                            drop_remainder=self.config.training.train_drop_remainder,
                            dataset_sampling_strategy=self.config.training.dataset_sampling_strategy,
                            max_batch_cost=int(self.config.training.max_batch_cost),
                            top_mass_sample_strategy=(
                                self.config.training.top_mass_sample_strategy
                                if self.config.training.use_top_mass_objective
                                else "full_molecule"
                            ),
                            max_tail_samples_per_molecule=(
                                self.config.training.max_tail_samples_per_molecule
                                if self.config.training.use_top_mass_objective
                                else None
                            ),
                            max_structure_cost_per_molecule=(
                                self.config.training.max_structure_cost_per_molecule
                                if self.config.training.use_top_mass_objective
                                else None
                            ),
                        )
                    )
                if self.config.training.dataset_sampling_strategy == "balanced":
                    return len(
                        self.pipeline.buildDatasetBalancedGroupedBatchIndices(
                            chunk_groups=grouped_chunks,
                            batch_size=batch_size,
                            shuffle=True,
                            seed=self.config.training.seed,
                            bucket_key=bucket_key,
                            drop_remainder=self.config.training.train_drop_remainder,
                        )
                    )
                if bucket_key is not None:
                    return len(
                        self.pipeline.buildGroupedBatchIndices(
                            chunk_groups=grouped_chunks,
                            batch_size=batch_size,
                            shuffle=True,
                            seed=self.config.training.seed,
                            bucket_key=bucket_key,
                            drop_remainder=self.config.training.train_drop_remainder,
                        )
                    )
            total_groups = len(grouped_chunks)
            if training and self.config.training.train_drop_remainder:
                return total_groups // batch_size
            return int(np.ceil(total_groups / batch_size))

        if training:
            bucket_key = self.trainBucketKey()
            if bucket_key is not None:
                return len(
                    self.pipeline.buildBucketedBatchIndices(
                        chunks=self.split_molecule_chunks[split_name],
                        batch_size=batch_size,
                        shuffle=True,
                        seed=self.config.training.seed,
                        bucket_key=bucket_key,
                        drop_remainder=self.config.training.train_drop_remainder,
                    )
                )

        total_chunks = len(self.split_molecule_chunks[split_name])
        if training and self.config.training.train_drop_remainder:
            return total_chunks // batch_size
        return int(np.ceil(total_chunks / batch_size))

    def chunkStructureCount(self, chunk) -> int:
        """
        Return the number of VB structures represented by one packed chunk record.
        """

        if hasattr(chunk, "num_structures"):
            return int(chunk.num_structures)
        return int(chunk.atom_n_node.shape[0])

    def collectEpochMetrics(self, batch_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Aggregate batch-level outputs into epoch metrics.

        Arguments:
        - batch_outputs: List of batch output dictionaries.

        Returns:
        - Metric dictionary with keys mae, rmse, spearman.
        """

        prediction = np.concatenate([entry["prediction"] for entry in batch_outputs], axis=0)
        target = np.concatenate([entry["target"] for entry in batch_outputs], axis=0)
        dataset_index_per_molecule = np.concatenate(
            [entry["dataset_index_per_molecule"] for entry in batch_outputs],
            axis=0,
        )
        num_structures_per_molecule = np.concatenate(
            [entry["num_structures_per_molecule"] for entry in batch_outputs],
            axis=0,
        )
        if self.config.training.use_top_mass_objective:
            top_mass_focus_mask = np.concatenate(
                [entry["top_mass_focus_mask"] for entry in batch_outputs],
                axis=0,
            )
            summary = self.metrics.summarize(
                prediction=prediction,
                target=target,
                num_structures_per_molecule=num_structures_per_molecule,
                top_mass_focus_mask=top_mass_focus_mask,
                focus_cumulative_mass=self.config.training.focus_cumulative_mass,
                dataset_index_per_molecule=dataset_index_per_molecule,
                dataset_names=self.dataset_ids,
            )
        else:
            summary = self.metrics.summarize(
                prediction=prediction,
                target=target,
                num_structures_per_molecule=num_structures_per_molecule,
                dataset_index_per_molecule=dataset_index_per_molecule,
                dataset_names=self.dataset_ids,
            )
        if "loss" in batch_outputs[0]:
            summary["loss"] = float(np.mean([float(np.asarray(entry["loss"])) for entry in batch_outputs]))
        else:
            summary["loss"] = summary["mae"]
        if "slot_diversity_penalty" in batch_outputs[0]:
            summary["slot_diversity_penalty"] = float(
                np.mean([float(np.asarray(entry["slot_diversity_penalty"])) for entry in batch_outputs])
            )
        if "regression_loss" in batch_outputs[0]:
            summary["regression_loss"] = float(
                np.mean([float(np.asarray(entry["regression_loss"])) for entry in batch_outputs])
            )
        if "rank_loss" in batch_outputs[0]:
            summary["rank_loss"] = float(
                np.mean([float(np.asarray(entry["rank_loss"])) for entry in batch_outputs])
            )
        if self.config.training.use_top_mass_objective:
            summary["focus_priority_score"] = self.focusPriorityScore(summary)
            per_dataset = summary.get("per_dataset")
            if isinstance(per_dataset, dict) and len(per_dataset) > 0:
                dataset_scores = []
                for dataset_name, dataset_metrics in per_dataset.items():
                    if not isinstance(dataset_metrics, dict):
                        continue
                    dataset_focus_score = self.focusPriorityScore(dataset_metrics)
                    dataset_metrics["focus_priority_score"] = dataset_focus_score
                    dataset_scores.append(float(dataset_focus_score))
                if dataset_scores:
                    summary["macro_focus_priority_score"] = float(np.mean(dataset_scores))
        summary["num_samples"] = int(target.shape[0])
        return summary

    def runSplit(self, split_name: str, epoch: int, training: bool) -> Dict[str, float]:
        """
        Run one full pass over a split.

        Arguments:
        - split_name: One of train, val, test.
        - epoch: Current epoch index.
        - training: Whether this pass updates parameters.

        Returns:
        - Metric dictionary with keys mae, rmse, spearman.
        """

        # Keep the iterator order stable across epochs to avoid triggering
        # fresh JAX/XLA compilations from new bucket/shuffle layouts.
        split_seed = self.config.training.seed
        batch_size = (
            self.config.training.batch_size
            if training
            else self.config.training.eval_batch_size
        )
        if self.usesMoleculeGroupedBatches():
            iterator = self.pipeline.createMoleculeIterator(
                molecule_groups=self.split_molecule_chunk_groups[split_name],
                batch_size=batch_size,
                shuffle=training,
                seed=split_seed,
                drop_remainder=training and self.config.training.train_drop_remainder,
                bucket_key=self.trainBucketKey() if training else None,
                fixed_bucket_config=self.fixedBucketConfig() if training else None,
                split_name=split_name,
                focus_cumulative_mass=self.config.training.focus_cumulative_mass,
                top_mass_sample_strategy=(
                    self.config.training.top_mass_sample_strategy
                    if training and self.config.training.use_top_mass_objective
                    else "full_molecule"
                ),
                mixed_tail_top_fraction=self.config.training.mixed_tail_top_fraction,
                max_tail_samples_per_molecule=(
                    self.config.training.max_tail_samples_per_molecule
                    if training and self.config.training.use_top_mass_objective
                    else None
                ),
                dataset_sampling_strategy=(
                    self.config.training.dataset_sampling_strategy if training else "natural"
                ),
                max_batch_cost=(
                    self.config.training.max_batch_cost
                    if training
                    else None
                ),
                max_structure_cost_per_molecule=(
                    self.config.training.max_structure_cost_per_molecule
                    if training and self.config.training.use_top_mass_objective
                    else None
                ),
            )
        else:
            iterator = self.pipeline.createIterator(
                sample_groups=self.split_molecule_chunks[split_name],
                batch_size=batch_size,
                shuffle=training,
                seed=split_seed,
                drop_remainder=training and self.config.training.train_drop_remainder,
                bucket_key=self.trainBucketKey() if training else None,
                fixed_bucket_config=self.fixedBucketConfig() if training else None,
                split_name=split_name,
            )
        outputs: List[Dict[str, np.ndarray]] = []
        total_batches = self.splitBatchCount(split_name=split_name, training=training)
        if self.smoke_mode:
            total_batches = min(total_batches, self.smoke_max_batches)
        batch_log_interval = int(self.config.training.batch_log_interval)
        for batch_id, batch in enumerate(iterator):
            should_log_batch = (
                batch_id == 0
                or ((batch_id + 1) % batch_log_interval) == 0
                or batch_id == max(total_batches - 1, 0)
            )
            if should_log_batch:
                self.log(
                    "SPLIT_BATCH "
                    f"split={split_name} "
                    f"epoch={epoch:04d} "
                    f"batch={batch_id:05d}/{max(total_batches - 1, 0):05d} "
                    f"phase={'train' if training else 'eval'} "
                    f"status=received"
                )
            if training:
                outputs.append(self.runTrainBatch(batch))
            else:
                outputs.append(self.runEvalBatch(batch))
            if should_log_batch:
                self.log(
                    "SPLIT_BATCH "
                    f"split={split_name} "
                    f"epoch={epoch:04d} "
                    f"batch={batch_id:05d}/{max(total_batches - 1, 0):05d} "
                    f"phase={'train' if training else 'eval'} "
                    f"status=finished"
                )
            if self.smoke_mode and (batch_id + 1) >= self.smoke_max_batches:
                break
        if len(outputs) == 0:
            raise RuntimeError(
                f"Split {split_name} produced zero batches. "
                "If train_drop_remainder is enabled, ensure the split has at least one full molecule batch."
            )
        return self.collectEpochMetrics(outputs)

    def saveBestCheckpoint(self, epoch: int, val_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
        """
        Save best-validation checkpoint.

        Arguments:
        - epoch: Current epoch.
        - val_metrics: Validation metrics dictionary.
        - test_metrics: Test metrics dictionary.

        Returns:
        - None.
        """

        metadata = {
            "epoch": epoch,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "config_path": self.config_path,
            "device_info": self.device_info,
            "log_path": self.runtime_log_path,
        }
        self.checkpoint.save(
            model=self.model,
            path=self.runtime_checkpoint_path,
            metadata=metadata,
        )

    def logEpochMetrics(self, epoch: int, split_name: str, metrics: Dict[str, float]) -> None:
        """
        Emit one stable epoch metric line.

        Arguments:
        - epoch: Epoch index.
        - split_name: Split name.
        - metrics: Metric dictionary.
        """

        if self.learning_rate_schedule is None:
            learning_rate = float(self.config.training.learning_rate)
        else:
            schedule_step = max(0, min(self.completed_train_steps - 1, self.total_train_steps - 1))
            learning_rate = float(self.learning_rate_schedule(schedule_step))
        focus_suffix = ""
        if "focus_weighted_mae" in metrics:
            focus_suffix = (
                f" focus_wmae={metrics['focus_weighted_mae']:.6f} "
                f"focus_pair_acc={metrics.get('focus_pair_acc', 0.0):.6f} "
                f"focus_spearman={metrics.get('focus_spearman', 0.0):.6f} "
                f"focus_recall={metrics.get('focus_recall', 0.0):.6f} "
                f"focus_precision={metrics.get('focus_precision', 0.0):.6f} "
                f"tail_fpr={metrics.get('tail_false_positive_rate', 0.0):.6f}"
            )
            if "focus_priority_score" in metrics:
                focus_suffix += f" focus_score={metrics['focus_priority_score']:.6f}"
            if "macro_focus_priority_score" in metrics:
                focus_suffix += f" macro_focus_score={metrics['macro_focus_priority_score']:.6f}"
        self.log(
            "EPOCH "
            f"epoch={epoch:04d} "
            f"split={split_name} "
            f"loss={metrics['loss']:.6f} "
            f"mae={metrics['mae']:.6f} "
            f"rmse={metrics['rmse']:.6f} "
            f"spearman={metrics['spearman']:.6f} "
            f"molecule_macro_spearman={metrics.get('molecule_macro_spearman', 0.0):.6f} "
            f"reg_loss={metrics.get('regression_loss', metrics['loss']):.6f} "
            f"rank_loss={metrics.get('rank_loss', 0.0):.6f} "
            f"slot_div={metrics.get('slot_diversity_penalty', 0.0):.6f} "
            f"lr={learning_rate:.8e} "
            f"samples={metrics['num_samples']}"
            f"{focus_suffix}"
        )

    def runTrainingLoop(self) -> None:
        """
        Execute training/validation/testing loops across epochs.

        Returns:
        - None.
        """

        monitor_name = self.validationMonitorName()
        best_val_score = float("inf")
        best_epoch = -1
        total_epochs = self.smoke_epochs if self.smoke_mode else self.config.training.epochs
        no_improvement_epochs = 0
        early_stopping_patience = int(self.config.training.early_stopping_patience)
        early_stopping_start_epoch = int(self.config.training.early_stopping_start_epoch)

        self.log(
            "DATA "
            f"train_molecules={len(self.split_molecule_ids['train'])} "
            f"train_chunks={len(self.split_molecule_chunks['train'])} "
            f"train_structures={self.split_structure_counts['train']} "
            f"val_molecules={len(self.split_molecule_ids['val'])} "
            f"val_chunks={len(self.split_molecule_chunks['val'])} "
            f"val_structures={self.split_structure_counts['val']} "
            f"test_molecules={len(self.split_molecule_ids['test'])} "
            f"test_chunks={len(self.split_molecule_chunks['test'])} "
            f"test_structures={self.split_structure_counts['test']}"
        )
        if self.split_molecule_ids_by_dataset is not None:
            for split_name in ["train", "val", "test"]:
                for dataset_id in self.dataset_ids:
                    molecule_count = len(
                        self.split_molecule_ids_by_dataset.get(split_name, {}).get(dataset_id, [])
                    )
                    structure_count = None
                    if self.split_structure_counts_by_dataset is not None:
                        structure_count = self.split_structure_counts_by_dataset.get(split_name, {}).get(dataset_id)
                    self.log(
                        "DATASET "
                        f"split={split_name} "
                        f"dataset_id={dataset_id} "
                        f"molecules={molecule_count} "
                        f"structures={0 if structure_count is None else int(structure_count)}"
                    )
        self.log(
            "BATCHING "
            f"train_molecules_per_batch={self.config.training.batch_size} "
            f"eval_molecules_per_batch={self.config.training.eval_batch_size} "
            f"batch_unit={'molecule_groups' if self.usesMoleculeGroupedBatches() else 'chunks'} "
            f"max_structures_per_chunk={self.config.data.max_structures_per_chunk} "
            f"dataset_sampling={self.config.training.dataset_sampling_strategy} "
            f"max_batch_cost={self.config.training.max_batch_cost} "
            f"max_structure_cost_per_molecule={self.config.training.max_structure_cost_per_molecule} "
            f"train_bucketed={str(self.config.training.bucketed_batching).lower()} "
            f"train_bucket_key={self.trainBucketKey()} "
            f"fixed_bucketed={str(self.config.training.fixed_bucket_batching).lower()} "
            f"train_drop_remainder={str(self.config.training.train_drop_remainder).lower()}"
        )
        if (not self.smoke_mode) and (early_stopping_patience > 0):
            self.log(
                "EARLY_STOPPING "
                f"monitor=val_{monitor_name} "
                f"patience={early_stopping_patience} "
                f"start_epoch={early_stopping_start_epoch}"
            )
        if self.smoke_mode:
            self.log(
                "SMOKE "
                f"epochs={self.smoke_epochs} "
                f"max_batches_per_split={self.smoke_max_batches} "
                f"sample_limit_per_split={self.sample_limit_per_split}"
            )
        self.log(
            "RUNTIME "
            f"eval_interval_epochs={self.config.training.eval_interval_epochs} "
            f"test_on_best_only={str(self.config.training.test_on_best_only).lower()} "
            f"batch_log_interval={self.config.training.batch_log_interval} "
            f"iterator_log_interval={self.config.training.iterator_log_interval} "
            f"max_focus_rank_samples_per_molecule="
            f"{self.config.training.max_focus_rank_samples_per_molecule} "
            f"max_tail_rank_samples_per_molecule="
            f"{self.config.training.max_tail_rank_samples_per_molecule} "
            f"max_rank_pairs_per_molecule={self.config.training.max_rank_pairs_per_molecule}"
        )

        for epoch in range(total_epochs):
            train_metrics = self.runSplit(split_name="train", epoch=epoch, training=True)
            self.logEpochMetrics(epoch=epoch, split_name="train", metrics=train_metrics)
            should_eval = (
                epoch == 0
                or ((epoch + 1) % int(self.config.training.eval_interval_epochs)) == 0
                or epoch == (total_epochs - 1)
            )
            if not should_eval:
                self.log(
                    "EVAL_SKIPPED "
                    f"epoch={epoch:04d} "
                    f"next_eval_in={int(self.config.training.eval_interval_epochs) - ((epoch + 1) % int(self.config.training.eval_interval_epochs))}"
                )
                continue

            val_metrics = self.runSplit(split_name="val", epoch=epoch, training=False)
            self.logEpochMetrics(epoch=epoch, split_name="val", metrics=val_metrics)

            improved = (not self.smoke_mode) and (val_metrics[monitor_name] < best_val_score)
            should_run_test = (not self.config.training.test_on_best_only) or improved or self.smoke_mode
            test_metrics = None
            if should_run_test:
                test_metrics = self.runSplit(split_name="test", epoch=epoch, training=False)
                self.logEpochMetrics(epoch=epoch, split_name="test", metrics=test_metrics)
            else:
                self.log(
                    "TEST_SKIPPED "
                    f"epoch={epoch:04d} "
                    f"reason=val_not_improved"
                )

            if improved:
                best_val_score = float(val_metrics[monitor_name])
                best_epoch = epoch
                if epoch >= early_stopping_start_epoch:
                    no_improvement_epochs = 0
                if test_metrics is None:
                    test_metrics = self.runSplit(split_name="test", epoch=epoch, training=False)
                    self.logEpochMetrics(epoch=epoch, split_name="test", metrics=test_metrics)
                self.saveBestCheckpoint(epoch, val_metrics, test_metrics)
                self.log(
                    "BEST "
                    f"epoch={epoch:04d} "
                    f"val_mae={val_metrics['mae']:.6f} "
                    f"val_{monitor_name}={val_metrics[monitor_name]:.6f} "
                    f"test_mae={test_metrics['mae']:.6f} "
                    f"checkpoint={self.runtime_checkpoint_path}"
                )
            elif (
                (not self.smoke_mode)
                and (early_stopping_patience > 0)
                and (epoch >= early_stopping_start_epoch)
            ):
                no_improvement_epochs += 1
                if no_improvement_epochs >= early_stopping_patience:
                    self.log(
                        "EARLY_STOP "
                        f"epoch={epoch:04d} "
                        f"best_epoch={best_epoch:04d} "
                        f"best_val_{monitor_name}={best_val_score:.6f} "
                        f"patience={early_stopping_patience} "
                        f"start_epoch={early_stopping_start_epoch}"
                    )
                    break

        if self.smoke_mode:
            self.log("SMOKE status=finished")
        else:
            self.log(
                "TRAINING_DONE "
                f"best_epoch={best_epoch:04d} "
                f"best_val_{monitor_name}={best_val_score:.6f} "
                f"checkpoint={self.runtime_checkpoint_path}"
            )

    def run(self) -> None:
        """
        Run full experiment setup and training flow.

        Returns:
        - None.
        """

        try:
            self.loadConfig()
            self.runtime_checkpoint_path = self.resolveRuntimeCheckpointPath()
            self.setupLogging()
            self.log(
                "RUN "
                f"config={self.config_path} "
                f"log_path={self.runtime_log_path} "
                f"checkpoint_path={self.runtime_checkpoint_path}"
            )
            self.inspectTrainingDevice()
            self.buildDataset()
            self.buildModel()
            self.runTrainingLoop()
        finally:
            if self.logger is not None:
                self.logger.close()


def main(
    mode: str,
    config_path: str,
    smoke: bool,
    smoke_epochs: int,
    smoke_max_batches: int,
    allow_cpu: bool,
    sample_limit_per_split: int | None,
    checkpoint_path: str | None,
    log_path: str | None,
    quiet_stdout: bool,
    force_preprocess: bool,
) -> None:
    """
    Launch preprocess or training from one YAML config path.

    Arguments:
    - mode: Either preprocess or train.
    - config_path: YAML config file path.
    - smoke: Whether to run low-cost smoke mode.
    - smoke_epochs: Number of smoke epochs.
    - smoke_max_batches: Maximum batches per split in smoke mode.
    - allow_cpu: Whether to bypass the default GPU requirement.
    - sample_limit_per_split: Optional cap applied to each split in molecule units.
    - checkpoint_path: Optional runtime checkpoint path override.
    - log_path: Optional runtime log path override.
    - quiet_stdout: Whether to suppress mirrored training logs on stdout.
    - force_preprocess: Whether to rebuild offline caches during preprocess.

    Returns:
    - None.
    """

    if mode == "preprocess":
        runner = PreprocessRunner(
            config_path=config_path,
            log_path_override=log_path,
            quiet_stdout=quiet_stdout,
            force_rebuild=force_preprocess,
        )
        runner.run()
        return

    runner = ExperimentRunner(
        config_path=config_path,
        smoke_mode=smoke,
        smoke_epochs=smoke_epochs,
        smoke_max_batches=smoke_max_batches,
        allow_cpu=allow_cpu,
        sample_limit_per_split=sample_limit_per_split,
        checkpoint_path_override=checkpoint_path,
        log_path_override=log_path,
        quiet_stdout=quiet_stdout,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end JAX E3VB preprocess/train entry")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train"],
        default="train",
        help="Run offline preprocess export or GPU/CPU training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config_e2e.yaml"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run low-cost smoke mode for pipeline connectivity checks.",
    )
    parser.add_argument(
        "--smoke_epochs",
        type=int,
        default=1,
        help="Number of epochs used when --smoke is enabled.",
    )
    parser.add_argument(
        "--smoke_max_batches",
        type=int,
        default=2,
        help="Max batches per split used when --smoke is enabled.",
    )
    parser.add_argument(
        "--sample_limit_per_split",
        type=int,
        default=None,
        help="Optional post-split cap applied to train/val/test molecule counts.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional runtime override for best-checkpoint output path.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional runtime override for preprocess/train log file path.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU execution for non-formal diagnostics. Formal training should not use this.",
    )
    parser.add_argument(
        "--quiet_stdout",
        action="store_true",
        help="Write training logs only to --log_path and suppress mirrored stdout lines.",
    )
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Rebuild offline preprocess artifacts even if existing caches are present.",
    )
    arguments = parser.parse_args()
    main(
        mode=arguments.mode,
        config_path=arguments.config,
        smoke=arguments.smoke,
        smoke_epochs=arguments.smoke_epochs,
        smoke_max_batches=arguments.smoke_max_batches,
        allow_cpu=arguments.allow_cpu,
        sample_limit_per_split=arguments.sample_limit_per_split,
        checkpoint_path=arguments.checkpoint_path,
        log_path=arguments.log_path,
        quiet_stdout=arguments.quiet_stdout,
        force_preprocess=arguments.force_preprocess,
    )
