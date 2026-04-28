"""Main training entry for end-to-end JAX + Flax NNX E3VB."""

import argparse
from datetime import datetime
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
from utils import ConfigFactory, NNXCheckpointManager, RegressionMetrics, pairwiseRankLoss, weightedMae


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
        mae = jnp.mean(jnp.abs(prediction - batch.targets))
        regression_loss, _ = weightedMae(
            prediction=prediction,
            target=batch.targets,
            power=target_weight_power,
            offset=target_weight_offset,
        )
        rank_loss = pairwiseRankLoss(
            prediction=prediction,
            target=batch.targets,
            num_structures_per_molecule=batch.num_structures_per_molecule,
            margin=rank_loss_margin,
            min_delta=rank_loss_min_delta,
            pair_power=rank_pair_power,
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
    return prediction, batch.targets


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
                f"train_chunks={len(packed_cache.packed_chunks['train'])} "
                f"train_structures={packed_cache.split_structure_counts['train']} "
                f"val_molecules={len(processed_cache.split_ids['val'])} "
                f"val_chunks={len(packed_cache.packed_chunks['val'])} "
                f"val_structures={packed_cache.split_structure_counts['val']} "
                f"test_molecules={len(processed_cache.split_ids['test'])} "
                f"test_chunks={len(packed_cache.packed_chunks['test'])} "
                f"test_structures={packed_cache.split_structure_counts['test']}"
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
        self.split_structure_counts = None
        self.split_molecule_ids = None
        self.split_molecule_chunks = None
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
        self.split_molecule_ids = {
            split_name: list(packed_cache.split_ids[split_name])
            for split_name in ["train", "val", "test"]
        }
        self.split_molecule_chunks = {
            split_name: list(packed_cache.packed_chunks[split_name])
            for split_name in ["train", "val", "test"]
        }
        self.split_structure_counts = dict(packed_cache.split_structure_counts)
        if self.sample_limit_per_split is not None:
            limit = int(self.sample_limit_per_split)
            for split_name in ["train", "val", "test"]:
                allowed_ids = set(self.split_molecule_ids[split_name][:limit])
                self.split_molecule_ids[split_name] = self.split_molecule_ids[split_name][:limit]
                self.split_molecule_chunks[split_name] = [
                    chunk
                    for chunk in self.split_molecule_chunks[split_name]
                    if chunk.molecule_id in allowed_ids
                ]
                self.split_structure_counts[split_name] = int(
                    sum(int(chunk.atom_n_node.shape[0]) for chunk in self.split_molecule_chunks[split_name])
                )

        self.log(
            "CACHE "
            f"processed_cache={self.config.data.processed_cache_path} "
            f"packed_cache={self.config.data.packed_cache_path}"
        )
        adapter = GraphPackingAdapter(lap_pe_k=self.config.model.atom.lap_pe_k)
        self.pipeline = GrainPipeline(
            adapter=adapter,
            orbital_feature_dim=self.config.model.orbital.orbital_feature_dim,
        )

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
        transformation = optax.adamw(
            learning_rate=self.learning_rate_schedule,
            weight_decay=self.config.training.weight_decay,
        )
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
        loss = jnp.mean(jnp.abs(prediction - batch.targets))
        return loss, prediction

    def runTrainBatch(self, batch) -> Dict[str, np.ndarray]:
        """
        Run one optimization step for one batch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - Dictionary with keys loss, prediction, target.
        """

        loss, prediction, target, mae, regression_loss, rank_loss, slot_diversity_penalty = trainStep(
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
        return {
            "loss": np.asarray(loss),
            "prediction": np.asarray(prediction),
            "target": np.asarray(target),
            "mae": np.asarray(mae),
            "regression_loss": np.asarray(regression_loss),
            "rank_loss": np.asarray(rank_loss),
            "slot_diversity_penalty": np.asarray(slot_diversity_penalty),
        }

    def runEvalBatch(self, batch) -> Dict[str, np.ndarray]:
        """
        Run forward-only evaluation for one batch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - Dictionary with keys prediction and target.
        """

        prediction, target = evalStep(self.model, batch)
        return {
            "prediction": np.asarray(prediction),
            "target": np.asarray(target),
        }

    def splitBatchCount(self, split_name: str, training: bool) -> int:
        """
        Estimate the number of batches for one split.
        """

        total_chunks = len(self.split_molecule_chunks[split_name])
        batch_size = self.config.training.batch_size
        if training and self.config.training.train_drop_remainder:
            return total_chunks // batch_size
        return int(np.ceil(total_chunks / batch_size))

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
        summary = self.metrics.summarize(prediction=prediction, target=target)
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
        iterator = self.pipeline.createIterator(
            sample_groups=self.split_molecule_chunks[split_name],
            batch_size=self.config.training.batch_size,
            shuffle=training,
            seed=split_seed,
            drop_remainder=training and self.config.training.train_drop_remainder,
            bucket_key=self.trainBucketKey() if training else None,
        )
        outputs: List[Dict[str, np.ndarray]] = []
        total_batches = self.splitBatchCount(split_name=split_name, training=training)
        if self.smoke_mode:
            total_batches = min(total_batches, self.smoke_max_batches)
        for batch_id, batch in enumerate(iterator):
            if training:
                outputs.append(self.runTrainBatch(batch))
            else:
                outputs.append(self.runEvalBatch(batch))
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
        self.log(
            "EPOCH "
            f"epoch={epoch:04d} "
            f"split={split_name} "
            f"loss={metrics['loss']:.6f} "
            f"mae={metrics['mae']:.6f} "
            f"rmse={metrics['rmse']:.6f} "
            f"spearman={metrics['spearman']:.6f} "
            f"reg_loss={metrics.get('regression_loss', metrics['loss']):.6f} "
            f"rank_loss={metrics.get('rank_loss', 0.0):.6f} "
            f"slot_div={metrics.get('slot_diversity_penalty', 0.0):.6f} "
            f"lr={learning_rate:.8e} "
            f"samples={metrics['num_samples']}"
        )

    def runTrainingLoop(self) -> None:
        """
        Execute training/validation/testing loops across epochs.

        Returns:
        - None.
        """

        best_val_mae = float("inf")
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
        self.log(
            "BATCHING "
            f"molecules_per_batch={self.config.training.batch_size} "
            f"max_structures_per_chunk={self.config.data.max_structures_per_chunk} "
            f"train_bucketed={str(self.config.training.bucketed_batching).lower()} "
            f"train_bucket_key={self.trainBucketKey()} "
            f"train_drop_remainder={str(self.config.training.train_drop_remainder).lower()}"
        )
        if (not self.smoke_mode) and (early_stopping_patience > 0):
            self.log(
                "EARLY_STOPPING "
                f"monitor=val_mae "
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

        for epoch in range(total_epochs):
            train_metrics = self.runSplit(split_name="train", epoch=epoch, training=True)
            self.logEpochMetrics(epoch=epoch, split_name="train", metrics=train_metrics)
            val_metrics = self.runSplit(split_name="val", epoch=epoch, training=False)
            self.logEpochMetrics(epoch=epoch, split_name="val", metrics=val_metrics)
            test_metrics = self.runSplit(split_name="test", epoch=epoch, training=False)
            self.logEpochMetrics(epoch=epoch, split_name="test", metrics=test_metrics)

            if (not self.smoke_mode) and (val_metrics["mae"] < best_val_mae):
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch
                if epoch >= early_stopping_start_epoch:
                    no_improvement_epochs = 0
                self.saveBestCheckpoint(epoch, val_metrics, test_metrics)
                self.log(
                    "BEST "
                    f"epoch={epoch:04d} "
                    f"val_mae={val_metrics['mae']:.6f} "
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
                        f"best_val_mae={best_val_mae:.6f} "
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
                f"best_val_mae={best_val_mae:.6f} "
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
