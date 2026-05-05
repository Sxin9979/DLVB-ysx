"""Metric utilities for normalized structure-weight regression."""

import numpy as np

from utils.top_mass import buildPredictedTopMassMask


class RegressionMetrics:
    """
    Compute MAE, RMSE, and Spearman correlation for regression outputs.
    """

    def computeMAE(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute mean absolute error.

        Arguments:
        - prediction: Predicted values, shape [batch_size].
        - target: Ground-truth values, shape [batch_size].

        Returns:
        - MAE scalar.
        """

        return float(np.mean(np.abs(prediction - target)))

    def computeRMSE(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute root mean square error.

        Arguments:
        - prediction: Predicted values, shape [batch_size].
        - target: Ground-truth values, shape [batch_size].

        Returns:
        - RMSE scalar.
        """

        return float(np.sqrt(np.mean((prediction - target) ** 2)))

    def computeSpearman(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute Spearman rank correlation for one-dimensional arrays.

        Arguments:
        - prediction: Predicted values, shape [num_samples].
        - target: Ground-truth values, shape [num_samples].

        Returns:
        - Spearman rank correlation scalar.
        """

        prediction_rank = prediction.argsort().argsort().astype(np.float64)
        target_rank = target.argsort().argsort().astype(np.float64)

        prediction_rank = prediction_rank - prediction_rank.mean()
        target_rank = target_rank - target_rank.mean()
        numerator = np.sum(prediction_rank * target_rank)
        denominator = np.sqrt(
            np.sum(prediction_rank**2) * np.sum(target_rank**2)
        )
        return float(numerator / max(denominator, 1e-12))

    def computeWeightedMAE(self, prediction: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
        """
        Compute weighted mean absolute error with a normalized non-negative weight.
        """

        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        total_weight = float(np.sum(weight))
        if total_weight <= 1.0e-12:
            return 0.0
        return float(np.sum(weight * np.abs(prediction - target)) / total_weight)

    def computeFocusWeightedMAE(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        top_mass_focus_mask: np.ndarray,
    ) -> float:
        """
        Compute weighted MAE restricted to the top-mass focus set.
        """

        focus_mask = np.asarray(top_mass_focus_mask, dtype=np.float64) > 0.5
        if not np.any(focus_mask):
            return 0.0
        focus_target = np.asarray(target, dtype=np.float64)[focus_mask]
        focus_prediction = np.asarray(prediction, dtype=np.float64)[focus_mask]
        focus_weight = np.maximum(focus_target, 0.0)
        if float(np.sum(focus_weight)) <= 1.0e-12:
            focus_weight = np.ones_like(focus_target, dtype=np.float64)
        return self.computeWeightedMAE(focus_prediction, focus_target, focus_weight)

    def computeTopMassMetrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        num_structures_per_molecule: np.ndarray,
        top_mass_focus_mask: np.ndarray,
        focus_cumulative_mass: float,
    ) -> dict:
        """
        Compute task-aligned top-mass metrics aggregated across molecules.
        """

        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        counts = np.asarray(num_structures_per_molecule, dtype=np.int32)
        focus_mask = np.asarray(top_mass_focus_mask, dtype=np.float64) > 0.5
        predicted_focus_mask = buildPredictedTopMassMask(
            prediction=prediction,
            num_structures_per_molecule=counts,
            cumulative_mass=focus_cumulative_mass,
            fallback_focus_mask=focus_mask.astype(np.float32),
        ) > 0.5

        focus_pair_accuracy_parts = []
        focus_spearman_parts = []
        recall_parts = []
        precision_parts = []
        tail_false_positive_parts = []
        offset = 0
        for count in counts.tolist():
            count = int(count)
            if count <= 0:
                continue
            pred_slice = prediction[offset : offset + count]
            target_slice = target[offset : offset + count]
            focus_slice = focus_mask[offset : offset + count]
            predicted_focus_slice = predicted_focus_mask[offset : offset + count]

            true_focus_index = np.flatnonzero(focus_slice)
            tail_index = np.flatnonzero(~focus_slice)
            predicted_focus_index = np.flatnonzero(predicted_focus_slice)

            if true_focus_index.size > 0:
                overlap = np.intersect1d(true_focus_index, predicted_focus_index, assume_unique=False)
                recall_parts.append(float(overlap.size / max(true_focus_index.size, 1)))
            if predicted_focus_index.size > 0:
                overlap = np.intersect1d(true_focus_index, predicted_focus_index, assume_unique=False)
                precision_parts.append(float(overlap.size / max(predicted_focus_index.size, 1)))
            if tail_index.size > 0:
                tail_false_positive_parts.append(float(np.mean(predicted_focus_slice[tail_index])))

            if true_focus_index.size >= 2:
                focus_pred = pred_slice[true_focus_index]
                focus_target = target_slice[true_focus_index]
                focus_spearman_parts.append(self.computeSpearman(focus_pred, focus_target))

                pair_hits = 0
                pair_total = 0
                for left in range(true_focus_index.size):
                    for right in range(left + 1, true_focus_index.size):
                        left_index = true_focus_index[left]
                        right_index = true_focus_index[right]
                        target_delta = target_slice[left_index] - target_slice[right_index]
                        if abs(target_delta) <= 1.0e-12:
                            continue
                        pred_delta = pred_slice[left_index] - pred_slice[right_index]
                        pair_hits += int(np.sign(pred_delta) == np.sign(target_delta))
                        pair_total += 1
                if pair_total > 0:
                    focus_pair_accuracy_parts.append(float(pair_hits / pair_total))
            offset += count

        return {
            "focus_weighted_mae": self.computeFocusWeightedMAE(
                prediction=prediction,
                target=target,
                top_mass_focus_mask=focus_mask.astype(np.float32),
            ),
            "focus_pair_acc": float(np.mean(focus_pair_accuracy_parts)) if focus_pair_accuracy_parts else 0.0,
            "focus_spearman": float(np.mean(focus_spearman_parts)) if focus_spearman_parts else 0.0,
            "focus_recall": float(np.mean(recall_parts)) if recall_parts else 0.0,
            "focus_precision": float(np.mean(precision_parts)) if precision_parts else 0.0,
            "tail_false_positive_rate": (
                float(np.mean(tail_false_positive_parts)) if tail_false_positive_parts else 0.0
            ),
        }

    def computeMoleculeMacroSpearman(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        num_structures_per_molecule: np.ndarray,
    ) -> float:
        """
        Compute Spearman inside each molecule, then average across molecules.
        """

        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        counts = np.asarray(num_structures_per_molecule, dtype=np.int32)

        molecule_spearman_parts = []
        offset = 0
        for count in counts.tolist():
            count = int(count)
            if count <= 1:
                offset += max(count, 0)
                continue
            pred_slice = prediction[offset : offset + count]
            target_slice = target[offset : offset + count]
            molecule_spearman_parts.append(self.computeSpearman(pred_slice, target_slice))
            offset += count
        return float(np.mean(molecule_spearman_parts)) if molecule_spearman_parts else 0.0

    def splitPayloadByDataset(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        num_structures_per_molecule: np.ndarray | None,
        top_mass_focus_mask: np.ndarray | None,
        dataset_index_per_molecule: np.ndarray,
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        Partition one split payload into per-dataset arrays using molecule counts.
        """

        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if num_structures_per_molecule is None:
            counts = np.ones((prediction.shape[0],), dtype=np.int32)
        else:
            counts = np.asarray(num_structures_per_molecule, dtype=np.int32)
        dataset_index_per_molecule = np.asarray(dataset_index_per_molecule, dtype=np.int32)
        focus_mask = (
            None
            if top_mass_focus_mask is None
            else np.asarray(top_mass_focus_mask, dtype=np.float64)
        )

        grouped_prediction: dict[int, list[np.ndarray]] = {}
        grouped_target: dict[int, list[np.ndarray]] = {}
        grouped_counts: dict[int, list[int]] = {}
        grouped_focus: dict[int, list[np.ndarray]] = {}

        offset = 0
        for molecule_index, count in enumerate(counts.tolist()):
            count = int(count)
            dataset_index = int(dataset_index_per_molecule[molecule_index])
            if count <= 0:
                continue
            next_offset = offset + count
            if dataset_index < 0:
                offset = next_offset
                continue
            grouped_prediction.setdefault(dataset_index, []).append(prediction[offset:next_offset])
            grouped_target.setdefault(dataset_index, []).append(target[offset:next_offset])
            grouped_counts.setdefault(dataset_index, []).append(count)
            if focus_mask is not None:
                grouped_focus.setdefault(dataset_index, []).append(focus_mask[offset:next_offset])
            offset = next_offset

        dataset_payloads: dict[int, dict[str, np.ndarray]] = {}
        for dataset_index in grouped_prediction:
            payload = {
                "prediction": np.concatenate(grouped_prediction[dataset_index], axis=0),
                "target": np.concatenate(grouped_target[dataset_index], axis=0),
                "num_structures_per_molecule": np.asarray(
                    grouped_counts[dataset_index],
                    dtype=np.int32,
                ),
            }
            if focus_mask is not None:
                payload["top_mass_focus_mask"] = np.concatenate(grouped_focus[dataset_index], axis=0)
            dataset_payloads[dataset_index] = payload
        return dataset_payloads

    def summarize(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        num_structures_per_molecule: np.ndarray | None = None,
        top_mass_focus_mask: np.ndarray | None = None,
        focus_cumulative_mass: float = 0.98,
        dataset_index_per_molecule: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
    ) -> dict:
        """
        Compute metric summary dictionary.

        Arguments:
        - prediction: Predicted values, shape [num_samples].
        - target: Ground-truth values, shape [num_samples].

        Returns:
        - Dictionary with keys: mae, rmse, spearman.
        """

        summary = {
            "mae": self.computeMAE(prediction, target),
            "rmse": self.computeRMSE(prediction, target),
            "spearman": self.computeSpearman(prediction, target),
        }
        if num_structures_per_molecule is not None:
            summary["molecule_macro_spearman"] = self.computeMoleculeMacroSpearman(
                prediction=prediction,
                target=target,
                num_structures_per_molecule=num_structures_per_molecule,
            )
        if (num_structures_per_molecule is not None) and (top_mass_focus_mask is not None):
            summary.update(
                self.computeTopMassMetrics(
                    prediction=prediction,
                    target=target,
                    num_structures_per_molecule=num_structures_per_molecule,
                    top_mass_focus_mask=top_mass_focus_mask,
                    focus_cumulative_mass=focus_cumulative_mass,
                )
            )
        if (dataset_index_per_molecule is not None) and (dataset_names is not None):
            dataset_payloads = self.splitPayloadByDataset(
                prediction=prediction,
                target=target,
                num_structures_per_molecule=num_structures_per_molecule,
                top_mass_focus_mask=top_mass_focus_mask,
                dataset_index_per_molecule=dataset_index_per_molecule,
            )
            per_dataset = {}
            macro_metrics: dict[str, list[float]] = {}
            for dataset_index, payload in dataset_payloads.items():
                if dataset_index < 0 or dataset_index >= len(dataset_names):
                    continue
                dataset_summary = self.summarize(
                    prediction=payload["prediction"],
                    target=payload["target"],
                    num_structures_per_molecule=payload.get("num_structures_per_molecule"),
                    top_mass_focus_mask=payload.get("top_mass_focus_mask"),
                    focus_cumulative_mass=focus_cumulative_mass,
                    dataset_index_per_molecule=None,
                    dataset_names=None,
                )
                per_dataset[dataset_names[dataset_index]] = dataset_summary
                for metric_name, value in dataset_summary.items():
                    if isinstance(value, (int, float, np.floating)):
                        macro_metrics.setdefault(metric_name, []).append(float(value))
            if per_dataset:
                summary["per_dataset"] = per_dataset
                for metric_name, values in macro_metrics.items():
                    summary[f"macro_{metric_name}"] = float(np.mean(values))
        return summary
