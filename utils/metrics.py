"""Metric utilities for normalized structure-weight regression."""

import numpy as np


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

    def summarize(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        """
        Compute metric summary dictionary.

        Arguments:
        - prediction: Predicted values, shape [num_samples].
        - target: Ground-truth values, shape [num_samples].

        Returns:
        - Dictionary with keys: mae, rmse, spearman.
        """

        return {
            "mae": self.computeMAE(prediction, target),
            "rmse": self.computeRMSE(prediction, target),
            "spearman": self.computeSpearman(prediction, target),
        }
