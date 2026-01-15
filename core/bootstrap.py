from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def bootstrap_prediction_bands(
    x: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_resamples: int = 200,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap prediction bands by resampling indices.

    Args:
        x: Input array.
        y: Observed array.
        model_fn: Function that returns predictions given x and y sample.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence interval.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    preds = []
    n = len(x)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        preds.append(model_fn(x[idx], y[idx]))
    pred_stack = np.vstack(preds)
    mean = pred_stack.mean(axis=0)
    alpha = (1 - ci) / 2
    lower = np.quantile(pred_stack, alpha, axis=0)
    upper = np.quantile(pred_stack, 1 - alpha, axis=0)
    return BootstrapResult(mean=mean, lower=lower, upper=upper)
