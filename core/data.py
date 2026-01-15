from __future__ import annotations

import numpy as np


def add_noise(y: np.ndarray, noise: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y + rng.normal(0, noise, size=len(y))


def synthetic_time_axis(n_points: int = 30, t_max: float = 10.0) -> np.ndarray:
    return np.linspace(0, t_max, n_points)
