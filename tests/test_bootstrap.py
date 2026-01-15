import numpy as np

from core.bootstrap import bootstrap_prediction_bands


def test_bootstrap_shapes():
    x = np.linspace(0, 1, 10)
    y = x * 2

    def model_fn(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
        coef = np.polyfit(x_sample, y_sample, 1)
        return np.polyval(coef, x)

    result = bootstrap_prediction_bands(x, y, model_fn, n_resamples=20, seed=1)
    assert result.mean.shape == x.shape
    assert result.lower.shape == x.shape
    assert result.upper.shape == x.shape
