import numpy as np

from core.metrics import mae, rmse


def test_rmse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    assert mae(y_true, y_pred) == 2.0 / 3.0
