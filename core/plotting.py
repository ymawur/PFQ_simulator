from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Parity plot") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig


def residual_plot(x: np.ndarray, residuals: np.ndarray, title: str = "Residual plot") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.axhline(0, color="gray", linestyle="--")
    ax.scatter(x, residuals, alpha=0.7)
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    return fig


def multi_line_plot(
    x: np.ndarray,
    series: Iterable[np.ndarray],
    labels: Iterable[str],
    title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots()
    for y, label in zip(series, labels):
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return fig
