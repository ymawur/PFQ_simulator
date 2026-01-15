from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from core.bootstrap import bootstrap_prediction_bands
from core.data import add_noise, synthetic_time_axis
from core.metrics import mae, rmse
from core.plotting import parity_plot, residual_plot

st.set_page_config(page_title="Two-stage / Multi-process Degradation", layout="wide")

st.title("Two-stage / Multi-process Degradation")

with st.sidebar:
    st.header("Dataset controls")
    model_name = st.selectbox(
        "True process",
        ["Lag + decay", "Parallel exponentials", "Sequential A → B"],
    )
    noise = st.slider("Noise (std dev)", 0.0, 0.5, 0.05)
    n_points = st.slider("Points", 15, 60, 30)

    st.header("Parameter sliders")
    lag = st.slider("Lag (time)", 0.0, 5.0, 1.0)
    k1 = st.slider("k1", 0.05, 1.0, 0.3)
    k2 = st.slider("k2", 0.05, 1.0, 0.15)
    a0 = st.slider("A0", 1.0, 10.0, 5.0)
    b0 = st.slider("B0", 0.5, 5.0, 2.0)


def lag_decay(t: np.ndarray, a_val: float, k_val: float, lag_val: float) -> np.ndarray:
    return np.where(t < lag_val, a_val, a_val * np.exp(-k_val * (t - lag_val)))


def parallel_exp(t: np.ndarray, a_val: float, b_val: float, k1_val: float, k2_val: float) -> np.ndarray:
    return a_val * np.exp(-k1_val * t) + b_val * np.exp(-k2_val * t)


def sequential_ab(t: np.ndarray, a_val: float, b_val: float, k1_val: float, k2_val: float) -> np.ndarray:
    return a_val * np.exp(-k1_val * t) + b_val * (1 - np.exp(-k1_val * t)) * np.exp(-k2_val * t)


model_map = {
    "Lag + decay": (lag_decay, [a0, k1, lag]),
    "Parallel exponentials": (parallel_exp, [a0, b0, k1, k2]),
    "Sequential A → B": (sequential_ab, [a0, b0, k1, k2]),
}

x = synthetic_time_axis(n_points, 12)
true_model, params = model_map[model_name]
true_y = true_model(x, *params)
obs = add_noise(true_y, noise, seed=4)

st.subheader("Synthetic dataset")
st.dataframe(pd.DataFrame({"time": x, "observed": obs}), use_container_width=True)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.line_chart(pd.DataFrame({"time": x, "Observed": obs, "True": true_y}).set_index("time"))

with tabs[1]:
    st.markdown("### Fit Model")
    try:
        bounds = (0, np.inf)
        popt, _ = curve_fit(true_model, x, obs, p0=params, bounds=bounds)
        pred = true_model(x, *popt)
        metrics = {"RMSE": rmse(obs, pred), "MAE": mae(obs, pred)}
        param_df = pd.DataFrame([popt], columns=[f"p{i+1}" for i in range(len(popt))])
        st.dataframe(pd.concat([param_df, pd.DataFrame([metrics])], axis=1), use_container_width=True)

        fig_parity = parity_plot(obs, pred)
        st.pyplot(fig_parity, use_container_width=True)

        fig_resid = residual_plot(x, obs - pred)
        st.pyplot(fig_resid, use_container_width=True)
    except RuntimeError:
        st.error("Nonlinear fit failed for this dataset.")

with tabs[2]:
    st.markdown("### Uncertainty")

    def model_fn(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
        popt, _ = curve_fit(true_model, x_sample, y_sample, p0=params, bounds=(0, np.inf))
        return true_model(x, *popt)

    bands = bootstrap_prediction_bands(x, obs, model_fn)
    band_df = pd.DataFrame(
        {
            "time": x,
            "Mean": bands.mean,
            "Lower": bands.lower,
            "Upper": bands.upper,
            "Observed": obs,
        }
    ).set_index("time")
    st.line_chart(band_df)

with tabs[3]:
    st.markdown("### Compare")
    compare_rows = []
    for name, (model_fn_local, initial) in model_map.items():
        try:
            popt, _ = curve_fit(model_fn_local, x, obs, p0=initial, bounds=(0, np.inf))
            pred = model_fn_local(x, *popt)
            compare_rows.append({"Model": name, "RMSE": rmse(obs, pred), "MAE": mae(obs, pred)})
        except RuntimeError:
            compare_rows.append({"Model": name, "RMSE": np.nan, "MAE": np.nan})
    st.dataframe(pd.DataFrame(compare_rows).set_index("Model"), use_container_width=True)
