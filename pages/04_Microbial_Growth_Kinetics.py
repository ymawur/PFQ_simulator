from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from core.bootstrap import bootstrap_prediction_bands
from core.data import add_noise, synthetic_time_axis
from core.metrics import mae, rmse
from core.plotting import parity_plot, residual_plot

st.set_page_config(page_title="Microbial Growth Kinetics", layout="wide")

st.title("Microbial Growth Kinetics")

with st.sidebar:
    st.header("Dataset controls")
    model_name = st.selectbox("True model", ["Logistic", "Gompertz"])
    n_points = st.slider("Points", 20, 80, 40)
    noise = st.slider("Noise (std dev)", 0.0, 0.5, 0.05)

    st.header("Parameter sliders")
    k_max = st.slider("Carrying capacity", 5.0, 15.0, 10.0)
    rate = st.slider("Growth rate", 0.1, 2.0, 0.6)
    t_mid = st.slider("Inflection (time)", 1.0, 8.0, 4.0)
    lag = st.slider("Lag (Gompertz)", 0.0, 4.0, 1.0)
    threshold = st.slider("Threshold", 1.0, 15.0, 8.0)


def logistic(t: np.ndarray, k_val: float, r_val: float, t0_val: float) -> np.ndarray:
    return k_val / (1 + np.exp(-r_val * (t - t0_val)))


def gompertz(t: np.ndarray, k_val: float, r_val: float, lag_val: float) -> np.ndarray:
    return k_val * np.exp(-np.exp((r_val * np.e / k_val) * (lag_val - t) + 1))


model_map = {
    "Logistic": (logistic, [k_max, rate, t_mid]),
    "Gompertz": (gompertz, [k_max, rate, lag]),
}

x = synthetic_time_axis(n_points, 10)
true_model, params = model_map[model_name]
true_y = true_model(x, *params)
obs = add_noise(true_y, noise, seed=5)

st.subheader("Synthetic dataset")
st.dataframe(pd.DataFrame({"time": x, "observed": obs}), use_container_width=True)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.line_chart(pd.DataFrame({"time": x, "Observed": obs, "True": true_y}).set_index("time"))

with tabs[1]:
    st.markdown("### Fit Model")
    popt, _ = curve_fit(true_model, x, obs, p0=params, bounds=(0, np.inf))
    pred = true_model(x, *popt)
    metrics = {"RMSE": rmse(obs, pred), "MAE": mae(obs, pred)}
    param_df = pd.DataFrame([popt], columns=["K", "rate", "lag/t0"])
    st.dataframe(pd.concat([param_df, pd.DataFrame([metrics])], axis=1), use_container_width=True)

    fig_parity = parity_plot(obs, pred)
    st.pyplot(fig_parity, use_container_width=True)

    fig_resid = residual_plot(x, obs - pred)
    st.pyplot(fig_resid, use_container_width=True)

    t_threshold = np.interp(threshold, pred, x) if threshold <= pred.max() else np.nan
    if np.isnan(t_threshold):
        st.warning("Threshold above predicted maximum; adjust threshold or parameters.")
    else:
        st.metric("Time to threshold", f"{t_threshold:.2f}")

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
        popt, _ = curve_fit(model_fn_local, x, obs, p0=initial, bounds=(0, np.inf))
        pred = model_fn_local(x, *popt)
        compare_rows.append({"Model": name, "RMSE": rmse(obs, pred), "MAE": mae(obs, pred)})
    st.dataframe(pd.DataFrame(compare_rows).set_index("Model"), use_container_width=True)
