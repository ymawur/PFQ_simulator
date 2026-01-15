from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from core.bootstrap import bootstrap_prediction_bands
from core.data import add_noise
from core.metrics import mae, rmse
from core.plotting import parity_plot, residual_plot

st.set_page_config(page_title="Enzyme Kinetics", layout="wide")

st.title("Enzyme Kinetics")

with st.sidebar:
    st.header("Dataset controls")
    inhibition = st.selectbox("Inhibition mode", ["None", "Competitive", "Noncompetitive"])
    n_points = st.slider("Points", 12, 40, 20)
    noise = st.slider("Noise (std dev)", 0.0, 0.5, 0.05)

    st.header("Parameter sliders")
    vmax_true = st.slider("Vmax", 1.0, 10.0, 5.0)
    km_true = st.slider("Km", 0.5, 10.0, 3.0)
    inhibitor = st.slider("Inhibitor [I]", 0.0, 5.0, 1.0)
    ki_true = st.slider("Ki", 0.5, 10.0, 2.0)
    show_lb = st.toggle("Show Lineweaver-Burk plot")


def mm_rate(s: np.ndarray, vmax: float, km: float) -> np.ndarray:
    return vmax * s / (km + s)


def competitive_rate(s: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    return vmax * s / (km * (1 + inhibitor / ki) + s)


def noncompetitive_rate(s: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    return (vmax / (1 + inhibitor / ki)) * s / (km + s)


model_map = {
    "None": (mm_rate, [vmax_true, km_true]),
    "Competitive": (competitive_rate, [vmax_true, km_true, ki_true]),
    "Noncompetitive": (noncompetitive_rate, [vmax_true, km_true, ki_true]),
}

substrate = np.linspace(0.5, 10.0, n_points)
true_model, params = model_map[inhibition]
true_v = true_model(substrate, *params)
obs = add_noise(true_v, noise, seed=6)

st.subheader("Synthetic dataset")
st.dataframe(pd.DataFrame({"[S]": substrate, "v": obs}), use_container_width=True)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(substrate, obs, label="Observed", alpha=0.7)
    ax.plot(substrate, true_v, label="True", color="tab:orange")
    ax.set_xlabel("[S]")
    ax.set_ylabel("v")
    ax.set_title("Enzyme kinetics simulation")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    if show_lb:
        lb_df = pd.DataFrame({"1/[S]": 1 / substrate, "1/v": 1 / np.clip(obs, 1e-6, None)})
        st.line_chart(lb_df.set_index("1/[S]"))

with tabs[1]:
    st.markdown("### Fit Model")
    bounds = (0, np.inf)
    popt, _ = curve_fit(true_model, substrate, obs, p0=params, bounds=bounds)
    pred = true_model(substrate, *popt)
    metrics = {"RMSE": rmse(obs, pred), "MAE": mae(obs, pred)}
    columns = ["Vmax", "Km"] + (["Ki"] if len(popt) == 3 else [])
    st.dataframe(pd.concat([pd.DataFrame([popt], columns=columns), pd.DataFrame([metrics])], axis=1), use_container_width=True)

    fig_parity = parity_plot(obs, pred)
    st.pyplot(fig_parity, use_container_width=True)

    fig_resid = residual_plot(substrate, obs - pred)
    st.pyplot(fig_resid, use_container_width=True)

with tabs[2]:
    st.markdown("### Uncertainty")
    def model_fn(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
        popt, _ = curve_fit(true_model, x_sample, y_sample, p0=params, bounds=bounds)
        return true_model(substrate, *popt)

    bands = bootstrap_prediction_bands(substrate, obs, model_fn)
    band_df = pd.DataFrame(
        {
            "[S]": substrate,
            "Mean": bands.mean,
            "Lower": bands.lower,
            "Upper": bands.upper,
            "Observed": obs,
        }
    ).set_index("[S]")
    st.line_chart(band_df)

with tabs[3]:
    st.markdown("### Compare")
    compare_rows = []
    for name, (model_fn_local, initial) in model_map.items():
        popt, _ = curve_fit(model_fn_local, substrate, obs, p0=initial, bounds=bounds)
        pred = model_fn_local(substrate, *popt)
        compare_rows.append({"Model": name, "RMSE": rmse(obs, pred), "MAE": mae(obs, pred)})
    st.dataframe(pd.DataFrame(compare_rows).set_index("Model"), use_container_width=True)
