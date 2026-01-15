from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from core.bootstrap import bootstrap_prediction_bands
from core.data import add_noise, synthetic_time_axis
from core.metrics import mae, rmse
from core.plotting import parity_plot, residual_plot

st.set_page_config(page_title="Kinetic Sandbox", layout="wide")

st.title("Kinetic Sandbox")

with st.sidebar:
    st.header("Dataset controls")
    model_name = st.selectbox("True model", ["Zero order", "First order", "Second order"])
    y0 = st.slider("Initial concentration", 1.0, 20.0, 10.0)
    k_true = st.slider("Rate constant", 0.01, 1.0, 0.2)
    noise = st.slider("Noise (std dev)", 0.0, 1.0, 0.1)
    n_points = st.slider("Points", 10, 60, 30)

    st.header("Model selection")
    fit_models = st.multiselect(
        "Fit models", ["Zero order", "First order", "Second order"], default=["Zero order", "First order"]
    )


def zero_order(t: np.ndarray, y0_val: float, k_val: float) -> np.ndarray:
    return y0_val - k_val * t


def first_order(t: np.ndarray, y0_val: float, k_val: float) -> np.ndarray:
    return y0_val * np.exp(-k_val * t)


def second_order(t: np.ndarray, y0_val: float, k_val: float) -> np.ndarray:
    return y0_val / (1 + k_val * y0_val * t)


model_map = {
    "Zero order": zero_order,
    "First order": first_order,
    "Second order": second_order,
}

x = synthetic_time_axis(n_points, t_max=10)
true_model = model_map[model_name]
true_y = true_model(x, y0, k_true)
obs = add_noise(true_y, noise, seed=1)

st.subheader("Synthetic dataset")
obs_df = pd.DataFrame({"time": x, "observed": obs})
st.dataframe(obs_df, use_container_width=True)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.line_chart(pd.DataFrame({"time": x, "Observed": obs, "True": true_y}).set_index("time"))

with tabs[1]:
    st.markdown("### Fit Model")
    fit_results = []
    for name in fit_models:
        try:
            popt, _ = curve_fit(model_map[name], x, obs, p0=[y0, k_true], bounds=(0, np.inf))
            pred = model_map[name](x, *popt)
            rss = np.sum((obs - pred) ** 2)
            n = len(obs)
            k_params = len(popt)
            aic = n * np.log(rss / n) + 2 * k_params
            bic = n * np.log(rss / n) + k_params * np.log(n)
            fit_results.append(
                {
                    "Model": name,
                    "y0": popt[0],
                    "k": popt[1],
                    "RMSE": rmse(obs, pred),
                    "MAE": mae(obs, pred),
                    "AIC": aic,
                    "BIC": bic,
                }
            )
        except RuntimeError:
            st.warning(f"Fit failed for {name} model.")

    if fit_results:
        result_df = pd.DataFrame(fit_results).set_index("Model")
        st.dataframe(result_df.style.format("{:.3f}"), use_container_width=True)

        best_model = result_df["RMSE"].idxmin()
        st.success(f"Best RMSE: {best_model}")

        fig_parity = parity_plot(obs, model_map[best_model](x, result_df.loc[best_model, "y0"], result_df.loc[best_model, "k"]))
        st.pyplot(fig_parity, use_container_width=True)

        residuals = obs - model_map[best_model](x, result_df.loc[best_model, "y0"], result_df.loc[best_model, "k"])
        fig_resid = residual_plot(x, residuals)
        st.pyplot(fig_resid, use_container_width=True)

with tabs[2]:
    st.markdown("### Uncertainty")
    base_model = model_map[model_name]

    def model_fn(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
        popt, _ = curve_fit(base_model, x_sample, y_sample, p0=[y0, k_true], bounds=(0, np.inf))
        return base_model(x, *popt)

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
    st.line_chart(band_df[["Observed", "Mean", "Lower", "Upper"]])

with tabs[3]:
    st.markdown("### Compare")
    if fit_models:
        compare_df = pd.DataFrame(fit_results).set_index("Model")
        st.bar_chart(compare_df[["RMSE", "AIC", "BIC"]])
