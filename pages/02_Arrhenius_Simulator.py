from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from core.bootstrap import bootstrap_prediction_bands
from core.data import add_noise, synthetic_time_axis
from core.metrics import mae, rmse
from core.plotting import parity_plot, residual_plot

st.set_page_config(page_title="Arrhenius Simulator", layout="wide")

st.title("Arrhenius Simulator")

R_GAS = 8.314

with st.sidebar:
    st.header("Dataset controls")
    mode = st.selectbox("Input mode", ["Single k + temperature", "Multi-temp time series"])
    base_temp = st.slider("Base temperature (K)", 280.0, 330.0, 300.0)
    ea_true = st.slider("True Ea (kJ/mol)", 20.0, 120.0, 60.0)
    a_true = st.slider("True A (1/min)", 1e2, 1e8, 1e5, format="%.1e")
    noise = st.slider("Noise (std dev)", 0.0, 0.5, 0.05)

    st.header("Model selection")
    temp_to_predict = st.slider("Predict k at T (K)", 270.0, 360.0, 310.0)


def arrhenius_k(temp_k: np.ndarray, a_val: float, ea_val: float) -> np.ndarray:
    return a_val * np.exp(-ea_val / (R_GAS * temp_k))


if mode == "Single k + temperature":
    temps = np.array([base_temp])
    k_true = arrhenius_k(temps, a_true, ea_true * 1000)
    k_obs = add_noise(k_true, noise, seed=2)
    data = pd.DataFrame({"Temp(K)": temps, "k": k_obs})
else:
    temps = np.linspace(base_temp - 15, base_temp + 15, 5)
    t_axis = synthetic_time_axis(20, 8)
    k_true = arrhenius_k(temps, a_true, ea_true * 1000)
    k_obs = add_noise(k_true, noise, seed=3)
    data = pd.DataFrame({"Temp(K)": temps, "k": k_obs})

st.subheader("Arrhenius data")
st.dataframe(data, use_container_width=True)




def linear_arrhenius(x_vals: np.ndarray, a_val: float, ea_val: float) -> np.ndarray:
    return np.log(a_val) - ea_val / (R_GAS * x_vals)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.scatter_chart(data, x="Temp(K)", y="k")

with tabs[1]:
    st.markdown("### Fit Model")
    if len(temps) > 1:
        popt, _ = curve_fit(arrhenius_k, temps, k_obs, p0=[a_true, ea_true * 1000], bounds=(0, np.inf))
        a_fit, ea_fit = popt
        k_pred = arrhenius_k(temps, a_fit, ea_fit)
        metrics = {"RMSE": rmse(k_obs, k_pred), "MAE": mae(k_obs, k_pred)}
        st.write(pd.DataFrame({"A": [a_fit], "Ea (kJ/mol)": [ea_fit / 1000], **metrics}))

        fig_parity = parity_plot(k_obs, k_pred, title="Observed vs Predicted k")
        st.pyplot(fig_parity, use_container_width=True)

        fig_resid = residual_plot(temps, k_obs - k_pred, title="Residuals vs Temperature")
        st.pyplot(fig_resid, use_container_width=True)

        arr_df = pd.DataFrame({"1/T": 1 / temps, "ln(k)": np.log(k_obs)})
        st.line_chart(arr_df.set_index("1/T"))

        k_at_temp = arrhenius_k(np.array([temp_to_predict]), a_fit, ea_fit)[0]
        if temp_to_predict < temps.min() or temp_to_predict > temps.max():
            st.warning("Prediction is extrapolating beyond training temperatures.")
        st.metric("Predicted k", f"{k_at_temp:.4f}")
    else:
        st.info("Provide multi-temperature data to fit Arrhenius parameters.")

with tabs[2]:
    st.markdown("### Uncertainty")
    if len(temps) > 1:
        def model_fn(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
            popt, _ = curve_fit(arrhenius_k, x_sample, y_sample, p0=[a_true, ea_true * 1000], bounds=(0, np.inf))
            return arrhenius_k(temps, *popt)

        bands = bootstrap_prediction_bands(temps, k_obs, model_fn)
        band_df = pd.DataFrame({
            "Temp(K)": temps,
            "Mean": bands.mean,
            "Lower": bands.lower,
            "Upper": bands.upper,
            "Observed": k_obs,
        }).set_index("Temp(K)")
        st.line_chart(band_df)
    else:
        st.info("Uncertainty bands require multiple temperatures.")

with tabs[3]:
    st.markdown("### Compare")
    if len(temps) > 1:
        k_pred = arrhenius_k(temps, a_fit, ea_fit)
        compare_df = pd.DataFrame(
            {
                "Metric": ["RMSE", "MAE"],
                "Value": [rmse(k_obs, k_pred), mae(k_obs, k_pred)],
            }
        )
        st.dataframe(compare_df, use_container_width=True)
