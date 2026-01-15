from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA/PLS Visualization Lab", layout="wide")

st.title("PCA/PLS Visualization Lab")

with st.sidebar:
    st.header("Synthetic spectral data")
    n_samples = st.slider("Samples", 40, 150, 80)
    n_features = st.slider("Spectral variables", 20, 200, 60)
    noise = st.slider("Noise (std dev)", 0.0, 0.5, 0.1)
    n_components = st.slider("Components", 2, 10, 4)
    cv_toggle = st.toggle("Use cross-validation", value=True)

rng = np.random.default_rng(8)
wavelengths = np.linspace(400, 700, n_features)
base_spectra = np.sin(wavelengths / 40) + np.cos(wavelengths / 70)
X = base_spectra + rng.normal(0, noise, size=(n_samples, n_features))
latent = rng.normal(0, 1, size=(n_samples, 3))
y = latent @ np.array([1.2, -0.7, 0.4]) + rng.normal(0, 0.2, size=n_samples)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=n_components)
scores = pca.fit_transform(X_scaled)
loadings = pca.components_.T
explained = pca.explained_variance_ratio_

pls = PLSRegression(n_components=n_components)
pls.fit(X_scaled, y)


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.line_chart(pd.DataFrame(X_scaled[:10, :5]))

with tabs[1]:
    st.markdown("### Fit Model")
    score_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
    st.scatter_chart(score_df, x="PC1", y="PC2")

    load_df = pd.DataFrame(loadings[:, :2], columns=["PC1", "PC2"])
    st.line_chart(load_df)

    explained_df = pd.DataFrame({"Component": np.arange(1, n_components + 1), "Explained": explained})
    st.bar_chart(explained_df.set_index("Component"))

    y_pred = pls.predict(X_scaled).ravel()
    calib_df = pd.DataFrame({"Observed": y, "Predicted": y_pred})
    st.scatter_chart(calib_df, x="Observed", y="Predicted")

with tabs[2]:
    st.markdown("### Uncertainty")
    st.write("PLS cross-validation error shows prediction uncertainty across components.")
    if cv_toggle:
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        cv_errors = []
        for comp in range(1, n_components + 1):
            model = PLSRegression(n_components=comp)
            preds = cross_val_predict(model, X_scaled, y, cv=cv)
            cv_errors.append(mean_squared_error(y, preds))
        cv_df = pd.DataFrame({"Components": np.arange(1, n_components + 1), "CV MSE": cv_errors})
        st.line_chart(cv_df.set_index("Components"))

with tabs[3]:
    st.markdown("### Compare")
    if cv_toggle:
        st.write("Overfitting trend as components increase (CV error).")
    else:
        st.info("Enable cross-validation to compare component counts.")
