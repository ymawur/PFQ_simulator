from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.metrics import mae, rmse

st.set_page_config(page_title="Regularized Linear Regression", layout="wide")

st.title("Regularized Linear Regression")

with st.sidebar:
    st.header("Dataset controls")
    n_samples = st.slider("Samples", 40, 200, 80)
    noise = st.slider("Noise (std dev)", 0.0, 2.0, 0.5)
    standardize = st.toggle("Standardize features", value=True)

    st.header("Model selection")
    model_type = st.selectbox("Model", ["OLS", "Ridge", "Lasso"])
    lambda_val = st.slider("Lambda", -4.0, 2.0, -1.0, help="Log10 scale")


rng = np.random.default_rng(7)
X = rng.normal(0, 1, size=(n_samples, 5))
true_beta = np.array([2.0, -1.5, 0.0, 0.5, 0.0])
y = X @ true_beta + rng.normal(0, noise, size=n_samples)

if standardize:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

alpha = 10 ** lambda_val

if model_type == "OLS":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha, max_iter=10000)

model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

metrics = {
    "Train RMSE": rmse(y_train, train_pred),
    "Test RMSE": rmse(y_test, test_pred),
    "Train MAE": mae(y_train, train_pred),
    "Test MAE": mae(y_test, test_pred),
}

coef_df = pd.DataFrame({"Feature": [f"x{i+1}" for i in range(X.shape[1])], "Coefficient": model.coef_})


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.dataframe(pd.DataFrame({"y": y, "y_hat": model.predict(X)}).head(10), use_container_width=True)

with tabs[1]:
    st.markdown("### Fit Model")
    st.dataframe(coef_df, use_container_width=True)
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

with tabs[2]:
    st.markdown("### Uncertainty")
    st.write("Use cross-validated error to contextualize uncertainty in predictions.")

with tabs[3]:
    st.markdown("### Compare")
    lambdas = np.logspace(-4, 2, 20)
    coef_paths = []
    train_errors = []
    test_errors = []
    for lam in lambdas:
        if model_type == "OLS":
            temp_model = LinearRegression()
        elif model_type == "Ridge":
            temp_model = Ridge(alpha=lam)
        else:
            temp_model = Lasso(alpha=lam, max_iter=10000)
        temp_model.fit(X_train, y_train)
        coef_paths.append(temp_model.coef_)
        train_errors.append(mean_squared_error(y_train, temp_model.predict(X_train)))
        test_errors.append(mean_squared_error(y_test, temp_model.predict(X_test)))

    coef_paths = np.array(coef_paths)
    coef_df = pd.DataFrame(coef_paths, columns=[f"x{i+1}" for i in range(X.shape[1])])
    coef_df["lambda"] = lambdas
    st.line_chart(coef_df.set_index("lambda"))

    error_df = pd.DataFrame({"lambda": lambdas, "Train MSE": train_errors, "Test MSE": test_errors})
    st.line_chart(error_df.set_index("lambda"))
