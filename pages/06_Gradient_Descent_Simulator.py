from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gradient Descent Simulator", layout="wide")

st.title("Gradient Descent Simulator")

with st.sidebar:
    st.header("Loss surface")
    surface = st.selectbox("Surface", ["Convex quadratic", "Rosenbrock", "Noisy bowl"])
    learning_rate = st.slider("Learning rate", 0.001, 0.5, 0.05)
    momentum = st.slider("Momentum", 0.0, 0.99, 0.2)
    iterations = st.slider("Iterations", 10, 200, 80)

    st.header("Initialization")
    x0 = st.slider("x0", -2.0, 2.0, -1.2)
    y0 = st.slider("y0", -2.0, 2.0, 1.0)


def loss_fn(x: float, y: float) -> float:
    if surface == "Convex quadratic":
        return x**2 + 2 * y**2
    if surface == "Rosenbrock":
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2
    rng = np.random.default_rng(0)
    return x**2 + y**2 + rng.normal(0, 0.1)


def grad_fn(x: float, y: float) -> tuple[float, float]:
    if surface == "Convex quadratic":
        return 2 * x, 4 * y
    if surface == "Rosenbrock":
        return (
            -2 * (1 - x) - 400 * x * (y - x**2),
            200 * (y - x**2),
        )
    return 2 * x, 2 * y


positions = [(x0, y0)]
vel_x, vel_y = 0.0, 0.0
for _ in range(iterations):
    x, y = positions[-1]
    grad_x, grad_y = grad_fn(x, y)
    vel_x = momentum * vel_x - learning_rate * grad_x
    vel_y = momentum * vel_y - learning_rate * grad_y
    positions.append((x + vel_x, y + vel_y))

pos_arr = np.array(positions)
losses = np.array([loss_fn(x, y) for x, y in positions])

status = "Converging" if losses[-1] < losses[0] else "Diverging"


tabs = st.tabs(["Simulation", "Fit Model", "Uncertainty", "Compare"])

with tabs[0]:
    st.markdown("### Simulation")
    st.metric("State", status)
    st.line_chart(pd.DataFrame({"iter": np.arange(len(losses)), "loss": losses}).set_index("iter"))

with tabs[1]:
    st.markdown("### Fit Model")
    st.write("This module demonstrates iterative optimization rather than parameter fitting.")

    grid = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(grid, grid)
    Z = np.vectorize(loss_fn)(X, Y)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=20)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.plot(pos_arr[:, 0], pos_arr[:, 1], marker="o", color="red")
    ax.set_title("Gradient descent path")
    st.pyplot(fig, use_container_width=True)

with tabs[2]:
    st.markdown("### Uncertainty")
    st.write("Noise in the loss surface (if selected) highlights uncertainty in convergence.")

with tabs[3]:
    st.markdown("### Compare")
    st.write("Compare learning rate and momentum settings by adjusting sidebar sliders.")
