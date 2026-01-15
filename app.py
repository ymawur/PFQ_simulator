from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Food Quality Modelling Simulator", layout="wide")

st.title("Food Quality Modelling Simulator")
st.markdown(
    """
Welcome to the **Food Quality Modelling Simulator**. Use the pages in the left-hand
navigation to explore interactive modules covering kinetics, microbial and enzymatic
models, and degradation pathways.

Each page includes tabs for Simulation, Fit Model, Uncertainty, and Compare (when
multiple models are available). Use the sidebar controls to generate synthetic datasets
and tune model parameters.
"""
)

st.info(
    "Tip: start with the Kinetic Sandbox or Arrhenius Simulator to practice fitting and\n"
    "interpreting model parameters before moving into multi-process degradation modules."
)
