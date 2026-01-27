# Food Quality Modelling Simulator

Interactive Streamlit web application for teaching modelling basics, kinetics, microbial/enzymatic kinetics, machine learning basics, and chemometrics in a food quality modelling course.

## Features
- Multi-page simulator with eight modules:
  1. Kinetic Sandbox
  2. Arrhenius Simulator
  3. Two-stage / Multi-process Degradation
  4. Microbial Growth Kinetics
  5. Enzyme Kinetics
  6. Gradient Descent Simulator
  7. Regularized Linear Regression
  8. PCA/PLS Visualization Lab
- Shared utilities for metrics, plotting, bootstrap uncertainty, and report export.
- Synthetic data generation on every page.
- Example datasets in `data/examples/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pytest
```

## Deploy to Vercel

This repo includes a `vercel.json` plus an ASGI wrapper that starts Streamlit and
proxies requests to it. Deploy the project in Vercel as a Python project; no
additional build command is required.
