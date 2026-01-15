# Food Quality Modelling Simulator

Interactive Streamlit web application for teaching modelling basics, kinetics, microbial/enzymatic kinetics, and multi-process degradation in a food quality modelling course.

## Features
- Multi-page simulator with five modules:
  1. Kinetic Sandbox
  2. Arrhenius Simulator
  3. Two-stage / Multi-process Degradation
  4. Microbial Growth Kinetics
  5. Enzyme Kinetics
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
