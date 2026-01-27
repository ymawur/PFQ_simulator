# Food Quality Modelling Simulator

Interactive web application for teaching modelling basics, kinetics,
microbial/enzymatic kinetics, machine learning basics, and chemometrics in a
food quality modelling course.

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

## Run locally

Use any static server to preview the site. For example:

```bash
npx serve .
```

## Deploy to Vercel

The project is a static site, so Vercel can deploy it directly. Point Vercel at
the repository root and it will serve `index.html` automatically.
