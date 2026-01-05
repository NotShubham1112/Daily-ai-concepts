# Optimization Algorithms for Machine Learning — From First Principles

This repository implements **core optimization algorithms used to train machine learning models**, built entirely **from scratch using NumPy**.

The focus is not on using optimizers, but on **understanding their dynamics, geometry, strengths, and failure modes** across convex and non-convex loss landscapes.

This module forms **LEVEL 2** of a structured, research-oriented ML foundations series:
> Linear Algebra → Probability → Optimization → Learning Theory

---

## Motivation

Every machine learning model is, at its core, an **optimization problem**.

Yet most practitioners:
- tune learning rates blindly  
- rely on Adam without understanding why it works  
- cannot diagnose divergence or slow convergence  
- misunderstand convex vs non-convex behavior  

This repository exists to fix that.

It builds **optimizer intuition**, not optimizer dependency.

---

## Repository Structure

```text
optimization-algorithms-ml/
│
├── optimizers.py              # GD, SGD, Momentum, RMSProp, Adam (from scratch)
├── loss_surfaces.py           # Convex and non-convex objective functions
├── optimizer_comparison.py    # Optimizer trajectories on identical loss surfaces
├── convergence_failures.py    # Divergence and instability case studies
├── utils.py                   # Loss surface generation utilities
├── requirements.txt
└── README.md
