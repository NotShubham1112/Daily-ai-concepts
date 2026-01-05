# Information Theory for Machine Learning — From First Principles

This repository implements **core information-theoretic concepts** that govern how machine learning systems represent, compress, and transmit information.

All algorithms are implemented **from scratch using NumPy**, with controlled experiments and visualizations to build **intuition about uncertainty, dependence, and learning efficiency**.

This module completes a critical pillar of ML foundations:
> Linear Algebra → Probability → Optimization → **Information Theory**

---

## Motivation

Modern machine learning is fundamentally about **information**:
- reducing uncertainty about targets
- extracting informative features
- compressing representations without losing meaning

Information theory provides the mathematical language to answer questions like:
- *How much does a feature tell us about the target?*
- *Why is cross-entropy the correct loss for classification?*
- *How do neural networks trade compression for prediction?*

This repository focuses on **understanding these questions mathematically and empirically**, not heuristically.

---

## Repository Structure

```text
information-theory-ml/
│
├── entropy_metrics.py        # Entropy, cross-entropy, KL divergence
├── mutual_information.py    # Mutual information estimation
├── feature_selection.py     # Information gain for feature relevance
├── loss_comparison.py       # Cross-entropy vs MSE analysis
├── utils.py
├── requirements.txt
└── README.md
