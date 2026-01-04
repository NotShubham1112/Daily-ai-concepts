# Linear Algebra for Machine Learning — From First Principles

This repository provides **mathematically grounded, from-scratch implementations** of core linear algebra concepts used in modern machine learning.  
The objective is **conceptual mastery**, not library usage.

All algorithms are implemented using **NumPy only**, with explicit attention to:
- mathematical correctness
- numerical stability
- geometric intuition
- research relevance

---

## Motivation

Most machine learning failures stem not from poor models, but from **poor understanding of the mathematics** underneath them.

This repository is designed to:
- eliminate black-box thinking
- bridge theory ↔ implementation
- prepare for ML research, quant roles, and advanced systems design

---

## Repository Structure

```text
linear-algebra-ml/
│
├── pca_from_scratch.py        # PCA via eigen decomposition
├── eigen_visualization.py     # Geometric intuition of eigenvectors
├── pca_vs_autoencoder.py     # Linear autoencoder ≈ PCA proof
├── utils.py
├── requirements.txt
└── README.md
