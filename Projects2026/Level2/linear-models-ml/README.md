# Linear Models from Scratch — Classical Machine Learning (Level 2)

This repository implements **core linear models from first principles**, without using high-level machine learning libraries such as `scikit-learn`.

The goal of this module is to build **deep theoretical and practical understanding** of classical machine learning by implementing models exactly as they are derived mathematically.

This project is part of a larger **Machine Learning Foundations roadmap**, progressing from mathematical foundations to full learning systems.

---

## 📌 Module Scope

This module covers:

- Linear Regression (OLS)
- Logistic Regression (Binary Classification)
- Bias–Variance Tradeoff
- Regularization Techniques (L1 vs L2)
- Optimization-based learning

All models are implemented using:
- **NumPy** for numerical computation
- **Matplotlib** for visualization

No external ML frameworks are used.

---

## 📂 Repository Structure

```text
linear-models-ml/
│
├── linear_regression.py          # Linear regression from scratch
├── logistic_regression.py        # Logistic regression from scratch
├── bias_variance.py              # Bias–variance tradeoff experiments
├── regularization_comparison.py  # L1 vs L2 regularization analysis
├── utils.py                      # Loss functions and helpers
├── requirements.txt
└── README.md
