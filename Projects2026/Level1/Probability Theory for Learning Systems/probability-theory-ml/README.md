# Probability Theory for Learning Systems — From First Principles

This repository implements **core probability and information theory concepts** that underpin modern machine learning systems.  
All components are implemented **from scratch using NumPy**, with visualizations via Matplotlib and minimal reliance on external abstractions.

The focus is on **distributional thinking**, **uncertainty modeling**, and **information-theoretic reasoning**—skills essential for ML research, Bayesian methods, and advanced AI systems.

---

## Motivation

Machine learning models do not learn numbers — they learn **distributions**.

A strong understanding of probability theory is mandatory for:
- Bayesian inference
- Generative models
- Reinforcement learning
- Variational methods
- Uncertainty-aware decision systems

This repository is designed to remove black-box intuition and replace it with **mathematical clarity and empirical verification**.

---

## Repository Structure

```text
probability-theory-ml/
│
├── distributions.py        # Random variable & distribution simulators
├── kl_divergence.py        # Information-theoretic distance analysis
├── bayesian_inference.py   # Bayesian updating experiments
├── utils.py                # Entropy, expectation, variance
├── requirements.txt
└── README.md
