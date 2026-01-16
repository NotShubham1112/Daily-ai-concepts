# Self-Supervised Learning — Research-Oriented Topics (Level 5)

This module implements key concepts in **Self-Supervised Learning (SSL)**, where representations are learned from the data itself without explicit external labels.

## 📌 Module Scope

- **Contrastive Learning**: Implementing the InfoNCE (SimCLR-style) loss function.
- **Masked Prediction**: Demonstrating how models learn by predicting masked parts of the input (BERT/MAE style).
- **SSL vs Supervised Comparison**: Conceptual analysis of feature quality.

---

## 📂 Repository Structure

```text
self-supervised-learning/
│
├── contrastive_loss.py        # InfoNCE loss implementation
├── masked_prediction.py       # Input masking and reconstruction logic
├── ssl_vs_supervised.py       # Performance comparison metrics
├── utils.py                   # Data augmentation helpers
├── requirements.txt
└── README.md
```
