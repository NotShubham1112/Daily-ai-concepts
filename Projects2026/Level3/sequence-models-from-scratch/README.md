# 🧠 Sequence Models From Scratch (RNN, LSTM, Temporal Learning)

This repository implements **sequence models from first principles using NumPy only**, without relying on deep learning frameworks such as TensorFlow or PyTorch.

The goal is **conceptual mastery**, not abstraction.

---

## 📌 Module Overview

**Sequence Models** are designed to process ordered data where **temporal dependency** matters.

Examples:
- Text
- Speech
- Time-series
- Sensor data
- Financial signals

This module covers:
- Vanilla Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) networks
- Temporal forward propagation
- Vanishing gradient problem
- Real-world sequence modeling tasks

---

## 📂 Project Structure

```text
sequence-models-from-scratch/
│
├── rnn.py                     # Vanilla RNN implementation
├── lstm.py                    # LSTM cell from scratch
├── text_generation.py         # Character-level text generation
├── time_series.py             # Time-series forecasting
├── vanishing_gradients.py     # Gradient decay visualization
├── utils.py                   # Helper functions
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
