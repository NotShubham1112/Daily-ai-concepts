# Neural Networks from Scratch — Deep Learning Foundations (Level 3)

This module implements **Neural Networks completely from scratch**, focusing on the mathematical and algorithmic foundations of deep learning.

All components — forward propagation, backpropagation, activation functions, and gradient flow — are implemented using **NumPy only**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

This module serves as the **bridge between classical machine learning and modern deep learning systems**.

---

## 📌 Module Scope

This module covers:

- Fully connected (dense) neural networks
- Forward propagation
- Backpropagation via chain rule
- Activation functions and their gradients
- Gradient flow analysis
- Training dynamics and convergence behavior

---

## 📂 Repository Structure

```text
neural-networks-from-scratch/
│
├── neural_network.py          # Core neural network + backpropagation
├── activations.py             # Activation functions and gradients
├── losses.py                  # Loss functions and derivatives
├── gradient_visualization.py  # Gradient flow analysis per layer
├── activation_study.py        # Activation function performance comparison
├── utils.py                   # Helper utilities
├── requirements.txt
└── README.md
