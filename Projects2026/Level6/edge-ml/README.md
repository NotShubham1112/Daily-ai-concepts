# Edge & Resource-Constrained ML — ML Systems & Deployment (Level 6)

This module explores techniques for optimizing machine learning models to run efficiently on low-power devices like mobile phones, IoT sensors, and Raspberry Pis.

## 📌 Module Scope

- **Model Quantization**: Implementing 8-bit integer quantization to reduce model size by 4x.
- **Weight Pruning**: Removing redundant parameters based on magnitude to speed up inference.
- **Knowledge Distillation**: Training a small "student" model to mimic a larger "teacher" model.

---

## 📂 Repository Structure

```text
edge-ml/
│
├── quantization.py            # Weight quantization logic (Int8)
├── pruning.py                 # Structural and magnitude pruning
├── distillation_logic.py      # Student-Teacher loss implementation
├── utils.py                   # Model size/latency benchmarks
├── requirements.txt
└── README.md
```
