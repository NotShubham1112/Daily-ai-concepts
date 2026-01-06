# Support Vector Machines — Classical Machine Learning (Level 2)

This module implements **Support Vector Machines (SVMs) from scratch**, focusing on the geometric, optimization, and theoretical foundations that make SVMs one of the most important classical machine learning algorithms.

All implementations avoid high-level ML libraries and are written using **NumPy and Matplotlib only**, ensuring a deep understanding of margin maximization, hinge loss, and the kernel trick.

---

## 📌 Module Scope

This module covers:

- Linear SVM (hard and soft margins)
- Margin maximization and hinge loss
- Primal vs dual optimization intuition
- Kernel trick (linear, polynomial, RBF)
- Non-linear decision boundaries using kernels

---

## 📂 Repository Structure

```text
svm-from-scratch/
│
├── svm_linear.py                 # Linear SVM (primal form)
├── svm_kernel.py                 # Kernelized SVM (dual intuition)
├── kernels.py                    # Kernel functions
├── kernel_visualization.py       # Kernel similarity visualization
├── nonlinear_classification.py  # Non-linear classification experiments
├── requirements.txt
└── README.md
