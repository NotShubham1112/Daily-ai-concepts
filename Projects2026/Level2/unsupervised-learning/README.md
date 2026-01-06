# Unsupervised Learning — From-Scratch Implementations & Comparative Analysis

A comprehensive, math-first implementation of core **unsupervised learning algorithms**, built from scratch and rigorously compared against industry-standard libraries.  
This repository prioritizes **algorithmic understanding, reproducibility, and visual interpretability**.

---

## 📌 Objectives

- Implement foundational unsupervised learning algorithms **from first principles**
- Develop intuition behind **clustering and representation learning**
- Compare scratch implementations with **scikit-learn and UMAP**
- Provide a **clean, extensible research-grade codebase**

---

## 📂 Repository Structure

unsupervised-learning/
│
├── README.md
│
├── data/
│   ├── synthetic_blobs.py
│   └── load_data.py
│
├── kmeans/
│   ├── kmeans_from_scratch.py
│   └── kmeans_sklearn_compare.py
│
├── dbscan/
│   ├── dbscan_from_scratch.py
│   └── dbscan_vs_kmeans.py
│
├── dimensionality_reduction/
│   ├── pca_from_scratch.py
│   ├── tsne_visualization.py
│   ├── umap_visualization.py
│   └── comparison.py
│
├── utils/
│   ├── distance_metrics.py
│   └── visualization.py
│
├── requirements.txt
└── run_all.py
