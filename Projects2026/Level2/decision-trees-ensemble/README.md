# Decision Trees & Ensemble Methods — Classical Machine Learning (Level 2)

This module focuses on **tree-based learning algorithms**, which form the backbone of many high-performing classical machine learning systems used in industry and research.

All models in this repository are implemented **from scratch**, without relying on machine learning libraries such as `scikit-learn`, in order to build a precise understanding of how tree-based models learn, generalize, and scale.

---

## 📌 Module Scope

This module covers:

- Decision Trees (Classification & Regression)
- Feature selection using information gain
- Ensemble learning principles
- Random Forest intuition
- Gradient Boosting fundamentals
- Bias–variance behavior in ensembles

Only the following libraries are used:
- **NumPy** (numerical computation)
- **Matplotlib** (visualization)

---

## 📂 Repository Structure

```text
decision-trees-ensemble/
│
├── decision_tree.py              # Decision Tree from scratch
├── split_criteria.py             # Gini, Entropy, Information Gain
├── feature_importance.py         # Feature importance analysis
├── random_forest.py              # Random Forest (conceptual implementation)
├── ensemble_bias_variance.py     # Bias–variance study for ensembles
├── utils.py                      # Helper functions
├── requirements.txt
└── README.md
