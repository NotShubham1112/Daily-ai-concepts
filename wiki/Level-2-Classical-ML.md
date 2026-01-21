# 🧮 Level 2: Classical Machine Learning (From Scratch)

<div align="center">
  <h2>⚙️ Core Algorithms Without High-Level Libraries</h2>
  <p><em>First-principles understanding of fundamental ML techniques</em></p>
</div>

---

## 📋 **Navigation**
- **[🏠 Wiki Home](Home.md)** - Repository overview
- **[🗺️ Learning Roadmap](Roadmap.md)** - Study progression
- **[📖 Research Notes](../research-notes/LEVEL_2_RESEARCH_NOTES.md)** - Technical insights
- **[💻 Code Projects](../Projects2026/Level2/)** - From-scratch implementations

---

## 🎯 **Level Overview**

**Level 2** implements core machine learning algorithms **from scratch** using only NumPy and basic libraries. This approach reveals the **fundamental assumptions** and **inherent trade-offs** underlying each algorithm, providing deep insight into bias-variance decomposition, model capacity, and geometric interpretations of learning.

### 🏗️ **Core Focus Areas**
- **Linear Models** - Regression and classification from first principles
- **Decision Trees** - Information-theoretic splitting and ensemble methods
- **Support Vector Machines** - Maximum margin classification
- **Unsupervised Learning** - Clustering and dimensionality reduction

---

## 📚 **Algorithm Implementations**

### 📈 **Linear & Logistic Regression**

#### **Key Concepts**
- **Maximum Likelihood Estimation** - Parameter learning from data
- **Gradient Descent** - Optimization for parameter fitting
- **Regularization** - L1/L2 penalties for complexity control
- **Probabilistic Interpretation** - Logistic as sigmoid + cross-entropy

#### **From-Scratch Implementation**
- **Matrix operations** for efficient computation
- **Numerical stability** considerations
- **Convergence monitoring** and early stopping

#### **Resources**
- **[📖 Implementation Details](../research-notes/LEVEL_2_RESEARCH_NOTES.md)**
- **[💻 Linear Regression](../Projects2026/Level2/linear-models-ml/linear_regression.py)**
- **[📊 Regularization Comparison](../Projects2026/Level2/linear-models-ml/regularization_comparison.py)**

---

### 🌳 **Decision Trees & Ensemble Methods**

#### **Key Concepts**
- **Information Gain** - Entropy reduction for splitting
- **Gini Impurity** - Alternative splitting criteria
- **Random Forests** - Bootstrap aggregation and feature subsampling
- **Gradient Boosting** - Sequential error correction

#### **From-Scratch Implementation**
- **Recursive tree building** with stopping criteria
- **Ensemble aggregation** strategies
- **Feature importance** calculation

#### **Resources**
- **[📖 Tree Algorithms Research](../research-notes/LEVEL_2_RESEARCH_NOTES.md#key-mathematical--algorithmic-insights)**
- **[💻 Decision Tree](../Projects2026/Level2/decision-trees-ensemble/decision_tree.py)**
- **[🎯 Random Forest](../Projects2026/Level2/decision-trees-ensemble/random_forest.py)**

---

### 🎯 **Support Vector Machines**

#### **Key Concepts**
- **Maximum Margin Classification** - Optimal hyperplane finding
- **Kernel Methods** - Non-linear classification through feature mapping
- **Soft Margins** - Handling non-separable data
- **Dual Formulation** - Computational efficiency

#### **From-Scratch Implementation**
- **Quadratic programming** for optimal hyperplane
- **Kernel functions** (linear, polynomial, RBF)
- **SMO algorithm** for large-scale optimization

#### **Resources**
- **[📖 SVM Theory](../research-notes/LEVEL_2_RESEARCH_NOTES.md#common-failure-modes-observed)**
- **[💻 SVM Implementation](../Projects2026/Level2/svm-from-scratch/svm_linear.py)**
- **[🔄 Kernel Methods](../Projects2026/Level2/svm-from-scratch/svm_kernel.py)**

---

### 🎨 **Unsupervised Learning**

#### **Key Concepts**
- **K-Means Clustering** - Centroid-based partitioning
- **DBSCAN** - Density-based clustering
- **Dimensionality Reduction** - PCA and t-SNE
- **Evaluation Metrics** - Silhouette scores, purity measures

#### **From-Scratch Implementation**
- **Distance metrics** and similarity measures
- **Expectation-maximization** for probabilistic clustering
- **Manifold learning** techniques

#### **Resources**
- **[📖 Unsupervised Methods](../research-notes/LEVEL_2_RESEARCH_NOTES.md#empirical-observations)**
- **[💻 K-Means Algorithm](../Projects2026/Level2/unsupervised-learning/kmeans/kmeans_from_scratch.py)**
- **[🗺️ Dimensionality Reduction](../Projects2026/Level2/unsupervised-learning/dimensionality_reduction/)**

---

## 🔬 **Research Insights**

### 💡 **Algorithmic Deep Dives**
- **Bias-Variance Tradeoff** - Model complexity vs generalization
- **Geometric Interpretation** - Decision boundaries as mathematical objects
- **Computational Complexity** - Training vs prediction efficiency
- **Robustness Analysis** - Sensitivity to hyperparameters and data

### ⚠️ **Failure Modes & Debugging**
- **Overfitting** - High variance on unseen data
- **Underfitting** - High bias, inability to capture patterns
- **Numerical Issues** - Ill-conditioned optimization
- **Hyperparameter Sensitivity** - Critical parameter tuning

### 🤔 **Design Decisions**
- **Model Selection** - When to use trees vs linear models vs SVMs
- **Feature Engineering** - Preprocessing impact on algorithm performance
- **Scalability Considerations** - Computational requirements

---

## 🧭 **Learning Progression**

### 📚 **Building on Level 1**
- **Optimization Theory** → Gradient descent in practice
- **Linear Algebra** → Matrix operations for efficiency
- **Probability** → Maximum likelihood parameter estimation
- **Information Theory** → Decision tree splitting criteria

### 🎯 **Level 2 Mastery Goals**
- **Implement algorithms** from mathematical definitions
- **Diagnose model behavior** using theoretical understanding
- **Compare approaches** based on problem characteristics
- **Select appropriate methods** for specific applications

### 🚀 **Bridge to Advanced Topics**
- **Neural Networks** - Non-linear extensions of linear models
- **Deep Learning** - Hierarchical feature learning
- **Probabilistic Methods** - Uncertainty quantification

---

## 📖 **Implementation Details**

### 💻 **Code Structure**
Each algorithm includes:
- **Core implementation** with detailed comments
- **Visualization utilities** for understanding
- **Comparison scripts** against scikit-learn
- **Performance analysis** tools

### 🧪 **Empirical Validation**
- **Convergence testing** - Optimization behavior
- **Accuracy comparison** - Against established implementations
- **Robustness analysis** - Parameter sensitivity
- **Computational benchmarking** - Speed and memory usage

### 🔧 **Extensibility**
- **Modular design** - Easy to modify and extend
- **Hyperparameter interfaces** - Systematic tuning
- **Custom loss functions** - Algorithm adaptation

---

## 📋 **Complete Project Catalog**

### 📈 **Linear Models**
```
linear-models-ml/
├── linear_regression.py          # OLS implementation
├── logistic_regression.py        # Sigmoid classification
├── regularization_comparison.py  # L1/L2 analysis
└── bias_variance.py              # Decomposition study
```

### 🌳 **Decision Trees & Ensembles**
```
decision-trees-ensemble/
├── decision_tree.py              # ID3/CART algorithm
├── random_forest.py              # Bootstrap aggregation
├── feature_importance.py         # Variable significance
└── split_criteria.py             # Information gain/Gini
```

### 🎯 **Support Vector Machines**
```
svm-from-scratch/
├── svm_linear.py                 # Hard/soft margins
├── svm_kernel.py                 # Kernel trick
├── kernel_visualization.py       # Decision boundaries
└── nonlinear_classification.py   # RBF applications
```

### 🎨 **Unsupervised Learning**
```
unsupervised-learning/
├── kmeans/                       # Centroid clustering
├── dbscan/                       # Density clustering
├── dimensionality_reduction/     # PCA, t-SNE, UMAP
└── evaluation/                   # Clustering metrics
```

---

## 🔗 **Quick Navigation**

### 📚 **Level Progression**
- **[← Level 1](Level-1-Mathematical-Foundations.md)** - Mathematical foundations
- **[Level 3 →](Level-3-Neural-Networks.md)** - Neural architectures
- **[Level 4 →](Level-4-Probabilistic-ML.md)** - Advanced probabilistic methods

### 📖 **Research & Code**
- **[📖 Full Research Notes](../research-notes/LEVEL_2_RESEARCH_NOTES.md)**
- **[💻 All Level 2 Projects](../Projects2026/Level2/)**
- **[🧪 Comparison Scripts](../Projects2026/Level2/linear-models-ml/utils.py)**

### 🏠 **Repository Navigation**
- **[🏠 Wiki Home](Home.md)** - Complete overview
- **[🗺️ Roadmap](Roadmap.md)** - Learning progression

---

<div align="center">
  <strong>🚀 **Mastered classical ML?** Advance to [Level 3: Neural Networks & Deep Learning](Level-3-Neural-Networks.md)</strong>
</div>