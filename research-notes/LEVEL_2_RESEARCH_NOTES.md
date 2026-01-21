# LEVEL 2 Research Notes: Classical Machine Learning (From Scratch)

## Core Concepts Mastered

- **Bias-Variance Decomposition**: Systematic breakdown of generalization error into reducible and irreducible components, revealing fundamental limits of learning
- **Geometric Interpretation of Classification**: Linear models as hyperplane separators in feature space, with logistic regression emerging as maximum likelihood estimation under Bernoulli noise
- **Decision Tree Optimization**: Recursive binary splitting as greedy optimization of information gain, balancing depth vs overfitting through pruning
- **Ensemble Learning Mechanics**: Bootstrap aggregation reducing variance through uncorrelated weak learners, gradient boosting as functional gradient descent
- **Support Vector Machines**: Maximum margin classification as constrained optimization, kernel trick extending linear separability to infinite dimensions
- **Unsupervised Learning Objectives**: K-means as EM algorithm for Gaussian mixtures, spectral clustering revealing manifold structure through graph Laplacians

## Key Mathematical / Algorithmic Insights

Linear regression converges to minimum variance unbiased estimator under Gaussian noise assumptions, but real data violates these assumptions spectacularly. The "best linear unbiased estimator" guarantee holds only when model specification matches data generating process—rarely true in practice. Logistic regression's sigmoid emerges naturally from maximum likelihood under Bernoulli likelihood, but gradient descent on cross-entropy loss reveals optimization landscapes with multiple local minima despite global convexity in parameter space.

Decision trees minimize impurity measures (Gini, entropy) through greedy splitting, but this myopic optimization often produces suboptimal global structure. Ensemble methods compensate through variance reduction: random forests decorrelate trees via bootstrap sampling and random feature selection, while gradient boosting sequentially corrects residual errors through functional gradient descent.

SVMs maximize geometric margin through quadratic programming, with kernel methods implicitly mapping to high-dimensional spaces where linear separation becomes possible. The mathematical elegance hides computational intractability—SMO algorithm's coordinate descent emerges as practical necessity rather than theoretical optimality.

## Common Failure Modes Observed

**Overfitting in High Dimensions**: Linear models with more features than samples exhibit perfect training fit but catastrophic generalization, violating fundamental assumptions of statistical learning theory.

**Decision Tree Instability**: Small data perturbations cause radically different tree structures, with greedy splitting amplifying variance through cascading decisions.

**Ensemble Diversity Collapse**: Random forests fail when features are highly correlated, reducing ensemble diversity and negating variance reduction benefits.

**SVM Sensitivity to Scaling**: Unnormalized features create artificial importance hierarchies, with optimization converging to suboptimal margins in transformed spaces.

**Clustering Initialization Dependence**: K-means convergence depends critically on centroid initialization, often trapping in poor local optima despite theoretical guarantees.

**Class Imbalance Exploitation**: Algorithms optimize global objectives while minority classes suffer silent degradation, revealing objective function misalignment with application requirements.

## Trade-offs & Design Decisions

**Model Capacity vs Generalization**: Simpler models (linear regression) exhibit better generalization bounds but fail on complex patterns; complex models (SVMs with kernels) fit intricate decision boundaries but risk overfitting with limited data.

**Computational Efficiency vs Accuracy**: Exact optimization (SVM quadratic programming) scales poorly with data size; approximate methods (stochastic gradient descent) sacrifice convergence guarantees for practicality.

**Interpretability vs Performance**: Decision trees provide transparent decision rules but suffer from instability; ensemble methods boost accuracy through aggregation but obscure individual predictions.

**Memory vs Prediction Speed**: Kernel methods enable non-linear classification without explicit feature expansion but require storing entire training set for prediction, creating deployment bottlenecks.

**Robustness vs Sensitivity**: Parametric models (linear regression) handle noise gracefully through regularization; non-parametric methods (k-means) remain sensitive to outliers and initialization.

## Empirical Observations

Cross-validation reveals that bias-variance trade-off manifests differently across domains: financial data shows high irreducible error due to market noise, while image classification demonstrates low bias with sufficient model capacity. Bootstrap sampling exposes that ensemble methods' effectiveness depends on base learner instability—stable algorithms (linear regression) benefit less from aggregation than unstable ones (decision trees).

Real datasets violate independence assumptions spectacularly: temporal correlations in time series, spatial dependencies in images, and hierarchical structures in text all create challenges that theoretical guarantees ignore. Feature engineering emerges as crucial art rather than science, with domain knowledge often outweighing algorithmic sophistication.

## Open Questions & Research Curiosity

Why does ensemble diversity matter more than individual learner strength? Can we quantify the relationship between base learner correlation and ensemble performance bounds?

How do optimization landscapes differ between convex (linear regression) and non-convex (decision trees) objectives? What topological properties determine convergence to global vs local optima?

Can we develop theoretically grounded feature selection beyond wrapper methods? How does mutual information capture non-linear feature relevance?

What are the fundamental limits of unsupervised learning without ground truth? How can we evaluate clustering quality beyond heuristic metrics?

How should we handle distribution shift between training and deployment? Current methods assume stationarity, but real systems face continuous concept drift.