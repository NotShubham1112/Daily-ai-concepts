# LEVEL 4 Research Notes: Probabilistic & Advanced Machine Learning

## Core Concepts Mastered

- **Bayesian Inference Framework**: Probabilistic modeling with uncertainty quantification, posterior computation through likelihood and priors
- **Markov Chain Monte Carlo**: Sampling from complex distributions via Metropolis-Hastings and Gibbs sampling for intractable posteriors
- **Variational Inference**: Approximation of posterior distributions through optimization, trading exactness for computational tractability
- **Graph Neural Networks**: Message passing on graph structures, convolution operations on relational data
- **Causal Inference**: Directed acyclic graphs (DAGs) representing causal relationships, intervention analysis through do-operator calculus
- **Counterfactual Reasoning**: Potential outcomes framework for understanding treatment effects beyond correlation

## Key Mathematical / Algorithmic Insights

Bayesian inference reveals that uncertainty quantification emerges naturally from probability theory: posterior distributions encode belief updates, with credible intervals providing calibrated uncertainty estimates. Markov chains converge to target distributions through detailed balance, but mixing times determine practical feasibility—poor mixing creates samples that fail to explore the distribution adequately.

Variational inference reformulates posterior computation as optimization: evidence lower bound (ELBO) maximization approximates intractable integrals through tractable distributions. The mathematical elegance hides approximation quality dependence on variational family expressiveness—mean-field assumptions often prove too restrictive for correlated variables.

Graph convolutions extend CNN principles to irregular structures: node features aggregate through neighborhood summation weighted by edge attributes, creating hierarchical representations of relational data. Spectral graph theory provides foundation: graph Laplacians encode connectivity structure, with eigenvectors representing frequency components on graph manifolds.

Causal inference distinguishes correlation from causation through intervention: do-operator severs incoming edges in causal graphs, enabling counterfactual predictions. Simpson's paradox reveals how conditioning can reverse observed associations, necessitating careful consideration of confounding variables.

## Common Failure Modes Observed

**Posterior Collapse in Variational Autoencoders**: Latent variables become deterministic during training, negating uncertainty quantification benefits and reducing generative capacity.

**Graph Neural Network Over-smoothing**: Repeated message passing homogenizes node features across distant parts of the graph, destroying structural information through over-regularization.

**Confounding Variable Omission**: Unobserved confounders create spurious correlations that persist even after conditioning, violating causal inference assumptions.

**MCMC Poor Mixing**: Chains remain trapped in local modes of multimodal distributions, producing biased samples that fail to represent true posterior uncertainty.

**Plateau Effects in Causal Discovery**: Score-based structure learning algorithms converge to local optima in DAG space, producing incorrect causal graphs despite asymptotic guarantees.

**Distribution Shift in Probabilistic Models**: Training and test distributions diverge, causing calibrated uncertainty estimates on training data to become miscalibrated on deployment.

## Trade-offs & Design Decisions

**Exact vs Approximate Inference**: MCMC provides asymptotically exact samples but scales poorly with dimension; variational methods offer computational efficiency at the cost of approximation bias.

**Model Expressiveness vs Computational Tractability**: Rich Bayesian models capture complex uncertainty structures but require sophisticated inference algorithms; simpler approximations sacrifice accuracy for practicality.

**Causal Assumptions vs Statistical Power**: Rigorous causal inference requires strong assumptions about data generating process; relaxed assumptions enable broader applicability but weaken causal claims.

**Graph Structure Learning vs Fixed Topology**: Learning graph structure from data enables adaptability but introduces optimization challenges; fixed graphs provide computational stability but limit expressiveness.

**Frequentist vs Bayesian Approaches**: Frequentist methods offer computational simplicity and clear hypothesis testing; Bayesian methods provide uncertainty quantification but require prior specification and complex inference.

## Empirical Observations

Bayesian neural networks exhibit better calibration than frequentist counterparts, but computational overhead often negates benefits in production settings. Uncertainty estimates correlate with prediction accuracy, but the relationship breaks down under distribution shift—models become overconfident on out-of-distribution samples.

Graph neural networks demonstrate surprising generalization across domains: message passing architectures trained on molecular data transfer effectively to social network analysis, suggesting shared structural learning principles. However, performance degrades dramatically on graphs with different degree distributions, revealing sensitivity to topological assumptions.

Causal effect estimation reveals that naive observational studies overestimate treatment effects by 2-3x compared to randomized controlled trials, quantifying the value of experimental design over statistical adjustment. Instrumental variable methods recover true effects only when instruments satisfy exclusion restrictions—violations create worse bias than unadjusted estimates.

## Open Questions & Research Curiosity

Can we develop scalable Bayesian inference that maintains uncertainty quantification without prohibitive computational costs? What architectural innovations could make Bayesian deep learning practical?

How do graph neural networks balance local structure learning with global topology awareness? What are the fundamental limits of message passing on different graph classes?

What constitutes sufficient causal assumptions for real-world applications? Can we develop methods that quantify assumption violations rather than assuming their satisfaction?

How should uncertainty estimation adapt to distribution shift? Current calibration methods fail catastrophically—can we develop robust uncertainty quantification?

Can counterfactual reasoning scale beyond individual predictions? How might we enable population-level causal inference for policy decisions?

What are the theoretical limits of learning causal structure from observational data? When does structure learning become fundamentally impossible?