# LEVEL 3 Research Notes: Neural Networks & Deep Learning

## Core Concepts Mastered

- **Automatic Differentiation**: Chain rule application through computational graphs, reverse-mode differentiation for efficient gradient computation
- **Convolutional Architectures**: Translation equivariance, hierarchical feature extraction, and receptive field design
- **Sequence Processing**: Recurrent connections, gating mechanisms (LSTM/GRU), and gradient flow in temporal dependencies
- **Attention Mechanisms**: Scaled dot-product attention, multi-head processing, and transformer architectures
- **Normalization Techniques**: Batch normalization for training stability, layer normalization for sequence processing
- **Regularization Methods**: Dropout for ensemble effects, weight decay for parameter constraints, data augmentation for invariance

## Key Mathematical / Algorithmic Insights

Backpropagation reveals that deep networks learn through gradient propagation, but vanishing gradients in recurrent architectures create temporal credit assignment problems that LSTMs solve through gated memory cells. The mathematical insight shows that gradient flow depends on condition numbers of weight matrices—orthogonal initialization prevents explosion while maintaining expressiveness.

Convolutional networks exploit local structure through weight sharing: each filter learns translation-invariant features, creating hierarchical representations where early layers detect edges and later layers recognize complex patterns. The insight reveals why depth matters: stacking convolutions creates exponentially large receptive fields with logarithmic parameter growth.

Transformers eliminate recurrence through self-attention: every position attends to all others simultaneously, creating quadratic complexity but enabling parallelization. The mathematical formulation shows that attention computes pairwise similarities, with multi-head processing capturing different relation types simultaneously.

## Common Failure Modes Observed

**Gradient Vanishing/Explosion**: Deep networks fail to train without careful initialization and normalization, with gradients becoming zero or infinite through repeated multiplication.

**Internal Covariate Shift**: Batch statistics change during training, destabilizing optimization and requiring normalization techniques that introduce mini-batch dependencies.

**Attention Over-parameterization**: Transformers with excessive heads and dimensions overfit to training data while failing to generalize, creating parameter inefficiency.

**Sequential Processing Limitations**: RNNs forget distant information exponentially, with gradient truncation creating ineffective credit assignment across long sequences.

**Mode Collapse in Generative Models**: Autoencoders and GANs produce limited diversity, memorizing training examples rather than learning true data distribution.

**Adversarial Vulnerability**: Small input perturbations cause confident misclassifications, revealing that networks learn superficial patterns rather than robust features.

## Trade-offs & Design Decisions

**Depth vs Optimization Stability**: Deeper networks capture complex hierarchies but require sophisticated initialization and normalization to prevent gradient issues.

**Width vs Parameter Efficiency**: Wide networks train easily but become parameter-inefficient; narrow deep networks offer better scaling but demand careful optimization.

**Recurrence vs Parallelization**: RNNs model sequential dependencies naturally but cannot parallelize across time; transformers enable parallel processing but lose inductive biases.

**Local vs Global Processing**: Convolutional networks exploit spatial locality efficiently but miss long-range dependencies; attention captures everything but scales poorly.

**Deterministic vs Stochastic Regularization**: Dropout provides ensemble effects but slows training; batch normalization stabilizes optimization but creates training-serving discrepancies.

**Homogeneous vs Heterogeneous Architectures**: Uniform layer stacking simplifies design but may not match task structure; specialized layers (convolutional, attentional) require careful architecture engineering.

## Empirical Observations

Network depth matters less than architectural design: ResNets with 50 layers often outperform custom shallow networks, but transformer depth correlates strongly with performance up to certain limits. Pre-training consistently improves performance across tasks, suggesting that learned representations matter more than architecture novelty.

Optimization proves more important than architecture: Adam with proper scheduling outperforms SGD with sophisticated models despite weaker theoretical guarantees. Data quality dominates model sophistication—well-designed networks fail on noisy data while simple models succeed on clean datasets.

Attention mechanisms generalize surprisingly well: transformers trained on language transfer to vision and reinforcement learning, suggesting that self-attention captures fundamental computation patterns rather than domain-specific structure.

## Open Questions & Research Curiosity

Can we develop architectures that maintain gradient flow indefinitely? What are the fundamental limits of depth in neural networks?

Why do transformers work so well despite quadratic complexity? What properties of natural data make attention efficient in practice?

How do different architectural choices affect generalization? Can we predict which architectures will work on new tasks?

What causes adversarial vulnerability? Are there architectural solutions beyond adversarial training?

Can we develop theoretically grounded network design? Current architectures emerge through trial-and-error—what principles should guide future designs?

How should architectures adapt to different data modalities? Vision, language, and reinforcement learning seem to benefit from different inductive biases.