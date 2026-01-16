import numpy as np

def estimate_ate_ipw(y, t, propensity_scores):
    """
    Estimates Average Treatment Effect (ATE) using Inverse Probability Weighting (IPW).
    ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
    """
    n = len(y)
    
    # Weight for Treated
    w_t = t / propensity_scores
    # Weight for Control
    w_c = (1 - t) / (1 - propensity_scores)
    
    ate = (1/n) * np.sum(y * w_t) - (1/n) * np.sum(y * w_c)
    return ate

if __name__ == "__main__":
    # Synthetic dataset
    N = 1000
    # Confounder X
    X = np.random.normal(0, 1, N)
    # Treatment T influenced by X (non-randomized)
    prob_t = 1 / (1 + np.exp(-X)) # Sigmoid
    T = (np.random.rand(N) < prob_t).astype(float)
    # Outcome Y influenced by T and X
    # True Treatment Effect = 2.0
    Y = 2.0 * T + 0.5 * X + np.random.normal(0, 0.1, N)
    
    # Simple estimate (biased due to confounder X)
    naive_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    
    # IPW Estimate
    prop_scores = prob_t # Assuming we know the propensity scores
    ipw_ate = estimate_ate_ipw(Y, T, prop_scores)
    
    print(f"Naive ATE: {naive_ate:.4f}")
    print(f"IPW ATE:   {ipw_ate:.4f}")
    print(f"True ATE:  2.0")
