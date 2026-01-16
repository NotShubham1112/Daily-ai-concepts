import numpy as np

def metropolis_hastings(target_pdf, initial_state, n_samples=1000, proposal_width=0.5):
    """
    Standard Metropolis-Hastings MCMC implementation.
    """
    samples = []
    current_state = initial_state
    
    for _ in range(n_samples):
        # Propose new state
        proposal = current_state + np.random.normal(0, proposal_width)
        
        # Calculate acceptance probability
        p_current = target_pdf(current_state)
        p_proposal = target_pdf(proposal)
        
        acceptance_ratio = p_proposal / (p_current + 1e-10)
        
        # Accept or reject
        if np.random.rand() < acceptance_ratio:
            current_state = proposal
            
        samples.append(current_state)
        
    return np.array(samples)

def gaussian_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Usage example
if __name__ == "__main__":
    # Sample from a custom distribution (e.g., standard normal)
    samples = metropolis_hastings(gaussian_pdf, initial_state=0.0, n_samples=5000)
    print(f"MCMC Mean: {np.mean(samples):.4f} (Ideal: 0)")
    print(f"MCMC Std: {np.std(samples):.4f} (Ideal: 1)")
