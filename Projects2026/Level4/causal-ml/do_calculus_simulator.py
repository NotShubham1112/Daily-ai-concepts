import numpy as np

class StructuralCausalModel:
    """
    Simulates a chain: X -> Y -> Z
    """
    def __init__(self, n_samples=1000):
        self.n = n_samples
        
    def sample(self, intervention=None):
        """
        intervention: dict like {'Y': value}
        """
        # Exogenous noise
        u_x = np.random.normal(0, 1, self.n)
        u_y = np.random.normal(0, 1, self.n)
        u_z = np.random.normal(0, 1, self.n)
        
        # X = U_x
        X = u_x
        
        # Y = f(X, U_y)
        if intervention and 'Y' in intervention:
            Y = np.full(self.n, intervention['Y'])
        else:
            Y = 0.8 * X + u_y
            
        # Z = f(Y, U_z)
        Z = 1.2 * Y + u_z
        
        return {'X': X, 'Y': Y, 'Z': Z}

if __name__ == "__main__":
    scm = StructuralCausalModel()
    
    # Observational data
    obs = scm.sample()
    print(f"Corr(X, Z) observational: {np.corrcoef(obs['X'], obs['Z'])[0,1]:.4f}")
    
    # Interventional data: do(Y=2)
    inter = scm.sample(intervention={'Y': 2.0})
    print(f"Corr(X, Z) interventional do(Y=2): {np.corrcoef(inter['X'], inter['Z'])[0,1]:.4f}")
    print(f"Mean Z interventional: {np.mean(inter['Z']):.4f} (Ideal: 1.2 * 2 = 2.4)")
