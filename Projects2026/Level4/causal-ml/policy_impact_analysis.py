import numpy as np
from do_calculus_simulator import StructuralCausalModel

def analyze_policy_impact():
    scm = StructuralCausalModel(n_samples=5000)
    
    # Policy A: No intervention
    baseline = scm.sample()
    avg_z_baseline = np.mean(baseline['Z'])
    
    # Policy B: Fix Y to 1.0 (e.g., standardizing a process)
    policy_b = scm.sample(intervention={'Y': 1.0})
    avg_z_policy_b = np.mean(policy_b['Z'])
    
    # Policy C: Fix Y to 5.0 (e.g., aggressive investment)
    policy_c = scm.sample(intervention={'Y': 5.0})
    avg_z_policy_c = np.mean(policy_c['Z'])
    
    print("Policy Impact Analysis (Targeting Z):")
    print(f"Baseline (None): {avg_z_baseline:.4f}")
    print(f"Policy B (Y=1):  {avg_z_policy_b:.4f}")
    print(f"Policy C (Y=5):  {avg_z_policy_c:.4f}")

if __name__ == "__main__":
    analyze_policy_impact()
