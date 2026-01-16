import numpy as np

def detect_drift(baseline, current, threshold=0.1):
    """
    Simple drift detection using mean/std shift.
    In production, use KS-test or PSI (Population Stability Index).
    """
    mu_b, std_b = np.mean(baseline), np.std(baseline)
    mu_c, std_c = np.mean(current), np.std(current)
    
    # Z-score of the change in mean
    shift = np.abs(mu_b - mu_c) / (std_b + 1e-10)
    
    is_drifted = shift > threshold
    return is_drifted, shift

if __name__ == "__main__":
    baseline_data = np.random.normal(0, 1, 1000)
    # Simulate serving data with shifted mean
    current_data = np.random.normal(0.5, 1, 1000)
    
    drifted, score = detect_drift(baseline_data, current_data)
    print(f"Drift detected: {drifted} (Score: {score:.4f})")
