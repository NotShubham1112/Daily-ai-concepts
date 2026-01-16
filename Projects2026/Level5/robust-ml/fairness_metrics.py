import numpy as np

def calculate_disparate_impact(preds, sensitive_attr):
    """
    Disparate Impact = (P(Y=1|S=1)) / (P(Y=1|S=0))
    Ideal value: 1.0 (Fair)
    Value < 0.8 is often considered biased.
    """
    group_1 = preds[sensitive_attr == 1]
    group_0 = preds[sensitive_attr == 0]
    
    rate_1 = np.mean(group_1)
    rate_0 = np.mean(group_0)
    
    return rate_1 / (rate_0 + 1e-10)

if __name__ == "__main__":
    # 100 individuals, sensitive attribute S (e.g., gender)
    S = np.random.choice([0, 1], size=100)
    # Predictions Y_hat (biased towards group 1)
    Y_hat = np.zeros(100)
    Y_hat[S == 1] = np.random.choice([0, 1], size=sum(S==1), p=[0.3, 0.7])
    Y_hat[S == 0] = np.random.choice([0, 1], size=sum(S==0), p=[0.6, 0.4])
    
    di = calculate_disparate_impact(Y_hat, S)
    print(f"Disparate Impact Score: {di:.4f}")
    if di < 0.8 or di > 1.25:
        print("Model shows significant bias based on the 80% rule.")
