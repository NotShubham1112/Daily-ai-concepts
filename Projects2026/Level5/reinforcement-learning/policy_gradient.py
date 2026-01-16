import numpy as np

def update_policy(params, gradients, rewards, learning_rate=0.01):
    """
    Conceptual REINFORCE update step:
    theta = theta + alpha * grad(log pi(a|s)) * G
    """
    # G = return (discounted sum of rewards)
    G = np.sum(rewards)
    params += learning_rate * gradients * G
    return params

if __name__ == "__main__":
    # Simulate a policy parameter (e.g., weights of a linear policy)
    weights = np.array([0.5, -0.2])
    # Simulate some log-probability gradients and the reward obtained
    grads = np.array([0.1, 0.05])
    rewards = [1.0, 1.0, 2.0]
    
    new_weights = update_policy(weights, grads, rewards)
    print(f"Old Weights: {weights}")
    print(f"New Weights: {new_weights}")
    
