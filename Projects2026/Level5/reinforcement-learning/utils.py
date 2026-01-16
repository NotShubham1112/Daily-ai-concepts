import numpy as np

def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=float)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add
    return discounted
