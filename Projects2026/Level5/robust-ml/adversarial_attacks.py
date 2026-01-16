import numpy as np

def fgsm_perturbation(x, epsilon, data_grad):
    """
    Fast Gradient Sign Method (FGSM)
    x_adv = x + epsilon * sign(gradient)
    """
    # Collect the sign of the data gradient
    sign_data_grad = np.sign(data_grad)
    # Create the perturbed image
    perturbed_x = x + epsilon * sign_data_grad
    # Clamping is often used to keep value in [0,1] or [-1, 1]
    return np.clip(perturbed_x, 0, 1)

if __name__ == "__main__":
    # Simulate a 1D input and its gradient w.r.t. loss
    x = np.array([0.5, 0.2, 0.8])
    grad = np.array([0.1, -0.05, 0.02]) # Grad w.r.t loss
    
    x_adv = fgsm_perturbation(x, epsilon=0.1, data_grad=grad)
    print(f"Original:  {x}")
    print(f"Adversarial: {x_adv}")
