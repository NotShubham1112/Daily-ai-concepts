import numpy as np
import matplotlib.pyplot as plt
from bayesian_linear_regression import BayesianLinearRegression

def plot_bayesian_uncertainty():
    # Generate noisy synthetic data
    np.random.seed(42)
    def target_func(x): return 0.5 * x + np.sin(x)
    
    X_train = np.sort(np.random.uniform(-5, 5, 10))
    y_train = target_func(X_train) + np.random.normal(0, 0.4, size=X_train.shape)
    
    # Feature mapping (Polynomial)
    def get_features(X, degree=4):
        return np.vander(X, degree + 1, increasing=True)
    
    X_poly = get_features(X_train)
    
    # Model
    model = BayesianLinearRegression(alpha=0.1, beta=1.0)
    model.fit(X_poly, y_train)
    
    # Predict
    X_plot = np.linspace(-7, 7, 200)
    X_plot_poly = get_features(X_plot)
    mu_y, std_y = model.predict(X_plot_poly)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X_plot, target_func(X_plot), 'r--', label='True function', alpha=0.5)
    plt.scatter(X_train, y_train, c='black', label='Observed Data')
    plt.plot(X_plot, mu_y, 'b-', label='Mean Prediction')
    plt.fill_between(X_plot, mu_y - 2*std_y, mu_y + 2*std_y, color='blue', alpha=0.2, label='95% Confidence')
    
    plt.title("Bayesian Linear Regression Uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_bayesian_uncertainty()
