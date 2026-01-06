import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def generate_data(n=30, noise=0.3):
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(X).ravel() + noise * np.random.randn(n)
    return X, y

degrees = [1, 3, 9]
x_test = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.figure(figsize=(10, 6))

for deg in degrees:
    preds = []

    for _ in range(50):
        X, y = generate_data()
        X_poly = np.hstack([X ** i for i in range(1, deg + 1)])
        model = LinearRegression(lr=0.01, epochs=2000)
        model.fit(X_poly, y)

        X_test_poly = np.hstack([x_test ** i for i in range(1, deg + 1)])
        preds.append(model.predict(X_test_poly))

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)

    plt.plot(x_test, mean_pred, label=f"Degree {deg}")

plt.title("Bias–Variance Tradeoff via Polynomial Regression")
plt.legend()
plt.grid()
plt.show()
