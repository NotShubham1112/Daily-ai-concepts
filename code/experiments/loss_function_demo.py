"""
Loss Function Demo
------------------
Mean Squared Error from scratch.
"""

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

y_true = 10
predictions = [2, 5, 8, 10, 12]

for y_pred in predictions:
    print(f"Prediction: {y_pred}, Loss: {mse(y_true, y_pred)}")
