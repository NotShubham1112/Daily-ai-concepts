import numpy as np
from rnn import RNN

X = np.sin(np.linspace(0, 10, 100))
y = X[1:]

rnn = RNN(1, 8, 1)
h = np.zeros((8, 1))

inputs = [np.array([[v]]) for v in X[:-1]]
outputs, _ = rnn.forward(inputs, h)

preds = [o[0, 0] for o in outputs.values()]
print("Predictions:", preds[:10])
