import numpy as np
from rnn import RNN

text = "hello world"
chars = list(set(text))
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for ch,i in char_to_ix.items()}

rnn = RNN(len(chars), 16, len(chars))
h = np.zeros((16, 1))

def one_hot(ix):
    x = np.zeros((len(chars), 1))
    x[ix] = 1
    return x

inputs = [one_hot(char_to_ix[ch]) for ch in text[:-1]]
outputs, _ = rnn.forward(inputs, h)

print("Generated logits:")
for o in outputs.values():
    print(ix_to_char[np.argmax(o)])
