"""
Generalization Check
--------------------
Seen vs unseen data behavior.
"""

def model(x):
    return 2 * x

seen = [1, 2, 3]
unseen = [10, 20]

print("Seen data predictions:")
for x in seen:
    print(x, model(x))

print("\nUnseen data predictions:")
for x in unseen:
    print(x, model(x))
