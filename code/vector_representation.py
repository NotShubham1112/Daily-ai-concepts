"""
Vector Representation Experiment
--------------------------------
Shows how real-world data is represented as vectors in AI.
"""

import numpy as np

# Example: student features
# [hours studied, sleep hours, mock score]
student_a = np.array([5, 7, 65])
student_b = np.array([8, 6, 85])

print("Student A vector:", student_a)
print("Student B vector:", student_b)

# Difference between students
difference = student_b - student_a
print("Difference vector:", difference)
