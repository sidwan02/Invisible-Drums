import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

b = np.array([1, 1, 1, 1])

c = sum(a * np.expand_dims(b, axis=1)) / sum(b)

print(c)
