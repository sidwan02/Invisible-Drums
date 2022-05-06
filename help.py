import numpy as np

a = np.array([1, 2, 3, 4])

a[a > 3] *= 2
a[a > 2] *= 2
a[a > 1] *= 2

print(a)
