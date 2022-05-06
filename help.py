import numpy as np


def func3(x):
    y = 1 / (np.e ** -((x - 20) / 26)) + 50
    return int(y)


print(func3(0))
print(func3(50))
print(func3(100))
print(func3(150))
print(func3(200))
print(func3(250))
