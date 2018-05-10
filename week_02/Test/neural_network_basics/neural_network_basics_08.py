
import numpy as np

a = np.random.rand(3,4)
b = np.random.rand(4,1)
c = np.zeros((3,4))

for i in range(3):
    for j in range(4):
        c[i][j] = a[i][j] + b[j]

print(a+b.T)
print(c)