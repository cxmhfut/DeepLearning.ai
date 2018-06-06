import numpy as np
import time

size = 1000000

v = np.random.rand(size)
u = np.zeros(size)

tic = time.time()
for i in range(size):
    u[i] = np.exp(v[i])
toc = time.time()

print(u)
print("For loop:"+str(1000*(toc-tic))+"ms")

tic = time.time()
u = np.exp(v)
toc = time.time()

print(u)
print("Vectorized version:"+str(1000*(toc-tic))+"ms")