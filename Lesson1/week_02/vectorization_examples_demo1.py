
import numpy as np
import time

#u = A*v

row1 = 1
col1 = 1000000
row2 = 1000000
col2 = 2

A = np.random.rand(row1,col1)
v = np.random.rand(row2,col2)
u = np.zeros([row1,col2])

tic = time.time()
for i in range(row1):
    for j in range(col2):
        u[i][j] = 0
        for k in range(col1):
            u[i][j]+=A[i][k]*v[k][j]
toc = time.time()

print(u)
print("For loop:"+str(1000*(toc-tic))+"ms")

tic = time.time()
u = np.dot(A,v)
toc = time.time()

print(u)
print("Vectorized version:"+str(1000*(toc-tic))+"ms")