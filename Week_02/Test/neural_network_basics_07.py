
import numpy as np

a = np.random.rand(12288,150) #a.shape = (12288,150)
b = np.random.rand(150,45) #b.shape = (150,45)
c = np.dot(a,b)

print(c.shape)