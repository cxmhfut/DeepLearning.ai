#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

x = np.linspace(-10,10)
y = sigmoid(x)

plt.plot(x,y,label='sigmoid',color='blue')
plt.show()