#encoding:utf-8

import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def loss(a,y):
    return -(y*np.log(a)+(1-y)*np.log(1-a))

#z=w*x+b
def gradient_descent(X,Y,w=0,b=0,alpha=0.1):
    J = 0
    dw = 0
    db = 0
    m = len(X)

    for i in range(m):
        z = w*X[i]+b
        a = sigmoid(z)
        J += loss(a,Y[i])
        dz = a - Y[i]
        dw += X[i]*dz
        db += dz
    J /= m
    dw /= m
    db /= m

    w = w-alpha*dw
    b = b-alpha*db
    return w,b

X = [0,1,2,4,5,6,7,8]
Y = [1,0,1,0,1,0,1,0]

w,b = gradient_descent(X,Y)

for i in range(10):
    w,b = gradient_descent(X,Y,w,b)

print(w,b)