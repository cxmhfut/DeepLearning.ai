import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.where(x<=0,0,x)

def leaky_relu(x):
    return np.where(x<=0,0.1*x,x)

x = np.linspace(-10,10,1000)
y = relu(x)
z = leaky_relu(x)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

#设置显示边界
plt.xlim(-11,11)
plt.ylim(-11,11)

#将上方和下方的边界隐藏
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

#将下边界设置为x轴并移动到数据0的位置
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10,-5,0,5,10]) #设置坐标轴刻度
#将左边界设置为y轴并移动到数据0的位置
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-10,-5,5,10])

plt.plot(x,y,label='relu',color='blue')
plt.plot(x,z,label='Leaky Relu',color='red')
plt.legend() #显示标签
plt.show()