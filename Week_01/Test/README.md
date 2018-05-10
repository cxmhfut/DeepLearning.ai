## Introduction to Deep Learning

<h3> 1 What does the analogy “AI is the new electricity” refer to? </h3>

- A Similar to electricity starting about 100 years ago, AI is transforming multiple industries.
- B Through the “smart grid”, AI is delivering a new wave of electricity.
- C AI runs on computers and is thus powered by electricity, but it is letting computers do things not possible before.
- D AI is powering personal devices in our homes and offices, similar to electricity.

```
AI是新一轮电力革命比喻的是什么？
A 与大约100年前开始的电力类似，人工智能正在改变多个行业。
B 通过“智能电网”，AI正在推出新一波电力。
C AI在计算机上运行，并由电力驱动，但它使计算机无法做到以前不可能做的事情。
D AI正在为家庭和办公室的个人设备供电，类似于电力。
```
就像一百年前电力改造了每个主流行业，当今的 AI 技术在做着相同的事。好几个大
型科技公司都设立了 AI 部门，用 AI 革新他们的业务。接下来的几年里，各个行业、规
模大小各不相同的公司也都会意识到-----在由 AI 驱动的未来，他们必须成为其中的一份
子。

<h5>Answer:A</h5>

<h3> 2 Which of these are reasons for Deep Learning recently taking off? (Check the three options that apply.) </h3>

- A Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition.
- B We have access to a lot more data.
- C We have access to a lot more computational power.
- D Neural Networks are a brand new field.

```
下面那些是深度学习时下流行的原因？
A 深度学习已经在重要的应用程序，如在线广告，语音识别和图像识别方面取得重大进展。
B 我们可以访问更多的数据。
C 我们可以获得更多的计算能力。
D 神经网络是一个全新的领域。
```

<h5>Answer:ABC</h5>

<h3> 3 Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.) </h3>

![introduction_to_deep_learning_03](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_03.png)

- A Being able to try out ideas quickly allows deep learning engineers to iterate more quickly.
- B Faster computation can help speed up how long a team takes to iterate to a good idea.
- C It is faster to train on a big dataset than a small dataset.
- D Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware).

```
回想一下这个迭代不同ML想法的图表。 以下哪项陈述属实？
A 能够迅速尝试创意使得深度学习工程师能够更快地进行迭代。
B 更快的计算可以帮助加快团队花费多少时间来迭代一个好主意。
C 在大数据集上训练比在小数据集上训练要快。
D 深度学习算法的最新进展使我们能够更快地训练出好的模型（即使不改变CPU / GPU硬件）。
```

<h5>Answer:ABD</h5>

<h3> 4 When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False? </h3>

- A True
- B False

```
当一位经验丰富的深度学习工程师研究一个新问题时，他们通常可以利用以前问题的见解来第一次尝试训练出一个好的模型，
而不需要通过不同的模型多次迭代。 真假？
```

<h5>Answer:B</h5>

<h3> 5 Which one of these plots represents a ReLU activation function? </h3>

- A Figure 1:

![introduction_to_deep_learning_05_a](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_05_a.png)
- B Figure 2:

![introduction_to_deep_learning_05_b](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_05_b.png)
- C Figure 3:

![introduction_to_deep_learning_05_c](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_05_c.png)
- D Figure 4:

![introduction_to_deep_learning_05_d](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_05_d.png)

A is tanh, B is sigmoid, C is ReLU, D is Leaky ReLU

<h5>Answer:C</h5>

<h3> 6 Images for cat recognition is an example of “structured” data, because it is represented as a structured array in a computer. True/False? </h3>

- A True
- B False

<h5>Answer:B</h5>

<h3> 7 A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data because it contains data coming from different sources. True/False? </h3>

- A True
- B False

结构化数据意味着数
据的基本数据库。例如在房价预测中，你可能有一个数据库，有专门的几列数据告诉你卧室
的大小和数量，这就是结构化数据。或预测用户是否会点击广告，你可能会得到关于用户的
信息，比如年龄以及关于广告的一些信息，然后对你的预测分类标注，这就是结构化数据，
意思是每个特征，比如说房屋大小卧室数量，或者是一个用户的年龄，都有一个很好的定义。

相反非结构化数据是指比如音频，原始音频或者你想要识别的图像或文本中的内容。这
里的特征可能是图像中的像素值或文本中的单个单词。

<h5>Answer:B</h5>

<h3> 8 Why is an RNN (Recurrent Neural Network) used for machine translation, say translating English to French? (Check all that apply.) </h3>

- A It can be trained as a supervised learning problem.
- B It is strictly more powerful than a Convolutional Neural Network (CNN).
- C It is applicable when the input/output is a sequence (e.g., a sequence of words).
- D RNNs represent the recurrent process of Idea->Code->Experiment->Idea->....

RNN在语音识别，语言建模，翻译，图片描述等问题上已经取得一定成功。它是一种监督学习，比如输入数据英文，标签为法文。RNN 可以被看做是同一神经网络的多次赋值，
每个神经网络模块会把消息传递给下一个，所以它是链式的，链式的特征揭示了 RNN 本质上是与序列和列表相关的，所以它在解决sequence上是毫无问题的。
要说哪个完全比另一个强，基本都是错的。

<h5>Answer:AC</h5>

<h3> 9 In this diagram which we hand-drew in lecture, what do the horizontal axis (x-axis) and vertical axis (y-axis) represent? </h3>

![introduction_to_deep_learning_09](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/introduction_to_deep_learning_09.png)

- A 
>* x-axis is the performance of the algorithm
>* y-axis (vertical axis) is the amount of data.
- B
>* x-axis is the input to the algorithm
>* y-axis is outputs.
- C
>* x-axis is the amount of data
>* y-axis is the size of the model you train.
- D
>* x-axis is the amount of data
>* y-axis (vertical axis) is the performance of the algorithm.

在过去的几年里，很多人都问我为什么深度学习能够如此有效。当我回答这个问题时，
我通常给他们画个图，在水平轴上画一个形状，在此绘制出所有任务的数据量，而在垂直轴
上，画出机器学习算法的性能。比如说准确率体现在垃圾邮件过滤或者广告点击预测，或者
是神经网络在自动驾驶汽车时判断位置的准确性，根据图像可以发现，如果你把一个传统机
器学习算法的性能画出来，作为数据量的一个函数，你可能得到一个弯曲的线，就像图中这
样，它的性能一开始在增加更多数据时会上升，但是一段变化后它的性能就会像一个高原一
样。假设你的水平轴拉的很长很长，它们不知道如何处理规模巨大的数据，而过去十年的社
会里，我们遇到的很多问题只有相对较少的数据量。

<h5>Answer:D</h5>

<h3> 10 Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right), which of the following are true? (Check all that apply.) </h3>

- A Decreasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
- B Decreasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
- C Increasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
- D Increasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.

总的来说，对于相同的数据量，只要足够多了，那么大型神经网络的表现更好。对同一个神经网络，数据量越多，其表现越好。

<h5>Answer:CD</h5>