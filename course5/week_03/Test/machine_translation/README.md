# Neural Machine Translation

## 0 Load all the packages you will need for this assignment

导入所有需要导入的包

```python
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from course5.week_03.Test.machine_translation.nmt_utils import *
import matplotlib.pyplot as plt
#%matplotlib inline
```

## 1 Translating human readable dates into machine readable dates

我们的模型将人类可读(e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987")
的日期格式翻译为机器可读(e.g. "1958-08-29", "1968-03-30", "1987-06-24")的日期格式

<h3> 1.1 Dataset </h3>

```python
from course5.week_03.Test.machine_translation.nmt_utils import *

m = 10000
dataset,human_vocab,machine_vocab,inv_vocab = load_dataset(m)

print(dataset[:10])

Tx = 30
Ty = 10
X,Y,Xoh,Yoh = preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty)

print(X.shape)
print(Y.shape)
print(Xoh.shape)
print(Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
```
```
[('9 may 1998', '1998-05-09'), 
 ('10.09.70', '1970-09-10'), 
 ('4/28/90', '1990-04-28'), 
 ('thursday january 26 1995', '1995-01-26'), 
 ('monday march 7 1983', '1983-03-07'), 
 ('sunday may 22 1988', '1988-05-22'), 
 ('tuesday july 8 2008', '2008-07-08'), 
 ('08 sep 1999', '1999-09-08'), 
 ('1 jan 1981', '1981-01-01'), 
 ('monday may 22 1995', '1995-05-22')]
 
 (10000, 30)
 (10000, 10)
 (10000, 30, 37)
 (10000, 10, 11)
 Source date: 9 may 1998
 Target date: 1998-05-09

 Source after preprocessing (indices): [12  0 24 13 34  0  4 12 12 11 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36
  36 36 36 36 36]
 Target after preprocessing (indices): [ 2 10 10  9  0  1  6  0  1 10]

 Source after preprocessing (one-hot): 
 [[ 0.  0.  0. ...,  0.  0.  0.]
  [ 1.  0.  0. ...,  0.  0.  0.]
  [ 0.  0.  0. ...,  0.  0.  0.]
    ..., 
  [ 0.  0.  0. ...,  0.  0.  1.]
  [ 0.  0.  0. ...,  0.  0.  1.]
  [ 0.  0.  0. ...,  0.  0.  1.]]
 Target after preprocessing (one-hot): 
 [[ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
```
- data_size: 1000，数据集大小
- dataset: (human readable data,machine readable data)，数据集表示为一个元组，(人类可读格式日期，机器可读格式日期)。
- human_vocab: {human readable data character:integer-value index}，人类可读格式日期字典，将人类可读格式日期的字符
映射为整型数字下标(eg,{'a':1,'u':2,'g':3,'s':4,'t':5,...})。
- machine_vocab: {machine readable data character:integer-value index}，机器可读格式日期字典，将机器可读格式日期的
字符映射为整型数字下标。
- inv_machine_vocab: {integer-value index:machine readable data character},机器可读格式日期字典的翻转对应字典，将整型
下标作为key，机器可读格式日期字符作为value

- X: 通过human_vocab将人类可读格式日期转化为一个长度为Tx(=30)数组，
长度不足的序列用特殊的字符(<pad>)pad到指定的长度。X.shape = (m,Tx)
- Y: 类似于X，通过machine_vocab将原本字符序列转化为对应整型下标的序列，
长度不足的用特殊字符pad到指定的最大长度。Y.shape = (m,Ty)
- Xoh: X的one-hot表示，将X中的整型下标转化为维度为len(human_vocab)大小的
one-hot表示，Xoh.shape = (m,Tx,len(human_vocab))
- Yoh: Y的one-hot表示，Yoh.shape = (m,Ty,len(machine_vocab))

## 2 Neural machine translation with attention

如果你不得不把一个书的段落从法文翻译成英文，你不会阅读整段，然后关闭书本并翻译。 
即使在翻译过程中，您也会阅读/重读，并专注于与您正在写下的英语部分相对应的法语段落的部分。

Attention机制告诉我们的神经模型，在不同的时间步应该关注哪一个词

<h3> 2.1 Attention mechanism </h3>

在这部分，我们将会实现课程视频中所提出的注意力机制。

下面是你在实现模型的时候应该关注的相关细节：

![attention_model_intuition_03](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/attention_model_intuition_03.png)

![attention_model_intuition_04](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/attention_model_intuition_04.png)

- 这个模型中有两个单独的LSTM，在图片底部有一个双向LSTM，我们将其称为pre-attention Bi-LSTM，在顶部的一个LSTM出现在
Attention机制完成后，我们将其称为post-attention LSTM。pre-attention Bi-LSTM经过Tx个时间步，post-attention LSTM
经过Ty个时间步。
- post-attention LSTM 通过输入S<sup>\<t></sup>,C<sup>\<t></sup>一步步向后传播，在视频课程中，sequence模型用的是basic RNN
输出激活值S<sup>\<t></sup>。在这里我们应用的是LSTM输出激活值S<sup>\<t></sup>和hidden cell state C<sup>\<t></sup>，和序列生成
任务不同，我们在时间步t计算的时候，y<sup>\<t-1></sup>不会作为LSTM的输入。基于以上这些原因，我们重新设计了我们的模型，因为序列翻译不像
序列生成任务有着更强的依赖性。
- 我们用a<sup>\<t></sup> = \[a_pre<sup>\<t></sup>:a_post<sup>\<t></sup>]，将pre-attention Bi-LSTM的前向激活值和逆向激活值进行拼接。
- 将S<sup>\<t-1></sup>的值用RepeatVector拷贝Tx，将S<sup>\<t-1></sup>和a<sup>\<t></sup>进行拼接计算求出e<sup>\<t,t'></sup>,然后
e<sup>\<t,t'></sup>通过一个softmax计算出a<sup>\<t,t'></sup>，我们将会说明如何在keras中实现矩阵复制(RepeatVector)和矩阵拼接(Concatenation)
操作。

让我们来开始实现我们的模型，首先我们来实现两个函数one_step_attention()和model()

- 1)one_step_attention():输入pre-attention Bi-LSTM的所有隐藏层值a=\[a<sup>\<1></sup>,a<sup>\<2></sup>,...,a<sup>\<Tx></sup>]，然后输入
post-attention LSTM上一个时间步的隐藏层的输出S<sup>\<t-1></sup>，one_step_attention()计算出attention的权重
(\[α<sup>\<t,1></sup>,α<sup>\<t,2></sup>,...,α<sup>\<t,Tx></sup>])然后输出上下文向量context vector