## Recurrent Neural Networks

<h3> 1 Suppose your training example are sentences(sequences of words). 
Which of the following refers to the j<sup>th</sup> word in the i<sup>th</sup> 
training example?</h3>

- A x<sup>(i)\<j></sup>
- B x<sup>\<i>(j)</sup>
- C x<sup>(j)\<i></sup>
- D x<sup>\<j>(i)</sup>

Answer:A
we index into the i<sup>th</sup> row first to get the i<sup>th</sup> training example (represented by 
parentheses), then the jth column to get the j<sup>th</sup> word(represented the by the brackets).

<h3> 2 Consider this RNN: </h3>

![recurrent_neural_networks_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/recurrent_neural_networks_02.png)

<h3>This specific type of architecture is appropriate when:</h3>

- A T<sub>x</sub> = T<sub>y</sub>
- B T<sub>x</sub> < T<sub>y</sub>
- C T<sub>x</sub> > T<sub>y</sub>
- D T<sub>x</sub> = 1

Answer:A
It is appropriate when every input should be matched to an output.

<h3> 3 To which of these tasks would you apply a many-to-one RNN architecture?(Check all that apply). </h3>

- A Speech recognition (input an audio clip and output a transcript)
- B Sentiment classification (input a piece of text and output a 0/1 to denote positive or negative sentiment)
- C Image classification (input an image and output a label)
- D Gender recognition from speech (input an audio clip and output a label indicating the speaker's gender)

Answer:BD

<h3> 4 You are training this RNN language model. At the t<sup>th</sup> step, what is the RNN doing? Choose the best answer. </h3>

- A Estimating P(y<sup><1></sup>,y<sup><2></sup>,...,y<sup><t-1></sup>)
- B Estimating P(y<sup>\<t></sup>)
- C Estimating P(y<sup>\<t></sup> | y<sup><1></sup>,y<sup><2></sup>,...,y<sup><t-1></sup>)
- D Estimating P(y<sup>\<t></sup> | yy<sup><1></sup>,y<sup><2></sup>,...,y<sup><t></sup>)

Answer:C

<h3> 5 You have finished training a language model RNN and are using it to sample random sentences, as follows: 
what are you doing at each time step t?</h3>

- A 
(i)Use the probabilities output by the RNN to pick the highest probability word for that time-step as ŷ<sup>\<t></sup>.
(ii)Then pass the ground-truth word from the training set to the next time-step.
- B
(i)Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as ŷ<sup>\<t></sup>.
(ii)Then pass the ground-truth word from the training set to the next time-step.
- C
(i)Use the probabilities output by the RNN to pick the highest probability word for that time-step as ŷ<sup>\<t></sup>.
(ii)Then pass this selected word to the next time-step.
- D 
(i)Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as ŷ<sup>\<t></sup>.
(ii)Then pass this selected word to the next time-step.

Answer:D

<h3> 6 You are training an RNN, and find that your weights and activations are all taking on the value of NaN("Not a Number"). 
which of these is the most likely cause of this problem? </h3>

- A Vanishing gradient problem.
- B Exploding gradient problem.
- C ReLU activation function g(.) used to compute g(z), where z is too large.
- D Sigmoid activation function g(.) used to compute g(z), where z is too large.

Answer:B

<h3> 7 Suppose you are training a LSTM. You have 10000 word vacabulary, 
and are using an LSTM with 100-dimensional activations a<sup>&lt;t&gt;</sup>.
What is the dimension of Γ<sub>u</sub> at each time step?</h3>

- A 1
- B 100
- C 300
- D 10000

Answer:B

<h3> 8 Here are update equations for the GRU. </h3>
<h3> Alice proposes to simplify the GRU by always removing the Γ<sub>u</sub>.l.e., setting Γ<sub>u</sub> = 1,
Betty proposes to simplify the GRU by removing the Γ<sub>r</sub>.l.e., setting Γ<sub>r</sub> = 1 always.
Which of these models is more likely to work without vanishing gradient problems even when trained on every 
long input sequences?</h3>

- A Alice's model(removing Γ<sub>u</sub>), because if Γ<sub>r</sub>≈0 for time-step, 
the gradient can propagate back through that time-step without much decay.
- B Alice's model(removing Γ<sub>u</sub>), because if Γ<sub>r</sub>≈1 for time-step, 
the gradient can propagate back through that time-step without much decay.
- C Betty's model(removing Γ<sub>r</sub>), because if Γ<sub>u</sub>≈0 for time-step, 
the gradient can propagate back through that time-step without much decay.
- D Betty's model(removing Γ<sub>r</sub>), because if Γ<sub>u</sub>≈1 for time-step, 
the gradient can propagate back through that time-step without much decay.

Answer:C

<h3> 9 Here are the equations for the GRU and the LSTM:</h3>
<h3> From these, we can see that the Update Gate and Forget Gate in the LSTM play a role similar to
_____ and _____ in the GRU. What should go in the blanks?</h3>

- A Γ<sub>u</sub> and 1 - Γ<sub>u</sub>
- B Γ<sub>u</sub> and Γ<sub>r</sub>
- C 1 - Γ<sub>u</sub> and Γ<sub>u</sub>
- D Γ<sub>r</sub> and Γ<sub>u</sub>

Answer:A

<h3> 10 You have a pet dog whose mood is heavily dependent on the current and past few day's 
weather. You've collected data for the past 365 days on the weather, which you represent a sequence 
 as x<sup>&lt;1&gt;</sup>,...,x<sup>&lt;365&gt;</sup>. You've also collect data on your dog's mood, 
 which you represent as y<sup>&lt;1&gt;</sup>,...,y<sup>&lt;365&gt;</sup>. You'd like to build a model 
 to map from x→y. Should you use a Unidirectional RNN or Bidirectional RNN for this problem?</h3>

- A Bidirectional RNN, because this allows the prediction of mood on day t to take into account more information.
- B Bidirectional RNN, because this allows back propagation to compute more accurate gradients.
- C Unidirectional RNN, because the value of y<sup>\<t></sup> depends only on x<sup>&lt;1&gt;</sup>,...,x<sup>&lt;t&gt;</sup> 
but not on x<sup>&lt;t+1&gt;</sup>,...,x<sup>&lt;365&gt;</sup>
- D Unidirectional RNN, because the value of y<sup>\<t></sup> depends only on x<sup>\<t></sup>, and not 
other day's weather.

Answer:C