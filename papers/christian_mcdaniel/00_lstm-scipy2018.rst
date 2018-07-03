:author: Christian McDaniel
:email: clm121@uga.edu
:institution: University of Georgia

:author: Shannon Quinn, PhD
:email: spq@uga.edu
:institution: University of Georgia
:bibliography: citations

--------------------------------------------------
Developing an LSTM Pipeline for Accelerometer data
--------------------------------------------------

.. class:: abstract

Increased prevalence of smartphones and wearable devices has facilitated the collection of triaxial accelerometer data for numerous Human Activity Recognition (HAR) endeavors. Concurrently, advances in the theory and implementation of Long Short-Term Memory (LSTM) recurrent neural networks (RNNs) has made it possible to process this data in its raw form, making possible on-device online analysis. In this two-part experiment, we have first amassed the results from thirty studies and reported their methods and key findings in a meta-analysis style review. We then used these findings to guide our development of a start-to-finish data analysis pipeline, which we implemented in part two of our experiment on a commonly used open-source dataset. The pipeline addresses the large disparities in model hyperparameter settings and ensures the avoidance of potential sources of data leakage that were identified in the literature. Our pipeline uses a heuristic-based algorithm to tune a baseline LSTM model over an expansive hyperparameter search space and trains the resulting model on standardized windowed accelerometer signals alone. We find that we are able to compete with benchmark results from complex models trained on higher-dimensional data.

.. class:: keywords

Neural Network, Human Activity Recognition, Recurrent Neural Network, Long Short-Term Memory, Accelerometer, Machine Learning, Data Analysis, Data Science, Hyperparameter Optimization, Hyperparameter

Introduction
------------

Human Activity Recognition (HAR) is a time series classification problem in which a classifier attempts to discern distinguishable features from movement-capturing on-body sensors :cite:`KimCook2010`. Typical sensors record changes in velocity through time in the x- y- and z-directions, i.e., accelerometers. Accelerometer output consists of high-frequency (30-200Hz) triaxial time series recordings, often containing noise, imprecision, missing data, and long periods of inactivity (i.e., the Null class) between meaningful segments :cite:`Ravietal2005,BaoIntille2004,OrdonezRoggen2016`. Consequently, attempts to use traditional classifiers typically require significant preprocessing and technical engineering of hand crafted features from raw data, resulting in a barrier to entry to the field and making online and on-device data processing impractical :cite:`Gaoetal2016,ManniniSabatini2010,Gjoreskietal2016,Ravietal2005,OrdonezRoggen2016`.

The limitations of classical methods in this domain have been alleviated by concurrent theoretical and practical advancements in artificial neural networks (ANNs), which are more suited for complex non-linear data. While convolutional neural networks (CNNs) are attractive for their automated feature extraction capabilities during convolution and pooling operations :cite:`Sjostrum2017,Rassemetal2017,Fiterauetal2016,Seoketal2018,Zebinetal2017,Gaoetal2016,OrdonezRoggen2016,Gjoreskietal2016`, recurrent neural networks (RNNs) are specifically designed to extract information from time series data due to the recurrent nature of their data processing and weight updating operations :cite:`WilliamsZipser1989`. Furthermore, whereas earlier implementations of RNNs suffered from vanishing and exploding gradients during training, the incorporation of a multi-gated memory cell in long short-term memory recurrent neural networks (LSTMs) :cite:`HochreiterSchmidhuber1997` along with other regularization schemes helped alleviate these issues.

As RNN usage continues, numerous studies have emerged to address various aspects of understanding and implementing these complex models, namely regarding the vast architectural and hyperparameter combinations that are possible :cite:`Gersetal2002,ReimersGurevych2017,PressWolf2017,Karpathyetal2015,Merityetal2017`. Unfortunately, these pioneering studies tend to focus on tasks other than HAR, leaving the time series classification tasks of HAR without domain-specific architecture guidance or insights into the models’ representation of the data.

In a meta-analysis style overview of the use of LSTM RNNs for HAR experiments across 30 reports (discussed below), we found a general lack of consensus regarding the various model architectures and hyperparameters used. Often, a given pair of experiments explored largely or entirely non-overlapping ranges for a single hyperparameter. Many architectural and procedural details are not included in various reports, making reproducibility nearly impossible. The analysis pipelines employed are often lacking detail and sources of data leakage, where details from the testing data are exposed to the model during training, appear to be overlooked in certain cases. Without clear justifications for model implementations and deliberate, reproducible data analysis pipelines, objective model comparisons and inferences from results cannot be made. For these reasons, the current report seeks to summarize the previous implementations of LSTMs for HAR research available in literature and outline a structured data analysis pipeline for this domain. We implement our pipeline, optimizing a baseline LSTM over an expansive hyperparameter search space, and obtain results on par with benchmark studies. We suspect that our efforts will encourage scientific rigor in the field going forward and initiate the exploration inward as we understand these powerful data analysis tools within this domain.

Background
-------------
This section is intended to give the reader a digestible introduction to ANNs, RNNs and the LSTM cell. The networks will be discussed as they relate to multi-class classification problems as is the task in HAR.

*Artificial Neural Networks* The first ANN architecture was proposed by Drs. Warren McCulloch and Walter Pitts in 1943 as a means to emulate the cumulative semantic functioning of groups of neurons via propositional logic :cite:`McCullochPitts1943,Geron2017`. Frank Rosenblatt subsequently developed the Perceptron in 1957 :cite:`Rosenblatt1957`. This ANN variation carries out its step-wise operations via mathematical constructs known as linear threshold units (LTUs). The LTU operates by aggregating multiple weighted inputs and feeding this summation u through an activation function :math:`f(u)` or step function :math:`\text{step}(u)`, generating an interpretable output :math:`yp` (e.g. 0 or 1).

.. math::
  :type: eqnarray

  yp &=& f(u) \\
     &=& f(w^T \cdot x)

where :math:`w^T` is the transpose of the weight vector :math:`w` and :math:`\cdot` is the dot product operation from vector calculus. :math:`x` is a single instance of the training data, containing values for all :math:`n` attributes of the data. As such, :math:`w` is also of length :math:`n`, and the entire training data set for all :math:`m` instances is a matrix :math:`X` of dimensions :math:`m` by :math:`n` (i.e., :math:`m` x :math:`n`).

A 2-layer ANN can be found in :ref:`ANN` A. Each attribute in instance :math:`x(i)` represents a node in the perceptron's input layer, which simply provides the raw data to the the output layer where the LTU resides. Often more than one LTU is used in the output layer to represent multiple target classes. Each data instance has a one-hot target vector :math:`y(i)` the length of the number of classes :math:`k` containing all zeros except at the index corresponding to the instance's class. Each LTU node corresponds to a single class in :math:`y` and each LTU's prediction :math:`yp` indicates the predicted probability that the training instance belongs to the corresponding class. Given the predictions at each LTU, the prediction with the largest value - :math:`\text{max}(yp)` - is taken as the overall predicted class for the instance of the data being analyzed. Taken over the entire dataset, each LTU has a prediction vector :math:`yp_{k}` length :math:`m` and the entire output layer produces a prediction matrix :math:`Yp` with dimensions :math:`m` x :math:`k`. Additionally, each LTU contains its own weight vector :math:`w_{k}` of length :math:`n` (i.e., a fully-connected network), resulting in a weight matrix :math:`W` of dimensions :math:`n` x :math:`k`. The weight vector at each LTU is what is iteratively adjusted during training to apply a class-specific weighting of the data and yield a class-specific prediction.

ANN often contain complex architectures with additional layers, which allow for nonlinear transformations of the data and increase the flexibility and robustness of the model. If we look at a simple three-layer neural network (see :ref:`ANN` B), we see input and output layers as described above, as well as a layer in the middle, termed a *hidden layer*. This layer acts much like the output layer, except that its outputs :math:`z` for each training instance are fed into the output layer, which then generates predictions :math:`yp` from :math:`z` alone. The complete processing of all instances of the dataset, or all instances of a portion of the dataset called a *mini-batch*, through the input layer, the hidden layer, and the output layer marks the completion of a single *forward pass*. For the model to improve, the outputs generated by this forward pass must be evaluated somehow and the model updated in an attempt to improve the model's predictive power on the data. An error term (e.g., sum of squared error (:math:`sse`)) is calculated by comparing individual predictions :math:`yp_{k}` to corresponding ground truth target values in :math:`y_{k}`. Thus, an error matrix :math:`E` is generated composed of error terms over all :math:`k` classes for all :math:`m` training instances. This error matrix is used as an indicator for how to adjust the weight matrix in the output layer so as to yield more accurate predictions, and the corrections made to the output layer give an indication of how to adjust the weights in the hidden layer so as to further help transform the data in a way that leads to improved accuracy of the model. This process of carrying the error backward from the output layer through the hidden layer(s) is known as *backpropogation*. One forward pass and subsequent backpropogation makes up a single *epoch*, and the training process consists of many epochs repeated in succession to iteratively improve the model.

.. figure:: ANN.png

    **A.** A two-layer network and associated dimensions of the components. **B.** A three-layer network showing a single data instance x(*i*) being fed in as input. :label:`ANN`

The iterative improvements are known as *optimization*, and many methods exist to carry this process out. The common example is stochastic gradient descent (SGD), which calculates the gradient, or the collection of partial derivatives from all dimensions of the input, of the error matrix and adjusts the weight matrices at each layer in a direction opposite this gradient. The change to be applied to weight matrix is mediated via a learning rate :math:`\eta`.

.. math::

  E = Y - f(X W)

optimization:

.. math::

  \text{min}_{W} \|E\|_{F}

.. math::

  hsse_{W} = \frac{1}{2} \displaystyle\sum_{c=0}^{k-1} (y_{c} - f(X \cdot w_{c}) \cdot (y_{c} - f(X \cdot w_{c})))

.. math::

  \frac{\partial hsse} {\partial w_{k}} = X^T*[ f'( X \cdot w_{k} )*e_{k} ]* \eta = -X^T*\delta_{k}* \eta

where :math:`f(...)` represents the activation function, :math:`min_{W}` represents the objective function of minimizing with respect to :math:`W`, and :math:`\|E\|_{F}` stands for the Frobenius norm on the error matrix :math:`E`. :math:`\text{hsse}_{W}` represents the halved (for mathematical convenience) sum of squared error, calculated for all :math:`k` nodes in the output layer. :math:`f'(...)` represents the derivative of the activation function over term in the parentheses.

Looking at our three-layer neural network depicted in :ref:`ANN`, a single epoch would proceed as follows:

1. Compute :math:`yp` and compare with :math:`y` to generate the error term:

.. math::

  z_{h} = f_{1} ( a_{_h} \cdot x )

.. math::

  y_{pk} = f_{2} ( b_{_k} \cdot z )

.. math::

  e_{k} = y_{k} - yp_{k}

2. Backpropogate the error regarding the correction needed for :math:`yp`.

3. Backpropogate the correction to the hidden layer.

4. update :math:`A` and :math:`B` via :math:`\delta^y` and :math:`\delta^z`:

.. math::
  :type: eqnarray

  b_{hk} &=& b_{hk} - z_{h} \delta^y_{k} * \eta \\
         &=& b_{hk} - \frac{ \partial hsse} {\partial b_{hk}} * \eta

.. math::
  :type: eqnarray

  a_{jh} &=& a_{jh} - x_{j} \delta^z_{h} * \eta \\
         &=& a_{jh} - \frac{ \partial hsse} {\partial a_{jh}} * \eta

:math:`sse` is commonly used as the error term for regression problems, whereas squared error or *cross entropy* is typical for classification problems.

.. math::

  \text{cross entropy} = -\displaystyle\sum_{i=1}^m \displaystyle\sum_{c=1}^k y_ic * log( f_{c}(x_{i}))

where the first sum is taken over all :math:`m` training instances in the data set or mini-batch and the second sum is taken over all :math:`k` classes.

The high flexibility of neural networks increases the chances of overfitting, and there are various ways to avoid this. *Early stopping* is a technique that monitors the change in performance on a validation set (subset of the training set) and stops training once improvement slows sufficiently. *Weight decay* helps counter large updates to the weights during backpropogation and slowly shrinks the weights toward zero in proportion to their relative sizes. Similarly, the *dropout* technique "forgets" a specified proportion of the outputs from a layer's neurons by not passing those values on to the next layer. *Standardizing* the input is important, as it encourages all inputs to be treated equally during the forward pass by scaling and mitigating outliers' effects :cite:`Miller2018`.

Other hyperparameters tend to affect training efficiency and effectiveness and tend to differ with different datasets and types of data. Hammerla, et. al. found *learning rate* :math:`\eta` to be an important hyperparameter in terms of its effect on performance :cite:`Hammerlaetal2016`. Too small a learning rate and the model will exhibit slow convergence during training, while too large a value will lead to wild oscillations :cite:`Miller2018`. Hammerla, et. al. also find the *number of units* per layer :math:`n` to be important, and Miller adds that too many hidden units is better than too few. The former will lead to extra weights, which will likely be pushed to zero, while the latter restricts the flexibility of the model. *Bias* helps account for irreducible error in the data and is implemeneted in an ANN via giving it its own node (top node in the input layer of :ref:`ANN` A) sending all ones to the next layer. Reimers and Gurevych emphasize the importance of weight initialization for model performance in their survey of the importance of hyperparameter tuning for using LSTMs for language modeling :cite:`ReimersGurevych2017`. Pascanu, et. al. explain the downside of using an L1 or L2 penalty to regularize the recurrent weights during back propagation. Initially formulated to help with exploding gradients, this technique causes exponential loss of temporal information as a function of time, making long term dependencies difficult to learn :cite:`Pascanuetal2013`. Jozefowicz, et. al. cite the initialization of the forget gate bias to 1 as a major factor in LSTM performance :cite:`Jozefowiczetal2015`.

*Recurrent Neural Networks (RNNs)* The recurrent neuron is extremely useful in training a model on sequence data. Sequence data differs from the usual training data in that a single time series input is a vector that may contain patterns and dependencies across multiple indices or time steps. Recurrent neurons address these temporal dependencies by sending their outputs both forward to the next layer and "backward throught time," looping the neuron's output back to itself as input paired with new input from the previous layer. Thus, a component of the input to the neuron is an accumulation of activated inputs from each previous time step. :ref:`RNN` depicts a recurrent neuron.

.. figure:: RNN.png

  The recurrent neuron from three perspectives. **A.** A single recurrent neuron, taking input from X, aggregating this input over all timesteps in a summative fashion and passing the summation through an activation function at each timestep. **B.** The same neuron unrolled through time, making it resemble a multilayer network with a single neuron at each layer. **C.** A recurrent layer containing five recurrent nodes, each of which processes the entire dataset X through all time point. :label:`RNN`

Instead of a single weight vector as in ANN neurons, RNN neurons have two sets of weights, one (:math:`wx`) for the inputs :math:`x_{t}` and one (:math:`wy`) for the outputs of the previous time step :math:`y_{(t-1)}`, where :math:`t` represents the current time step. These become matrices :math:`W_{x}` and :math:`W_{y}` when taken over the entire layer. The portion of the neuron which retains a running record of the previous time steps is the *memory cell* or just the *cell*.

Outputs of the recurrent layer:

.. math::

  y_{(t)} = \phi(W_{x}^T \cdot x_{(t)} + W_{y}^T \cdot Y_{(t-1)} + b)

where :math:`\phi` is the activation function and :math:`b` is the bias vector of length :math:`n` (the number of neurons).

The *hidden state*, or the *state*, of the cell (:math:`h_{(t)}`) is the information that is kept in memory over time.

To train these neurons, we "unroll" the neurons following a complete forward pass to reveal a chain of linked neurons the length of time steps in a single input. We then apply standard backpropogation to these links, calling the process backpropogation through time (BPTT). This works relatively well for very short time series, but once the number of time steps increases to tens or hundreds of time steps, the network essentially becomes very deep during BPTT and problems arise such as very slow training and exploding and vanishing gradients. Various hyperparameter and regularization schemes exist to alleviate exploding/vanishing gradients, including *gradient clipping* :cite:`Pascanuetal2013`, *batch normalization*, dropout, and the long short-term memory (LSTM) cell originally developed by Sepp Hochreiter and Jurgen Schmidhuber in 1997 :cite:`HochreiterSchmidhuber1997`.

*Long Short-Term Memory (LSTM) RNNs* The LSTM cell achieves faster training and better long-term memory than vanilla RNN neurons by maintaining two state vectors, the short-term state :math:`h_{(t)}` and the long-term state :math:`c_{(t)}`, mediated by a series of inner gates, layers, and other functions. These added features allow the cell to process the time series in a deliberate manner, recognizing meaningful input to store long-term and later extract when needed, and forget unimportant information or that which is no longer needed.

.. figure:: LSTMcell.png

  The inner mechanisms of an LSTM cell. From outside the cell, information flows similarly as with a vanilla cell, except that the state now exists as two parts, one for long-term memory (:math:`c_{(t)}`) and the other for short-term memory (:math:`h_{(t)}`). Inside the cell, four different sub-layers and associated gates are revealed. :label:`LSTM`

As can be seen in :ref:`LSTM`, when the forward pass advances by one time step, the new time step's input enters the LSTM cell and is copied and fed into four independent fully-connected layers (each with its own weight matrix and bias vector), along with the short-term state from the previous time step, :math:`h_{(t-1)}`. The main layer is :math:`g_{(t)}`, which processes the inputs via :math:`tanh` activation function. In the basic cell, this is sent straight to the output; in the LSTM cell, part of this is incorporated in the long-term memory as decided by the *input gate*. The input gate also takes input from another layer, :math:`i_{(t)}`, which processes the inputs via the sigmoid activation function :math:`\sigma` (as do the next two layers). A third layer, :math:`f_{(t)}`, processes the inputs, combines them with :math:`c_{(t-1)}`, and passes this combination through a *forget gate* which drops a portion of the information therein. Finally, the fourth fully-connected layer :math:`o_{(t)}` processes the inputs and passes them through the *output gate* along with a copy of the updated long-term state :math:`c_{(t)}` after its additions from :math:`f_{(t)}`, deletions by the forget gate, further additions from the filtered :math:`g_{(t)}`-:math:`i_{(t)}` combination and a final pass through a :math:`tanh` activation function. The information that remains after passing through the output gate continues on as the short-term state :math:`h_{(t)}`.

.. math::

  i_{(t)} = \sigma (W){xi}^T . x_{(t)} + W_{hi}^T . h_{(t-1)} + b_{i}

.. math::

  f_{(t)} = \sigma (W){xf}^T . x_{(t)} + W_{hf}^T . h_{(t-1)} + b_{f}

.. math::

  o_{(t)} = \sigma (W){xo}^T . x_{(t)} + W_{ho}^T . h_{(t-1)} + b_{o}

.. math::

  g_{(t)} = \sigma (W){xg}^T . x_{(t)} + W_{hg}^T . h_{(t-1)} + b_{g}

.. math::

  c_{(t)} = f_{(t)} \otimes c_{(t-1)} + i_{(t)} \otimes g_{(t)}

.. math::

  y_{(t)} = h_{(t)} = o_{(t)} \otimes \tanh(c_{(t)})

where :math:`\otimes` represents element-wise multiplication.

Related Works
-------------
The following section outlines the nuanced hyperparameter combinations used by 30 studies available in literature in a meta-analysis style survey. Published works as well as pre-published and academic research projects were included so as to gain insight into the state-of-the-art methodologies at all levels and increase the volume of works available for review. It should be noted that the following summaries are not necessarily entirely exhaustive regarding the specifications listed. Additionally, many reports did not include explicit details of many aspects of their research.

The survey of previous experiments in this field provided blueprints for constructing an adequate search space of hyperparameters. We have held our commentary on the findings of this meta-study until the Discussion section.

*Experimental Setups*

Across the 30 studies, each used a unique implementation of LSTMs for the research conducted therein. Many reports used the open-source OPPORTUNITY Activity Recognition dataset :cite:`OrdonezRoggen2016,Riveraetal2017,Gaoetal2016,Zhaoetal2017,Broome2017,GuanPlotz2017`, while other datasets used include PAMAP2 :cite:`OrdonezRoggen2016,Setterquist2018,GuanPlotz2017,Zhangetal2018`, Skoda :cite:`OrdonezRoggen2016,GuanPlotz2017`, WISDM :cite:`Chenetal2016,U2018`, ChaLearn LAP large-scale Isolated Gesture dataset (IsoGD) :cite:`Zhangetal2017`, Sheffield Kinect Gesture (SKIG) dataset :cite:`Zhangetal2017`, UCI HAR dataset :cite:`U2018,Zhaoetal2017`, a multitude of fall-related datasets :cite:`Muscietal2018`, and various study-specific internally-collected datasets. Most studies used the Python programming language. Neural network libraries employed include Theano Lasagne, RNNLib, and Keras with TensorFlow. While most of the studies we examined trained models on tasks under the broad umbrella of “Activities of Daily Life” (ADL) – e.g., opening a drawer, climbing stairs, walking, or sitting down – several of the studies focused on more specific human activities such as smoking :cite:`Bergelin2017`, cross-country skiing :cite:`Rassemetal2017`, eating :cite:`Kyritsisetal2017`, nighttime scratching :cite:`Moreauetal2016`, and driving :cite:`Carvalhoetal2017`.

Numerous experimental data analysis pipelines were used, including cross validation :cite:`Lefebvreetal2015`, repeating experiments :cite:`ShinSung2016`, and various train-validation-test splitting procedures :cite:`Sjostrum2017,WuAdu2017,Huetal2018`.

*Preprocessing* Before training the proposed models, each study performed some degree of preprocessing. Some reports kept preprocessing to a minimum, e.g., linear interpolation to fill missing values :cite:`OrdonezRoggen2016`, per-channel normalization :cite:`OrdonezRoggen2016,Huetal2018`, and standardization :cite:`Chenetal2016,Zhaoetal2017`. Typically, data is standardized to have zero mean, i.e., centering the amplitude around zero :cite:`Broome2017`, and unit standard deviation, whereas Zhao, et. al. standardized the data to have 0.5 standard deviation :cite:`Zhaoetal2017`, citing Wiesler, et. al. as supporting this nuance for deep learning implementations :cite:`Wiesleretal2014`.

Other noise reduction strategies employed include kernel smoothing :cite:`Gaoetal2016`, removing the gravity component :cite:`Moreauetal2016`, applying a low-pass filter :cite:`Lefebvreetal2015`, removing the initial and last 0.5 seconds :cite:`Huetal2018`. Moreau, et. al. used the derivative of the axis-wise gravity component in order to group together segments of data from different axes, tracking a single motion across axes as the sensor rotated during a gesture :cite:`Moreauetal2016`.

For feeding the data into the models, the sliding window technique was commonly used, with vast discrepancy in the optimal size of the window (reported both as units of time and number of time points) and step size. Window sizes used range from 30 :cite:`Broome2017` to 100 :cite:`Zhaoetal2016` time points, and 32 :cite:`Muscietal2018`to 5000 :cite:`Zhaoetal2017` milliseconds (ms). Using a step size of 50% of the window size was typical :cite:`Rassemetal2017,Sjostrum2017,Broome2017,OrdonezRoggen2016`. Finally, Guan and Plotz ran an ensemble of models, each using a random sampling of a random number of frames with varying sample lengths and starting points. This method is similar to the bagging scheme of random forests and was implemented to increase robustness of the model :cite:`GuanPlotz2017`.

Once a window is generated it must be assigned a class and labeled as such. Labeling schemes used include using the last data point's class :cite:`OrdonezRoggen2016` or the majority class within the window :cite:`Broome2017`.

*Architectures* Numerous architectural and hyperparameter choices were made among the various studies. Most studies used two LSTM layers :cite:`OrdonezRoggen2016,Chenetal2016,Kyritsisetal2017,Zhangetal2017,Riveraetal2017,U2018,Zhaoetal2017,GuanPlotz2017,Huetal2018,Muscietal2018`, while others used a single layer :cite:`WuAdu2017,Broome2017,ShinSung2016,Carvalhoetal2017,Zhaoetal2016,Zhangetal2018,Seoketal2018`, three layers :cite:`Zhaoetal2016`, or four layers :cite:`MuradandPyun2017`.

Several studies designed or utilized novel LSTM architectures that went beyond the simple tuning of hyperparameters. Before we list them, note that the term “deep” in reference to neural network architectures indicates the use of multiple layers of hidden connections; for LSTMs, an architecture generally qualifies as “deep” if it has three or more hidden layers. Architectures tested include the combination of CNNs with LSTMs such as ConvLSTM :cite:`Zhangetal2017,Gaoetal2016`, DeepConvLSTM :cite:`OrdonezRoggen2016,Sjostrum2017,Broome2017`, and the multivariate fully convolutional LSTM network (MLSTM-FCN) :cite:`Karimetal2018`; innovations related to the connections between hidden units including the bidirectional LSTM (b-LSTM) :cite:`Rassemetal2017,Broome2017,Moreauetal2016,Lefebvreetal2015,Hammerlaetal2016`, hierarchical b-LSTM :cite:`LeeCho2012`, deep residual b-LSTM (deep-res-bidir LSTM) :cite:`Zhaoetal2017`, and LSTM with peephole connections (p-LSTM) :cite:`Rassemetal2017`; and other nuanced architectures such as ensemble deep LSTM :cite:`GuanPlotz2017`, weighted-average spatial LSTM (WAS-LSTM) :cite:`Zhangetal2018`, deep-Q LSTM :cite:`Seoketal2018`, the multivariate squeeze-and-excite fully convolutional network ALSTM (MALSTM-FCN) :cite:`Karimetal2018`, and similarity-based LSTM :cite:`Fiterauetal2016`. The use of densely-connected layers before or after the LSTM layers was also common. Kyritsis, et. al. added a dense layer with ReLU activation after the LSTM layers, Zhao, et. al. included a dense layer with tanh activation after the LSTMs, and Musci, et. al. used a dense layer before and after its two LSTM layers :cite:`Kyritsisetal2017,Zhaoetal2016,Muscietal2018`. The WAS-LSTM, deep-Q LSTM, and the similarity-based LSTM used a combination of dense and LSTM hidden layers.

Once the number of layers is determined, the number of units per LSTM layer must be set. The number of units per layer specified by various studies range from 3 :cite:`Moreauetal2016` to 512 :cite:`Setterquist2018`. Several studies used different numbers of units for different circumstances – e.g., three units per layer for unilateral movement (one arm) and four units per layer for bilateral movement (both arms) :cite:`Moreauetal2016` or 28 units per layer for the UCI HAR dataset (lower dimensionality) versus 128 units per layer for the Opportunity dataset :cite:`Zhaoetal2017`. Others used different numbers of units for different layers of the same model – e.g., 14-14-21 for a 3-layer model :cite:`Zhaoetal2016`.

Almost all of the reports used the sigmoid activation for the recurrent connections within cells and the tanh activation function for the LSTM cell outputs, as these are the activation functions used the original paper :cite:`HochreiterSchmidhuber1997`. Other activation functions used for the cell outputs include ReLU :cite:`Zhaoetal2017,Huetal2018` and sigmoid :cite:`Zhangetal2018`.

*Training* For trainint, weights are often initialized using specific strategies, for example random orthogonal initialization :cite:`OrdonezRoggen2016,Sjostrum2017`, fixed random seed :cite:`Setterquist2018`, the Glorot uniform initialization :cite:`Broome2017`, random uniform initialization on [-1, 1] :cite:`Moreauetal2016`, or using a random normal distribution :cite:`Huetal2018`. For mini-batch training, batch sizes reported range from 32 :cite:`Riveraetal2017,Setterquist2018` to 450 :cite:`Bergelin2017`.

To calculate the amount of change needed for each training epoch, different loss functions are used. Categorical cross-entropy is the most widely used method :cite:`OrdonezRoggen2016,MuradandPyun2017,Chenetal2016,Sjostrum2017,Kyritsisetal2017,Setterquist2018,Broome2017,Huetal2018,Zhangetal2018`, but F1 score loss :cite:`GuanPlotz2017`, mean squared error (MSE) :cite:`Carvalhoetal2017`, and mean absolute error :cite:`Zhaoetal2016` were also used with varying degrees of success. During back propagation, various updating rules – e.g. RMSProp :cite:`OrdonezRoggen2016,Setterquist2018,Broome2017`, Adam :cite:`MuradandPyun2017,Kyritsisetal2017,Broome2017,Huetal2018,Zhangetal2018`, and Adagrad :cite:`ShinSung2016,Hammerlaetal2016` – and learning rates – 10^-7 :cite:`ShinSung2016`, 10^-4 :cite:`Sjostrum2017,GuanPlotz2017`, 2e-4 :cite:`Moreauetal2016`, 5e-4 :cite:`Lefebvreetal2015`, and 10^-2 :cite:`OrdonezRoggen2016` are used.

Regularization techniques employed include weight decay of 90% :cite:`OrdonezRoggen20161,Sjostrum2017`; update momentum of 0.9 :cite:`Moreauetal2016`, 0.2 :cite:`Lefebvreetal2015`, or the Nesterov implementation :cite:`ShinSung2016`; dropout (e.g., 50% :cite:`OrdonezRoggen2016,Sjostrum2017` or 70% :cite:`Zhaoetal2016`) between various layers; batch normalization :cite:`Zhaoetal2017`; or gradient clipping using the norm :cite:`Zhaoetal2017,Huetal2018,Zhangetal2018`. Broome 2017 chose to use the stateful configuration for its baseline LSTM :cite:`Broome2017`. In this configuration, unit memory cell weights are maintained between each training example instead of resetting them to zero after each forward pass.

The number of epochs specified ranged from 100 :cite:`Broome2017` to 10,000 :cite:`Huetal2018`. Many studies chose to use early stopping to prevent overfitting :cite:`Garethetal2017`. Various patience schemes, specifying how many epochs with no improvement above a given threshold the model should allow, were chosen.

*Performance measures*

Once the model has been trained, it is given a set of examples it has not yet seen and predicts the target class that each example belongs to. Various performance measures are used to assess the performance of the model on this test set. The measures used include the F1 score - used by most :cite:`OrdonezRoggen2016,Broome2017,Gaoetal2016,Zhaoetal2017,Broome2017`, classification error :cite:`Rassemetal2017`, accuracy :cite:`Sjostrum2017,Setterquist2018`, and ROC :cite:`Moreauetal2016,Huetal2018`.

*Benchmark Performances*
We focus on the performances of models trained and tested using the the UCI HAR dataset, publicly available on the University of California at Irvine (UCI) Machine Learning Repository, as that is the dataset we utilize in our study. Initial benchmark results include the use of classical methods and 551 hand crafted features. Anguita, et. al. released three studies in 2013 following their release of the dataset. Using a multi-class SVM (MC-SVM) classifier, they reach F1 score of 0.96 :cite:`Anguitaetal2013ESANN`. They also reached an F1 score of 89.0 using a hardware-friendly MC-SVM (HF-MC-SVM) :cite:`AGO+13a`. Finally, they released the results from a competition using the dataset. Accuracies reached include 96.5% by a one-vs-one SVM (OVO SVM), 96.35% by a kernelized matrix learning vector quantized (LVQ) model, 94.33% by a confidence-based model (Conf-AdaBoost.M1), 93.7% by one-vs-all SVM (OVA SVM), and 90.6% by KNN :cite:`ReyesOrtizetal2013`.

As LSTMs rise in usage, we see competitive results using lower dimensional data. Most models make use of acceleration and gyroscope data. Accuracies reached consist of 96.7% by a four-layer LSTM model :cite:`MuradandPyun2017`, 96.71% by a multivariate LSTM + fully convoluted network (MLSTM-FCN), 96.71% by multivariate squeeze-and-excite ALSTM with fully convoluted network (MALSTM-FCN) :cite:`Karimetal2018`, 93.57% by the Deep-Res-Bidir LSTM, and 90.77% by the baseline LSTM :cite:`Zhaoetal2017`. Only one study seems to have used solely the accelerometer data, although it is not explicitly stated. This study reports a testing accuracy of 85.34% from their LSTM model :cite:`U2018`.

As this meta-analysis style overview has shown, there are many different model constructions being employed for HAR tasks. The work by the aforementioned studies as well as others have laid the groundwork for this field of research.

Experimental Setup
------------------

*Data* Although many studies use the gyroscope- and magnetometer-supplemented records from complex inertial signals, accelerometer data alone is more ubiquitous in this field and the decreased feature space helps illuminate the robustness of the model and requires lower computational complexity (i.e., more applicable to online and on-device classifications). As such, this report trains its models on triaxial accelerometer data alone.

The primary dataset used for our experiments is the Human Activity Recognition Using Smartphones Data Set (UCI HAR Dataset) from Anguita, et. al. :cite:`Anguitaetal2013ESANN`. This is a publicly available dataset that can be downloaded via the University of California at Irvine (UCI) online Machine Learning Repository.

*UCI HAR Dataset* Classes include walking, climbing stairs, descending stairs, sitting, standing, and laying down. This dataset was collected from built-in accelerometers and gyroscopes (not used in current study) in smartphones worn on the waists of participants. The collectors of this data manually extracted over 500 features from the raw data; however, this study only utilizes the raw accelerometer data itself.

A degree of preprocessing was applied to the raw signals themselves by the data collectors. The accelerometer data was recorded at 50Hz and was preprocessed to remove noise by applying a third order low pass Butterworth filter with corner frequecy of 20Hz and a median filter. The cleaned data were then separated into body motion and gravity components via a second application of a low pass Butterworth filter with 0.3Hz cuttoff. A sliding window was applied to the data using a window size of 2.56 seconds (128 time points) and a 50% stride. The data for the total accelerometer signals and the body-movement only (gravity component removed) signals are provided separately, with the windowed data from each axis (x, y, and z) contained in a separate file. The participant ID number and activity label corresponding to each window have their own respective files. Finally, the data were split into training (70%) and testing (30%) folders. See :ref:`HAR` A.

*Preprocessing* Preprocessing was kept to a minimum. Before any scaling or windowing was performed, we attempted to “undo” as much of the preprocessing already performed on the data before reformatting the data for feeding it into the network. First, the training and testing sets were combined into a single dataset (Figure :ref:`HAR` B). The windows were effectively removed from the data by concatenating together time points from every other window, reforming contiguous time series Figure :ref:`HAR` C. We then combined each axis-specific time series to form the desired triaxial data format, where each time point consists of the accelerometer values along the x-, y-, and z-axes as a 3-dimensional array (Figure :ref:`HAR` D). One-hot labels were also generated in that step. The participant to which each record belongs is kept track of (Figure :ref:`HAR` E) so that no single participant is later split into both training and testing sets.

.. figure:: HAR.png

  Depiction of the "undoing" procedure to return the data in the UCI HAR Dataset to its unprocessed form. **A.** Data is provided as train/test-split single-axis windowed acccelerometer signals. **B.** Combine train and test sets. **C.** Remove windows; reformat labels and subject include's accordingly. **D.** Axes are combined into a three-dimensional time series; one-hot labels are generated. **E.** 3-D time series and labels are grouped by subject to emulate subject-wise data acquisition. :label:`HAR`

For optimizing our model architecture, we used a single 80:20 training-to-testing split; whereas for the testing of the optimized model, we used 5-fold cross validation. After splitting into training and testing sets (Figure :ref:`Pipeline` A-D), the data is standardized by first fitting the standardization parameters (i.e., mean and standard deviation) to the training data and then using these parameters to standardize the training and testing sets separately (Fig. :ref:`Pipeline` E1). This prevents exposing any summary information about the testing set to the model before training, i.e., data leakage. Finally, a fixed-length sliding window was applied (Fig. :ref:`Pipeline` E2), the windows were shuffled to avoid localization during backpropagation (Fig. :ref:`Pipeline` F), and the data was ready to feed into the LSTM neural network.

.. figure:: Pipeline.png

  Outline of the proposed data analysis pipeline. **A.** The data should start as raw tri-axial data files separated into individual records; one record per individual. **B.** Shuffle the records. **C.** Partition the records into k equal groupings for the k-fold cross validation. **D.** Concatenate the records end-to-end within the train and test sets (for feeding in to the LSTM). **E.** Standardize the data, careful to avoid data leakage; subsequently window the data. **F.** Shuffle the windowed data sets. **G.** Train the model on the training data. **H.** Predict outcomes for the testing data using the trained model and score the results. :label:`Pipeline`

*Training* This experiment was broken up into two sections. The first section consisted of hyperparameter optimization. In the past, we have used randomized grid search with cross validation for each model to tune neural network hyperparameters. However, due to the vastness of the search space, it is difficult to assess even 10% of the possible architectures in a reasonable amount of time and computing resources. Thus, for this experiment we turned to heuristic-based search, namely the tree-structured Parzen (TPE) expected improvement (EI) algorithm. EI algorithms estimate the ability of supposed model :math:`M` to outperform some threshold :math:`y^*`, and TPE aims to assist this expectation by modeling the search space. TPE iteratively substitutes equally-weighted prior distributions over hyperparameters with Gaussians centered on examples it sees over time. This re-weighting of the search space allows TPE to estimate :math:`p(y)` and :math:`p(x|y)` for a performance :math:`y` via model :math:`x` for use by EI as :math:`p(y|x)` via Baye's Theorem :cite:`Bergstraetal2011`.

.. math::

  EI_{y^*}(x) := \int_{-\infty}^\infty \text{max}(y^* - y, 0) p_M(y|x)dy

.. math::
  :type: eqnarray

  EI_{y^*}(x) &=& \int_{-\infty}^{y^*} \text{max}(y^* - y, 0) p_M(y|x)dy \\
              &=& \int_{-\infty}^{y^*} \frac{p(x|y)p(y)}{p(x)}dy \\
              &=& \frac{\gamma y^* l(x) \int_-\infty^{y^*} p(y)dx}{y l(x) + (1-\gamma)g(x)} \\
              &\propto& (\gamma + frac{g(x)}{l(x)} (1-\gamma))^-1

where

.. math::

  \gamma = p(y^* < y)

.. math::

  l(x) = p(x|y) \text{if} y<y^*

.. math::

  g(x) = p(x|y) \text{if} y\geq y^*

and :math:`p(a|b)` is the conditional probability of :math:`a` given event :math:`b`.

The ranges of hyperparameters were devised to include all ranges explored by the various reports reviewed in the above section of this paper, as well as any other well-defined range or setting used in the field. The hyperparameters tested are listed in Table :ref:`hyperparameters`. Due to constraints in the Python package used for hyperparameter optimization (i.e., hyperas from hyperopt), the window size, stride length and number of layers were optimized on the highest performing combination of all other hyperparameters via randomized grid search. Thus, for initial optimization, data was partitioned using a window size of 128 with 50% stride length and fed into a 2-layer LSTM network.

.. table:: The various hyperparameters addressed in this experiement, and their respective ranges. :label:`hyperparameters`

  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  | Category           | Hyperparameter                                 | Range                                                                                                            |
  +====================+================================================+==================================================================================================================+
  | Data Processing    | Window Size                                    | 24, 48, 64, 128, 192, 256                                                                                        |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Stride                                         | 25%, 50%, 75%                                                                                                    |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Batch Size                                     | 32, 64, 128, ..., 480                                                                                            |
  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  | Archi-tecture      | Units                                          | 2, 22, 42, 62, ..., 522                                                                                          |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Layers                                         | 1, 2, 3                                                                                                          |
  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  | Forward Processing | Activation Function (unit, state)              | softmax, tanh, sigmoid, ReLU, linear                                                                             |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Bias                                           | True, False                                                                                                      |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Weight Initialization (cell, state)            | zeros, ones, random uniform dist., random normal dist., constant (0.1), orthogonal, Lecun normal, Glorot uniform |
  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  | Regular-ization    | Regularization (cell, state, bias, activation) | None, L2 Norm, L1 Norm                                                                                           |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Weight Dropout (unit, state)                   | uniform distribution (0, 1)                                                                                      |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Batch normalization                            | True, False                                                                                                      |
  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  | Learning           | Optimizers                                     | SGD, RMSProp, Adagrad, Adadelta, Nadam, Adam                                                                     |
  |                    +------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
  |                    | Learning Rate                                  | :math:`10^{-7}, 10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}`                                            |
  +--------------------+------------------------------------------------+------------------------------------------------------------------------------------------------------------------+


For the second portion of the experiment, the highest performing model was assessed using 5-fold cross validation, where the folds were made at the participant level so that no single participant's data ended up in both training and testing sets.

All models were written in the Python programming language. The LSTMs were built and run using the Keras library and TensorFlow as the backend heavy lifter. Hyperas from Hyperopt was used to optimize the network. Sci-kit learn provided the packages for cross validation, randomized grid search and standardization of data. Numpy and Pandas were used to read and reformat the data among various other operations.

*Performance Measures*
During hyperparameter optimization, backpropagation was set to minimize cross-entropy. The best model was selected using the accuracy from the test trial after each training run. During cross-validation, the F1 Score and accuracy are compiled and summed across all folds.

Results
-------
During preliminary testing of a baseline model to ensure the code would run, we found that the model performed better on the raw accelerometer data compared to the data with the gravity-component removed. As such, we used the total accelerometer signal in our experiment. The hyperparameter optimization explored a search space with billions of possible parameter combinations. Due to time constraints, we had to stop the search after two full days (hundreds of training iterations) and use the best-found model up to that point. The model parameters are as follows: window_size=128; stride_length=50% of window size; n_layers = 128; units_per_layer = 128 for layer1, 114 for layer2; cell_output_activation = tanh; recurrent_activation = sigmoid; use_bias = True; unit_forget_bias = True; kernel_initializer = Glorot uniform; cell dropout = 0.5; recurrent_dropout = 0.5; no other regularization used; optimizer = RMSprop; batch_size = 64. The two LSTM layers fed into a single Dense layer with linear activation to reshape the data before passing through a softmax activation function.

During optimization, test accuracies ranged from 16% to 91%.

We ran 5-fold CV on the optimized model and computed the overall and class-wise F1 scores and accuracies. Cross validation yielded an average accuracy of 90.97% and F1 score of 0.90968.

.. table:: Results table including classical benchamarks on 551 hand-crafted (HC) features, various complex and baseline LSTM models on all 9 features provided in the dataset - total accelerometer signals (T), body accelerometer signals (gravity component removed, B), gyroscope signals (G). One of the baseline LSTM's did not explicitly specify the number of features used but only mentioned accelerometer signals. The performances marked with an asterisk a FScores, all others are accuracies. :label:`results`

  +-----------+---------------------+-----------------+-----------+
  |           | Model               | Performance     | Features  |
  +===========+=====================+=================+===========+
  | Classical | MC-SVM              | 0.96^*          | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | HF-MC-SVM           | 0.89^*          | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | OVO SVM             | 96.5%           | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | LVQ                 | 96.35%          | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | Conf-Adaboost.M1    | 94.33%          | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | OVA SVM             | 93.7%           | 551 HC    |
  +           +---------------------+-----------------+-----------+
  |           | KNN                 | 90.6%           | 551 HC    |
  +-----------+---------------------+-----------------+-----------+
  | LSTM RNN  | 4-layer LSTM        | 96.7%           | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | MLSTM-FCN           | 96.71%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | MALSTM-FCN          | 96.71%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | Deep-Res-Bidir LSTM | 93.57%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | b-LSTM              | 91.09%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | Residual LSTM       | 91.55%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | Baseline LSTM 1     | 90.77%          | 9 (T,B,G) |
  +           +---------------------+-----------------+-----------+
  |           | Baseline LATM 2     | 85.35%          | 3-9 (?)   |
  +           +---------------------+-----------------+-----------+
  |           | **Ours (CV)**       | **90.97%**      | **3**     |
  |           |                     +-----------------+-----------+
  |           |                     | **0.90968**     | **3**     |
  |           +---------------------+-----------------+-----------+
  |           | Ours (Single best)  | 95.25%          | 3         |
  |           |                     +-----------------+-----------+
  |           |                     | 0.9572^*        | 3         |
  +-----------+---------------------+-----------------+-----------+

Discussion
-----------
The execution of HAR research in various settings from the biomedical clinic early on :cite:`Bussmanetal2001,Ravietal2005,Bussmanetal98` to current-day innovative settings such as the automobile :cite:`Carvalhoetal2017`, the bedroom :cite:`Moreauetal2016`, the dining room :cite:`Kyritsisetal2017`, and outdoor sporting environments :cite:`Rassemetal2017` justifies the time spent expanding this area of research. As LSTM models are increasingly demonstrated to have potential for HAR research, the importance of deliberate and reproducible works is paramount.

*Review of previous works*
A survey of the literature revealed a lack of cohesiveness regarding the use of LSTMs for accelerometer data and the overall data analysis pipeline. We grew concerned with possible sources of data leakage. For example, test set data should come from different participants than those used for the training data :cite:`Hastieetal2017`, and no information from the test set should be exposed to the model before training.

Regarding preprocessing, we were surprised to see some of the more advanced techniques being employed. These methods require a degree of domain knowledge in signal processing and are more computationally expensive and less realistic for online and on-device implementations than is desired. Much of the appeal of non-linear models such as neural networks is their ability to learn from raw data itself and independently perform smoothing and feature extraction on noisy data through parameterized embedding of the data. For example, Karpathy's 2015 study of LSTMs for language modeling showed specific neurons being activated when quotes were opened and deactivated when the quotes were closed, among other specialized functions :cite:`Karpathyetal2015`. That being said, when dealing with more complex and noisy data, standardization is often important for data-dependent models such as LSTMs since the presence of outliers and skewed distributions may distort the weight embeddings :cite:`Garethetal2017`.

The use of different loss functions and performance measures makes comparisons across studies difficult. Kline and Berardi demonstrate that categorical cross-entropy as the objective function to minimize during training has advantages over more standard error terms such as squared error :cite:`KlineandBerardi`. Furthermore, we view the F1 score, calculated for each class individually and then averaged across classes, as a superior performance measure for the testing set compared to the accuracy for multi-class problems. F1 score combines two nuanced measures of performance, namely the precision and the recall. Precision measures the exactness of the positive predictions by measuring the proportion of correct positive predictions for each class. Recall measures completeness of the positive predictions by measuring the proportions of positive examples identified from the test set. However, since accuracy is more intuitive and commonly used, we feel that reporting both F1 score and accuracy may be useful :cite:`Garethetal2017`.

*Hyperparameter optimization and data analysis pipeline*
We structured our experiments from start to finish with the objective of maintaining simplicity, relying as much as possible on the baseline model itself, and maximizing generalizability of our results. These objectives resonate with the widespread use of smartphones as a source of large amounts of real-world data and efforts by many to apply online and on-device HAR systems. The finding that training the model on the total accelerometer signal outperformed using the signal processed to have the gravity component removed demonstrates a promising potential of non-linear data-dependent models such as neural networks to classify complex noisy data in real-time settings and supports our claim that extensive preprocessing is not necessary.

We demonstrate the ability of these models to perform competitively with benchmark experiments even after extreme care is taken to prevent data leakage. We outperformed the only other study possibly identified to use solely accelerometer signals from this dataset :cite:`U2018`. Among the other LSTMs that were trained using more features from this same dataset, our averaged cross validation results slightly outperformed the baseline LSTM trained on this data :cite:`Zhaoetal2017` and scored competitively with the b-LSTM (91.09%), the residual LSTM (91.55%), and the deep res-bidir-LSTM (93.57%) published in the same report. Additionally, we found no evidence of cross validation in the benchmark reports that utilized the UCI HAR dataset. As such, we compare our single best-performing test's accuracy of 95.25% and F1 score of 0.9572 and find it to compete with the highest scoring models, which used higher dimensional data and additional complexity in their models: 4 layer LSTM (96.7% accuracy, 0.96 F1score), MLSTM-FCN and MALSTM-FCN (96.71% accuracy), and OVO SVM (96.4% accuracy, 551 features).

Although we were unable to complete our TPE based search over the entire hyperparameter search space, the algorithm was able to find a well-performing model, and the data analysis pipeline was demonstrated from start to finish.

Conclusion/Future Work
--------------------------------

We have used a data-centered approach to optimize an LSTM neural network for HAR research. As opposed to taking steps to improve the data quality or increase the complexity of our model, we worked with the baseline LSTM to allow it to fit the specific dataset given to it.

Additionally, we have demonstrated one implementation of a well-defined data analysis pipeline which will foster reproducibility and deliberate progression of the field. This pipeline focuses on simplicity and maintaining data science good practices.

This initial experiment has laid the groundwork for further exploration and understanding of LSTMs for HAR research. We would like to complete the hyperparameter search for multiple datasets so as to assess the resulting differences. Inspired by Karpathy’s 2015 paper, we would also like to dig deeper into the networks and explore the neurons’ representations of the data across time, comparing these weight embeddings and activation patterns with hand crafted features of the data.
