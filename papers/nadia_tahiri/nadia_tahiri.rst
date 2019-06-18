:author: Nadia Tahiri
:email: tahiri.nadia@courrier.uqam.ca
:institution: Département d’Informatique, Université du Québec à Montréal, Case postale 8888, Succursale Centre-ville, H3C 3P8 Montréal, Canada
:corresponding:



-----------------------------------------------------------------------------------------------------
An intelligent shopping list based on the application of partitioning and machine learning algorithms
-----------------------------------------------------------------------------------------------------

.. class:: abstract
   
  A grocery list is an integral part of the shopping experience for many consumers. Several mobile retail studies of grocery apps indicate that potential customers place the highest priority on features that help them to create and to manage personalized shopping lists. 
  First, we propose the development of a new machine learning model written in Python 3 that predicts which grocery products the consumer will buy again or will try to buy for the first time, and in which store(s) they will shop. 
  Second, we developed a smart shopping list template to provide consumers with a personalized weekly shopping list based on their shopping history and known preferences. 
  As the explanatory variables, we used available grocery shopping history, store promotion information for the given region, 
  as well as product price statistics.

.. class:: keywords

   Machine Learning, Prediction, Long short-term memory, Convolutional Neural Network, Gradient Tree Boosting, :math:`F_1`, Python, Sklearn, Tensorflow

Introduction
------------

A typical grocery retailer offers consumers thousands of promotions every week 
to attract more consumers and thus improve their economic performance [TTR16]_. 
The studies by Walters and Jamil (2002, 2003) of [WJ02]_ and [WJ03]_ found that about 39% of all items purchased 
during an inter-category grocery were weekly specials and promotions, and about 30% of consumers 
surveyed were very sensitive to product prices, buying more promotional items than regular ones. 
With the recent expansion of machine learning methods, including deep learning, 
it seems appropriate to develop a series of methods that allow retailers to offer consumers attractive 
and cost-effective shopping baskets, as well as to propose consumers tools 
to create smart personalized weekly shopping lists based on historical purchases, 
known preferences and specials available in local stores.

A grocery list is an integral part of the shopping experience for many consumers. 
The lists serve, for example, as a reminder, a budgeting tool, 
or an effective way to organize weekly grocery shopping. 
In addition, several mobile retail studies indicate that potential customers place 
the highest priority on features that help them to create and to manage personalized 
shopping lists interactively [NPS03]_ and [Faz17]_.

Problem statement and proposal
------------------------------

In this section, we present the problem statement and describe the considered machine learning architecture.
First, by using the grocery shopping database in Canada `MyGroceryTour` [#]_ (see Figure :ref:`MyGroceryTour`), 
we looked for the partitioning of consumers into classes that will group 
them efficiently based on purchases made. 
Then, this classification was used in the prediction stage. 
Since the real consumer data contains thousands of individual articles; we focus primarily on product categories. 
A principal component analysis (linear and polynomial PCA [Jol11]_) was first computed to visualize the initial data  
and to choose the number of main components to use when partitioning consumers into classes. 
The application of efficient partitioning methods, such as K-means [Jai10]_ and X-means [PM+00]_, 
made it possible to determine the number of classes of consumers, as well as their distribution by class.

.. figure:: figures/trois_magasins.png
   :align: center
   
   MyGroceryTour website for the postal code H2Y 1C6 in Montreal. 
   The website was created to test the new machine learning model. 
   It was written in Javascript, Bootstrap and Python. :label:`MyGroceryTour` 

.. [#] MyGroceryTour.ca

Second, we developed a statistical model to predict which products previously purchased will be 
in the next order of the consumer. By using explanatory variables, such as available grocery shopping histories, 
information on current promotions in stores in the given region, and commodity price statistics, 
we developed a machine learning model that is able to:

i. Predict which groceries the consumer will want to buy again or will try to buy for the first time, as well as which store(s) (within the area they usually shop in);
ii. Create a smart shopping list by providing the consumer with a weekly shopping list customized based on their purchase history and known preferences. 

This list also includes recommendations regarding the optimal quantity of every product suggested and the store(s)  
where these products are to be purchased. We also calculate the consumer's optimal weekly commute 
using the generalized commercial traveler algorithm (see Figure :ref:`circuit`).

.. figure:: figures/mygrocerytour_circuit.png
   :align: center
   
   Screenshot of MyGroceryTour website with an optimal shopping journey using the generalized commercial traveler algorithm. :label:`circuit`

An :math:`F_1` statistics maximization algorithm [NCLC12]_, 
based on dynamic programming, was used to achieve objective (i), 
which will be of major interest to retailers and distributors. 
A deep learning method [Gir16]_, based on Recurrent Neuron Networks (RNN) 
and Convolutional Neuron Network (CNN), and both implemented using the TensorFlow library [Gir16]_, 
were used to achieve objective (ii). Those implementations can provide significant benefits to consumers.

The problem can be reformulated as a binary prediction task. Given a consumer, 
the history of their previous purchases and a product with their price history, 
predict whether or not the given product will be included in the grocery list of the consumer. 
Our approach adapted a variety of generative models to existing data, i.e., 
first-level models, and to use the internal representations of 
these models as features of the second-level models. 
RNNs and CNNs were used at the first learning level 
and forward propagation neural networks (Feed-forward NN) 
was applied at the second learning level.

Depending on the user :math:`u` and the user purchase history
(:math:`shop_{t-h:t}`, :math:`h > 0`), we predict the probability that a product :math:`i` is included 
in the next shop :math:`_{t+1}` of :math:`u`

Dataset
-------
In this section we discuss the details of synthetic and real datasets,
the latter obtained from `MyGroceryTour`.

Features
========

For the project, we consider only the significant features, 
such as `distance`, `special` rate, `products`, and `store`. 
All features, we used, are described as follow:

- **user\_id**: user number. We take care of anonymized all data set. :math:`user\_id \in \underbrace{\{1 \cdots 374\}}_{\text{reals}} \cup \underbrace{\{375 \cdots 1374\}}_{\text{generated}}`
- **order\_id**: unique number of the basket. :math:`order\_id \in \mathbb{Z}`
- **store\_id**: unique number of the store. :math:`store\_id \in \{1 \cdots 10\}` 
- **distance**: distance to the store. :math:`distance \in \mathbb{R}^+`
- **product\_id**: unique number of the product. :math:`product\_id = 46 000`
- **category\_id**: unique category number for a product. :math:`category\_id \in \{1 \cdots 24\}`  
- **reorder**: the reorder is equal to 1 if this product has been ordered by this user in the past, 0 else. :math:`reorders \in \{0,1\}`
- **special**: discount percentage applied to the product price at the time of purchase. :math:`special \in \{[0\%,15\%[, [15\%,30\%[, [30\%,50\%[, [50\%,100\%[\}`
     
Consumer profile
================

We found that there are 3 consumer profiles see [WJ03]_, [WJ02]_, and [TNTK16]_. 
The first group is consumers who only buy products on promotion.
The second group is consumers who always buy the same products (without considering promotions).
Finally, the third group is consumers who buy products whether there is a promotion or not.
On the model, we plan to consider that information and make the prediction more personalized on the consumer profile.

Data Synthesis
==============

Since the real dataset was not enough to complete correctly the project, we increased it.
We described the sets of data simulated in the study, 
and we presented in detail the results of the simulation step.
For :math:`store\_id`, we started with an initial store and changed stores based on the proportion of common products between baskets.
If we assumed that the store coordinates are normally distributed :math:`\mathcal{N}(0,\sigma^2)` independently, 
the distance between this store and the consumer home located originally :math:`(0,0)` follows a Rayleigh distribution [KR05]_ with the :math:`\sigma` parameter.
Finally, we increased the `special` feature. This variable is based on the composition of the baskets, choosing a special random proportional to the Boltzmann distribution [AAR+18]_.
We observed that the sizes of the generated baskets follow the same distribution as the original basket sizes
(see Figure :ref:`orderfrequency`).

.. figure:: figures/order_frequency.png
   :align: center
     
   Difference of basket size distribution between **Baskets generated** in blue and **Original baskets** in red.  :label:`orderfrequency`

Preprocessing dataset
=====================

We launched the preprocessing dataset tasks on the servers of Compute Canada. This step was carried out using 172 nodes 
and 40 cores with an Intel Gold 6148 Skylake CPU(2.4 GHz) and  NVidia V100SXM2(16G memory). We preprocessed the user data, 
the product data, and the department data. The bash script has given as follow:

.. code-block:: bash

   #!/bin/bash
   #!SBATCH --time=48:00:00
   #SBATCH --account=def-jgnes
   #!SBATCH --job-name=market_cpu
   #SBATCH --output=market_out_cpu
   #SBATCH --error=market_err_cpu
   #!SBATCH --mem=32000M
   #SBATCH --mail-user=tahiri.nadia@courrier.uqam.ca
   #SBATCH --mail-type=BEGIN
   #SBATCH --mail-type=END
   #SBATCH --mail-type=FAIL

Models
------

In this section, we described the workflow (see Figure :ref:`workflow`) and the models we used.

Long short-term memory (LSTM) network
=====================================

The LSTM [HS97]_ is a recurrent neural network (RNN) that has an input, hidden (memory block), and an output layer. 
The memory block contains 3 gate units namely the input, forget, 
and output with a self-recurrent connection neuron [HS97]_.

- **Input gate** learns what information is to be stored in the memory block.
- **Forget gate** learns how much information to be retained or forgotten from the memory block.
- **Output gate** learns when the stored information can be used.

Figure :ref:`lstm` illustrates the proposed architecture and summarizes the detail involved in the structure. 

A combined RNN and CNN was trained to predict the probability that a user will order a product at each timestep. 
The RNN is a single-layer LSTM and the CNN is a 6-layer causal CNN with dilated convolutions.
The last layer is a fully-connected layer which makes the classification.
The CNN was used as a feature extractor and the LSTM network as a sequential learning.

.. figure:: figures/lstm.png
   :align: center 
  
   This figure shows a chain-structured LSTM network. An LSTM architecture contains forget, learn, remember and use gates that determine whether an input is so important  that  it  can  be  saved.  
   In  the  LSTM  unit representing in this figure, four different functions: sigmoid (:math:`\sigma`), hyperbolic tangent (:math:`tanh`), multiplication (:math:`*`), and sum (:math:`+`) are used, 
   which make it easier to update the weights during the backpropagation process. :label:`lstm`

Overall characteristics of the neuron networks which used in this project are described as follow:

.. code-block:: python

    nn = rnn(
     reader=dr,
     log_dir=os.path.join(base_dir, 
                          'logs'),
     checkpoint_dir=os.path.join(base_dir, 
                                'checkpoints'),
     prediction_dir=os.path.join(base_dir, 
                                'predictions'),
     optimizer='adam',
     learning_rate=.001,
     lstm_size=512,
     batch_size=64,
     num_training_steps=300,
     early_stopping_steps=10,
     warm_start_init_step=0,
     regularization_constant=0.0,
     keep_prob=1.0,
     enable_parameter_averaging=False,
     num_restarts=2,
     min_steps_to_checkpoint=100,
     log_interval=20,
     num_validation_batches=4,
    )

Gradient Boosted Tree (GBT) network
===================================

GBT [Fri02]_ is an iterative algorithm that combines simple parameterized functions with low performance 
(i.e. high prediction error) to produce a highly accurate prediction rule. GBT utilizes an ensemble of weak
learners to boost performance; this makes it a good candidate model for predicting the grocery shopping list. 
It requires little data preprocessing and tuning of parameters while yielding interpretable results, 
with the help of partial dependency plots and other investigative tools. 
Further, GBT can model complex interactions in a simple fashion and be applied in both classification and 
regression with a variety of response distributions including Gaussian [Car03]_, Bernoulli [CMW16]_, Poisson [PJ73]_, and Laplace [Tay19]_. 
Finally, missing values in the collected data can be easily managed.
Moreover, in this study, we denote frequently missing data in the history grocery list by the user, that is why this technique is more adapted.

The data is divided into 2 groups (training and validation) which comprise 90% and 10% of the data respectively.
The final model has two neural networks and a GBT classifier.
Once trained, it was used to predict in real time what would be the consumer's basket, based on their history of purchases and current promotions in neighboring stores.
Based on the validation loss function, we eliminated the LSTM Rays and LSTM model size (see Figure :ref:`lstm`).

First level model (feature extraction)
======================================

Our goal is to find a diverse set of representations using neural networks (see Table 1). 
Table 1 summarizes top-level models used by the algorithm and we described each type of model used for every representation (e.g. `Products`, `Category`, `Size of the basket`, and `Users`).
We estimated the probability of the :math:`product_i` to be include to 
the next basket :math:`order_{t+1}` with :math:`orders_{t-h}`, 
with :math:`t` represents the actual time, 
:math:`t+1` represents the next time,
and :math:`t-h` represents all previous time (i.e. historical time).
We decomposed the matrix {user,product} by two matrices one corresponding to the user and another to the product.
We predicted the probability to have the :math:`product_i` on the next :math:`order_{t+1}` 
knowing the historical purchases of this user. We used one LSTM with 300 neurons.
We also predicted the probability that the :math:`product_i` is included for which category. 
Finally, we estimated the size of the next order minimizing the root mean square error (RMSE).

.. raw:: latex

   \begin{table}

     \begin{longtable}{lcc}
     \hline
     \textbf{Representation} & \textbf{Description} & \textbf{Type}\tabularnewline
     \hline
     \textcolor{blue}{Products} & \textcolor{blue}{\begin{tabular}{@{}c@{}} Predicts P$(\text{product}_{i}\in \text{order}_{t+1})$\\ with orders$_{t-h,t}$, $h>0$.\end{tabular}}& \textcolor{blue}{\begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}} \\
     \hline
     Categories & Predicts P$(\exists i:\text{product}_{i,t+1} \in \text{category}_r)$. & \begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}\\
     \hline
     Size & Predict the size of the order$_{t+1}$. & \begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}\\
     \hline
     \textcolor{blue}{\begin{tabular}{@{}c@{}}Users \\ Products \end{tabular}} & \textcolor{blue}{Decomposed $V_{(u \times p)}=W_{(u \times d)} H^T_{(p \times d)}$} & \textcolor{blue}{\begin{tabular}{@{}c@{}}Dense\\ (50 neurons)\end{tabular}}\\
     \hline
     \end{longtable}

     \caption{Top-level models used.}
         \label{tab:model1}

   \end{table}

Latent representations of entities (embeddings)
===============================================

For each :math:`a \in \mathcal{A}`, an embedding :math:`T:\mathcal{A} \rightarrow \mathbb{R}^{d}` returns a vector :math:`d`-dimensionel.
If :math:`\mathcal{A} \subset \mathbb{Z}`, :math:`T` is a matrix :math:`|\mathcal{A}|\times d` learned by backpropagation. We represented in Table 2 all dimensions of each model used.

.. raw:: latex

    \begin{table}
        
        \begin{longtable}{lcc}
        \hline
        \textbf{Model} & \textbf{Embedding} & \textbf{Dimensions}\tabularnewline
        \hline
        LSTM Products & Products & $49,684 \times 300$\\
        \hline
        LSTM Products & Catégories & $24 \times 50$\\
        \hline
        LSTM Products & Departments & $50 \rightarrow 10$\\
        \hline
        LSTM Products & Users & $1,374 \times 300$\\
        \hline
        NNMF & Users & $1,374 \times 25$\\
        \hline
        NNMF & Products & $49,684 \times 25$\\
        \hline        
        \end{longtable}

        \caption{Dimensions of the representations learned by different models.}
        \label{tab:model2}

    \end{table}

Second level model: Composition of baskets
==========================================

The final basket is chosen according to the final reorganization probabilities, choosing the subset of products with the expected maximum :math:`F_1` score, see [LEN14]_ and [NCLC12]_.
This score is frequently used especially when the relevant elements are scarce.

.. math::
   
   \max_\mathcal{P} \mathbb{E}_{p'\in \mathcal{P}}[F_1(\mathcal{P})]=\max_\mathcal{P}\mathbb{E}_{p'\in \mathcal{P}}\bigg[\frac{2\sum_{i\in \mathcal{P}}\text{TP}(i)}{\sum_{i\in \mathcal{P}}(2\text{VP}(i)+\text{FN}(i)+\text{FP}(i))}\bigg],

where True Positive :math:`(TP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=1]`, False Negative :math:`(FN)=\mathbb{I}[\lfloor p(i)\rceil=0]\mathbb{I}[R_i=1]`, False Positive :math:`(FP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=0]` and :math:`R_i=1` if the product :math:`i` was bought in the basket :math:`p'\in \mathcal{P}`, else :math:`0`.\\
We used :math:`\mathbb{E}_{X}[F_1(Y)]=\sum_{x\in X}F_1(Y=y|x)P(X=x)`

.. figure:: figures/workflow.png
   :align: center
   :scale: 29%
   
   The graphical illustration of the proposed model trying to predict the next basket in term of the list of product. 
   The first level of the model used LSTM and NNMF. 
   The second level of the model applied GBT.
   Finally, the last test considered to predict the next basket by using :math:`F_1`. :label:`workflow`

Statistics
==========

We present the obtained results using proposed method in this section. 
As well as the metrics (see Equations 1-6) that are utilized to evaluate the performance of methods.

Statistic score
===============

The *accuracy* of a test is its capability to recognize the classes properly. 
To evaluate the accuracy of the model, we should define the percentage 
of true positive and true negative in all estimated cases, 
i.e. the sum of true positive, true negative, false positive, and false negative.
Statistically, this metric can be identified as follow:

.. math::
   :label: e:matrix
   
   Accuracy = \frac{(TP+TN)}{(TP+TN+FP+FN)}

where:

- **TP** is True Positive, i.e. the number of positively labeled data, which have been classified as `Positive`, correct class,
- **FP** is False Positive, i.e. the number of negatively labeled data, which falsely have been classified as `Positive`,
- **TN** is True Negative, i.e. the number of negatively labeled data, which have been classified as `Negative`, correct class, and 
- **FN** is False Negative, i.e.  the number of positively labeled data, which falsely have been classified as `Negative`.

The *precision* is a description of random errors, a measure of statistical variability.
The formula of precision is the ratio between TP with all truth data (positive or negative). 
The Equation is described as follow:

.. math::
   :label: e:matrix
   
   Precision = \frac{TP}{(TP+FP)}

The *recall* or *sensitivity* or *TP Rate* is defined as the number of true positive data labeled divided by 
the total number of TP and FN labeled data.

.. math::
  :label: e:matrix
  
   Recall = Sensitivity = TP Rate = \frac{TP}{(TP+FN)}

The *F-measure* or :math:`F_1` precise the classifier, as well as how robust it is (does not miss a significant number of instances).

.. math::
   :label: e:matrix
   
   F-measure = F1 = \frac{2TP}{(2TP + FP + FN)} 

Finally, we evaluated the model by *FP Rate* which corresponds to the ratio between FP and sum of TN and FP.

.. math::
   :label: e:matrix
   
   FP Rate = FPR = \frac{FP}{(TN+FP)} 
   
We examined these six evaluation metrics on this study.

Python Script
-------------

The final reorder probabilities are a weighted average of the outputs from the second-level models. The final basket is chosen by using these probabilities and choosing the product subset with maximum expected F1-score.
The select_products function in Python script is the following:

.. code-block:: python
    :linenos:
    
    from f1_optimizer import F1Optimizer

    def select_products(x):
     series = pd.Series()

     for prod in x['product_id'][x['label'] > 0.5:
       if prod != 0:
        true_products = [str(prod)].values]
       else:
        true_products = ['None'].values]

     if true_products:
      true_products = ' '.join(true_products)
     else:
      true_products = 'None'

     prod_preds_dict = dict(zip(x['product_id'].values,
                                x['prediction'].values))
     none_prob = prod_preds_dict.get(0, None)
     del prod_preds_dict[0]

     other_products = np.array(prod_preds_dict.keys())
     other_probs = np.array(prod_preds_dict.values())

     idx = np.argsort(-1*other_probs)
     other_products = other_products[idx]
     other_probs = other_probs[idx]

     opt = F1Optimizer.max_expectation(other_probs,
                                       none_prob)

     best_prediction = ['None'] if opt[1] else []
     best_prediction += list(other_products[:opt[0]])

     if best_prediction:
      predicted_products = ' '.join(map(str, 
                                    best_prediction))
     else:
      predicted_products = 'None'

     series['products'] = predicted_products
     series['true_products'] = true_products

     return true_products, predicted_products, opt[-1]

Results
-------

Figure :ref:`productpca` illustrates PCA of 20 random products projected in 2 dimensions. 
The results show clearly the cluster of Pasta sauce with Pasta group. 
In fact, this result can identify consumer buying behavior.

.. figure:: figures/product_pca.png
   :align: center
   :scale: 25%
   
   Embeddings of 20 random products projected in 2 dimensions. :label:`productpca`

:math:`F_1` in Figure :ref:`violon` (a) shows that the profiles of all promotions are similar. 
In the perspective of this work, it will be interesting to include weight base on statistic value. 
In Statistic Canada - 2017, only 5% of all promotions are more than 50% promoted, 95% of all promotions are less than 50%. 
Weightings are needed to give the model more robust. 
Figure :ref:`violon` (b) indicates that all shops follow the same profiles in the model. 

.. figure:: figures/violon.png
   :align: center
   :scale: 20%
   :figclass: wt
   
   Distribution of :math:`F_1` measures against stores (a) and rebates (b). :label:`violon`

Figure :ref:`productsF1` and Table 3 indicates that the values of :math:`F_1` metric to all products. 
Some products are easy to predict with the value of :math:`F_1` > 0 and 
some products are so hard to predict with the value of :math:`F_1` < 0. 
For the first group, they are products included on restriction regimes 
such as `diet cranberry fruit juice`, `purified water`, and `total 0% blueberry acai greek yogurt`.

.. raw:: latex
    
    \begin{table}

        \begin{longtable}{lc}
        \hline
                                      \textbf{Product} &        \textbf{$F_1$} \\
        \hline
    Gogo Squeez Organic Apple Strawberry Applesauce &  0.042057 \\
            Organic AppleBerry Applesauce on the Go &  0.042057 \\
                           Carrot And Celery Sticks &  0.042057 \\
             Gluten Free Peanut Butter Berry  Chewy &  0.042057 \\
                   Organic Italian Balsamic Vinegar &  0.049325 \\ 
        \hline
                         Diet Cranberry Fruit Juice &  0.599472 \\
                                     Purified Water &  0.599472 \\
     Vanilla Chocolate Peanut Butter Ice Cream Bars &  0.599472 \\
  Total 0\% with Honey Nonfat Greek Strained Yogurt &  0.590824 \\
              Total 0\% Blueberry Acai Greek Yogurt &  0.590824 \\
        \hline
        \end{longtable}
		\caption{The average value of $F_1$ for all products considered.}
    \end{table}   

.. figure:: figures/products_F1.png
   :align: center
   :scale: 20%
   
   Distribution of :math:`F_1` measures relative to products around average. :label:`productsF1`
	
.. raw:: latex
    
    \begin{table}

        \begin{longtable}{|l|c|}
        \hline
           \textbf{Product} &  \textbf{Number of baskets} \\
        \hline
                     Banana &   6138 \\
               Strawberries &   3663 \\
       Organic Baby Spinach &   1683 \\
                      Limes &   1485 \\
                 Cantaloupe &   1089 \\
              Bing Cherries &    891 \\
         Small Hass Avocado &    891 \\
         Organic Whole Milk &    891 \\
                Large Lemon &    792 \\
 Sparkling Water Grapefruit &    792 \\
        \hline
        \end{longtable}
        \caption{The 10 most popular products included in the predicted baskets.}
  \end{table}
	
.. figure:: figures/pearsonr.png
   :align: center

   Distribution of :math:`F_1` measures against consumers and products. :label:`pearsonr`

We evaluated the model with the statistics score given in Section 'Statistic score'.

Conclusions and Future Work
---------------------------

We analyzed grocery shopping data generated by the consumers of the site `MyGroceryTour`.
We developed a new machine learning model to predict which grocery products the consumer will
buy and in which store(s) of the region he/she will do grocery shopping.
We created an intelligent shopping list based on the shopping history of consumer and his/her
known preferences.
The originality of the approach, compared to the existing algorithms, is that in addition to the
purchase history we also consider promotions, possible purchases in different stores and the
distance between these stores and the home of the consumer.

We have modelled the habits of the site's consumers
MyGroceryTour with the help of deep neural networks.
We used two types of neural networks during
Learning: Recurrent Neural Networks (RNN) and Networks
forward-propagating neurons (Feedforward NN).
The value of the :math:`F_1` statistic that represents the quality of the model
need to be increasing on the next step. The constant influx of new data on *MyGroceryTour*
improved the model over time.
The originality of the approach, compared to existing algorithms,
is that in addition to the purchase history we also consider the
promotions, possible purchases in different stores and distance
between these stores and the consumer's home.

In future work, we plan to predict the grocery store that will visited next, and to include the product quantities in the basket proposed to the user. 
We suggest also to ponderate the algorithm with the distance between shop and user home coordinates to the promotion rate.

Acknowledgments
---------------

The authors thank PyCon Canada for their valuable comments on this project. This work used
resources of the Calcul Canada. This work was supported by Natural Sciences 
and Engineering Research Council of Canada and Fonds de Recherche sur la Nature et Technologies of Quebec. 
The funds provided by these funding institutions have been used. We would like to thanks SciPy conference 
and anonymous reviewers for their valuable comments on this manuscript.

Abbreviations
-------------

- CNN - Convolutional Neural Network
- GBT  - Gradient Tree Boosting
- LSTM - Long short-term memory
- ML - Machine Learning
- NN - Neuron Networks
- PCA - Principal Component Analysis
- RMSE - Root Mean Square Error
- RNN - Recurrent Neuron Networks


References
----------

.. [AAR+18] Amin, Mohammad H., Evgeny Andriyash, Jason Rolfe, Bohdan Kulchytskyy, and Roger Melko. 
            Quantum boltzmann machine.
            Physical Review X, 8(2):021050, 2018.
.. [Car03] Rasmussen, Carl Edward. Gaussian processes in machine learning.
           In Summer School on Machine Learning, pages 63:71. Springer, Berlin, Heidelberg, 2003.
.. [CMW16] Maddison, Chris J., Andriy Mnih, and Yee Whye Teh. 
           The concrete distribution: A continuous relaxation of discrete random variables. 
           arXiv preprint arXiv:1611.00712, 2016.
.. [Faz17] Fatlume Fazliu. En utforskande studie: inköpslistor som app, 2017.
.. [Fri02] Jerome H Friedman. Stochastic gradient boosting. Computational
           Statistics & Data Analysis, 38(4):367–378, 2002.
.. [GBC16] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep
           learning. MIT press, 2016.
.. [Gir16] Sanjay Surendranath Girija. Tensorflow: Large-scale machine
           learning on heterogeneous distributed systems. Software available
           from tensorflow. org, 2016.
.. [HS97] Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory.
          Neural computation, 9(8):1735–1780, 1997.
.. [Jai10] Anil K Jain. Data clustering: 50 years beyond k-means. Pattern
           recognition letters, 31(8):651–666, 2010.
.. [Jol11] Ian Jolliffe. Principal component analysis. Springer, 2011.
.. [KR05] Debasis Kundu and Mohammad Z Raqab. Generalized rayleigh
          distribution: different methods of estimations. Computational
          statistics & data analysis, 49(1):187–200, 2005.
.. [LEN14] Zachary C Lipton, Charles Elkan, and Balakrishnan
           Naryanaswamy. Optimal thresholding of classifiers to maximize
           f1 measure. In Joint European Conference on Machine Learning
           and Knowledge Discovery in Databases, pages 225–239. Springer,
           2014.
.. [NCLC12] Ye Nan, Kian Ming Chai, Wee Sun Lee, and Hai Leong Chieu.
            Optimizing f-measure: A tale of two approaches. arXiv preprint
            arXiv:1206.4625, 2012.
.. [NPS03] Erica Newcomb, Toni Pashley, and John Stasko. Mobile computing
           in the retail arena. In Proceedings of the SIGCHI Conference
           on Human Factors in Computing Systems, pages 337–344. ACM,
           2003.
.. [PJ73] Consul, Prem C., and Gaurav C. Jain. 
          A generalization of the Poisson distribution. 
          Technometrics 15(4):791-799, (1973).
.. [PM+00] Dan Pelleg, Andrew W Moore, et al. X-means: extending kmeans
           with efficient estimation of the number of clusters. In Icml,
           volume 1, pages 727–734, 2000.
.. [Tay19] Taylor, James W. Forecasting value at risk and expected shortfall using a 
           semiparametric approach based on the asymmetric Laplace distribution.
           Journal of Business & Economic Statistics 37(1):121-133, 2019.
.. [TNTK16] Arry Tanusondjaja, Magda Nenycz-Thiel, and Rachel Kennedy.
            Understanding shopper transaction data: how to identify crosscategory
            purchasing patterns using the duplication coefficient.
            International Journal of Market Research, 58(3):401–419, 2016.
.. [TTR16] Arry Tanusondjaja, Giang Trinh, and Jenni Romaniuk. Exploring
           the past behaviour of new brand buyers. International Journal of
           Market Research, 58(5):733–747, 2016.
.. [WJ02] Rockney Walters and Maqbul Jamil. Measuring cross-category
          specials purchasing: theory, empirical results, and implications.
          Journal of Market-Focused Management, 5(1):25–42, 2002.
.. [WJ03] Rockney G Walters and Maqbul Jamil. Exploring the relationships
          between shopping trip type, purchases of products on promotion,
          and shopping basket profit. Journal of Business Research,
          56(1):17–29, 2003.

