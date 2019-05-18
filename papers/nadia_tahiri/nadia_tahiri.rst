:author: Nadia Tahiri
:email: tahiri.nadia@courrier.uqam.ca
:institution: University of Quebec at Montreal
:corresponding:

:bibliography: nTahiriSciPy

-----------------------------------------------------------------------------------------------------
An intelligent shopping list based on the application of partitioning and machine learning algorithms
-----------------------------------------------------------------------------------------------------

.. class:: abstract
   
   A shopping list is an integral part of the shopping experience for many consumers. 
   Several mobile retail studies indicate that potential customers place the highest 
   priority on features that help them create and manage personalized shopping lists. 
   First, we propose to develop a new model of machine learning written in Python3.6 
   and plan to predict which grocery products the consumer in buy again or 
   will try to buy for the first time, and in which store(s) will the area he will shop. 
   Second, we developped a smart shopping list template to provide consumers with a 
   personalized weekly shopping list based on their historical shopping history and 
   known preferences. As explanatory variables, we used available grocery shopping histories, 
   store promotion information for the given region, as well as product price statistics.

.. class:: keywords

   Machine Learning, prediction, python, Long short-term memory, Convolutional Neural Network, Gradient Tree Boosting, :math:`F_1`, sklearn, tensorflow

Introduction
------------

A typical grocery retailer offers consumers thousands of promotions every week 
to attract more consumers and thus improve their economic performance :cite:`tanusondjaja2016exploring`. 
Studies by Walters and Jamil (2002, 2003) of :cite:`walters2002measuring` and :cite:`walters2003exploring` found that about 39% of all items purchased 
during an inter-category grocery were specials of the week and about 30% of consumers 
surveyed were very sensitive to product prices, buying more promotional items than regular items. 
With the recent expansion of machine learning methods, including deep learning, 
it seems appropriate to develop a series of methods that allow retailers to offer consumers attractive 
and cost-effective shopping baskets, as well as to offer consumers tools 
to create smart personalized weekly shopping lists based on historical purchases, 
known preferences and specials available in local stores.

A shopping list is an integral part of the shopping experience for many consumers. 
Shopping lists serve, for example, as a reminder, a budgeting tool, 
or an effective way to organize weekly grocery shopping. 
In addition, several mobile retail studies indicate that potential customers place 
the highest priority on features that help them create and manage personalized 
shopping lists interactively :cite:`newcomb2003mobile` and :cite:`fazliu2017utforskande`.

Our proposal
------------

By using the grocery shopping database in Canada MyGroceryTour.ca (see Figure :ref:`circuitpromo`), 
first we looked for the partitioning of consumers into classes that will group 
them efficiently based on purchases made. 
Then, this classification was used in the prediction stage. 
Since real consumer data contains thousands of individual articles, 
we focus primarily on categories of articles. 
The principal component analysis (linear and polynomial PCA :cite:`jolliffe2011principal`) was first be performed to better visualize the initial data 
and to choose the number of main components to use when partitioning consumers into classes. 
The application of efficient partitioning methods, such as K-means :cite:`jain2010data` and X-means :cite:`pelleg2000x`, 
made it possible to determine the number of classes of consumers, 
as well as their distribution by class.

.. figure:: figures/trois_magasins.png
   :align: center
   
   CircuitPromo.ca website for the postal code H2Y 1C6 in Montreal. :label:`circuitpromo` 

Secondly, we developed a statistical model to predict which products previously purchased will be 
in the next order of the consumer. By using explanatory variables, such as available grocery shopping histories, 
information on current promotions in stores in the given region, and commodity price statistics, 
we developed a model of machine learning and able to:

1. Predict which groceries the consumer will want to buy again or will try to buy for the first time, and in which store(s) in the area he will shop;
2. Create a smart shopping list by providing the consumer with a weekly shopping list customized based on their purchase history and known preferences. 

This list was also include recommendations regarding the optimal quantity of each product suggested and the store(s) 
where these products are to be purchased. We also calculated the consumer's optimal weekly commute 
using the generalized commercial traveller algorithm (see Figure :ref:`circuit`).

.. figure:: figures/mygrocerytour_circuit.png
   :align: center
   
   Screenshot of CircuitPromo.ca website with an optimal shopping journey. :label:`circuit`

:math:`F_1` statistics maximization algorithm :cite:`nan2012optimizing`, 
based on dynamic programming, was used to achieve objectives (i), 
which will be of major interest to retailers and distributors. 
A deep learning method :cite:`goodfellow2016deep`, based on recurrent neuron networks (RNN) 
and convolutional neuron network (CNN), and implemented in Google's TensorFlow tool :cite:`girija2016tensorflow`, 
was used to achieve objectives (ii), which will be of major interest to consumers.

The problem can be reformulated as a binary prediction task: given a consumer, 
the history of his previous purchases and a product with his price history, 
to predict whether or not the given product will be included in the grocery list of the consumer. 
Our approach adapted a variety of generative models to existing data, i.e., 
first-level models, and to use the internal representations of 
these models as features of the second-level models. 
Recurrent neural networks and convolutional neural networks was used at the first learning level 
and forward propagation neural networks (Feedforward NN) 
was used at the second level of learning.

Depending on the user :math:`u` and the user purchase history
(shop :math:`_{t-h:t}`, :math:`h>0`), we predict the probability that a product :math:`i` is included 
in the next shop :math:`_{t+1}` of :math:`u`

Dataset
-------
In this section we discuss the details of our set synthetic and real datasets.
The real datasets was obtained from CircuitPromo.ca as basic data.

*Features*

The features are described as follow:

- **user\_id**: user number. :math:`user\_id \in \underbrace{\{1 \cdots 374\}}_{\text{reals}} \cup \underbrace{\{375 \cdots 1374\}}_{\text{generated}}`
- **order\_id**: unique number of the basket. :math:`order\_id \in \mathbb{Z}`
- **store\_id**: unique number of the store. :math:`store\_id \in \{1 \cdots 10\}` 
- **distance**: distance to the store. :math:`distance \in \mathbb{R}^+`
- **product\_id**: unique number of the product.
- **category\_id**: unique category number for a product. :math:`aisle\_id \in \{1 \cdots 24\}`  
- **reorder**: 1 if this product has been ordered by this user in the past, 0 else. :math:`reorders \in \{0,1\}`
- **special**: discount percentage applied to the product price at the time of purchase. :math:`special \in \{[0\%,15\%[, [15\%,30\%[, [30\%,50\%[, [50\%,100\%[\}`
     
*Consumer profile*

We found that there are 3 consumer profiles see :cite:`walters2003exploring`, :cite:`walters2002measuring`, and :cite:`tanusondjaja2016understanding`. 
The first group is consumers who buy only the products on promotion. 
The second group is consumers who always buy the same products (without considering promotions).
Finally, the third group is consumers who buy products on promotion or not.

Since our real dataset was not enough to complete correctly our project, we increased it.
We described the sets of data simulated in our study, 
and we presented in detail the results of our simulations

*Data increase*

For :math:`store\_id`, we started with an initial store and changed stores based on the proportion of common products between baskets.
If we assumed that the store coordinates are normally distributed :math:`\mathcal{N}(0,\sigma^2)` independently, 
the distance between this store and the consumer home located originally :math:`(0,0)` follows a Rayleigh distribution :cite:`kundu2005generalized` with the :math:`\sigma` parameter.
Finally, we increased the `special` feature. This variable is based on the composition of the baskets, choosing a special random proportional to the Boltzmann distribution.
We observed that our baskets generated follow the same distribution that original basket in term of the basket size 
(see Figure :ref:`orderfrequency`).

.. figure:: figures/order_frequency.png
   :align: center
     
   Basket size distribution. :label:`orderfrequency`

Models
------

In this section, we described the workflow (see Figure :ref:`workflow`) and models we used.

*Long short-term memory (LSTM) network*

The LSTM :cite:`hochreiter1997long` is a recurrent neural network (RNN) that has an input, hidden (memory block), and an output layer. 
The memory block contains 3 gate units namely the input, forget, 
and output with a self-recurrent connection neuron :cite:`hochreiter1997long`.

- Input gate: learns what information is to be stored in the memory block.
- Forget gate: learns how much information to be retained or forgotten from the memory block.
- Output gate: learns when the stored information can be used.

Fig. :ref:`lstm` illustrates the proposed architecture and summarizes the detail involved in the structure. 

A combined RNN and CNN trained to predict the probability that a user will order a product at each timestep. 
The RNN is a single-layer LSTM and the CNN is a 6-layer causal CNN with dilated convolutions.
The last layer is a fully-connected layer which makes the classification.
The CNN was used as a feature extractor and the LSTM network as a sequential learning.

.. figure:: figures/lstm.png
   :align: center 
  
   This figure shows circuit using generalized commercial traveller algorithm. the improvement over the course of this study in the DESI 
   spectral extraction throughput. :label:`lstm`

*Gradient Boosted Tree (GBT) network*

GBT :cite:`friedman2002stochastic` is an iterative algorithm that combines simple parameterized functions with “poor” performance 
(high prediction error) to produce a highly accurate prediction rule. GBT utilizes an ensemble of weak
learners to boost performance; this makes it a good candidate model for predicting credit card fraud. 
It requires little data preprocessing and tuning of parameters while yielding interpretable results, 
with the help of partial dependency plots and other investigative tools. 
Further, GBT can model complex interactions in a simple fashion and be used in both classification and 
regression with a variety of response distributions including Gaussian, Bernoulli, Poisson, and Laplace. 
Finally, missing values in the collected data can be easily managed.

The data is divided into 2 groups (training and validation) which comprise 90% and 10% of the data respectively.
The final model has two neuron networks and a GBT classifier.
Once trained, it was used to predict in real time what will be the consumer's basket, based on the history of purchases and current promotions in neighboring stores.
Based on the validation loss function, we eliminated the LSTM Rays and LSTM model size (see Figure :ref:`lstm`).

*First level model (feature extraction)*

Our goal is to find a diverse set of representations using neural networks (see Table 1). 
Table 1 summarizes top-level models used by our algorithm and we described each type of model used for each representation (e.g. Products, Category, Size of basket, Products and Users).
We estimated the probability of the :math:`product_i` to be include to 
the next basket :math:`order_{t+1}` with :math:`orders_{t-h}`, 
with :math:`t` represents the actual time, 
:math:`t+1` represents the next time,
and :math:`t-h` represents all previous time (i.e. historical time).
We decomposed the matrix {user,product} by two matrices one corresponding to user and the other to product.
We predicted the probability to have the :math:`product_i` on the next :math:`order_{t+1}` 
knowing the historical purchases of this user. We used one LSTM with 300 neurons.
We also predicted the probability that the :math:`product_i` is include for which category. 
Finally, we estimated the size of the next order minimizing root mean square error (RMSE).

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

*Latent representations of entities (embeddings)*

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

*Second level model: Composition of baskets*

The final basket is chosen according to the final reorganization probabilities, choosing the subset of products with the expected maximum :math:`F_1` score (:cite:`lipton2014optimal` and :cite:`nan2012optimizing`).
This score is frequently used especially when the relevant elements are scarce.

.. math::
   
   \max_\mathcal{P} \mathbb{E}_{p'\in \mathcal{P}}[F_1(\mathcal{P})]=\max_\mathcal{P}\mathbb{E}_{p'\in \mathcal{P}}\bigg[\frac{2\sum_{i\in \mathcal{P}}\text{TP}(i)}{\sum_{i\in \mathcal{P}}(2\text{VP}(i)+\text{FN}(i)+\text{FP}(i))}\bigg],

where True Positive :math:`(TP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=1]`, False Negative :math:`(FN)=\mathbb{I}[\lfloor p(i)\rceil=0]\mathbb{I}[R_i=1]`, False Positive :math:`(FP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=0]` and :math:`R_i=1`if the product :math:`i` was bought in the basket :math:`p'\in \mathcal{P}`, else :math:`0`.\\
We used :math:`\mathbb{E}_{X}[F_1(Y)]=\sum_{x\in X}F_1(Y=y|x)P(X=x)`

.. figure:: figures/workflow.png
   :align: center
   :scale: 25%
   
   Model used in the classification. :label:`workflow`

*Results*

We present the obtained results using proposed method in this section. 
As well as the metrics (see Equations 1-6) that are used to evaluate the performance of methods.

*Statistic score*
The *accuracy* of a test is its capability to recognize the classes properly. 
To evaluate the accuracy of our model, we should define the percentage 
of true positive and true negative in all estimated cases, 
i.e. the sum of true positive, true negative, false positive, and false negative.
Statistically, this can be identified as follow:

.. math::
   :label: e:matrix
   
   Accuracy = \frac{(TP+TN)}{(TP+TN+FP+FN)}

where:

- *TP* is True Positive, i.e. the number of positively labeled data, which have been classified as "True", correct class,
- *FP* is False Positive, i.e. the number of negatively labeled data, which falsely have been classified as "Positive",
- *TN* is True Negative, i.e. the number of negatively labeled data, which have been classified as "Negative", correct class, and 
- *FN* is False Negative, i.e.  the number of positively labeled data, which falsely have been classified as "Negative".

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

Finally, we evaluated our model by *FP Rate* which corresponds to the ratio between FP and sum of TN and FP.

.. math::
   :label: e:matrix
   
   FP Rate = FPR = \frac{FP}{(TN+FP)} 

Python Script
-------------

The final reorder probabilities are a weighted average of the outputs from the second-level models. The final basket is chosen by using these probabilities and choosing the product subset with maximum expected F1-score.
The select_products function in Python script is the following:

.. code-block:: python
    :linenos:
    
    from multiprocessing import Pool, cpu_count

    import numpy as np
    import pandas as pd

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
We can see cluster of Pasta sauce with Pasta group.

.. figure:: figures/product_pca.png
   :align: center
   
   Embeddings of 20 random products projected in 2 dimensions. :label:`productpca`

:math:`F_1` in Figure :ref:`violon` (a) shows that the profiles of all promotions are similar. 
In the perspective of this work, it will be interesting to include weight base on statistic value. 
In Statistic Canada - 2017, only 5% of all promotions are more than 50% promoted, 95% of all promotions are less than 50%. 
Weightings are needed to give our model more robust. 
Figure :ref:`violon` (a) indicates that all shops follow the same profiles in our model. 

.. figure:: figures/violon.png
   :align: center
   :scale: 20%
   :figclass: wt
   
   Distribution of :math:`F_1` measures against stores (a) and rebates (b). :label:`violon`

Figure :ref:`productsF1` and Table 3 indicates :math:`F_1` to all products. 
Some products are easy to predict with value of :math:`F_1` >0 and 
some products are so hard to predict with value of :math:`F_1` <0. 
For the first group, they are products includes on restriction regime 
such as diet cranberry fruit juice, purified water, and total 0% blueberry acai greek yogurt.

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
     
   Distribution of :math:`F_1` measures relative to products, around average. :label:`productsF1`
	
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

We evaluated our model with the statistics score given in Section 'Statistic score'.
We obtained Precision value equal to  0.779, Recall equal to 0.967, and F1-score = 0.871.
	
Conclusions and Future Work
---------------------------

We analyzed grocery shopping data generated by the consumers of the site MyGroceryTour.ca.
We developed a new machine learning model to predict which grocery products the consumer will
buy and in which store(s) of the region he/she will do grocery shopping.
We created an intelligent shopping list based on the shopping history of consumer and his/her
known preferences.
The originality of our approach, compared to the existing algorithms, is that in addition to the
purchase history we also consider promotions, possible purchases in different stores and the
distance between these stores and the home of consumer.

We have modeled the habits of the site's consumers
CircuitPromo.ca with the help of deep neural networks.
We used two types of neural networks during
Learning: Recurrent Neural Networks (RNN) and Networks
forward-propagating neurons (Feedforward NN).
The value of the :math:`F_1` statistic that represents the quality of our model
is 0.22. The constant influx of new data on *CircuitPromo*
improved the model over time.
The originality of our approach, compared to existing algorithms,
is that in addition to the purchase history we also consider the
promotions, possible purchases in different stores and distance
between these stores and the consumer's home.

Acknowledgments
---------------

The authors thank PyCon Canada for their valuable comments on this project. This work used
resources of the Calcul Canada. This work was supported by Natural Sciences 
and Engineering Research Council of Canada and Fonds de Recherche sur la Nature et Technologies of Quebec. 
The funds provided by these funding institutions have been used. We would like to thanks SciPy conference 
and anonymous reviewers for their valuable comments on this manuscript.

Abbreviations
-------------

- ML - Machine Learning
- LSTM - Long short-term memory
- CNN - Convolutional Neural Network
- GBT  - Gradient Tree Boosting
- PCA - Principal Component Analysis
- RMSE - Root Mean Square Error
- RNN - recurrent neuron networks



