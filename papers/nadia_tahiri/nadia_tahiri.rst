:author: Nadia Tahiri
:email: tahiri.nadia@courrier.uqam.ca
:institution: University of Quebec at Montreal
:corresponding:

:bibliography: scipy

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
   Second, we will develop a smart shopping list template to provide consumers with a 
   personalized weekly shopping list based on their historical shopping history and 
   known preferences. As explanatory variables, we will use available grocery shopping histories, 
   store promotion information for the given region, as well as product price statistics.

.. class:: keywords

   machine learning, prediction, pandas, numpy, scipy, sklearn, tensorflow
   

Introduction
------------

A typical grocery retailer offers consumers thousands of promotions every week 
to attract more consumers and thus improve their economic performance (Tanusondjaja et al., 2016). 
Studies by Walters and Jamil (2002, 2003) found that about 39% of all items purchased 
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
shopping lists interactively (Newcomb et al., 2003, Fazliu 2017).


Methods
-------

By using the grocery shopping database in Canada MyGroceryTour.ca (see Figure :ref:`circuitpromo`), 
we will first look for the partitioning of consumers into classes that will group 
them efficiently based on purchases made. 
This classification will then be used in the prediction stage. 
Since real consumer data contains thousands of individual articles, 
we focus primarily on categories of articles. 
The principal component analysis (linear and polynomial PCA) will first be performed to better visualize the initial data 
and to choose the number of main components to use when partitioning consumers into classes. 
The application of efficient partitioning methods, such as K-means and X-means, 
will make it possible to determine the number of classes of consumers, 
as well as their distribution by class.

Secondly, we will develop a statistical model to predict which products previously purchased will be 
in the next order of the consumer. By using explanatory variables, such as available grocery shopping histories, 
information on current promotions in stores in the given region, and commodity price statistics, 
we will develop a model of machine learning and able to:

(i) Predict which groceries the consumer will want to buy again 
or will try to buy for the first time, and in which store(s) in the area he will shop;
(ii) Create a smart shopping list by providing the consumer 
with a weekly shopping list customized based on their purchase history and known preferences. 
This list will also include recommendations regarding the optimal quantity of each product suggested and the store(s) 
where these products are to be purchased. We will also calculate the consumer's optimal weekly commute 
using the generalized commercial traveller algorithm (see Figure :ref:`circuit`).

.. figure:: figures/trois_magasins.png
   
	 CircuitPromo.ca website for the postal code H2Y 1C6 in Montreal. :label:`circuitpromo` 

.. figure:: figures/mygrocerytour_circuit.png
	 
	 Screenshot of CircuitPromo.ca website with an optimal shopping journey. :label:`circuit`

:math:`F_1` statistics maximization algorithm (Ye et al., 2012), 
based on dynamic programming, will be used to meet the objective (i), 
which will be of major interest to retailers and distributors. 
A deep learning method (Goodfellow et al., 2016), based on recurrent neuron networks (RNN) 
and convolutional neuron network (CNN), and implemented in Google's TensorFlow tool (Dean et al., 2015), 
will be used to meet objective (ii), which will be of major interest to consumers.

The problem will be reformulated as a binary prediction task: given a consumer, 
the history of his previous purchases and a product with his price history, 
to predict whether or not the given product will be included in the grocery list of the consumer. 
Our approach will be to adapt a variety of generative models to existing data, i.e., 
first-level models, and to use the internal representations of 
these models as features of the second-level models. 
Recurrent neural networks and convolutional neural networks will be used at the first learning level 
and forward propagation neural networks (Feedforward NN) 
will be used at the second level of learning.

Depending on the user :math:`u` and the user purchase history
(shop :math:`_{t-h:t}`, :math:`h>0`), we predict the probability that a product :math:`i` is included 
in the next shop :math:`_{t+1}` of :math:`u`


Dataset
-------

We used the data from CircuitPromo.ca as basic data. 

*Features*

The features is described as follow:

- **user\_id**: user number. :math:`user\_id \in \underbrace{\{1 \cdots 374\}}_{\text{reals}} \cup \underbrace{\{375 \cdots 1374\}}_{\text{generated}}`
- **order\_id**: unique number of the basket. :math:`order\_id \in \mathbb{Z}`
- **store\_id**: unique number of the store. :math:`store\_id \in \{1 \cdots 10\}` 
- **distance**: distance to the store. :math:`distance \in \mathbb{R}^+`
- **product\_id**: unique number of the product.
- **category\_id**: unique category number for a product. :math:`aisle\_id \in \{1 \cdots 24\}`  
- **reorder**: 1 if this product has been ordered by this user in the past, 0 else. :math:`reorders \in \{0,1\}`
- **special**: discount percentage applied to the product price at the time of purchase. :math:`special \in \{[0\%,15\%[, [15\%,30\%[, [30\%,50\%[, [50\%,100\%[\}`
	 
*Consumer profile*

We found that there are 3 consumer profiles ({walters2003exploring, walters2002measuring, tanusondjaja2016understanding}). 
The first group is consumer who buy only the products on promotion. 
The second group is consumer who always buy the same products (without considering promotions).
Finally, the third group is consumer who buy products as well on promotion or not.

*Data increase*

We considered that our dataset is not enougth, and we decided to increase them by following statitic rules. 
For :math:`store_id`, we started with an initial store and changed stores based on the proportion of common products between baskets.
The strategy, we used for computed :math:`distance` if we assumed that the store coordinates are normally distributed :math:`\mathcal{N}(0,\sigma^2)` independently, 
the distance between this store and the consumer home located originally :math:`(0,0)` follows a Rayleigh distribution with the :math:`\sigma` parameter.
Finally, we increased `special` feature. This variable is based on the composition of the baskets, choosing a special random proportional to the Boltzmann distribution.


Models
------

In this section, we described the workflow and models we used.
The data is divided into 2 groups (training and validation) which comprise 90% and 10% of the data respectively.
The final model has two neuron networks and a Gradient Boosted Tree (GBT) classifier ({friedman2002stochastic}).
Once trained, it can be used to predict in real time what will be the consumer's basket, based on the history of purchases and current promotions in neighborhood stores.
Based on the validation loss function, we eliminated the LSTM Rays and LSTM model size.

*First level model (feature extraction)*
Our goal is to find a diverse set of representations using neural networks (see Table 1). 
Table 1 summarizes top-level models used by our algorithm and we described each type of model used for each representation (e.g. Products, Category, Size of basket, Products and Users).

.. raw:: latex

   \begin{table}

     \begin{longtable}{lcc}
     \hline
     \textbf{Representation} & \textbf{Description} & \textbf{Type}\tabularnewline
     \hline
     \textcolor{blue}{Products} & \textcolor{blue}{\begin{tabular}{@{}c@{}} Model P$(\text{product}_{i}\in \text{order}_{t+1})$\\ with orders$_{t-h,t}$, $h>0$.\end{tabular}}& \textcolor{blue}{\begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}} \\
     \hline
     Categories & Predicts P$(\exists i:\text{product}_{i,t+1} \in \text{category}_r)$. & \begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}\\
     \hline
     Size & Predict the size of the order$_{t+1}$. & \begin{tabular}{@{}c@{}}LSTM\\ (300 neurons)\end{tabular}\\
     \hline
     \textcolor{blue}{\begin{tabular}{@{}c@{}}Users \\ Products \end{tabular}} & \textcolor{blue}{Decomposed $V_{(u \times p)}=W_{(u \times d)}	H^T_{(p \times d)}$} & \textcolor{blue}{\begin{tabular}{@{}c@{}}Dense\\ (50 neurons)\end{tabular}}\\
     \hline
     \end{longtable}

     \caption{Top-level models used.}
		 \label{tab:model1}

   \end{table}


*Latent representations of entities (embeddings)*

For each :math:`a \in \mathcal{A}`, an embedding :math:`T:\mathcal{A} \rightarrow \mathbb{R}^{d}` returns a vector :math:`d`-dimensionel.
If :math:`\mathcal{A} \subset \mathbb{Z}`, :math:`T` is a matrix :math:`|\mathcal{A}|\times d` learned by backpropagation.

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

The final basket is chosen according to the final reorganization probabilities, choosing the subset of products with the expected maximum :math:`F_1` score ({lipton2014optimal} and {nan2012optimizing}).
This score is frequently used especially when the relevant elements are scarce.

.. math::
	\max_\mathcal{P} \mathbb{E}_{p'\in \mathcal{P}}[F_1(\mathcal{P})]=\max_\mathcal{P}\mathbb{E}_{p'\in \mathcal{P}}\bigg[\frac{2\sum_{i\in \mathcal{P}}\text{TP}(i)}{\sum_{i\in \mathcal{P}}(2\text{VP}(i)+\text{FN}(i)+\text{FP}(i))}\bigg],

where True Positive :math:`(TP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=1]`, False Negative :math:`(FN)=\mathbb{I}[\lfloor p(i)\rceil=0]\mathbb{I}[R_i=1]`, False Positive :math:`(FP)=\mathbb{I}[\lfloor p(i)\rceil=1]\mathbb{I}[R_i=0]` and :math:`R_i=1`if the product :math:`i` was bought in the basket :math:`p'\in \mathcal{P}`, else :math:`0`.\\
We used :math:`\mathbb{E}_{X}[F_1(Y)]=\sum_{x\in X}F_1(Y=y|x)P(X=x)`

.. figure:: figures/workflow.png
	 
	 Model used in the classification. :label:`workflow`

.. figure:: figures/order_frequency.png
	 
	 Basket size distribution. :label:`orderfrequency`

.. figure:: figures/product_pca.png
	 
	 Embeddings of 20 random products projected in 2 dimensions. :label:`productpca`


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
   
.. figure:: figures/lstm.png
   
   This figure shows circuit using generalized commercial traveller algorithm. the improvement over the course of this study in the DESI 
   spectral extraction throughput. :label:`lstm`

.. figure:: figures/products_F1.png
	 
	 Distribution of :math:`F_1` measures relative to products, around average. :label:`productsF1`
   
.. figure:: figures/violon.png
   :align: center
   :scale: 15%
   
   Distribution of :math:`F_1` measures against stores (a) and rebates (b). :label:`violon`

.. raw:: latex
	
	\begin{table}

		\begin{longtable}{lc}
		\hline
                                      \textbf{Product} &        \textbf{F} \\
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
	\end{table}   

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


Acknowledgments
---------------
The authors thank PyCon Canada for their valuable comments on this project. This work used
resources of the Calcul Canada. This work was supported by Natural Sciences 
and Engineering Research Council of Canada and Fonds de Recherche sur la Nature et Technologies of Quebec. 
The funds provided by these funding institutions have been used. 
We thank also reviewers and SciPy.


References
----------
