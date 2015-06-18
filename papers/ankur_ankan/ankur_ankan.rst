:author: Ankur Ankan
:email: ankurankan@gmail.com

:author: Abinash Panda
:email: mailme.abinashpanda@gmail.com

--------------------------------------------------
pgmpy: Probabilistic Graphical Models using Python
--------------------------------------------------

.. class:: abstract

   Probabilistic Graphical Models (PGM) is a technique of compactly representing   
   a joint distribution by exploiting dependencies between the random variables.     
   It also allows us to do inference on joint distributions in a computationally 
   cheaper way than the traditional methods. PGMs are widely used in the field 
   of speech recognition, information extraction, image segmentation, modelling 
   gene regulatory networks. 
   
   pgmpy [pgmpy] is a python library for working with graphical models. It allows the 
   user to create their own models and answer inference or map queries over 
   them. pgmpy has implementation of many inference algorithms like 
   VariableElimination, Belief Propagation etc.

   This paper first gives a short introduction to PGMs and various other python
   packages available for working with PGMs. Then we discuss about creating and
   doing inference over Bayesian Networks and Markov Networks using pgmpy.

.. class:: keywords

   Graphical Models, Bayesian Networks, Markov Networks, Variable Elimination

Introduction
------------

Probabilistic Graphical Model is a technique of representing Joint
Distributions over random variables in a compact way by exploiting the 
dependencies between the random variables. PGMs use a network structure and some 
parameters to represent the joint distribution. The network structure is used to
encode the relationships between the random variables. There are mainly two 
types of Graphical Models: Bayesian Networks and Markov Networks.


.. figure:: figure1.png
   :align: center
   :figclass: w

   A simple Bayesian Network. :label:'bayesian'

Bayesian Network: A Bayesian Network consists of a directed graph and a 
conditional distribution associated with each of the random variables. A 
Bayesian network is used mostly when there is a causal relationship between the
random variables. An example of a Bayesian Network representing a student taking 
some course is shown in Fig :ref:'bayesian'.

.. figure:: figure2.png
   :align: center
   :figclass: w

   A simple Markov Model :label:'markov'

Markov Network: A Markov Network consists of an undirected graph and a few 
factors are associated with it. Unlike in the case of Bayesian network, a factor
does not represent the probabilities of variables in the network. Rather it represents 
how much is a state of a random variable likely to agree to the state 
of the other random variable. An example of 4 friends A, B, C, D agreeing to
some concept is shown in Fig :ref:'markov'.

There are numerous packages available in Python for working with graphical 
models but most of them are focused towards some special case and doesn't 
give complete freedom to the user. Give some examples of some packages.
pgmpy tries to be a complete package for working with graphical models and gives 
the user full control on designing the model. Also pgmpy provides easy extensibility and 
the user can write his own inference algorithms or elimination orders without actually 
looking at the source code.
 
Getting Source Code and Installing
----------------------------------
pgmpy is released under MIT Licence and is hosted on github. We can simply clone the repository and install it::

    git clone https://github.com/pgmpy/pgmpy
    cd pgmpy
    [sudo] python3 setup.py install

Dependencies: pgmpy runs only on python3 and is dependent on networkx, numpy and scipy.

Creating Bayesian Models using pgmpy
------------------------------------

A Bayesian Network consists of a directed graph which connect the random variables based on
the relation between them. It is parameterized using Conditional Probability Distributions(CPD).
Each random variable in a Bayesian Network has a CPD associated with it. If the random varible 
has parents in the network then the CPD represents :math:`P(var| Par_var)` i.e. the probability
of that variable given its parents. In the case when the random variable has no parents it 
simply represents :math:`P(var)` i.e. the probability of that variable.

We can take the example of the CPD for the random variable grade in the student model :ref:'bayesian'.
A possible CPD for the grade variable is shown in the table :ref:'CPT'.

.. table:: Conditional Probability Table. :label:'CPT'

   +-------------------+------------+-------------+-----------+---------+
   | Intelligence (I)  |    i0      |     i0      |   i1      |   i1    |
   +-------------------+------------+-------------+-----------+---------+
   | Difficulty (D)    |    d0      |     d1      |   d0      |   d1    |
   +-------------------+------------+-------------+-----------+---------+
   | g0                |    0.3     |    0.05     |   0.9     |   0.5   |
   +-------------------+------------+-------------+-----------+---------+
   | g1                |    0.4     |    0.25     |   0.08    |   0.3   |
   +-------------------+------------+-------------+-----------+---------+
   | g2                |    0.3     |    0.7      |   0.02    |   0.2   |
   +-------------------+------------+-------------+-----------+---------+

We can represent the CPT :ref:'CPT' in pgmpy as follows:

.. code-block:: python

   from pgmpy.factors import TabularCPD
   grade_cpd = TabularCPD(variable='G',
			        variable_card=3,
                          values=[[0.3, 0.05, 0.9, 0.5],
                                  [0.4, 0.25, 0.08, 0.3],
                                  [0.3, 0.7, 0.02, 0.2]],
                          evidence=['I', 'D'],
                          evidence_card=[2, 2])

Now, coming back to defining a model using pgmpy. The general workflow for defining a
model in pgmpy is to first define the network structure and then add the parameters 
to it. We can create the student model :ref:'bayesian' in pgmpy as follows:

.. code-block:: python

   from pgmpy.models import BayesianModel
   from pgmpy.factors import TabularCPD
   student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'),
                                    ('I', 'S')])
   grade_cpd = TabularCPD(variable='G',
			        variable_card=3,
                          values=[[0.3, 0.05, 0.9, 0.5],
                                  [0.4, 0.25, 0.08, 0.3],
                                  [0.3, 0.7, 0.02, 0.2]],
                          evidence=['I', 'D'],
                          evidence_card=[2, 2])
   difficulty_cpd = TabularCPD(variable='D',
                               variable_card=2,
                               values=[[0.6, 0.4]])
   intel_cpd = TabularCPD(variable='I',
                          variable_card=2,
                          values=[[0.7, 0.3]])
   letter_cpd = TabularCPD(variable='L',
                           variable_card=2,
                           values=[[0.1, 0.4, 0.99],
                                   [0.9, 0.6, 0.01]],
                           evidence=['G'],
                           evidence_card=[3])
   sat_cpd = TabularCPD(variable='S',
                        variable_card=2,
                        values=[[0.95, 0.2],
                                [0.05, 0.8]],
                        evidence=['I'],
                        evidence_card=[2])
   student_model.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, 
			  letter_cpd, sat_cpd)

The network structure of a Graphical Model encodes the independence conditions between the 
random variables. pgmpy also has methods to determine the local independencies, D-Separation,
converting to a markov model etc. A few example are shown below:

.. code-block:: python

   student_model.get_cpds()
   [<TabularCPD representing P(G:3 | I:2, D:2) at 0x7f196c0b27b8>,
    <TabularCPD representing P(D:2) at 0x7f196c0b2828>,
    <TabularCPD representing P(I:2) at 0x7f196c0b2908>,
    <TabularCPD representing P(L:2 | G:3) at 0x7f196c0b2978>,
    <TabularCPD representing P(S:2 | I:2) at 0x7f196c0b27f0>]

   student_model.active_trail_nodes('D')
   {'D', 'G', 'L'}

   student_model.local_independencies('G')
   (G _|_ S | D, I)

   student_model.get_independencies()
   (S _|_ I, G, L | D)
   (S _|_ D, I | G)
   (S _|_ D, I, G | L)
   (D _|_ G, L | S)
   (D _|_ I, S | G)
   (D _|_ G, L | I)
   (D _|_ G, I, S | L)
   (G _|_ D, I, L | S)
   (G _|_ I, L, S | D)
   (G _|_ D, L | I)
   (G _|_ D, I, S | L)
   (I _|_ G, L | S)
   (I _|_ G, S, L | D)
   (I _|_ D, S | G)
   (I _|_ D, G, S | L)
   (L _|_ D, G, I | S)
   (L _|_ G, I, S | D)
   (L _|_ D, G | I)

   student_model.to_markov_model()
   <pgmpy.models.MarkovModel.MarkovModel at 0x7f196c0b2470>

Creating Markov Models in pgmpy
-------------------------------

A Markov Network consists of a undirected graph which connects the random variables according to 
the relation between them. A markov network is parameterized by factors which represent the likelihood
of a state of one variable to agree with some state of other variable. 

We can take the example of a Factor over variables A and B in the network. :ref:'markov'.
A possible Factor over variables A and B is shown in the table :ref:'FactorAB'.

.. table:: Factor over variables A and B. :label:'FactorAB'

   +-----------+-----------+-------------------+
   |  A        |  B        | :math:'phi(A, B)' |
   +-----------+-----------+-------------------+
   |:math:'a^0'|:math:'b^0'| 100               |
   +-----------+-----------+-------------------+
   |:math:'a^0'|:math:'b^1'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'a^1'|:math:'b^0'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'a^1'|:math:'b^1'| 100               |
   +-----------+-----------+-------------------+

We can represent the Factor in pgmpy as follows:

.. code-block:: python

   from pgmpy.factors import Factor
   phi_a_b = Factor(['A', 'B'], [2, 2], [100, 5, 5, 100])

.. table:: Factor over variables B and C. :label:'FactorBC'

   +-----------+-----------+-------------------+
   |  B        |  C        | :math:'phi(B, C)' |
   +-----------+-----------+-------------------+
   |:math:'b^0'|:math:'c^0'| 100               |
   +-----------+-----------+-------------------+
   |:math:'b^0'|:math:'c^1'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'b^1'|:math:'c^0'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'b^1'|:math:'c^1'| 100               |
   +-----------+-----------+-------------------+

.. table:: Factor over variables C and D. :label:'FactorCD'

   +-----------+-----------+-------------------+
   |  C        |  D        | :math:'phi(C, D)' |
   +-----------+-----------+-------------------+
   |:math:'c^0'|:math:'d^0'| 100               |
   +-----------+-----------+-------------------+
   |:math:'c^0'|:math:'d^1'| 5                 |
   +-----------+----------+--------------------+
   |:math:'c^1'|:math:'d^0'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'c^1'|:math:'d^1'| 100               |
   +-----------+-----------+-------------------+

.. table:: Factor over variables D and A. :label:'FactorDA'

   +-----------+-----------+-------------------+
   |  D        |  A        | :math:'phi(D, A)' |
   +-----------+-----------+-------------------+
   |:math:'d^0'|:math:'a^0'| 100               |
   +-----------+-----------+-------------------+
   |:math:'d^0'|:math:'a^1'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'d^1'|:math:'a^0'| 5                 |
   +-----------+-----------+-------------------+
   |:math:'d^1'|:math:'a^1'| 100               |
   +-----------+-----------+-------------------+

Assuming some other possible factors as in table 2, 3 and 4, we can define the complete
markov model as:

.. code-block:: python

   from pgmpy.models import MarkovModel
   from pgmpy.factors import Factor
   model = MarkovModel([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
   factor_a_b = Factor(['A', 'B'], [2, 2], [100, 5, 5, 100])
   factor_b_c = Factor(['B', 'C'], [2, 2], [100, 3, 2, 4])
   factor_c_d = Factor(['C', 'D'], [2, 2], [3, 5, 1, 6])
   factor_d_a = Factor(['D', 'A'], [2, 2], [6, 2, 56, 2])
   model.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)

Similar to Bayesian Networks, pgmpy also has the feature for computing independencies,
converting to Bayesian Network etc in the case of Markov Networks.

.. code-block:: python

   model.get_local_independencies()
   (D _|_ B | C, A)
   (C _|_ A | D, B)
   (A _|_ C | D, B)
   (B _|_ D | C, A)
   model.to_bayesian_model()
   <pgmpy.models.BayesianModel.BayesianModel at 0x7f196c084320>
   model.get_partition_function()
   10000

Doing Inference over models
---------------------------
pgmpy support various Exact and Approximate inference algorithms. The general API to run 
inference over models is to first create an inference object by passing the model to the
inference algorithm class. Then we can call the query method to find the probability of 
some variable given some evidence. Or else if we want to know the state of the variable 
having maximum probability we can call map_query method. Let's take an example of doing 
Variable elimination on the student model: :ref:'bayesian'

.. code-block:: python

   from pgmpy.inference import VariableElimination
   student_infer = VariableElimination(student_model)
   prob_G = student_infer.query('G')
   print(prob_G['G'])
   G       phi(G)
   G_0     0.4470
   G_1     0.2714
   G_2     0.2816

   prob_G = student_infer.query('G', evidence=[('I', 1), ('D', 0)])
   print(prob_G['G'])
   G       phi(G)
   G_0     0.0500
   G_1     0.2500
   G_2     0.7000

   student_infer.map_query('G')
   {'G': 0}   
   student_infer.map_query('G', evidence=[('I', 1), ('D', 0)])
   {'G': 2}

Fit and Predict Methods
-----------------------
In a general machine learning task we are given some data from which we want to compute
the parameters of the model. pgmpy simplifies working on these problems by providing 
fit and predict methods in the models. fit method accepts the given data as a pandas 
DataFrame object and learns all the parameters from it. The predict method also 
accepts a pandas DataFrame object and predicts values of all the missing variables using
the model. An example of fit and predict over the student model using some randomly 
generated data:

.. code-block:: python

   from pgmpy.models import BayesianModel
   import pandas as pd
   import numpy as np
   raw_data = np.random.randint(low=0, high=2, size=(1000, 6)) # Considering that each variable have only 2 states
   data = pd.DataFrame(raw_data, columns=['A', 'C', 'D', 'L', 'F', 'P'])
   data_train = data[: int(data.shape[0] * 0.75)]

   student_model = BayesianModel([('F', 'P'), ('A', 'P'), ('L', 'P'), ('C', 'L'), ('D', 'L')])
   student_model.fit(data_train)
   student_model.get_cpds()
   [<TabularCPD representing P(C:2) at 0x7f195ee5e400>,
    <TabularCPD representing P(A:2) at 0x7f195ee5e518>,
    <TabularCPD representing P(D:2) at 0x7f195ee5e2b0>,
    <TabularCPD representing P(F:2) at 0x7f195ee5e320>,
    <TabularCPD representing P(P:2 | F:2, A:2, L:2) at 0x7f195ed620f0>,
    <TabularCPD representing P(L:2 | C:2, D:2) at 0x7f195ed62048>]

   test_data = data[0.75 * data.shape[0] : data.shape[0]]
   test_data.drop('P', axis=1, inplace=True)
   student_model.predict(test_data)
     P
750  0
751  0
752  1
753  0
..  ..
996  0
997  0
998  0
999  0

[250 rows x 1 columns]

Extending pgmpy
---------------
One of the main features of pgmpy is its extensibility. pgmpy has been built in a way so that 
new algorithms can be directly written without needing to get familiar with the code base. 
For writing any new inference algorithm we can simply inherit the Inference class. 
Inheriting this base inference class exposes three variables to the class: self.variables,
self.cardinalities and self.factors and using these variables we can write our own 
inference algorithm. An example is shown:

.. code-block:: python

   from pgmpy.inference import Inference
   class MyNewInferenceAlgo(Inference):
       def print_variables(self):
           print(self.variables)
           print(self.cardinalities)
           print(self.factors)

   infer = MyNewInferenceAlgo(student_model)
   ['S', 'D', 'G', 'I', 'L']
   {'D': 2, 'G': 3, 'I': 2, 'S': 2, 'L': 2}
   defaultdict(<class 'list'>, {'D': [<Factor representing phi(D:2) at 0x7f195ed61c18>, <Factor representing phi(G:3, D:2, I:2) at 0x7f195ed61cf8>], 'I': [<Factor representing phi(S:2, I:2) at 0x7f195ed61a58>, <Factor representing phi(G:3, D:2, I:2) at 0x7f195ed61cf8>, <Factor representing phi(I:2) at 0x7f195ed61e10>], 'G': [<Factor representing phi(G:3, D:2, I:2) at 0x7f195ed61cf8>, <Factor representing phi(L:2, G:3) at 0x7f195ed61e48>], 'S': [<Factor representing phi(S:2, I:2) at 0x7f195ed61a58>], 'L': [<Factor representing phi(L:2, G:3) at 0x7f195ed61e48>]})

Similarly for adding any new variable elimination order algorithm we can simply inherit from
EliminationOrder and define a method named get_order in it. Below is an example for returning 
an elimination order in which the variables are sorted alphabetically.

.. code-block:: python

   from pgmpy.inference import EliminationOrder
   class MyEliminationAlgo(EliminationOrder):
       def get_order(self, variables):
           return sorted(variables)

   # finish this

Conclusion
----------
pgmpy is being currently rapidly developing and soon sampling algorithms, file format support will be added.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.
.. [pgmpy] pgmpy github page https://github.com/pgmpy/pgmpy
.. [student] Koller, D.; Friedman, N. (2009). Probabilistic Graphical Models. Massachusetts: MIT Press. p. 1208. ISBN 0-262-01319-3.
.. [markov] Koller, D.; Friedman, N. (2009). Probabilistic Graphical Models. Massachusetts: MIT Press. p. 1208. ISBN 0-262-01319-3.
