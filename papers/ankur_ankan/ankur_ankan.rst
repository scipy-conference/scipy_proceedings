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
   
   pgmpy is a python library for working with graphical models. It allows the 
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

Probabilistic Graphical Models is a technique of representing Joint
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

The general workflow for creating any model in pgmpy is to first define the 
network structure and then add the parameters to it.

A Bayesian Netowrk is parameterized using Conditional Probability Distributions.


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

   from pgmpy.models import BayesianModel
   from pgmpy.factors import TabularCPD
   student_model = BayesianNetwork([('D', 'G'), ('I', 'G'), ('G', 'L'),
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
   student_model.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, letter_cpd,
                          sat_cpd)

Various methods are available in pgmpy for checking the D-separation and independencies in the network.

Creating Markov Models in pgmpy
-------------------------------

 Should we go into the details of Markov Network here?
Short Intro to Markov Models.

Again taking an example of simple Markov model. It's all the same except the Markov models are parameterized using Factors instead of CPTs. So, we can define a Markov Model as:

.. code-block:: python

   from pgmpy.models import MarkovModel
   from pgmpy.factors import Factor
   model = MarkovModel([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
   factor_a_b = Factor(['A', 'B'], [2, 2], [100, 5, 5, 100])
   factor_b_c = Factor(['B', 'C'], [2, 2], [])
   factor_c_d = Factor(['C', 'D'], [2, 2], [])
   factor_d_a = Factor(['D', 'A'], [2, 2], [])
   model.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)

Doing Inference over models
---------------------------
pgmpy support various Exact and Approximate inference algorithms. The general API to run 
inference over models is to first create an inference object by passing the model to the
inference algorithm class. Then we can simply call the query method of the inference object
to query for the probability of some state of some variable given observations of other 
variables. Let's take an example of doing Variable elimination on the student model above:

.. code-block:: python

   from pgmpy.inference import VariableElimination
   student_infer = VariableElimination(student_model)
   student_infer.query('G')
   
   student_infer.query('G', evidence=[('I', 1), ('D', 0)])

   student_infer.map_query('G')

   student_infer.map_query('G', evidence=[('I', 1), ('D', 0)])

Fit and Predict Methods
-----------------------
While working with data it's difficult to compute the distributions by hand and is too
much work to create each of the factor/CPT by hand. So, pgmpy gives the option of fit 
and predict:

.. code-block:: python

   import numpy as np
   # Generate some random data
   student_model.fit(data)
   student_model.get_cpds()
   student_model.predict()


Conclusion
----------

References
----------
[pgmpy] pgmpy github page https://github.com/pgmpy/pgmpy
