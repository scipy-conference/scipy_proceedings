:author: Ankur Ankan
:email: ankurankan@gmail.com

:author: Abinash Panda
:email: 

--------------------------------------------------
pgmpy: Probabilistic Graphical Models using Python
--------------------------------------------------

.. class:: abstract

   Probabilistic Graphical Models is a probabilistic technique in machine learning which 
   allows us to represent the joint distribution over the random variables in a compactly 
   and do inference over the model keeping the computational costs low.

   pgmpy is a Python library for working with Graphical Models.

.. class:: keywords

   Graphical Models, Bayesian Networks, Markov Networks

Introduction
------------

pgmpy is a Python library for working with Probabilistic Graphical Models. It gives the 
user complete flexibility in building models which other packages do not provide. 

Getting Source Code and Installing
----------------------------------
pgmpy is hosted on github. So, it can be simply cloned and installed::

    git clone https://github.com/pgmpy/pgmpy
    cd pgmpy
    [sudo] python3 setup.py install

Dependencies: pgmpy runs only on python3 and needs networkx, numpy.

Creating Bayesian Models using pgmpy
------------------------------------
Short Intro to Bayesian Models

.. figure:: figure1.png
   :align: center
   :figclass: w
   
   A Bayesian Network :label:'student'

Let's take the example of a simple Bayesian Network as shown in the figure :ref:'student'.
The base structure representing the connections between the different variables in the model
is a Directed Acyclic Graph. In pgmpy we can define such a network structure as:

.. code-block:: python

   from pgmpy.models import BayesianNetwork
   student_model = BayesianNetwork([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

We also see that the network is parameterized using Conditional Probability Tables (CPT).
Let's take a simple example of a table to show how to represent CPTs using pgmpy.

.. table:: Conditional Probability Table. :label:'CPT'
   +-------------------+--------------------------+---------------------+
   | Intelligence (I)  |           i0             |          i1         |
   +-------------------+------------+-------------+-----------+---------+
   | Difficulty (D)    |   d0       |    d1       |   d0      |   d1    |
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
   grade_cpt = TabularCPD(variable='G', 
			  variable_card=3, 
                          values=[[0.3, 0.05, 0.9, 0.5],
                                  [0.4, 0.25, 0.08, 0.3],
                                  [0.3, 0.7, 0.02, 0.2]],
                          evidence=['I', 'D'],
                          evidence_card=[2, 2])

We can similarly define CPTs for all the variables in the model. For associating the CPTs to
a model structure we can do:

.. code-block:: python

   student_model.add_cpds(grade_cpt, diff_cpt, intel_cpt, letter_cpt, sat_cpt)

In this way we have created a complete Bayesian Network.

Creating Markov Models in pgmpy
-------------------------------
Short Intro to Markov Models.

Again taking an example of simple Markov model. It's all the same except the Markov
models are parameterized using Factors instead of CPTs. So, we can define a Markov 
Model as:

.. code-block:: python

   from pgmpy.models import MarkovModel
   from pgmpy.factors import Factor
   model = MarkovModel([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
   factor_a_b = Factor(['A', 'B'], [2, 2], [100, 5, 5, 100])
   ...
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

Refernces
---------
