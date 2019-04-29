:author: Scott Sievert
:email: scott@stsievert.com
:institution: University of Wisconsin–Madison

:author: Tom Augspurger
:email: taugspurger@anaconda.com
:institution: Anaconda, Inc.

:author: Matthew Rocklin
:email: mrocklin@gmail.com
:institution: NVIDIA

.. :bibliography: refs

-------------------------------------------
Better and faster model selection with Dask
-------------------------------------------

.. class:: abstract

   TODO

.. class:: keywords

   machine learning, model selection, distributed, dask

Introduction
============

.. Introduction
   Hyperparameters are input to machine learning workflow
   They require tuning
   For modern classifier: Difficult problem (continuous variables, deep learning, etc)

.. Problem statement
   Question 1: how good?
   Question 2: how fast?
   Question 3: inputs to selection?

.. Contributions
   A high performing algorithm, Hyperband, is implemented in Dask-ML
   * Has mathematical justification; specifically it performs well with high probability
   * Amendable to parallelism
   * Easy to use
   These provide progress towards all 3 questions above.
   We will walk through each of these sections.

.. Theoretical groundings
   Thm from Hyperband paper
   Depends on successive halving
   Runs many brackets in parallel
   Intuition: use bandit framework

.. Amendable to parallelism
   Two levels of parallelism
   Intuition: requires sweeping over how easy to use

.. Ease of use
   Requires one parameter
   Direct result of killing off models early and sweeping over parameter
   Here's how to specify

.. Simulations
   Walk through blog post example

.. Conclusion & Future work
   Conclusion
   Implement for black-box models
   Work on removing deepcopy
   Validate works well with large memory GPU models


There are three inputs to a machine learning pipeline: data, an untrained model
and "hyper-parameters", or parameters that are required for training and
influence the performance of the model. At the most basic level, these
hyper-parameters help adapt the model to the data. For example, the model might
have one hyper-parameter that adapts the prediction to the amount of noise
present in the data (e.g., the regularization parameter in ridge regression
:cite:`TODO` or LASSO :cite:`TODO`).

Model performance strongly depends on the hyper-parameters provided.

.. cite "those hyper-parameters really matter"

These hyper-parameters are typically assumed to be given. However, this often
requires solving the problem.

.. cite step size
.. cite lagrangian. Gubner's textbook (or notes).

In practice, this means that hyper-parameters have to be searched or tuned to
find the value that provides the highest performance.

Model selection has become more complicated as data has grown, especially with
the growth of deep learning.

.. cite automl, bayesian, etc

Contributions
=============

Model selection is required if high performance is desired. In practice, it's a
burden for machine learning researchers and practitioners. At best, model
selection wouldn't need to be performed. However, in practice it's required.
Ideally, a model selection algorithm should return high performing models.
High performing models make quality predictions on examples never observed
before, typically evaluated by scoring a validation dataset.

Returning the high performing model quickly will mean the user (e.g., a data
scientists) will not be blocked by model selection. They can use the tuned
model more quickly and move onto other more interesting tasks. Personal
experience suggests that model selection is time-consuming and routine but
required for quality performance.

Ideally, model selection algorithms return high performing models quickly and
are simple to use. This work

* implements a particular model selection algorithm, Hyperband, in Dask-ML.
  This algorithm returns models with a high validation score and is moderately
  amendable to parallelism.
* makes some simple modifications to increase amenability to parallelism
* provides an simple method to determine the parameters to Hyperband, which
  only requires knowing how many examples the model should observe and a rough
  estimate on how many parameters to sample

This algorithm treats each computation as a scarce resource. If computation is
not a scarce resource, there is no benefit from this algorithm. At it's core,
this algorithm is a fancy early-stopping scheme for a random search
:cite:`TODO`.

.. cite ben recht paper on randomized search

This paper will review other existing work for model selection before
detailing the Hyperband implementation in Dask. A realistic set of experiments
will be presented before mentioning ideas for future work.

Related work
============

Software for model selection
----------------------------

Hyperband
---------
This section is a short review of :cite:`TODO`. Hyperband

* returns high performing models. It adapts to all the scores received to help
  choose which models to train further.
* has minimal inputs. It requires one input parameter to control the amount of
  work performed, and has an optional input parameter that controls how
  aggressive the search is.

High performing models
^^^^^^^^^^^^^^^^^^^^^^

Input parameters
^^^^^^^^^^^^^^^^

Hyperband in Dask
=================
Hyperband architecture
----------------------

Dwindling number of models
--------------------------

Experiments
===========

Future work
===========

References
==========

