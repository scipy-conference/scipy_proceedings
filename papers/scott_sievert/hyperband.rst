:author: Scott Sievert
:email: scott@stsievert.com
:institution: University of Wisconsin–Madison

:author: Tom Augspurger
:email: taugspurger@anaconda.com
:institution: Anaconda, Inc.

:author: Matthew Rocklin
:email: mrocklin@gmail.com
:institution: NVIDIA

:bibliography: refs

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
:cite:`marquardt1975` or LASSO :cite:`tibshirani1996`).

Model performance strongly depends on the hyper-parameters provided, even for
the examples above. Performance depends strongly on the number of possible
hyper-parameter combinations, which grows exponentially as the number of
hyper-parameters grows. For example in a recent study of a particular
visualization tool input combinations of three different hyper-parameters, and
the first section is titled "Those hyper-parameters really matter"
:cite:`wattenberg2016`.

These hyper-parameters are typically assumed to be given. Even in the simple
ridge regression case above, a brute force search is required
:cite:`marquardt1975`. This gets more complex with many different
hyper-parameter values to input, and especially because there's often an
interplay between hyper-parameters. In practice, this means that
hyper-parameters have to be searched or tuned to find the value that provides
the highest performance.

Model selection has become more complicated as data has grown, especially with
the growth of deep learning. A good example is with the simplest optimization
hyper-parameter: learning rate. For convex problems with few data, a technique
called line search can be performed every iteration. However, with many data
this becomes too expensive to compute and the learning rate is another
hyper-parameter that needs to be tuned.

.. cite Steven Wright's book TODO

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
  amendable to parallelism, making a Dask implementation attractive.
* makes some simple modifications to increase amenability to parallelism
* provides an simple heuristic to determine the parameters to Hyperband, which
  only requires knowing how many examples the model should observe and a rough
  estimate on how many parameters to sample

This algorithm treats each computation as a scarce resource. If computation is
not a scarce resource, there is no benefit from this algorithm. At it's core,
this algorithm is a fancy early-stopping scheme for a random selection of
hyper-parameters.

This paper will review other existing work for model selection before
detailing the Hyperband implementation in Dask. A realistic set of experiments
will be presented before mentioning ideas for future work.

Related work
============

Software for model selection
----------------------------

A commonly used method for hyper-parameter selection is a random selection of
hyper-parameters followed by training each model to completion.  This offers
several advantages, including sampling "important parameters" more densely over
unimportant parameters :cite:`bergstra2012random` and being very amendable to
parallelism. This randomized search is implemented in Scikit-Learn
:cite:`pedregosa2011` and mirrored in Dask-ML.

These implementations are by definition passive: they do not adapt to previous
training. One popular class of adaptive algorithms are Bayesian model selection
algorithms. These algorithms treat the model as a black box and scores as a
noisy evaluation of that black box. These methods try to find the optimal set
of a hyper-parameters given a minimal number of observations by adapting to
previous evaluations.

Popular Bayesian searches include sequential model-based algorithm
configuration (SMAC) :cite:`hutter2011`, tree-structure Parzen estimator (TPE)
:cite:`bergstra2011`, and Spearmint :cite:`snoek2012`. Many of these are
available through the "robust Bayesian optimization" package RoBo
:cite:`kleinbayesopt17` through AutoML [#automl]_. This package also includes
Fabolas, a method that takes data-set size as input and allows for some
computational control :cite:`klein2016`.

.. [#automl] https://github.com/automl/

Hyperband
---------

Hyperband is an adaptive model selection algorithm. It can rely on each model
being iterative because it's grounded in a multi-armed bandit theoretical
framework. Hyperband treats model fitting as a scarce resource. One application of Hyperband is to assume the models are iterative. [#future-work]_

.. TODO cite bandits

.. [#future-work] One other application is mentioned as future work.

In this application, Hyperband is a principled early-stopping scheme for
randomized searches. Hyperband evaluates many models with different
hyper-parameters and selects to stop particular models at select times.

The Hyperband algorithm :cite:`li2016hyperband`

* returns high performing models. It adapts to all the scores received to help
  choose which models to train further.
* has minimal inputs. It requires one input parameter to control the amount of
  work performed, and has an optional input parameter that controls how
  aggressive the search is.

Bayesian searches and Hyperband can be combined by using the Hyperband bracket
framework `sequentially` and progressively tuning a Bayesian prior to select
parameters for each bracket :cite:`falkner2018`. This work is also available
through AutoML.

High performing models
^^^^^^^^^^^^^^^^^^^^^^

TODO

Input parameters
^^^^^^^^^^^^^^^^

TODO


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

