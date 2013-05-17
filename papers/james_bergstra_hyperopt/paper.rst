:author: James Bergstra
:email: james.bergstra@uwaterloo.ca
:institution: University of Waterloo

:author: Dan Yamins
:email: yamins@mit.edu
:institution: Massachusetts Institute of Technology

:author: David D. Cox
:email: davidcox@fas.harvard.edu
:institution: Harvard University


-------------------------------------------------------------------------------------------
Hyperopt: A Python Library for Optimizing the Hyperparamters of Machine Learning Algorithms
-------------------------------------------------------------------------------------------

.. class:: abstract

    Model-based optimization (i.e. based on Bayesian optimization) is one of the most efficient
    methods for optimizing expensive functions over awkward spaces.
    The Hyperopt library (Hyperopt) provides model-based optimization in Python.
    Hyperopt provides an implementation of the TPE algorithm, which has proven to be an effective strategy for optimizing neural networks, deep networks, and computer vision systems.
    Hyperopt supports parallel and asynchronous evaluation of hyperparameter configurations via its MongoDB backend.
    This paper describes the usage and architecture of hyperopt.

.. class:: keywords

    Bayesian optimization, hyperparameter optimization, model selection


Introduction
------------

Sequential model-based optimization (SMBO, also known as Bayesian optimization) is a family of optimization methods that includes some of the most efficient
(in terms of function evaluations) methods currently available.
Originally developed for oil exploration [mockus78]_
SMBO methods are generally applicable to scenarios in which a user wishes to minimize some
scalar-valued function :math:`f(x)` that is costly to evaluate, e.g. in terms of time or money.
Compared with standard optimization strategies such as conjugate gradient descent methods,
model-based optimization algorithms invest more time between function
evaluations in order to reduce the number of function evaluations overall.


Sequential Model-Based Optimization
-----------------------------------

* Does not require gradient

* Can leverage function smoothness even without analytic gradient

* Works over awkward spaces (product of real-valued, discrete, conditional variables)

* Can parallelize the evaluation of :math:`f` new points :math:`x`


Hyperopt Usage
--------------

Basic usage of the hyperopt library is illustrated by the following code.

.. code-block::python

    # define an objective function
    def objective(args):
        case, val = args
        if case == 'case 1':
            return val
        else:
            return val ** 2

    # define a search space
    from hyperopt import hp
    space = hp.choice('a',
        [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print best
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print hyperopt.space_eval(space, best)
    # -> ('case 2', 0.01420615366247227}


This code illustrates hyperopt's `fmin` function.
The `fmin` function is the main interface for both synchronous and asynchronous
(parallel, including across hosts)
execution.



* fmin
* configuration language
* returning more than the loss function
* Trials
* MongoDB
* Parallel/Asynchronous optimization
* MongoTrials
* algorithms
* pyll
* vectorization?
* TPE


Planned Future Work
-------------------

Drivers for other systems: 
* Jasper Snoek's "spearmint" package for Gaussian process-based Bayesian optimization
* Frank Hutter's SMAC and ROAR algorithms, as implemented in XXX.


Wrapper layer around sklearn.


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

