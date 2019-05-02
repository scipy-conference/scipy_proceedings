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
influence the performance of the model. A good example is with the
regularization parameter in ridge regression :cite:`marquardt1975` or LASSO
:cite:`tibshirani1996`). This value helps the model adapt to the noise levels
in the data.

Model performance strongly depends on the hyper-parameters provided, even for
the simple examples above with a convex optimization and one hyper-parameter.
This gets much more complex when more hyper-parameters are required.
For example, in a recent study of a particular visualization tool input
combinations of three different hyper-parameters and the the first section is
titled "Those hyper-parameters really matter" :cite:`wattenberg2016`.

These hyper-parameters are typically assumed to be given, so they require some
cross-validation search to find an estimate of the optimal value. Even in the
simple ridge regression case above, a brute force search is required
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

Dask
----

Dask is a distributed computation framework for Python. It integrates nicely
with Python, and especially with NumPy :cite:`vanderwalt2011` and Pandas
:cite:`mckinney2012`. It provides easy methods to easily scale Python data
analysis, either "up" to "more data" or "out" to "more machines".

Dask is a declarative distributed computation framework that works with many
distributed schedulers including SLURM, Spark and SGE. The data analyst describes
the distributed computation to perform, and Dask communicates with the
distributed scheduler to determine what communication and computation to
perform to compute the desired result.

Dask provides a host of convenient diagnostics, including a dashboard that
reports diagnostic information on memory and CPU usage, a timeline of running
tasks and profiling information.

Software for model selection
----------------------------

A commonly used method for hyper-parameter selection is a random selection of
hyper-parameters followed by training each model to completion.  This offers
several advantages, including sampling "important parameters" more densely over
unimportant parameters :cite:`bergstra2012random` and being very amendable to
parallelism. This randomized search is implemented in Scikit-Learn
:cite:`pedregosa2011` and mirrored in Dask-ML.

These implementations are passive by definition: they do not adapt to previous
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

Hyperband is an adaptive model selection algorithm :cite:`li2016hyperband`.
Hyperband is a principled early-stopping scheme for randomized searches in one
application of the algorithm. Hyperband trains many models in parallel and
decides to stop models at particular times to preserve computation. By
contrast, Bayesian searches tweak a set of hyper-parameters based on serial
evaluations of a model that's assumed to be a black box.

.. TODO cite bandits

The Hyperband algorithm

* returns high performing models. It adapts to all the scores received to help
  choose which models to train further.
* has minimal inputs. It requires one input parameter to control the amount of
  work performed, and has an optional input parameter that controls how
  aggressive the search is.

Hyperband is constructed to identify the best model with high probability. The
analysis relies on sweeping over the tradeoff between training time
and hyper-parameter importance. If training time only matters a little, it
makes sense to aggressively stop training models. On the flip side, if only
training time influence the score, it only makes sense to let the models finish
training.

This sweep of possible values for the hyper-parameter vs. training time
importance is illustrated by the algorithm:

.. code-block:: python

   def sha(n_models, calls):
       """Successive halving algorithm"""
       models = [get_model_w_random_params()
                 for _ in range(n_models)]
       while True:
           models = [train(m, calls) for m in models]
           models = top_k(len(models) // 3, models)
           if len(models) == 1:
               break
           calls *= 3
       return models

   def hyperband(max_iter):
       # More models => more aggressive pruning of models
       brackets = [num_models_initial_calls(b, max_iter)
                   for b in range(f(max_iter))]
       final_models = [sha(n, r) for n, r in brackets]
       return best_model(final_models)

Each bracket indicates a value in the tradeoff between hyper-parameter and
training time importance. With ``max_iter=243``, the least adaptive bracket runs
5 models until completion and the most adaptive bracket aggressively prunes off
81 models.

This allows a mathematical proof that Hyperband is near optimal in the number
of iterations in the optimization algorithm:

.. latex::
   :usepackage: amsthm


.. raw:: latex

   \newtheorem{thm}{Theorem}
   \newcommand{\Log}{\overline{\log}}
   \newcommand{\parens}[1]{\left( #1 \right)}
   \begin{thm}
   \label{thm:hyperband}
   (informal presentation of Theorem 5 from \cite{li2016hyperband})
   Assume the loss at iteration $k$ decays like $(1/k)^{1/\alpha}$, and
   the validation losses approximately follow the cumulative distribution
   function $F(\nu) = (\nu - \nu_*)^\beta$ with optimal validation loss $\nu_*$.

   Higher values of $\alpha$ mean slower
   convergence, and higher values of $\beta$ represent more difficult model
   selection problems because it's harder to obtain a validation loss close to
   the optimal validation loss $\nu_*$.
   If $\beta > 1$, the validation losses are not uniformly
   distributed. The commonly used stochastic gradient
   descent has convergence rates with $1 \le \alpha \le 2$ with lower values
   implying more structure and regularity
   \cite{bottou2012stochastic} \cite{shamir2013}.

   Then for any $T\in\mathbb{N}$, let $\widehat{i}_T$ be the empirically best
   performing model from the last round of the infinite horizon Hyperband
   algorithm when $T$ resources have been used to train models. Then,
   model $\widehat{i}_T$ has loss $$\nu_{\widehat{i}_T} \le \nu_* +
   c\parens{\frac{\Log(T)^3 b}{T}}^{1/\max(\alpha,~\beta)}$$ for some constant
   $c$ and $b = \Log(\log(T) / \delta)$ where $\Log(x) = \log(x \log(x))$.

    By comparison, the best model without early stopping (i.e., randomized
    searches) after $T$ resources have been used to train models only has loss
   $$\nu_{\widehat{i}_T} \le \nu_* + c \parens{\frac{\log(T) b}{T}}^{1 / (\alpha + \beta)}$$
   \end{thm}

For simplicity, only the infinite horizon case is presented though the finite
horizon case is implemented in Dask-ML. [#finite]_ Theorem :ref:`thm:hyperband`
only applies to the "infinite horizon" case when model selection continues
indefinitely, so it only makes sense to compare for large values of the number
of resources used :math:`T` is large. When this happens, the loss
:math:`\nu_{\widehat{i}_T}` of Hyperband is much smaller than the uniform
allocation scheme. [#sizes]_


.. [#finite] To prove results about the finite horizon algorithm Li et. al.
   only need the result in Corollary 9 :cite:`li2016hyperband`.
   In the discussion afterwards, they remark that with Corollary 9
   they can show a similar result to Theorem :ref:`thm:hyperband` but leave
   it as an exercise for the reader.

.. [#sizes] This is clear by examining :math:`\log(\nu_{\widehat{i}_T} -
   \nu_*)` for Hyperband and uniform allocation. For Hyperband, the slope
   approximately decays
   like :math:`-1 / \max(\alpha,~\beta)`, much faster than the approximate
   uniform allocation slope of :math:`-1 / (\alpha + \beta)`

This shows a definite advantage to performing early stopping on randomized
searches. In addition, Li et. al. note that the probability the best model is
identified with a (near) minimal number of pulls, within log factors of the
lower bound on number of resources required as noted by Kaufmann et. al.
:cite:`kaufmann2015complexity`.

Theorem :ref:`thm:hyperband` only applies to the infinite budget setting when
training continues indefinitely. They also analyze the finite budget setting
when training is limited, and much of their analysis carries over.

More relevant work involves combining Bayesian searches and Hyperband, which
can be combined by using the Hyperband bracket framework `sequentially` and
progressively tuning a Bayesian prior to select parameters for each bracket
:cite:`falkner2018`. This work is also available through AutoML.

Model selection in Dask
=======================

Model selection searches can be compute and/or memory constrained. Memory
constrained problems include data not fitting in memory. Compute constrained
involve searches of many hyper-parameters (e.g., in neural nets).  This paper
is focused on searches that are compute constrained searches and is agnostic to
if they're memory constrained.

Dask-ML has a prior implementation that alleviate some computational effort
though it can not be applied to memory-constrained problems. This
implementation has a drop-in replacement for Scikit-Learn's randomized search.
The Dask-ML implementation caches trained sections of pipelines, which can
result in much lower time to the same solution as Scikit-Learn. However, it
requires that the entire dataset fit into the memory of a single machine.

The implementation of Hyperband in Dask-ML is follows the Scikit-Learn API. It
expects the model passed to have ``partial_fit``, ``score`` and ``{get,
set}_params`` methods. The requirement of a ``partial_fit`` implementation is
natural because all optimization algorithms are iterative to the author's
knowledge.

Hyperband architecture
----------------------

The Hyperband algorithm involves two "embarassingly parallel" for-loops:

* the sweep over the possible values of hyper-parameter vs. training time
  importance
* in each call to successive halving, the models are trained completely
  independently

The one downside to the amount of parallelism is that the number of models
decays approximately in each call to the successive halving algorithm,
approximately like :math:`1 / k` (but rather quantized).

This lends itself well to Dask, an advanced distributed scheduler that can
handle many concurrent jobs. Dask Distributed is required because the
computation graph is not static: training stops on particular models. This
wouldn't be a problem if only one successive halving bracket ran; however,
those are also run in parallel.

Input parameters
----------------

Hyperband requires two input parameters:

1. the number of ``partial_fit`` calls for the best estimator (via ``max_iter``)
2. the number of examples that each ``partial_fit`` call sees (which is implicit
   via ``chunks``, the chunk size of the Dask array).

These two parameters rely on knowing how long to train the estimator
[#examples]_ and having a rough idea on the number of parameters to evaluate.
Trying twice as many parameters with the same amount of computation requires
halving ``chunks`` and doubling ``max_iter``.

In comparison, random searches require three inputs:

1. the number of ``partial_fit`` calls for `every` estimator (via ``max_iter``)
2. how many parameters to try (via ``num_params``).
3. the number of examples that each ``partial_fit`` call sees (which is implicit
   via ``chunks``, the chunk size of the Dask array).

Trying twice as many parameters with the same amount of computation requires
doubling ``num_params`` and halving either ``max_iter`` or ``chunks``, so every
estimator will see half as many data. This means a balance between training
time and hyper-parameter importance is implicitly being decided upon.
Hyperband has one fewer input because it sweeps over this balance's importance.

.. [#examples] e.g., something in the form "the most trained model should see 100 times the number of examples (aka 100 epochs)"
.. [#tolerance] Tolerance (typically via ``tol``) is a proxy for ``max_iter`` because smaller tolerance typically means more iterations are run.

Dwindling number of models
--------------------------

Experiments
===========

Future work
===========

References
==========

