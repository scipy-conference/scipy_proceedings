:author: Scott Sievert
:email: scott@stsievert.com
:institution: University of Wisconsin–Madison
:institution: Relevant work performed while interning for Anaconda, Inc.
:corresponding:

:author: Tom Augspurger
:email: taugspurger@anaconda.com
:institution: Anaconda, Inc.

:author: Matthew Rocklin
:email: mrocklin@gmail.com
:institution: NVIDIA
:institution: Relevant work performed while employed for Anaconda, Inc.

:bibliography: refs

--------------------------------------------------------
Better and faster hyperparameter optimization with Dask
--------------------------------------------------------

.. class:: abstract

    Nearly every machine learning model requires that the user specify certain
    parameters before training begins, aka "hyperparameters". Finding the
    optimal set of hyperparameters is often a time- and resource-consuming
    process. A recent breakthrough hyperparameter optimization algorithm,
    Hyperband :cite:`li2016hyperband`, can find high performing hyperparameters with minimal training
    and has theoretical foundations. This paper will provide intuition for
    Hyperband, explain the slightly modified implementation and why it's
    well-suited for Dask, a Python library that scales Python to larger
    datasets and more computational resources. Experiments find high performing
    hyperparameters more quickly in the presence of serial or parallel
    computational resources with deep learning models.

.. class:: keywords

   distributed computation, hyperparameter optimization, machine learning

Introduction
============

Training any machine learning pipeline requires data, an untrained model or
estimator and "hyperparameters", parameters chosen before training begins that
help with cohesion between the model and data. The user needs to specify values
for these hyperparameters in order to use the model. A good example is with
adapting the ridge regression or LASSO to the amount of noise in the
data with the regularization parameter :cite:`marquardt1975`
:cite:`tibshirani1996`.  Hyperparameter choice verification can not be
performed until model training is completed.

Model performance strongly depends on the hyperparameters provided, even for
the simple examples above. This gets much more complex when multiple
hyperparameters are required. For example, a particular visualization tool,
t-SNE requires (at least) three hyperparameters
:cite:`maaten2008visualizing` and the first section in a study on how to use this
tool effectively is titled "Those hyperparameters really matter"
:cite:`wattenberg2016`.

These hyperparameters need to be specified by the user. There are no good
heuristics for determining what the values should be.
These values are typically found through a search over possible values through
a "cross validation" search where models are scored on unseen holdout data.
These searches are part of a hyperparameter optimization, and for certain more import. Even in the simple ridge regression case
above, a brute force search is required :cite:`marquardt1975`. This brute force
search quickly grows infeasible as the number of hyperparameters grow.

Hyperparameter optimization gets more complex with many hyperparameters, and
especially because there's often interaction between hyperparameters. A good
example of hyperparameter optimization is with deep learning, which has
specialized techniques for handling many data :cite:`bottou2010large`. However,
these optimization methods can't provide basic hyperparameters because there
are too many data to perform optimization efficiently :cite:`bottou2010large`.
For example, the most basic hyperparameter "learning rate" or "optimization
step size" is a quick computation with few data but infeasible for many data
:cite:`maren2015prob`.

Contributions
=============

A hyperparameter optimization is required if high performance is desired. In
practice, it's expensive and time-consuming for machine learning researchers
and practitioners. Ideally, hyperparameter optimization algorithms return high
performing models quickly and are simple to use.

Quickly returning quality hyperparameters relies on making decisions about
which hyperparameters to devote training time to. This might mean progressively
choosing higher-performing hyperparameter values or stopping low-performing
models early during training.

Returning this high performing model quickly would lower the expense and/or
time barrier to performing hyperparameter optimization. This will allow the
user (e.g., a data scientist) to more easily use these algorithms.

This work

* provides an implementation of a particular hyperparameter optimization
  algorithm, Hyperband :cite:`li2016hyperband` in Dask :cite:`dask`, a Python
  library that provides advanced parallelism. Hyperband returns models with a
  high validation score with minimal training.  A Dask implementation is
  attractive because Hyperband is amenable to parallelism.
* makes a simple modifications to increase Hyperband's amenability to
  parallelism.
* provides an simple heuristic to determine the parameters Hyperband requires,
  which only requires knowing how many examples the model should observe and a
  rough estimate on how many parameters to sample
* provides validating experiments that also illustrate common use cases

Hyperband treats computation as a scarce resource [#scarce]_ and has parallel
underpinnings.  In experiments, Hyperband returns high performing models fairly
quickly with a simple heuristic for determining Hyperband's input parameters.
The implementation can be found in Dask's machine learning package,
Dask-ML [#dask-ml]_.

This paper will review other existing work for hyperparameter optimization before
detailing the Hyperband implementation in Dask. A realistic set of experiments
will be presented before mentioning ideas for future work.

.. [#scarce] If computation is not a scarce resource, there is little benefit from
   this algorithm.

.. [#dask-ml] https://ml.dask.org.

Related work
============

Hyperparameter optimization
----------------------------

Hyperparameter optimization finds the optimal set of hyperparameters for a given model.
These hyperparameters are chosen to maximize performance on unseen data.
The hyperparameter optimization process typically looks like

1. Split the dataset into the train dataset and test dataset. The test dataset
   is reserved for the final model evaluation.
2. Choose hyperparameters
3. Train models with those hyperparameters
4. Score those models with unseen data (a subset of the train dataset typically
   referred to as the "validation set")
5. Use the best performing hyperparameters to train a model with those
   hyperparameters on the
   complete train dataset
6. Score the model on the test dataset. This is the score that is reported.

The rest of this paper will focus on steps 2 and 3, which is where most of the
work happens in hyperparameter optimization.

A commonly used method for hyperparameter selection is a random selection of
hyperparameters, and is typically followed by training each model to
completion. This offers several advantages, including a simple implementation
that is very amenable to parallelism. Other benefits include sampling
"important parameters" more densely than unimportant parameters
:cite:`bergstra2012random`. This randomized search is implemented in many
places, including in Scikit-Learn :cite:`pedregosa2011`.

These implementations are by definition `passive` because they do not adapt to previous training. `Adaptive` algorithms can return a higher quality solution with less training
by adapting to previous training and choosing which hyperparameter values to
evaluate. This is especially useful for difficult hyperparameter optimization problems with
many hyperparameters and many values for each hyperparameter.

A popular class of adaptive hyperparameter optimization algorithms are Bayesian
algorithms. These algorithms treat the model as a black box and the model
scores as an evaluation of that black box. These algorithms have an
estimate of the optimal set of hyperparameters and use some probabilistic
methods to improve the estimate. The choice of which hyperparameter value to
evaluate depends on previous evaluations.

Popular Bayesian searches include sequential model-based algorithm
configuration (SMAC) :cite:`hutter2011`, tree-structure Parzen estimator (TPE)
:cite:`bergstra2011`, and Spearmint :cite:`snoek2012`. Many of these are
available through the "robust Bayesian optimization" package RoBo
:cite:`kleinbayesopt17` through AutoML [#automl]_. This package also includes
Fabolas, a method that takes dataset size as input and allows for some
computational control :cite:`klein2016`.

.. [#automl] https://github.com/automl/

Hyperband
---------

Hyperband is a principled early stopping scheme for randomized hyperparameter
selection [#resources]_ and an adaptive hyperparameter optimization algorithm :cite:`li2016hyperband`.
At the most basic level, it partially trains
models before stopping models with low scores, then
repeats. By default, it stops training the lowest performing 33% of the available models
at certain times. This means that the number of models decay over time, and
the surviving models have high scores.

Naturally, model quality depends on two factors: the amount of training
performed and the values of various hyperparameters. If training time only
matters a little, it makes sense to aggressively stop training models. On the
flip side, if only training time influences the score, it only makes sense to
let all models train for as long as possible and not perform any stopping.

Hyperband sweeps over the relative importance of hyperparameter choice and
amount of training.
This sweep over training time importance enables a theorem that Hyperband will return a much higher performing model than the
randomized search without early stopping returns. This is best characterized by
an informal presentation of the main theorem:

.. [#resources] In general, Hyperband is a resource-allocation scheme for model
   selection.

.. latex::
   :usepackage: amsthm


.. raw:: latex

   \newtheorem{cor}{Corollary}
   \newcommand{\Log}{\overline{\log}}
   \newcommand{\parens}[1]{\left( #1 \right)}
   \begin{cor}
   \label{thm:hyperband}
   (informal presentation of \cite[Theorem 5]{li2016hyperband})
   Assume the loss at iteration $k$ decays like $(1/k)^{1/\alpha}$, and
   the validation losses $\nu$ approximately follow the cumulative distribution
   function $F(\nu) = (\nu - \nu_*)^\beta$ with optimal
   validation loss $\nu_*$ with $\nu-\nu_*\in[0, 1]$ .

   Higher values of $\alpha$ mean slower convergence, and higher values of
   $\beta$ represent more difficult hyperparameter optimization problems because it's
   harder to obtain a validation loss close to the optimal validation loss
   $\nu_*$.  Taking $\beta > 1$ means the validation losses are not uniformly
   distributed and higher losses are more common. The commonly used stochastic
   gradient descent has convergence rates with $\alpha= 2$
   \cite{bottou2012stochastic} \cite[Corollary 6]{li2016hyperband}, and
   gradient descent has convergence rates with $\alpha = 1$ \cite[Theorem 3.3]{bubeck2015convex}.

   Then for any $T\in\mathbb{N}$, let $\widehat{i}_T$ be the empirically best
   performing model when models are stopped early according to the infinite
   horizon Hyperband
   algorithm when $T$ resources have been used to train models. Then
   with probability $1 -\delta$, the empirically best performing model
   $\widehat{i}_T$ has loss $$\nu_{\widehat{i}_T} \le \nu_* +
   c\parens{\frac{\Log(T)^3 \cdot a}{T}}^{1/\max(\alpha,~\beta)}$$ for some constant
   $c$ and $a = \Log(\log(T) / \delta)$ where $\Log(x) = \log(x \log(x))$.

   By comparison, finding the best model without the early stopping Hyperband
   performs (i.e., randomized searches and training until completion) after $T$
   resources have been used to train models has loss $$\nu_{\widehat{i}_T} \le
   \nu_* + c \parens{\frac{\log(T) \cdot a}{T}}^{1 / (\alpha + \beta)}$$
   \end{cor}

For simplicity, only the infinite horizon case is presented though much of the
analysis carries over to the practical finite horizon Hyperband. [#finite]_
Because of this, it only makes sense to compare the loss when the number of
resources used :math:`T` is large. When this happens, the validation loss of
the Hyperband produces :math:`\nu_{\widehat{i}_T}` decays much faster than the
uniform allocation scheme. [#sizes]_ This shows a definite advantage to
performing early stopping on randomized searches.

.. [#finite] To prove results about the finite horizon algorithm Li et. al.
   only need the result in Corollary 9 :cite:`li2016hyperband`.
   In the discussion afterwards they remark that with Corollary 9
   they can show a similar result but leave it as an exercise for the reader.

.. [#sizes] This is clear by examining :math:`\log(\nu_{\widehat{i}_T} -
   \nu_*)` for Hyperband and uniform allocation. For Hyperband, the slope
   approximately decays
   like :math:`-1 / \max(\alpha,~\beta)`, much faster than the
   uniform allocation's approximate slope of :math:`-1 / (\alpha + \beta)`.

Li et. al. show that the model Hyperband identifies as the best is identified
with a (near) minimal amount of training in Theorem 7 :cite:`li2016hyperband`,
within log factors of the known lower bound :cite:`kaufmann2015complexity`.

More relevant work involves combining Bayesian searches and Hyperband, which
can be combined by using the Hyperband bracket framework `sequentially` and
progressively tuning a Bayesian prior to select parameters for each bracket
:cite:`falkner2018`. This work is also available through AutoML.

There is little to no gain from adaptive searches if the passive search
requires little computational effort. Adaptive searches spends choosing which
models to evaluate to minimize the computational effort required; if that's not
a concern there's not much value the value in any adaptive search is limited.

Dask
----

Dask provides advanced parallelism for analytics, especially for NumPy, Pandas
and Scikit-learn :cite:`dask`. It is familiar to Python users and does not
require rewriting code or retraining models to scale to larger datasets or to
more machines. It can scale up to clusters or to a massive dataset but also works
on laptops and presents the same interface. Dask provides two components:

* Dynamic task scheduling optimized for computation. This low level scheduler
  provides parallel computation and is optimized for interactive computational
  workloads.
* "Big Data" collections like parallel arrays, or dataframes, and lists that
  extend common interfaces like NumPy, Pandas, or Python iterators to
  larger-than-memory or distributed environments. These parallel collections
  run on top of dynamic task schedulers.

Dask aims to be familiar and flexible: it aims to parallelize and distribute
computation or datasets easily while retaining a task scheduling interface for
custom workloads and integration into other projects. It is fast and the
scheduler has low overhead. It's implemented in pure Python and can scale
from massive datasets to a cluster with thousands of cores to a laptop running
single process. In addition, it's designed with interactive computing
and provides rapid feedback and diagnostics to aid humans.


Dask's implementation of Hyperband
==================================

Dask can scale up to clusters or to massive datasets. Hyperparameter optimization searches
often require significant amounts of computation and can involve large
datasets, and Hyperband is amenable to parallelism. Combining Dask
with Hyperband is a natural fit.

This work focuses on the case when the computation required is not
insignificant. Then, the existing passive hyperparameter optimization algorithms in
Dask-ML have limited use because they don't adapt to previous training to
reduce the amount of training required.  [#dasksearchcv]_

An adaptive hyperparameter optimization algorithm, Hyperband is implemented in Dask's
machine learning library, Dask-ML.  [#docs]_ This algorithm adapts to previous
training to minimize the amount of computation required. This section will
detail the Hyperband architecture, the input arguments required and some
small modifications to reduce time to solution.

.. [#dasksearchcv] Though the existing implementation can reduce the
   computation required when pipelines are used. This is particularly useful
   when tuning data preprocessing (e.g., with natural language processing).
   More detail at https://ml.dask.org/hyper-parameter-search.html.

.. [#docs] https://ml.dask.org/modules/generated/dask_ml.model_selection.HyperbandSearchCV

Hyperband architecture
----------------------

There are two levels of parallelism in Hyperband, which result in two for-loops:

* an "embarrassingly parallel" sweep over the different brackets of the
  hyperparameter vs. training time importance
* in each bracket, the models are trained independently. This would be
  embarrassingly parallel if not for ceasing training of low performing models
  at particular times.

The amount of parallelism makes a Dask implementation very attractive. Dask
Distributed is required because of the nested parallelism: the computational
graph is dynamic and depends on other nodes in the graph.

Of course, the number of models in each bracket decrease over time because
Hyperband is an early stopping strategy. This is best illustrated by the
algorithm's pseudo-code:

.. code-block:: python

   from sklearn.base import BaseEstimator

   def sha(n_models: int,
           calls: int,
           max_iter: int) -> BaseEstimator:
       """Successive halving algorithm"""
       # (model and params are specified by the user)
       models = [get_model(random_params())
                 for _ in range(n_models)]
       while True:
           models = [train(m, calls) for m in models]
           models = top_k(models, k=len(models) // 3)
           calls *= 3
           if len(models) <  3:
               return top_k(models, k=1)

   def hyperband(max_iter: int) -> BaseEstimator:
       # Different brackets have different values of
       # "training" and "hyperparameter" importance.
       # => more models means more aggressive pruning
       brackets = [(get_num_models(b, max_iter),
                    get_initial_calls(b, max_iter))
                   for b in range(formula(max_iter))]
       if max_iter == 243:  # for example...
           assert brackets == [(81, 3), (34, 9),
                               (15, 27), (8, 81),
                               (5, 243)]
       # Each tuple is (num_models, n_init_calls)
       final_models = [sha(n, r, max_iter)
                       for n, r in brackets]
       return top_k(final_models, k=1)

In this pseudo-code, the train set and validation data are hidden, which ``train``
and ``top_k`` rely on. ``top_k`` returns the ``k`` best performing
models on the validation data and ``train`` trains a model for a certain number
of calls to ``partial_fit``.

Each bracket indicates a value in the trade-off between hyperparameter and
training time importance, and is specified by the list of tuples in the example
above. Each bracket is specified so that the total number of ``partial_fit``
calls is approximately the same among different brackets. Then, having many
models requires pruning models very aggressively and vice versa with few
models. As an example, with ``max_iter=243`` the least adaptive bracket has 5
models and no pruning. The most adaptive bracket has 81 models and fairly
aggressive early stopping schedule.

The exact aggressiveness of the early stopping schedule depends one optional
input to ``HyperbandSearchCV``, ``aggressiveness``. The default value is 3,
which has some theoretical motivation :cite:`li2016hyperband`.
``aggressiveness=4`` is likely more suitable for initial exploration when not
much is known about the model, data or hyperparameters.


Input parameters
----------------

Hyperband is also fairly easy to use. It only requires two input parameters:

1. the number of ``partial_fit`` calls for the best model (via
   ``max_iter``)
2. the number of examples that each ``partial_fit`` call sees (which is
   implicit and referred to as ``chunks``, which can be the "chunk size" of the
   Dask array).

These two parameters rely on knowing how long to train the model
[#examples]_ and having a rough idea on the number of parameters to evaluate.
Trying twice as many parameters with the same amount of computation requires
halving ``chunks`` and doubling ``max_iter``.

The primary advantage to Hyperband's inputs is that they do not require
balancing training time importance and hyperparameter importance.

In comparison, random searches require three inputs:

1. the number of ``partial_fit`` calls for `every` model (via ``max_iter``)
2. how many parameters to try (via ``num_params``).
3. the number of examples that each ``partial_fit`` call sees (which is
   implicit and referred to as ``chunks``, which can be the "chunk size" of the
   Dask array).

Trying twice as many parameters with the same amount of computation requires
doubling ``num_params`` and halving either ``max_iter`` or ``chunks``, which
means every model will see half as many data. Implicitly, a balance between
training time and hyperparameter importance is being decided upon. Hyperband
has one fewer input because it sweeps over this balance's importance in
different brackets.

.. [#examples] e.g., something in the form "the most trained model should see
   100 times the number of examples (aka 100 epochs)"
.. [#tolerance] Tolerance (typically via ``tol``) is a proxy for ``max_iter``
   because smaller tolerance typically means more iterations are run.

Dwindling number of models
--------------------------

At first, Hyperband evaluates many models. As time progresses, the number of
models decay because Hyperband is an early stopping scheme.  This
means towards the end of the computation, a few (possibly high-performing)
models can be training while most of the computational hardware is free. This
is especially a problem when computational resources are not free (e.g., with
cloud platforms like Amazon AWS or Google Cloud Platform).

Hyperband is a principled early stopping scheme, but it doesn't protect against
at least two common cases:

1. when models have converged before training completes (i.e., the score stays
   constant)
2. when models have not converged and poor hyperparameters are chosen (i.e, the
   scores are not increasing).

Providing a "stop on plateau" scheme will protect against these cases because
training will be stopped if a model's score stops increasing
:cite:`prechelt1998automatic`. This will require two additional parameters:
``patience`` to determine how long to wait before stopping a model, and ``tol``
which determines how much the score should increase.

Hyperband's early stopping is designed to identify the highest performing model
with minimal training. Setting ``patience`` to be high avoids interference with
this scheme, protects against both cases above, and errs on the side of giving
models more training time. In particular, it also provides a basic early
stopping mechanism for the least adaptive bracket of Hyperband.

Serial Simulations
==================

This section focuses on a synthetic classification example between 4 classes.
Some
detail is mentioned in the appendix, though complete details can be found at
https://github.com/stsievert/dask-hyperband-comparison.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   X, y = make_4_circles(num=60e3)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=int(10e3)
   )

The complete dataset is shown in Figure :ref:`fig:synthetic-data`.

.. latex::
   :usepackage: subcaption

.. latex::
   :usepackage: graphicx

.. raw:: latex

   \begin{figure}  % figure* for horizontal figures
   \centering
   \begin{subfigure}{0.45\textwidth}
       \centering
       \includegraphics[width=0.70\linewidth]{imgs/synthetic-dataset.png}
       \caption{
           The synthetic dataset used as input. In addition to these two
           informative dimensions, there are 4 uninformative dimensions with
           uniformly distributed random noise. There are 60,000 examples in
           this dataset and 50,000 are used for training. The colors correspond
           to different class labels and all points are bounded between $-2$
           and $2$ for all dimensions.
       }
       \label{fig:synthetic-data}
   \end{subfigure}
   \begin{subfigure}{0.45\textwidth}
       \centering
       \includegraphics[width=0.95\linewidth]{imgs/synthetic-val-acc.pdf}
       \caption{
           The average best score from
           Hyperband's early stopping scheme (via \texttt{hyperband})
           and randomized search without any early stopping (via
           \texttt{passive}). The shaded regions
           correspond to the 25\% and 75\% percentiles over 40 runs.
           The mean is low for \texttt{passive} because two models had final best
           scores below 60\%. All other \texttt{passive} runs had final best scores about 70\%.
       }
       \label{fig:synthetic-performance}
   \end{subfigure}
   \caption{
       In this simulation, each call to \texttt{partial\_fit} sees about 1/3rd
       of examples in the complete train dataset. Each model completes no more
       than 81 passes through the data. The passive search performs no early
       stopping and trains 17 models to completion. Hyperband initially
       evaluates 143 models.
   }
   \label{fig:synthetic}
   \end{figure}



Model architecture & Hyperparameters
-------------------------------------

The model used is Scikit-learn's fully-connected neural network, their
``MLPClassifier``. In this, there are several hyperparameters.  Only one is
effects the architecture of the best model: ``hidden_layer_sizes``, which
controls the number of layers and number of neurons in each layer.

There are 5 values for the hyperparameter. It is varied so the neural network
has 24 neurons but varies the network depth and the width of each layer. Two
choices are 12 neurons in 2 layers or 6 neurons in four layers. One choice
has 12 neurons in the first layer, 6 in the second, and 3 in third and
fourth layers.

Six other hyperparameters have to be tuned and control finding the best model,
3 of which are continuous. There are 50 possible choices from all of the 3
discrete hyperparameters. Details are in the appendix. These hyperparameters
include the batch size, learning rate (and decay schedule) and a regularization
parameter:

.. code-block:: python

   from sklearn.neural_network import MLPClassifier
   model = MLPClassifier(...)
   params = {'batch_size': [32, 64, ..., 512], ...}
   print(params.keys())
   # dict_keys([
   #     "batch_size",  # 5 choices
   #     "learning_rate",  # 2 choices
   #     "hidden_layer_sizes",  # 5 choices
   #     "alpha",  # cnts
   #     "power_t",  # cnts
   #     "momentum",  # cnts
   #     "learning_rate_init"  # cnts
   # ])

Usage: rule of thumb on ``HyperbandSearchCV``'s inputs
------------------------------------------------------

``HyperbandSearchCV`` only requires `two` parameters besides the model and data
as discussed above: ``max_iter`` and the number of examples each call to
``partial_fit`` sees (which is implicit via the Dask array chunk size
``chunks``). These inputs control how many hyperparameter values are considered
and how long to train the models.

The values for ``max_iter`` and ``chunks`` can be specified by a rule-of-thumb
once the number of parameter to be sampled and the number of examples required
to be seen by at least one model, ``n_examples``. This rule of thumb is:

.. code-block:: python

   # Specify these two parameters
   n_params = 230
   n_examples = 81 * len(X_train)

   # Use this rule-of-thumb
   max_iter = n_params
   chunks = n_examples // n_params

Creation of a ``HyperbandSearchCV`` object and the Dask array is simple with
this:

.. code-block:: python

   from dask_ml.model_selection import HyperbandSearchCV
   search = HyperbandSearchCV(
       model, params,
       max_iter=max_iter, aggressiveness=4)

   X_train = da.from_array(X_train, chunks=chunks)
   y_train = da.from_array(y_train, chunks=chunks)
   search.fit(X_train, y_train)


``aggressiveness=4`` is chosen because this is my first time optimizing these
hyperparameters – I only made one small edit to the hyperparameter search space
[#change]_.  With ``max_iter``, no model sees more than ``n_examples`` examples
as desired and Hyperband evaluates (approximately) ``n_params`` hyperparameter
combinations.

.. [#change] For personal curiosity, I changed total number of neurons to 24
   from 20 to allow the ``[12, 6, 3, 3]`` configuration.


Performance
-----------

.. figure:: imgs/synthetic-priority
   :align: center
   :scale: 75%

   A visualization of how the Dask prioritization scheme influences the
   Hyperband's time to solution. Dask assigns prioritizes training models with
   higher scores (via ``high-scores``). When Dask uses the default priority
   scheme it fits models in the order they are received by Dask Distributed's
   scheduler (via  ``fifo``). Only the prioritization in the figure changes
   because both ``high-scores`` and ``fifo`` have the same hyperparameters,
   train/validation data, and assign the same internal random state to models.
   The hyperparameters are chosen from a run in Figure
   :ref:`fig:synthetic-performance`.
   :label:`fig:synthetic-priority`

Two hyperparameter optimizations are compared, Hyperband and random search and
is shown in Figure :ref:`fig:synthetic-performance`.
Recall from above that Hyperband is a principled early stopping scheme for
random search. The comparison mirrors that by sampling same hyperparameters
[#random-sampling-hyperband]_ and using the same validation set for each run.

Dask provides features that the Hyperband implementation can easily exploit.
Dask Distributed supports prioritizing different jobs, so it's simple to
prioritize the training of different models based on their most recent score.
Prioritizing high-performing models will almost certainly prioritize the
highest performing bracket finishes training more quickly. This highlights how
Dask is useful to Hyperband and shown in Figure :ref:`fig:synthetic-priority`.
These simulations are performed on a laptop with 4 Dask workers. This makes the
hyperparameter selection very serial and the number of ``partial_fit`` calls or
passes through the dataset a good proxy for time.

Dask's priority of training high scoring models works best in very serial
environments: priority makes no difference in very parallel environment when
every job can be run. To get around this in very parallel environments, the
worst performing models all have the same priority for each bracket.


.. [#random-sampling-hyperband] As much as possible – Hyperband evaluates more
   hyperparameter values. The random search without early stopping
   evaluates every hyperparameter value Hyperband evaluates.

Parallel Experiments
====================

This section will highlight a practical use of ``HyperbandSearchCV`` and use 25
Dask workers. A popular neural network library will be used PyTorch [#pytorch]_
:cite:`paszke2017automatic` (through the wrapper Skorch [#skorch]_) for an
image denoising task.


.. [#pytorch] https://pytorch.org
.. [#skorch] https://github.com/skorch-dev/skorch

The inputs and desired outputs are given in Figure :ref:`fig:io+est`. This is
an especially difficult problem because the noise variance varies slightly
between images. To protect against this, let's use a shallow neural network
that's slightly more complex than a linear model.  This means hyperparameter
optimization is not simple.

Again, some detail is mentioned in the appendix though complete details can be
found at https://github.com/stsievert/dask-hyperband-comparison.

.. raw:: latex

   \begin{figure}
   \centering
   \begin{subfigure}{0.45\textwidth}
       \centering
       \includegraphics[width=0.95\linewidth]{imgs/io+est}
       \caption{
   The rows show in the ground truth, input and output respectively for the
   denoising problem. The output is shown for the best model that Hyperband
   finds.
       }
       \label{fig:io+est}
   \end{subfigure}
   \begin{subfigure}{0.45\textwidth}
       \centering
       \includegraphics[width=0.95\linewidth]{imgs/2019-03-24-time.png}
       \caption{
   The time required to obtain a particular validation score (or negative loss). The legend labels are in
   Table \ref{table:legend}.
       }
       \label{fig:time}
   \end{subfigure}
   \caption{
       In this experiment, each call to \texttt{partial\_fit} uses 1/3 of the
       examples in the complete train dataset, so algorithm passes over the training data about 1,667 times in
       total, a.k.a.  1,667 epochs. Each model sees no more than 81 times the
       number of examples in the dataset because \texttt{max\_iter=243} for all
       searches.
   }
   \label{fig:img-exp}
   \end{figure}

.. TODO: figure out which model that is. Say a sentence about it (which bracket, etc)

.. figure:: imgs/2019-03-24-activity.png
   :align: center

   The activity over time for the 25 Dask workers for the run shown in
   Figure :ref:`fig:img-exp`.
   :label:`fig:activity`


Model architecture & Hyperparameters
-------------------------------------

To address that complexity, let's use an autoencoder. These are a type of neural
network that reduce the dimensionality of the input before expanding to the
original dimension. This can be thought of as a lossy compression. Let's create
that model and the images it'll denoise:

.. code-block:: python

   # custom model definition with PyTorch
   from autoencoder import Autoencoder
   from sklearn.model_selection import train_test_split
   import skorch  # scikit-learn API wrapper for PyTorch

   model = skorch.NeuralNetRegressor(Autoencoder, ...)

   X, y = noisy_mnist()
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.1
   )

.. This autoencoder has two layers that compress

Of course, this is a neural network so there are many hyperparameters to tune.
Only one hyperparameter affects the model architecture:
``estimator__activation``, which specifies the activation the neural network
should use.

This hyperparameter is varied between 4 different choices, all different types
of the rectified linear unit (ReLU) :cite:`relu`, including the leaky ReLU
:cite:`leaky-relu`, parametric ReLU :cite:`prelu` and exponential linear units
(ELU) :cite:`elu`.

There are 6 other hyperparameters do not influence the model architecture.
There are 3 discrete hyperparameters (and 160 combinations of all discrete
variables) and 3 continuous hyperparameters. These hyperparameters all control
finding the optimal model after the architecture is fixed. These includes
hyperparameter like the optimizer to use (stochastic gradient descent
:cite:`bottou2010large` a.k.a SGD or Adam :cite:`adam`), initialization,
regularization and optimizer hyperparameters like learning rate or momentum.
Here's a brief description:

.. code-block:: python

   params = {'optimizer': ['SGD', 'Adam'], ...}
   print(params.keys())
   # dict_keys([
   #     "optimizer",  # 2 choices
   #     "batch_size",  # 5 choices
   #     "module__init",  # 4 choices
   #     "module__activation",  # 4 choices
   #     "optimizer__lr",  # cnts
   #     "optimizer__momentum",  # cnts
   #     "optimizer__weight_decay"  # cnts
   # ])

Details are in the appendix.

Usage: plateau specification for non-improving models
-----------------------------------------------------

``HyperbandSearchCV`` supports specifying ``patience=True`` to make a decision
on how long to wait to see if scores stop increasing, as mentioned above. Let's
create a ``HyperbandSearchCV`` object that stops training non-improving models.

.. code-block:: python

    from dask_ml.model_selection import HyperbandSearchCV
    search = HyperbandSearchCV(
        model, params, max_iter=max_iter, patience=True
    )
    search.fit(X_train, y_train)

The current implementation uses ``patience=True`` to choose a high value of
``patience=max_iter // 3``. This is most useful for the least adaptive bracket
of Hyperband, which trains a couple models to completion. This has a large
impact on the time to solution with these parallel computational resources but
doesn't have a large impact on the number of ``partial_fit`` calls.

Performance
-----------

Anecdotally, ``HyperbandSearchCV`` performs well and beats manual hand-tuning
by a considerable margin. While manually tuning, I considered any scores/negative losses of about
:math:`-0.10` to be pretty good, and I obtained scores no higher than
:math:`-0.098`. That's the context necessary to interpret
``HyperbandSearchCV``'s score of :math:`-0.093` and ``IncrementalSearchCV``'s
score of :math:`-0.0975`.

A quantitative measure comes by comparing three algorithms with
the same model, parameters and validation data. The comparisons are shown in
Figures :ref:`fig:time` and :ref:`fig:activity` and the legends for these plots
is shown in Table :ref:`table:legend`. In these experiments, 25 workers are
used with Dask, meaning that 25 tasks can complete in parallel.

This section will focus on stopping non-improving models. Two schemes that stop
non-improving models will be compared with Hyperband (which is already an early
stopping scheme that will likely stop these models). These two schemes will be
``HyperbandSearchCV`` with ``patience=True`` and a randomized search that
trains all models to completion via ``IncrementalSearchCV``.

The inputs to ``IncrementalSearchCV`` illustrate the implicit balance between
hyperparameter vs. training time importance: training models for longer with
the same computational effort would require a higher value for ``num_params``
and a lower and more aggressive value for ``patience``.  For
``IncrementalSearchCV``, I choose an aggressive and low patience of
``patience=24`` 8 passes through the data. Because of this, I choose to
evaluate twice as many hyperparameters as the number required to have the same
number of ``partial_fit`` calls as Hyperband.

The data scientist cares about time to reach a particular score, not the number
of ``partial_fit`` calls required. Those are similar for a small personal
machine but may be very different in the presence of a large cluster or
supercomputer. The time required to reach a particular validation accuracy
that is shown in Figure :ref:`fig:time`.  This plot is shown with 25 workers, a
reasonable number of workers to expect, especially if each worker requires a
GPU.

``HyperbandSearchCV`` with ``patience=True`` and ``patience=False`` require a
similar number of calls to ``partial_fit``, within a 5% difference. However,
Figure :ref:`fig:time` shows a remarkable difference of specifying
``patience=True`` for Hyperband: specifying ``patience=True`` means that
Hyperband finishes in about 2/3rds of the time as the default Hyperband! This
is because one worker hold onto a single model for about 4 minutes as shown in
Figure :ref:`fig:activity`.


.. table:: A summary of the legends in Figures
           :ref:`fig:time` and :ref:`fig:activity`. ``IncrementalSearchCV``
           ``patience=24`` is an algorithm that stops training after the scores
           stop increasing or plateau, hence the label.
           :label:`table:legend`

   +---------------------+---------------------------------------------------+
   | Label               | Class                                             |
   +=====================+===================================================+
   | ``hyperband``       | ``HyperbandSearchCV``                             |
   +---------------------+---------------------------------------------------+
   | ``stop-on-plateau`` | ``IncrementalSearchCV``, ``patience=24``          |
   +---------------------+---------------------------------------------------+
   | ``hyperband+sop``   | ``HyperbandSearchCV``, ``patience=True``          |
   +---------------------+---------------------------------------------------+


Future work
===========

The biggest area for improvement is using another application of the Hyperband
algorithm: controlling the dataset size as the scarce resource.  This would
treat every model as a black box and vary the amount of data provided. This
would not require the model to implement ``partial_fit`` and would only require
a ``fit`` method.

Another area of future work is ensuring ``IncrementalSearchCV`` and all of its
children (including ``HyperbandSearchCV``) work well with large models.
Modern models often consume most of GPU memory, and currently
``IncrementalSearchCV`` requires making a copy the model. How much does this
hurt performance and can it be avoided?

References
==========


Appendix
========

This section expands upon the example given above. Complete details can be
found at
https://github.com/stsievert/dask-hyperband-comparison.


Serial Simulation
-----------------

Here are some of the other hyperparameters tuned, alongside descriptions of
their default values and the values chosen for tuning.

* ``alpha``, a regularization term that can affect generalization. This value
  defaults to :math:`10^{-4}` and is tuned logarithmically between
  :math:`10^{-6}` and :math:`10^{-3}`
* ``batch_size``, the number of examples used to approximate the gradient at
  each optimization iteration. This value defaults to 200 and is chosen to be one of :math:`[32, 64,
  \ldots, 512]`.
* ``learning_rate`` controls the learning rate decay scheme, either constant or
  via the "``invscaling``" scheme, which has the learning rate decay like
  :math:`\gamma_0/t^p` where :math:`p` and :math:`\gamma_0` are also tuned.
  :math:`\gamma_0` defaults to :math:`10^{-3}` and is tuned logarithmically
  between :math:`10^{-4}` and :math:`10^{-2}`. :math:`p` defaults to 0.5 and is
  tuned between 0.1 and 0.9.
* ``momentum``, the amount of momentum to include in Nesterov's momentum
  :cite:`nesterov2013a`. This value is chosen between 0 and 1.

The learning rate scheduler used is not Adam :cite:`adam` because it claims to
be most useful without tuning and has reportedly has marginal gain
:cite:`wilson2017b`.


Parallel Experiments
--------------------
Here are some of the other hyperparameters tuned:

* ``optimizer``: which optimization method should be used for training? Choices
  are stochastic gradient descent (SGD) :cite:`bottou2010large` and Adam
  :cite:`adam`. SGD is chosen with 5/7th probability.
* ``estimator__init``: how should the estimator be initialized before training?
  Choices are Xavier :cite:`xavier` and Kaiming :cite:`kaiming` initialization.
* ``batch_size``: how many examples should the optimizer use to approximate the
  gradient? Choices :math:`[32, 64, \ldots,  512]`.
* ``weight_decay``: how much of a particular type of regularization should the
  neural net have? Regularization helps control how well the model performs on
  unseen data. This value is chosen to be zero 1/6th of the time, and if not
  zero chosen uniformly at random between :math:`10^{-5}` and
  :math:`10^{-3}` logarithmically.
* ``optimizer__lr``: what learning rate should the optimizer use? This is the
  most basic hyperparameter for the optimizer. This value is tuned between
  :math:`10^{-1.5}` and :math:`10^{1}` after some initial tuning.
* ``optimizer__momentum``, which is a hyper-parameter for the SGD optimizer to
  incorporate Nesterov momentum :cite:`nesterov2013a`. This value is tuned
  between 0 and 1.
