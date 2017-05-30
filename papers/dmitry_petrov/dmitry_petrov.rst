:author: Dmitry Petrov
:email: to.dmitry.petrov@gmail.com
:institution: Imaging Genetics Center, University of Southern California, Los Angeles, USA
:institution: The Institute for Information Transmission Problems, Moscow, Russia
:corresponding:
:equal-contributor:

:author: Alexander Ivanov
:email: alexander.radievich@@gmail.com
:institution: The Institute for Information Transmission Problems, Moscow, Russia
:institution: Skoltech Institute of Science and Technology, Moscow, Russia
:equal-contributor:

:author: Daniel Moyer
:email: dcmoyer@gmail.com
:institution: Imaging Genetics Center, University of Southern California, Los Angeles, USA

:author: Mikhail Belyaev
:email: belyaevmichel@gmail.com
:institution: Skoltech Institute of Science and Technology, Moscow, Russia

:author: Paul Thompson
:email: pthomp@usc.edu
:institution: Imaging Genetics Center, University of Southern California, Los Angeles, USA

:video: https://github.com/neuro-ml/reskit

--------------------------------------------------------------------------------------------------
Reskit: a library for creating and curating reproducible pipelines for scientific machine learning
--------------------------------------------------------------------------------------------------

.. class:: abstract

In this work we introduce Reskit (researcher’s kit), a library for creating and curating reproducible pipelines for scientific machine learning. A natural extension of the Scikit Pipelines to general classes of pipelines, Reskit allows for the efficient and transparent optimization of each pipeline step. Its main features include data caching, compatibility with most of the scikit-learn objects, optimization constraints such as forbidden combinations, and table generation for quality metrics. Reskit’s design will be especially useful for researchers requiring pipeline versioning and reproducibility, while running large volumes of experiments.

.. class:: keywords

   data science, reproducibility, python

Introduction
------------

A central task in machine learning and data science is the comparison and selection of models. The evaluation of a single model is very simple, and can be carried out in a reproducible fashion using the standard scikit pipeline. Organizing the evaluation of a large number of models is tricky; while there are no real theory problems present, the logistics and coordination can be tedious. Evaluating a continuously growing zoo of models is thus an even more painful task. Unfortunately, this last case is also quite common.

The task is simple: find the best combination of pre-processing steps and predictive models with respect to an objective criterion. Logistically this can be problematic: a small example might involve three classification models, and two data preprocessing steps with two possible variations for each — overall 12 combinations. For each of these combinations we would like to perform a grid search of predefined hyperparameters on a fixed cross-validation dataset, computing performance metrics for each option (for example ROC AUC). Clearly this can become complicated quickly. On the other hand, many of these combinations share substeps, and re-running such shared steps amounts to a loss of compute time.

Reskit [1] is a Python library that helps researchers manage this problem. Specifically, it automates the process of choosing the best pipeline, i.e. choosing the best set of data transformations and classifiers/regressors. The researcher specifies the possible processing steps and the scikit objects involved, then Reskit expands these steps to each possible pipeline, excluding forbidden combinations. Reskit represents these pipelines in a convenient pandas dataframe, so the researcher can directly visualize and manipulate the experiments.

Reskit then runs each experiment and presents results which are provided to the user through a pandas dataframe. For example, for each pipeline’s classifier, Reskit could  grid search on cross-validation to find the best classifier’s parameters and report metric mean and standard deviation for each tested pipeline. Reskit also allows you to cache interim calculations to avoid unnecessary recalculations.

Main features of Reskit
-----------------------

- En masse experiments with combinatorial expansion of step options, running each option and returning results in a convenient format for human consumption (Pandas dataframe).

- Step caching. Standard SciKit-learn pipelines cannot cache temporary steps. Reskit includes the option  to save fixed steps, so in next pipeline specified steps won’t be recalculated.

- Forbidden combination constraints. Not all possible combinations of pipelines are viable or meaningfully different. For example, in a classification task comparing the performance of  logistic regression and decision trees the former requires feature scaling while the latter may not. In this case you can block the unnecessary pair. Reskit supports general tuple blocking as well.

- Full compatibility with scikit-learn objects. Reskit can use any scikit-learn data transforming object and/or predictive model, and many other libraries that uses the scikit template.

- Evaluation of multiple performance metrics simultaneously. Evaluation is simply another step in the pipeline, so we can specify a number of possible evaluation metrics and Reskit will expand out the computations for each metric for each pipeline.

- The DataTransformer class, which is Reskit’s simplfied interface for specifying fit/transform methods in pipeline steps. A DataTransformer subclass need only specify one function.

- Tools for learning on graphs. Due to our original motivations, Reskit includes a number of operations for network data. In particular, it allows  a variety of normalization choices for adjacency matrices, as well as built in  local graph metric calculations. These were implemented using DataTransformer and in some cases the BCTpy (the Brain Connectivity Toolbox python version)


How Reskit works
----------------

Pipeliner class
---------------

DataTransformer class
---------------------

MatrixTransformer class
-----------------------

Applications
------------

Reskit was originally developed for a brain network classification task. We have successfully applied it in our own research several times [8,9]. Code from two of these projects can be found at [10] and [11].  We believe the library is general enough to be useful in a variety of data science contexts, and we hope that other researchers will find this library useful in their studies.

Dependencies
------------

- Python 3.4 and higher.
- Scikit-learn [2] 0.18.1 and its dependencies. Our library was heavily inspired by scikit-learn Pipeline class and overall architecture of this library. One can think of Reskit as an extension of  scikit-learn pipelines.
- Pandas [4].
- SciPy [5], Python-Igraph [6] and NetworkX [7] for machine learning on networks.

Future plans
------------

- Ability to merge  multiple experiment plans.
- Distributed computing for calculation on computing clusters.
- Ability to calculate different quality metrics after one optimization.
- Public repository of DataTransformers for various purposes.
- Option to save best models/pipelines according to external criteria.
- Support for Python 2.7

Conclusion
----------

In this abstract we introduced Reskit, a library for creating and curating reproducible pipelines for scientific machine learning. Reskit allows for the efficient and transparent optimization of each pipeline step. Its main features include data caching, compatibility with most of the scikit-learn objects, optimization constraints, and table generation for quality metrics. Reskit’s design will be especially useful for researchers requiring pipeline versioning and reproducibility, while running large volumes of experiments.


References
----------

.. [reskit] https://github.com/neuro-ml/reskit/tree/master

.. [scikit] http://scikit-learn.org/stable/

.. [bct] https://sites.google.com/site/bctnet/

.. [pandas] http://pandas.pydata.org/

.. [scipy] https://www.scipy.org/

.. [igraph] http://igraph.org/python/

.. [networkx] https://networkx.github.io/

.. [PRNI2016] D. Petrov, Y. Dodonova, L. Zhukov, M. Belyaev, Boosting Connectome Classification via Combination of Geometric and Topological Normalization, 6th International Workshop on
   Pattern Recognition in Neuroimaging - 2016

.. [ISBI2017]  https://arxiv.org/abs/1701.07847

.. [PRNI_code] https://github.com/neuro-ml/PRNI2016

.. [ISBI_code] https://github.com/neuro-ml/structural-connectome-validation-pairwise
