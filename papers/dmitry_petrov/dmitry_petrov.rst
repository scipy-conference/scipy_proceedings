:author: Dmitry Petrov
:email: to.dmitry.petrov@gmail.com
:institution: Senate House, S.P.Q.R.
:institution: Egyptian Embassy, S.P.Q.R.
:corresponding:

:author: Alexander Ivanov
:email: alexander.radievich@@gmail.com
:institution: Egyptian Embassy, S.P.Q.R.

:author: Daniel Moyer
:email: dcmoyer@gmail.com
:institution: Egyptian Embassy, S.P.Q.R.
:institution: Yet another place, S.P.Q.R.
:equal-contributor:

:author: Mikhail Belyaev
:email: brutus@rome.it
:institution: Unaffiliated
:equal-contributor:

:author: Paul Thompson
:email: brutus@rome.it
:institution: Unaffiliated
:equal-contributor:

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

— En masse experiments with combinatorial expansion of step options, running each option and returning results in a convenient format for human consumption (Pandas dataframe).
— Step caching. Standard SciKit-learn pipelines cannot cache temporary steps. Reskit includes the option  to save fixed steps, so in next pipeline specified steps won’t be recalculated.
— Forbidden combination constraints. Not all possible combinations of pipelines are viable or meaningfully different. For example, in a classification task comparing the performance of  logistic regression and decision trees the former requires feature scaling while the latter may not. In this case you can block the unnecessary pair. Reskit supports general tuple blocking as well.
— Full compatibility with scikit-learn objects. Reskit can use any scikit-learn data transforming object and/or predictive model, and many other libraries that uses the scikit template.
— Evaluation of multiple performance metrics simultaneously. Evaluation is simply another step in the pipeline, so we can specify a number of possible evaluation metrics and Reskit will expand out the computations for each metric for each pipeline.
— The DataTransformer class, which is Reskit’s simplfied interface for specifying fit/transform methods in pipeline steps. A DataTransformer subclass need only specify one function.
— Tools for learning on graphs. Due to our original motivations, Reskit includes a number of operations for network data. In particular, it allows  a variety of normalization choices for adjacency matrices, as well as built in  local graph metric calculations. These were implemented using DataTransformer and in some cases the BCTpy (the Brain Connectivity Toolbox python version)


How Reskit works
----------------

Usage example:

.. code-block:: python
 :linenos:
 :linenostart: 2

  from sklearn.datasets import make_classification
  from reskit.core import Pipeliner

  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import MinMaxScaler

  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  from sklearn.model_selection import StratifiedKFold

  # specifing data
  X, y = make_classification()

  # setting steps
  classifiers = [('LR', LogisticRegression()),
                 ('SVC', SVC())]

  scalers = [('standard', StandardScaler()),
             ('minmax', MinMaxScaler())]

  steps = [('scaler', scalers),
           ('classifier', classifiers)]

  # setting grid search parameters
  param_grid = {'LR': {'penalty': ['l1', 'l2']},
                'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}}

  # setting cross-validations for grid search and for evaluation
  grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  # creation of Pipeliner object
  pipeliner = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
  # launching experiment
  pipeliner.get_results(X, y, scoring=['roc_auc'])


Result will be

.. code-block:: bash

  Line: 1/4
  Line: 2/4
  Line: 3/4
  Line: 4/4


.. csv-table::
  :file: papers/dmitry_petrov/overview_results.csv

When Pipeliner initializes dataframe with all possible combinations is created

.. code-block:: python

  pipeliner.plan_table

.. csv-table::
  :file: papers/dmitry_petrov/overview_plan_table.csv

Gives results dataframe by defined pipelines.

.. code-block:: python

    pipeliner.transform_with_caching(X, y, row_keys)

Description

.. code-block:: python

  pipeliner.get_grid_search_results(self, X, y, row_keys,scoring):

Description

.. code-block:: python

  pipeliner.get_scores(self, X, y, row_keys, scoring):

Description

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }

Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.
