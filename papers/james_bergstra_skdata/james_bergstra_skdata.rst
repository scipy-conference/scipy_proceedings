:author: James Bergstra
:email: james.bergstra@uwaterloo.ca
:institution: University of Waterloo

:author: Nicolas Pinto
:email: pinto@mit.edu
:institution: Massachusetts Institute of Technology

:author: David D. Cox
:email: davidcox@fas.harvard.edu
:institution: Harvard University


--------------------------------------------------------------
Skdata: Data Sets and Algorithm Evaluation Protocols in Python
--------------------------------------------------------------

.. class:: abstract

    Machine learning benchmark data sets come in all shapes and sizes, yet classification algorithm implementations often insist on operating on sanitized input, such as (x, y) pairs with vector-valued input x and integer class label y.
    Researchers and practitioners are well aware of how tedious it can be to get from the URL of a new data set to an ndarray fit for e.g. pandas or sklearn.
    The skdata library [1] handles that work for a growing number of benchmark data sets,
    so that one-off in-house scripts for downloading and parsing data sets can be replaced with library code that is reliable, community-tested, and documented.

    The skdata library provides both scripts and library routines for tasks related to data sets, including several large datasets of images that must be handled with care in order to fit in typical amounts of RAM.
    Scripts are provided for downloading, verifying, and visualizing many public benchmark datasets.
    Routines and data types are provided for interfacing to data sets as both raw and structured things.
    A low level interface provides relatively *raw* access to the contents of
    each data set, in terms that closely reflect what was (in most cases) downloaded from the web.
    Each data set may also be associated with zero or more high level interfaces.
    A high level interface provides a *sanitized* view of a data set, in which the data have been assembled into e.g. (X, y) pairs,
    in order to look like a standard machine learning problem such as classification, regression, or density estimation.
    The high level interface ensures that learning algorithms see exactly the right examples for training and testing,
    so that results are directly comparable to ones published in academic literature.

    This paper describes the architecture and usage of the skdata library.
    It is hoped that the library will make it easier for researchers to reproduce published results on public benchmarks,
    and to run valid experiments aimed at improving on those benchmarks.


.. class:: keywords

    machine learning, cross validation, reproducibility

Introduction
------------

One of the most basic testing strategies for an implementation of a machine learning algorithm,
whether it implements a new algorithm or one that is widely known,
is "How well does it perform on standard benchmarks?"
Answering this question for new algorithms is often a principal contribution of research papers in machine learning.
Anyone implementing an algorithm from scientific literature must first verify that their implementation matches the results reported in that literature.
Adherence to standard algorithm evaluation protocols is critical to
obtaining an accurate answer to this central question.

At the same time, it is not always obvious what exactly the standard protocol
is.
For example, the widely-used "Iris" data set is simply an enumeration of 150 specimens' petal and sepal measurements along with the label of which kind of iris each one is [Iris]_. 
If we are interested in matching the generalization error of our implementation to a generalization error in the literature, then we would like to know more than just the accuracy;
we would like to know exactly which examples were used for training, and which
examples were used for measuring that generalization error.
It would be tedious to write such detail into a paper (and to transcribe it back into code afterward!), but it is natural to express
this kind of precision in programming logic.
Indeed, the authors of every scientific papers with empirical results of this type used some programmatic logic to

1. obtain their data,
#. unpack it into some working directory,
#. load it into the data structures of their programming language of choice,
#. convert those examples into the training, validation, and testing sets used for cross-validation, and
#. provide those training examples as input to some machine learning algorithm.

These steps are typically not formalized by authors of scientific papers as
reusable software. We conjecture that instead, the vast majority of researchers use web
browsers, hand-typed unix shell commands, and one-off private scripts to accomplish these steps.
This practice stands as an obstacle to reproducibility in machine learning,
computer vision, natural language processing, and other applications of
machine learning.

The skdata library consolidates these ugly little details of machine learning practice
and packages them as a library of reusable code [skdata]_.
It serves as both a gateway to access a growing list of standard public data sets,
and as a framework for expressing precise evaluation protocols that
correspond to standard ways of using those data sets.

This paper provides an overview of the problem the skdata library aims to
solve, a description of the project's architecture, some example usage of the
low level and high level interfaces to the library, and a listing of the data
sets currently provided by the library.


Data Sets
---------

There is nothing standard about data sets.
The nature of data sets varies widely, from physical measurements of flower petals ([Iris]_),
to pixel values of tiny images scraped from the internet ([CIFAR-10]_),
to the movie watching habits of NetFlix users ([Netflix]_).
Some data sets are tiny, and others are too large to store in RAM.
Different data sets are used to test different algorithms' ability to make statistical inferences,
and often a single data set may be used in several such ways.
This flexibility and un-defined-ness makes it challenging to design software
abstractions for data sets.

Data sets come from a range of sources, and can be public, private, or semi-public.
Data sets are provided by academics who have developed them for their own
research purposes, or by organizations that release a sample of their
internal data for scientific analysis.
Public datasets (available to anonymous requests) are the most popular in
research, but there are private data sets too, as well as ones that are free
for research purposes, but only available to those who ask (not available to anonymous requests).
The skdata library is suitable for all of these types of data set,
but it is most useful as a means of working with public data sets
because it can automate the downloading of the data set as well as the other
processing steps.

Public data sets are typically distributed via a web page
which describes the nature of the data set and
provides links to compressed archive files containing
the data set itself.
The data set itself may be just about anything, but some of the more popular data sets in machine learning and computer vision
include one or more of:

* Comma Separated Value (CSV) text files,
* XML documents (with custom internal structure),
* Text files with ad-hoc formatting,
* Collections of image, movies, audio files,
* Matlab workspaces, and
* Pickled NumPy ndarray objects.

Correctly interpreting meta-data can be tricky (error-prone) and simply
loading media collections that include files with non-homogeneous
formats, encoding types, sampling frequencies, and color spaces can be
awkward.

The skdata library provides logic for dealing with the ideosyncracies of data
sets so that they are accessible via familiar Python data structures such as
lists, dictionaries, and NumPy ndarrays.  The sordid details of parsing e.g.
ad-hoc text files and turning them into appropriate data structures is
encapsulated in the submodules of the skdata library.

XXX Discuss UCI, MLData, and other central repos of data sets: how does skdata
library relate to those?


Machine Learning: Problems and Protocols
----------------------------------------


Unlike the data sets themselves, which appear at a steady pace and which vary
endlessly in their nature and formatting, the set of *machine learning algorithms*
that people tend to apply to those data sets is much more stable,
and the set of *machine learning problems* for which those algorithms have been
formulated changes more slowly still.

For example, a great deal of machine learning research addresses
the *classification problem* of assigning an integer-valued *label* (:math:`y`) to some vector of binary- or
real-valued *features* (:math:`X`).
Many classification algorithms have been developed in the last few
decades, including Support Vector Machines (SVMs), Decision Trees (DTs), Naive Bayes Classifiers (NBs), Neural Networks (NNets), Nearest Neighbors (NNeighbs), and various other more general graphical models.
The reason that they are all called classification algorithms is that they can
implement a common mathematical interface.
We can see each one of these algorithms as fitting a predictive model
:math:`\cal M` to a
given *training set* of :math:`(X, y)` pairs, so that :math:`\cal M` can make
accurate label predictions for feature vectors that were not included in the
training set.

The organization of the sklearn library reflects this commonality of
interfaces [sklearn]_. Continuing with classification as our working example,
the sklearn library defines an "Estimator" interface for predictive models with a
"fit" method and a "predict" method.
The fit method expects two arguments: a matrix `X` whose rows are independent examples and
whose columns correspond to each input feature, and a vector `y` of integer
target labels for each row in `X`.
When the fit method of a predictive model is called, the model adapts itself
to *learn* the pattern of association between the rows of `X` and the values
of `y`.
The predict method requires just one argument: another matrix `X_test` whose
rows are examples and columns are features.
When the predict method is called, it returns the models best guesses of the
correct label for each row of `X_test`.


Machine learning algorithms for classification (also "classification
algorithms) are often judged on their
accuracy on a *test set* of examples that were not part of the *training set*.
This type of evaluation is called *cross-validation*. Evaluating a
classification algorithm by cross-validation proceeds as follows:

1. Load a data set.
#. Choose some examples for training.
#. Choose remaining examples for testing.
#. Fit the classification model to training data.
#. Predict labels for the test data.
#. Count fraction of correct predictions on test data.

The skdata library provides two kinds of service to help researchers step
through this standard pattern: the low level data-wrangling that loads the data set,
and a high-level description of the entire 6-step protocol
(including the partitioning of data into particular training and testing sets).
The skdata library therefore fills in important gaps around the scope of the
sklearn library: it addresses how to get data into the `X` and `y` numpy
arrays expected by the `fit()` and `predict()` methods of `Estimator`
subclasses and it provides formal description of how machine learning
algorithms should be used to obtain standard measures of generalization error
that can be compared with results in scientific literature.

The simple cross-validation protocol described above is standard for many data
sets, but it is not unusual for a data set to suggest or require a
variation.
For example, when algorithms are evaluated on small data sets, a more
efficient "K-fold" cross-validation is typically used.
When model selection is part of the training process, training sets
must be further subdivided into a test for fitting and a *validation* set
used for the internal model selection.
Some data sets (e.g. related to face-pair match verification and music
style labeling) have non-i.i.d.
(non-independently identically drawn) examples that cannot be arbitrarily
partitioned into training and testing sets.
The high level protocol layer of the skdata library has been designed
to help researchers respect these more detailed protocols.

Beyond classification, there are many other kinds of machine learning problem.
More general regression problems include the prediction of real-valued
variables and structured mathematical objects.
Density estimation is the problem of predicting the probability of events
like the ones in the training data.
Matrix completion problems arise in recommendation settings,
and many information retrieval tasks can be described more accurately as
ranking problems than classification or regression.
The list goes on.
There are relationships between these different problems
([LangfordReductions_] has pointed out that many of them can be reduced to
binary classification, in theory) but often the best algorithms are the
most specialized ones.

Certainly skdata's low level data interface provides a natural place to put
code for loading the data sets used to evaluate algorithms for these other
kinds of machine learning problem.
Currently skdata's high level data interfaces do not have special support
these other kinds of protocols.
As far as we know there is nothing about these kinds of learning problems that
makes them incompatible with the encapsulation techniques used in skdata, but
the design has not been pushed in this direction.


High Level: Protocol Layer Usage
--------------------------------

This section describes the high-level *protocol* layer provided by the skdata
library.
The protocol layer provides users with a direct way to evaluate a particular learning algorithm
on a particular data set.
We will first look at how to use the protocol layer, and then
we will look at how it works.

To begin, let's look at how to use skdata to evaluate an SVM
as a model for predicting Iris labels from the features in the [Iris]_ data.
Fortunately, sklearn has several great SVM wrappers, so all we
need to type is the following:

.. code-block:: python
    :linenos:

    from sklearn.svm import LinearSVC
    from skdata.base import SklearnClassifier
    from skdata.iris.view import SimpleCrossValidation

    # Allocate an standard evaluation protocol
    iris_view = SimpleCrossValidation()

    # Choose a learning algorithm constructor.
    # Configure a generic skdata<->sklearn wrapper
    estimator = LinearSVC
    algo = SklearnClassifier(estimator)

    # Step through the evaluation protocol
    test_error = iris_view.protocol(algo)

    # See what happened:
    for report in algo.results['best_model']:
        print report['train_name'], report['model']

    for report in algo.results['loss']:
        print report['task_name'], report['err_rate']

    print "TL;DR: average test error:", test_error

The next few paragraphs will go over this code line by line,
explaining what happened.

The first statement creates a *view* of the Iris data set.

.. code-block:: python
    :linenos:
    :linenostart: 6

    iris_view = SimpleCrossValidation()

The `SimpleCrossValidation` class uses Iris data set's low level
interface to prepare it for usage by sklearn,
but we do not see any of that work at this level.
Any arguments to configure the evaluation protocol itself would
have been passed to this constructor, but our simple demonstration
protocol does not have any parameters.

The next two statements create a learning algorithm.

.. code-block:: python
    :linenos:
    :linenostart: 10

    estimator = LinearSVC
    algo = SklearnClassifier(estimator)

The `estimator` is treated by the `SklearnClassifier`
object as a parameter-free constructor that creates a new model, ready to be
fit to data.
Any classifier that behaves like an sklearn classifier (i.e. has the expected
kind of `fit` and `predict` methods) can be used to configure an
`SklearnClassifier` object.
The `algo` object represents our experiment, in the sense that it
will keep track of the interactions between the `iris_view` protocol object
and the `estimator` classifier object.

All of the actual computation of the evaluation process
is carried out by the `protocol` method in line 14, and we can see
the results from that work in the loops on lines 17-21.

.. code-block:: python
    :linenos:
    :linenostart: 14

    test_error = iris_view.protocol(algo)

    # See what happened:
    for report in algo.results['best_model']:
        print report['train_name'], report['model']

    for report in algo.results['loss']:
        print report['task_name'], report['err_rate']

The `protocol` method encapsulates a sort of dialog
between the `iris_view` object as a driver,
and the `algo` as a handler of commands from the driver.
The protocol in question (`iris.view.SimpleCrossValidation`)
happens to use just two kinds of command:
* Learn the best model for training data
* Evaluate a model on testing data

The first kind of command produces an entry in the
`algo.results['best_model']` list.
The second kind of command produces an entry in the
`algo.results['loss']` list.

So after the protocol method has returned,
we can loop over these lists to obtain a summary of what happened during our
evaluation protocol.
(Some data sets offer this protocol as an iterator so that very long sequences
of commands can be aborted early.)

The `SklearnClassifier` class serves two roles:
(a) it is meant to illustrate how to create an adapter between an
existing implementation of a machine learning algorithm, and the various
data sets defined in the skdata library;
(b) it is used for unit-testing the protocol classes in the library.
Researchers are encouraged to implement their own adapter classes
following the example of the `SklearnClassifier` class (i.e. by cut & paste)
to measure the statistics they care about when handling the various
methods (e.g. best_model_vector_classification) and to save those
statistics to a convenient place. The practice of appending a summary
dictionary to the lists in self.results has proved to be useful for our work,
but it likely not the best technique for all scenarios.



How the Protocol Layer Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `SklearnClassifier` serves as an adapter between the


XXX



(including very lightweight description (similar in some ways to a domain-specific
language) of all 6 steps


In the skdata library, this series of steps for evaluating an algorithm is called
an evaluation *protocol*.


Low Level: Data Layer Usage
---------------------------

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b



Skdata Project Architecture
---------------------------

The skdata library aims to provide two levels of interface to data sets.
The lower level interface provides a "raw" view of the underlying data set.


Skdata consists primarily of independent submodules that deal with individual data sets.
Each submodule has three important sub-sub-module files:

1. a 'dataset' file with the nitty-gritty details of how to download, extract,
   and parse a particular data set;

2. a 'view' file with any standard evaluation protocols from relevant
   literature; and

3. a 'main' file with CLI entry points for e.g. downloading and visualizing
   the data set in question.


The evaluation protocols represent the logic that turns parsed (but potentially ideosyncratic) data into one or more standardized learning tasks.


The skdata library provides two levels of interfacing to each data set
that it provides.
A low level interface provides relatively *raw* access to the contents of
a data set, in terms that reflect what was (in most cases) downloaded from the web.
The goal of the low level interface is save users the trouble of unpacking
and parsing downloaded files, while giving them direct acces to the
downloaded content.

A high level ("protocol") interface provides a sanitized version of a data
set, in which examples have been assembled into e.g. (X, y) pairs,
standard preprocessing has been applied, and the examples have been
partitioned into standard training, validation, and testing splits, where
applicable. The goal of this high level interface is to allow algorithm
designers to simply "plug in" classification and feature transformation algorithms,
and rest assured that they have trained and tested on the right examples
which allow them to make direct comparisons in academic literature.

Skdata consists primarily of independent submodules that deal with individual data sets.
Each submodule has three important sub-sub-module files:

The basic approach has been developed over years of combined experience by the authors, and used extensively in recent work (e.g. [2]).
The presentation will cover the design of data set submodules, and the basic interactions between a learning algorithm and an evaluation protocol.


.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`



Cache directory
~~~~~~~~~~~~~~~

Various skdata utilities help to manage the data sets themselves, which are stored in the user's "~/.skdata" directory.


Current list of data sets
-------------------------

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+-------+
   | Material   | Units |
   +------------+-------+
   | Stone      | 3     |
   +------------+-------+
   | Water      | 12    |
   +------------+-------+

We show the different quantities of materials required in Table
:ref:`mtable`.



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

.. [CIFAR-10] XXX
.. [Iris] The Iris data set: http://archive.ics.uci.edu/ml/datasets/Iris
.. [skdata] XXX
.. [sklearn] XXX
.. [LangfordReductions] XXX
.. [Netflix] XXX
