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

    Machine learning benchmark data sets come in all shapes and sizes,
    whereas classification algorithms assume sanitized input,
    such as (x, y) pairs with vector-valued input x and integer class label y.
    Researchers and practitioners know all too well how tedious it can be to
    get from the URL of a new data set to a NumPy ndarray suitable for e.g. pandas or sklearn.
    The skdata library handles that work for a growing number of benchmark data sets
    (small and large)
    so that one-off in-house scripts for downloading and parsing data sets can be replaced with library code that is reliable, community-tested, and documented.
    The skdata library also introduces an open-ended formalization of training and
    testing protocols that facilitates direct comparison with published
    research.
    This paper describes the usage and architecture of the skdata library.


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

At the same time, it is not always obvious what exactly the standard protocol is.
For example, the widely-used "Iris" data set is simply an enumeration of 150 specimens' petal and sepal measurements along with the label of which kind of iris each one is [Iris]_. 
If we are interested in matching the generalization error of our implementation to a generalization error in the literature, then we would like to know more than just the accuracy;
we would like to know exactly which examples were used for training, and which
examples were used for measuring that generalization error.
It would be tedious to write such detail into a paper (and to transcribe it back into code afterward!), but it is natural to express
this kind of precision in programming logic.
Indeed, the authors of every scientific paper with empirical results of this type used some programmatic logic to

1. Obtain their data,
#. Unpack it into some working directory,
#. Load it into the data structures of their programming language of choice,
#. Convert those examples into the training, validation, and testing sets used for cross-validation, and
#. Provide those training examples as input to some machine learning algorithm.

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
to pixel values of tiny public domain images ([CIFAR-10]_),
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

The skdata library provides logic for dealing with the idiosyncrasies of data
sets so that they are accessible via familiar Python data structures such as
lists, dictionaries, and NumPy ndarrays.  The sordid details of parsing e.g.
ad-hoc text files and turning them into appropriate data structures is
encapsulated in the submodules of the skdata library.

Relative to the well known UCI database [UCI]_, the sklearn library provides
logic for downloading and loading diverse data representations into more
standardized in-memory formats.
Relative to MLData (mldata.org) the sklearn library provides downloading and
loading logic, and a formal protocol for model selection and evaluation.


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
the sklearn library defines an ``Estimator`` interface for predictive models with ``fit`` and ``predict`` methods.
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


Machine learning algorithms for classification (or simply "classification
algorithms") are often judged on their
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
sklearn library: it addresses how to get data into the `X` and `y` NumPy
arrays expected by the `fit()` and `predict()` methods of `Estimator`
subclasses and it provides formal description of how machine learning
algorithms should be used to obtain standard measures of generalization error
that can be compared with results in scientific literature.

The simple cross-validation protocol described above is standard for many data
sets, but it is not unusual for a data set to suggest or require a
variation.
For example, when algorithms are evaluated on small data sets, a more
efficient *K-fold* cross-validation is typically used.
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

The `protocol` method encapsulates a sort of dialog between the `iris_view` object as a driver, and the `algo` as a handler of commands from the driver.
The protocol in question (`iris.view.SimpleCrossValidation`) happens to use just two kinds of command:

* Learn the best model for training data
* Evaluate a model on testing data

The first kind of command produces an entry in the `algo.results['best_model']` list.
The second kind of command produces an entry in the `algo.results['loss']` list.

After the protocol method has returned, we can loop over these lists to obtain a summary of what happened during our evaluation protocol.
(Some data sets offer this protocol as an iterator so that very long sequences of commands can be aborted early.)

The `SklearnClassifier` class serves two roles:
(a) it is meant to illustrate how to create an adapter between an existing implementation of a machine learning algorithm, and the various data sets defined in the skdata library;
(b) it is used for unit-testing the protocol classes in the library.
Researchers are encouraged to implement their own adapter classes
following the example of the `SklearnClassifier` class (e.g. by cut & paste)
to measure the statistics they care about when handling the various
methods (e.g. best_model_vector_classification) and to save those
statistics to a convenient place.
The practice of appending a summary dictionary to the lists in self.results has proved to be useful for our work, but it likely not the best technique for all scenarios.


How the Protocol Layer Works
----------------------------

The skdata library's protocol layer is built around a command-driven interface in which protocol objects (such as `iris.view.SimpleCrossValidation`)
walk a learning algorithm (e.g. `SklearnClassifier`) through the process of running an experiment.
In our example, the protocol object used two commands:

.. code-block:: python

    model = algo.best_model(task=training_data)
    err_rate = algo.loss(model, task=testing_data)

These commands involve arguments `training_data` and `testing_data` which are instances of a `Task` class, which we have not seen yet.
Before we go through the list of protocol commands in any more detail, it is important to understand what these Task objects are.


Task Objects: Protocol Layer Data Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The skdata.base file defines a class called `Task` that is used in all aspects of the protocol layer.
A `Task` instance represents a subsample from a data set.
In all settings so far, a Task instance represents *all* of the information about a *subset* of the examples in a data set
(although future protocols looking at e.g. user ratings data may define task semantics differently).
For example, in cross-validation the training set and the testing set would be represented by Task objects.
In a K-fold cross-validation setting, there would be 2K Task objects representing each of the training sets and each of the test sets
involved in the evaluation protocol.
Task objects may, in general, overlap in the examples they represent.

A `Task` class is simply a dictionary container with access to elements by object attribute,
but it has two required attributes: `name` and `semantics`.
The name is a string that uniquely identifies this Task among all tasks involved in a Protocol.
The semantics attribute is a string that identifies what kind of Task this is;
the identifiers we have used so far include:

* "vector_classification"
* "indexed_vector_classification"
* "indexed_image_classification"
* "image_match_indexed"

A task's semantics identifies (to the learning algorithm) which other attributes are present in the task object, and how they should be interpreted.
For example, if a task object has "vector_classification" semantics,
then it is expected to have (a) an ndarray `.x` attribute whose rows are examples and columns are features,
and (b) an ndarray vector `.y` attribute whose elements are the labels of the rows of `x`.
If a task object has "indexed_image_classification" semantics, then it is expected to have
(a) a sequence of RGBA image ndarrays in attribute `.all_images`,
(b) a corresponding sequence of labels `.all_labels`, and
(c) a sequence of integers `.idxs` that picks out the relevant items from `all_images` and `all_labels` as defined by NumPy's `take` function.


The Evaluation Protocol
~~~~~~~~~~~~~~~~~~~~~~~

The protocol objects (such as `iris.view.SimpleCrossValidation`) are responsible for fashioning their respective data sets (e.g. Iris) into Task objects
and passing these task objects as arguments to a relatively small number of possible learning commands:

best_model(task)
    Instruct a learning algorithm to find the best possible model for the given task, and return that model to the protocol driver.

loss(model, task)
    Instruct a learning algorithm to evaluate the given model for the given task. The returned value should be a floating point scalar,
    but the semantics of that scalar are defined by the semantics of the task.

forget_task(task)
    Instruct the learning algorithm to free any possible memory that has been used to cache computations related to this task,
    because the task will not be used again by the protocol.

retrain_classifier(model, task)
    Instruct the learning algorithm, to retrain only the classifier, and not repeat any internal model selection that has taken place.
    (This command will only be used by protocols that involve classification tasks!)


In our call above to `iris_view.protocol(algo)` what happened was that `iris_view` constructed two Task objects corresponding to the training and test sets,
and called

.. code-block:: python

    model = algo.best_model(train)
    err = algo.loss(model, test)
    return err

More elaborate protocols differ in constructing more task objects, and training and testing more models.

One of the strengths of using Python to glue these various components together is that very few things need to be carved in stone at the design phase.
Every data set has quirks, and there will be variations on the protocols we have used so far.
Certainly new semantics identifiers will be required to support a wider variety of machine learning applications.
For better or for worse, the protocol and the set of allowed semantics is not strictly defined anywhere;
"Adding a command to the protocol" is as simple as implementing and calling an unused attribute of the algo object passed to a protocol method.
Of course, if you add new commands to this protocol then you will not be able to use existing learning algorithms (e.g. `SklearnClassifier`).
Presumably though, you are adding a command because existing learning algorithms couldn't do what was necessary in the first place, so losing
compatibility is not a big loss.
A quick and dirty way to determine what semantics strings are in use is to apply a text search to the source tree (`grep -R semantics skdata`).
To see what protocol commands are supported by the SklearnClassifier,
look at its source definition in `skdata.base`.

.. The design of the protocol makes it natural to provide fallback implementations that allow more generic learning algorithms (e.g. SVC)
.. to serve in place of more specialized ones (e.g. image classification algorithms)

Dealing with Large Data
~~~~~~~~~~~~~~~~~~~~~~~

Some data sets are naturally large, and some datasets simply appear large by virtue of the way they are meant to be used
by experimental protocols.
Two techniques are used within the skdata library to keep memory usage under control.
The first technique is to use the "indexed" Task semantics to avoid
The second, related technique is to use the *lazy array* in `skdata.larray` to avoid allocating intermediate buffers for
certain kinds of transformations of original bulk data.

Indexed task semantics, such as "indexed_vector_classification" describe data subsets in terms of advanced NumPy indexing syntax
to reduce memory usage. NumPy's ndarrays are required to be layed out in a particular way in a computer's RAM,
so if we need to create many arbitrary subset views of an ndarray, it generally requires making many copies of that data.
Since the subsets involved in defining Tasks relative to a base set of examples only require manipulating set membership,
it is easier to leave the original base set of examples alone, and manipulate vectors of positions within that base set.
Making many Tasks simply means making many integer vectors that specify which examples are in which Task. These integer
vectors are much smaller than copies of the base set of examples would be, when the examples are associated with many features.

The *lazy array* described in `skdata.larray` makes it possible to evaluate certain transformations of ndarray data in an on-demand manner.
Lazy evaluation is done example by example, so if a protocol only requires the first 100 examples of a huge data set, then only those examples will be computed.
A lazy evaluation pipeline used together with appropriate cache techniques ensure that even when a data set is very large,
only those examples which are actually needed are loaded from disk and processed.
The lazy array does not make batch algorithms into online ones,
but it provides a mechanism for designing iterators
so that online algorithms can
traverse large numbers of examples in a cache-efficient way.

Low Level: Data Layer Usage
---------------------------

When the high level protocol layer does not suit your needs,
skdata also provides a lower-level interface that provides low level logic for each of the data sets in the library:

* Downloading
* Verifying archive integrity
* Decompressing
* Loading into Python

Whereas not all data sets have defined high-level protocol objects, all data sets define a low-level interface.
The high-level classes are implemented in terms of the low-level logic.

There is a convention that this low-level logic for each data (e.g. "foo") should be written in a Python file called "skdata.foo.dataset".
Technically, there is no requirement that the low-level routines adhere to any standard interface, because the skdata library has been
designed such that there are no functions that must work on any data set.
With that said, there are some common patterns, like downloading, deleting, and accessing whatever data a data set provides.
A data set wrapper for the Labeled Faces in the Wild (LFW) data set [lfw]_ provides a representative example of what low-level data set objects look like.
What follows is an abridged version of what appears in `skdata.lfw.dataset`.

.. code-block:: python

    """
    <Description of data set>
    <Citations to key publications>
    """

    url_to_data_file = ...
    sha1_of_data_file = ...

    class LFW(object):

        @property
        def home(self):
            """Return cache folder for this data set"""
            return os.path.join(
                skdata.data_home.get_data_home(),
                'lfw')

        def fetch(self, download_if_missing=True):
            """Return iff required data is in cache."""
            ...

        def clean_up(self):
            """Remove cached and downloaded files"""
            ...

        @property
        def meta(self):
            """Return data set meta-data as list of dicts"""
            ...

First, a dataset.py file includes a significant docstring describing the data set and providing some history / context regarding it's usage.
The docstring should always provide links to key publications that either introduced or used this data set.

When a public data set is free for download, the dataset file should include the URL of the original data,
and a checksum for verifying the correctness of downloaded data.

Most dataset files use the `skdata.data_home.get_data_home` mechanism to identify a local location for storing large files.
This location defaults to `.skdata/` but it can be set via a `$SKDATA_ROOT` environment variable.
In our code example, `LFW.home()` uses this mechanism to identify a location where it can store downloaded and decompressed data.

The `fetch` and `clean_up` methods download and delete the LFW data set, respectively.
The `fetch` method downloads, verifies the correctness-of, and decompresses the various files that make up the LFW data set.
It stores them all within the folder named by `LFW.home()`.
If `download_if_missing` is False, then `fetch` raises an exception if the data is not present.
The `clean_up` method recursively deletes the entire `LFW.home()` folder, erasing the downloaded data and all derived files.

The `meta` method parses a few text files and walks the directory structure within `LFW.home()` in order to provide a succint summary
of what images are present, what individual is in each image.

In the case of the LFW data set, an additional method called `parse_pairs_file` helps to parse some additional text files that describe
the train/test splits that the LFW authors recommend using for the development and evaluation of algorithms.
Generally, these low-level classes serve to support their corresponding high-level protocol objects (in e.g. `skdata.lfw.view`)


Command-Line Interface
----------------------

Some data sets also provide a `main.py` file that provides a command-line interface for certain operations, such as downloading and visualizing.
The LFW data set for example, has a simple main.py script that supports one command that downloads (if necessary) and visualzes
a particular variant of the LFW data set using [glumpy]_.

.. code-block:: python

    python -c skdata/lfw/main.py show funneled

Running a main.py file with no arguments should always print out a short description of usage,
but the files themselves are almost always very short and easy to read.


Current list of data sets
-------------------------

The skdata library currently provides some level of support for about 40 data sets.
The data sets marked with (*) provide the full set of low-level, high-level, and script interfaces described above.


Blobs
    Synthetic: isotropic Gaussian blobs

Boston
    Real-estate features and prices

Brodatz
    Texture images

CALTECH101
    Med-res Images of 101 types of object

CALTECH256
    Med-res Images of 256 types of object

CIFAR10 (*)
    Low-res images of 10 types of object

Convex
    Small images of convex and non-convex shapes

Digits
    Small images of hand-written digigs

Diabetes
    Small non-synthetic temporal binary classification

IICBU2008
    Benchark suite for biological image analysis

Iris (*)
    Features and labels of iris specimens

FourRegions
    Synthetic

Friedman{1, 2, 3}
    Synthetic

Labeled Faces in the Wild  (*)
    Face pair match verification

Linnerud
    Synthetic

LowRankMatrix
    Synthetic

Madelon
    Synthetic

MNIST (*)
    Small images of hand-written digigs

MNIST Background Images
    MNIST superimposed on natural images

MNIST Background Random
    MNIST superimposed on noise

MNIST Basic
    MNIST subset

MNIST Rotated
    MNIST digits rotated around

MNIST Rotated Background Images
    Rotated MNIST over natural images

MNIST Noise {1,2,3,4,5,6}
    MNIST with various amounts of noise

Randlin
    Synthetic

Rectangles
    Synthetic

Rectangles Images
    Synthetic

PascalVOC {2007, 2008, 2009, 2010, 2011}
    Labelled images from PascalVOC challenges

PosnerKeele (*)
    Dot pattern classification task

PubFig83
    Face identification

S Curve
    Synthetic

SampleImages
    Synthetic

SparseCodedSignal
    Synthetic

SparseUncorrelated
    Synthetic

SVHN (*)
    Street View House Numbers

Swiss Roll
    Synthetic dimensionality reduction test

Van Hateren Natural Images
    High-res natural images


Conclusions
-----------

Standard practice for handling data in machine learning and related research applications involves a significant amount of manual work.
The lack of formalization of data handling steps is a barrier to reproducible science in these domains.
The skdata library provides a host for both low-level data wrangling logic (downloading, decompressing, loading into Python) and high-level experimental protocols.
To date the development effort has focused on classification tasks, and image labeling problems in particular.
The abstractions used in the library should apply to natural language processing and audio information retrieval, as well as timeseries data.
The protocol layer of the skdata library (especially using the larray module) has been designed to accommodate large or infinite (virtual) data sets.
The library currently provides some degree of support for about 40 data sets, and about a dozen of those have full support for the high-level, low-level, and script APIs.



References
----------

.. [CIFAR-10] XXX
.. [Iris] The Iris data set: http://archive.ics.uci.edu/ml/datasets/Iris
.. [skdata] XXX
.. [lfw] XXX
.. [sklearn] XXX
.. [LangfordReductions] XXX
.. [Netflix] XXX
.. [glumpy] XXX
.. [UCI] http://archive.ics.uci.edu/ml/
