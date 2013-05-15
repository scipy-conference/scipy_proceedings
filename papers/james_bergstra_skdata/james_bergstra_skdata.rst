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
1. unpack it into some working directory,
1. load it into the data structures of their programming language of choice,
1. convert those examples into the training, validation, and testing sets used
   for cross-validation, and
1. provide those training examples as input to some machine learning algorithm.

These steps are typically not formalized by authors of scientific papers as
reusable software. We conjecture that the vast majority of researchers use web
browsers and hand-typed unix shell commands to accomplish the first two steps in this
sequence. We conjecture that the third step (loading into language data structures) is
typically accomplished with a
"write-once" script, after which the data is saved in a derived
language-specific storage format so that the script isn't necessary anymore.
(Further speculating, we wager that this script is then typically either lost, or kept in a private stash of similar scripts.)
Finally, the logic that divides examples into specific training and testing
"splits" is often messy code interwoven with various debugging statements and
diagnostic logic specific to the learning algorithm under investigation. It is
considered to be private research code, and not meant for distribution.

The skdata library takes these ugly little details of machine learning practice
and consolidates them into a library of reusable code.
It serves as both a gateway to access a growing list of standard public data sets,
and a conceptual framework for expressing the exact evaluation protocols that
should be used for those data sets.



Introduction
------------

The number of conceptually different kinds of statistical inferences that dominate the attention of machine learning researchers is not so large.
Some of the standard kinds of inferences include:

* *classification* is the prediction of a discrete label from some number of real-valued or discrete-valued *feature* variables
* *regression* the prediction of real-valued responses from some number of real-valued or discrete-valued *feature* variables
* *density estimation*
* *matrix completion* often arises in user rating scenarios
* *ranking and structured prediction*

These inference categories correspond to industrially relevant applications, and algorithms for solving them have seen a lot of attention for decades.
Consequently they are supported by many mature inference algorithms (see e.g. [sklearn]_), even while research continues to develop new ones.
Other kinds of machine learning problems exist (see e.g. [Langford]_, [bigPDFmaybeByRoweis]_) but skdata has been developed foremost to support
the popular inference types mentioned above.

In contrast to the relatively short list of common inference types,
the number of data sets in active use for evaluating relevant inference algorithms is quite a lot larger and grows every year.
As computers get faster, storage gets cheaper, and research interests shift, we
see a steady stream of new public data sets for that focus researchers on
industrially or scientifically relevant challenges.

There is nothing standard about data sets.
The nature of data sets varies widely, from features of flower petals (Iris) to pixel values of tiny images scraped from the internet (CIFAR-10),
but even such widely disparate data sets can be interpreted as providing features (X) and labels (y) for a machine learning *classification problem*, and be treated

Data sets are typically distributed via public web pages.
A data set's web page describes the nature of the data set and provides links to compressed archive files containing
the data set itself.

The data set itself may be just about anything, but some of the more popular data sets in machine learning and computer vision
include one or more of:
* comma-separated-value text files exported from a spreadsheet program,
* XML documents (with custom internal structure),
* text files with ad-hoc formatting,
* nested directory structures of images (which may or may not have identical encodings, shapes, and colour spaces),
* collections of MPG movies,
* collections of audio files,
* matlab workspaces, and
* pickled NumPy ndarray objects.





There is a typical pattern in the workflow of researchers as they approach a machine learning challenge.
(I am summarizing my observations of many graduate students as well as my own working habits.)

What typically happens when a machine learning researcher, say a graduate
student, starts to work on a new project is that they:

1. start with an idea for a new algorithm

1. read papers on the subject and learn what the standard benchmarks
   are

1. implement their idea and want to compare it with the previous work

1. download a relevant data set and spend a day or two ensuring that they
   have understood it correctly

1. try to determine from the paper exactly which examples were used for
   training and testing (possibly failing to do so, and guessing)

1. download another data set and spend another day or two on that one
   
1. download a third data set (etc.)

In terms of this workflow, the skdata library helps with steps 4-7.
The skdata library
Relative


Support Vector Machines, Decision Trees, Naive Bayes, Neural Networks, Nearest Neighbors, various Graphical Models,



Cross-validation
* training
* validation
* testing

Kfold cross-validation


Sklearn: input is sanitized


Pre-processing



Even within particular kinds of data, such as the prediction of a real valued response from another real-valued control variable,
different algorithms generalize from data differently.
For example, a linear model predicts very different values most points than a higher-order polynomial,
even if both the linear model and the polynomial have been fit to the same few examples (called "training data").

Machine learning benchmark data sets come in all shapes and sizes.



a
In most cases, they are made available via publicly visible web page with some explanation of what the data set contains,
and with links to download tar-gzip'd archive

Many are hosted on university web pages (some created many years ago).

They 
The easiest ones to deal with come as comma-separated value (CSV) text files.

Some data

It is well known [Atr03]_ 

that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

Project Architecture
--------------------

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

Script Usage
------------

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Library Usage
-------------

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b


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
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


