:author: Henry Schreiner
:email: henryfs@princeton.edu
:institution: Princeton University

:author: Jim Pivarski
:email: jpivarski@gmail.com
:institution: Princeton University

:author: Hans Dembinski
:email: hans.dembinski@gmail.com 
:institution: TU Dortmund


-------------------------------------------------------
Boost-histogram: High-Performance Histograms as Objects
-------------------------------------------------------

.. class:: abstract

    Unlike arrays and tables, histograms in Python have usually been denied
    their own object, and have been represented as a single operation producing
    several arrays. Boost-histogram is a new Python library that provides
    Histograms that can be filled, manipulated, sliced, and projected as
    objects. Building on top of the Boost libraries' Histogram in C++14
    provided interesting and distribution and design challenges with useful
    solutions. This is meant to be a foundation that others can build on; in
    the `Scikit-HEP project`_, a physicist friendly front-end "Hist" and a
    conversion package "Aghast" are already being designed around
    boost-histogram.

.. class:: keywords

   histogram, analysis, data processing, data reduction

Introduction
------------

.. Why is it necissary to come up with a histogram library?

There are many histogram libraries for Python (at least 24), but they all fall short in one or more of these four key areas: Design, Flexibility, Speed, and Distribution. Numpy, for example, is available everywhere, but does not have a true histogram object. In the High Energy Physics (HEP) community, histogramming is vital to most of our analysis. As part of building tools in Python to provide a friendly and powerful alternative to the ROOT C++ analysis stack[^1], histogramming was targeted as an area that needed significant improvement. The histograms as objects is a general, powerful tool that can be utilized in many other disciplines, as well.

.. Why use Boost.Histogram?

About a year ago, a new C++14 library was released as part of the Boost C++ libraries version 1.70; it was a well designed stand-alone histogram package that did exactly what we wanted, but in C++14 rather than Python. We built boost-histogram for Python in close collaboration with the original Histogram for Boost author.

.. Retrain your brain on what a histogram "is"

A histogram can be viewed as a collection of 1 or more axes and a bin storage. The storage has an accumulator, which could be an integer, a float, or something more complicated, like a mean (AKA profile). An axis can be regularly spaced, variably spaced, or a category, either with integer or string labels. Regular spacing has a complexity benefit, which can be exploited with transforms, such as log spacing, power spacing, or with a custom C callback, such as produced by Numba's cfunc decorator. Histograms have the same API, whether they have one axis or 32.

.. Indexing (and more) shareable concepts

The powerful indexing proposal, called Unified Histogram Indexing (UHI), allows Numpy-like slicing and cross-library tag usage. This can be used to select items from axes, sum over axes, and slice as well, in either data or bin coordinates. One of the benefits of the axes based design is that selections that traditionally would have required multiple histograms now can simply be represented as an axes in a single histogram and then UHI is used to select the subset of interest.

.. It is fast, too

Performance was a key design goal; the library is already 2-10 times faster than Numpy, without multithreading or axis-aware optimizations. Multithreading can be used to further gain 2-4x in performance in some cases. Future work could provide special performance boosts for common axes combinations.

.. Building wheels (ideas, contributions, using cibuildwheel now/soon)

Building a Python library on a C++14 library provided several challenges. Distributing wheels was automated through Azure DevOps and supports all major platforms; some tricks were employed to make the latest compilers available. The Azure build-system is now used in at least three other Scikit-HEP projects. Conda-Forge is also supported. Binding was done with PyBind11, all Boost dependencies are included, so a compatible compiler is the only requirement for building if a binary is not available.

.. Conclusion and plans, Hist and more

In conclusion, boost-histogram provides a powerful abstraction for histograms as a collection of axes and storage. Filling and manipulating histograms is simple and natural, while being highly performant. In the future, we are building on this foundation and expect other libraries may want to build on this as well.


.. code-block:: python

   def sum(a: int, b: int) -> int:
       """Sum two numbers."""

       return a + b

.. _Scikit-HEP project: https://scikit-hep.org
