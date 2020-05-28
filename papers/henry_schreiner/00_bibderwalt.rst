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

In the High Energy Physics (HEP) community, histogramming is vital to most of our analysis. As part of building tools in Python to provide a friendly and powerful alternative to the ROOT C++ analysis stack[^1], histogramming was targeted as an area in the Python ecosystem that needed significant improvement. The "histograms are objects" mindset is a general, powerful way of interacting with histograms that can be utilized across disciplines. We have built boost-histogram in cooperation with the Boost C++ community for general use, and also have separate more specialized tools built on top of boost-histogram that customize it for HEP analysis (which will be relegated to a brief mention at the end of this document).

At the start of the project, there were many existing histogram libraries for Python (at least 24 were identified by the authors), but none of them fulfilled the requirements and expectations of users coming from custom C++ analysis tools. Four key areas were identified as key to a good library for creating histograms: Design, Flexibility, Speed, and Distribution.

Before we continue, a brief description of a histogram should suffice to set the stage until we describe boost-histogram's approach in more detail. A histogram reduces an arbitrarily large dataset into a finite set of bins. A histogram consists of one or more *axes* (sometimes called "binnings") that describe a conversion from *data coordinates* to *bin coordinates*. The data coordinates may be continuous or discrete (often called categories); the bin coordinates are always discrete. In NumPy, this conversion is internally derived from a combination of the ``bin`` and ``range`` arguments. Each *bin* in the histogram stores some sort of aggregate information (a simple sum in NumPy) for each value that falls into it via the axes conversion. Histograms often have an extra "weight" value that is available to this aggregate (a weighted sum in NumPy).

Almost as important as defining a histogram is limiting what a histogram is. Notice the missing item above: a histogram, in this definition, is not a plot. It is not a plot any more than a NumPy array is a plot. You can plot a Histogram, certainly, and custom plotting is useful (much as Pandas has custom plotting for Series), but that is not part of a core histogram library, and is not part of boost-histogram (though most tutorials how how to plot using matplotlib).

.. HIII: Make sure that the tense remains consistent here.

The first area identified was *Design*; here many popular libraries fell short. Histograms need to be represented as an object, rather than a collection of NumPy arrays, in order to naturally manipulate histograms after filling. You should be able to continue to fill a histogram after creating it; filling in one pass is not always possible due to memory limits or in live data taking. Once a histogram is filled, it should be possible to perform common operations on it, such as rebinning to a courser binning scheme, projecting on a subset of axes, selecting a subset of bins and working with or summing over just that piece, and more. You should be able to fill histograms and then sum them, such as from different threads. You also should be able to access the transform between data coordinates and bin coordinates for each axes. Axis should be able to store extra information, such as a title or label of some sort, to assist the user and plotting tools.

The second area identified is *Flexibility*; there are a wide range of things a histogram should be able to do. Axes should support several forms of binning: variable width binnings, regularly spaced binnings (a speed-optimized subset of variable binning), and categorical binning. Out-of-range bins (discussed later) are also key for enabling lossless sums over a partial collection of axes. Axes should also be able to optionally grow when a fill is out of range instead. The bins themselves should support simple sums, like NumPy, but should also support means (sometimes called profile histograms). High-precision weighted summing is also useful. Finally, if you add a sample parameter to the fill, you can also keep track of the error for the bin.

The third area identified is *Speed*; when dealing with very large datasets that will not fit in memory, the filling performance becomes critical. High performance filling is also useful in real-time applications. A highly performance histogram library should support fast filling with a compiled loop, it should avoid reverting to a slower $\mathcal{O}(n)$ lookup when filling a regularly spaced axes, and it should be able to take advantage of multiple cores when filling from a large dataset. NumPy, for example, does do well for a single regularly spaced axes since version 1.13, but it still does not optimize for two regularly spaced axes (an image is an example of a common regularly spaced 2D histogram).

The fourth and final area identified was *Distribution*. A great library is not useful if no one can install it; it is especially important that students and inexperienced users be able to install the histogramming package. This is one of Python's strengths compared to something like C++, but the above requirements necessitate compiled components, so this is important to get right. It also needed to work flawlessly in virtual environments and in the Conda package manager

.. Why use Boost.Histogram?

About a year ago, a new C++14 library was being proposed to the Boost C++ libraries called Boost.Histogram. It would later be unanimously accepted and released as part of the Boost C++ libraries version 1.70. It was a well designed header-only package that fulfilled exactly what we wanted, but in C++14 rather than Python. A proposal was made to get a full-featured Python binding developed as part of IRIS-HEP, an institute for sustainable software for HEP, as one of the foundations for a Python based software stack. We built boost-histogram for Python in close collaboration with the original Histogram for Boost author, Hans Dembinski, who had always intended Boost.Histogram to be accessible from Python. Due to this close collaboration, concepts and design closely mimic the spirit of the Boost counterpart.

The Design of a Histogram
-------------------------

.. include figure here


Let's revisit our description of a histogram, this time mapping boost-histogram components to each piece.

The components in a bin are the smallest atomic piece of boost-histogram, and are called *Accumulators*. Four such accumulators are available. ``Sum`` just provides a high-accuracy floating point sum using the Neumaier algorithm, and is automatically used for floating point histograms. ``WeightedSum`` provides an extra term to allow sample sizes to be given. ``Mean`` stores a mean instead of a sum, created what is sometimes called a "profile histogram". And ``WeightedMean`` adds an extra term allowing the user to provide samples. Accumulators are like a 0D or scalar histogram, much like dtypes are like 0D scalar arrays in NumPy.

The above accumulators are then provided in a container called a *Storage*, of which boost-histogram provides several. The available storages include choices for the four accumulators listed above (the storage using ``Sum`` is just called ``Double()``, and is the default; unlike the other accumulator-based storages it provides a simple NumPy array rather than a specialized record array when viewed). Other storages include ``Int64()``, which stores integers directly, ``AtomicInt64``, which stores atomic integers, so can be filled from different threads concurrently, and ``Unlimited()``. which is a special growing storage that starts at 8-bit integers and grows as needed, or even converts to doubles if filled with a weighted fill or scaled with a float.

The next piece of a histogram is an *Axis*. A ``Regular`` axis describes an evenly spaced binning with start and end points, and takes advantage of the simplicity of the transform to provide $\mathcal{O}(1)$ computational complexity. You can also provide a ``Transform`` for a Regular axes; this is a pair of C function pointers (possibly generated by Numba) that can apply a function to the transform, allowing for things like log-scale axes to be supported at the same sort of complexity as a Regular axis. Several common transforms are supplied, including log and power spacings. You can also supply a list of bin edges with a ``Variable`` axis. If you want discrete axes, ``Integer`` provides a slightly simpler version of a Regular axes, and ``IntCategory``/``StrCategory`` provide true non-continuous categorical axes for arbitrary integers or strings, respectively. Most axes have configurable end behaviors for when a value is encountered by a fill that is outside the range described by the axis, allowing underflow/overflow bins to be turned off, or replaced with growing bins. All axes also have a metadata slot that can store arbitrary Python objects for each axis; no special meaning is applied by boost-histogram, but these can be used for titles, units, or other information.

A ``Histogram`` is the combination of a storage and one or more axes. Histograms always reserve their own memory, though they provide a view of that storage to Python via the buffer protocol and NumPy. Histograms have the same API regardless of whether they have one axes or thirty-two, and they have a rich set of interactions defined, which will be the topic of the next section.


Interactions with a Histogram
-----------------------------

.. Indexing (and more) shareable concepts

The powerful indexing proposal, called Unified Histogram Indexing (UHI), allows Numpy-like slicing and cross-library tag usage. This can be used to select items from axes, sum over axes, and slice as well, in either data or bin coordinates. One of the benefits of the axes based design is that selections that traditionally would have required multiple histograms now can simply be represented as an axes in a single histogram and then UHI is used to select the subset of interest.

Performance when Filling
------------------------

.. It is fast, too

Performance was a key design goal; the library is already 2-10 times faster than Numpy, without multithreading or axis-aware optimizations. Multithreading can be used to further gain 2-4x in performance in some cases. Future work could provide special performance boosts for common axes combinations.


Distributing
------------

.. Building wheels (ideas, contributions, using cibuildwheel now/soon)

Building a Python library on a C++14 library provided several challenges. Distributing wheels was automated through Azure DevOps and supports all major platforms; some tricks were employed to make the latest compilers available. The Azure build-system is now used in at least three other Scikit-HEP projects. Conda-Forge is also supported. Binding was done with PyBind11, all Boost dependencies are included, so a compatible compiler is the only requirement for building if a binary is not available.

Conclusion and Plans
--------------------

.. Conclusion and plans, Hist and more

In conclusion, boost-histogram provides a powerful abstraction for histograms as a collection of axes and storage. Filling and manipulating histograms is simple and natural, while being highly performant. In the future, we are building on this foundation and expect other libraries may want to build on this as well.


.. code-block:: python

   def sum(a: int, b: int) -> int:
       """Sum two numbers."""

       return a + b

.. _Scikit-HEP project: https://scikit-hep.org
