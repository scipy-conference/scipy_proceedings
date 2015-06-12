:author: Jean-Luc R. Stevens
:email: jlstevens@ed.ac.uk
:institution: Institute for Adaptive and Neural Computation, University of Edinburgh

:author: Philipp Rudiger
:email: p.rudiger@ed.ac.uk
:institution: Institute for Adaptive and Neural Computation, University of Edinburgh

:author: James A. Bednar
:email: jbednar@inf.ed.ac.uk
:institution: Institute for Adaptive and Neural Computation, University of Edinburgh

.. 
   Was "Effective and reproducible research with HoloViews", but I
   don't think they would want us to change the title so much from the
   one in the conference schedule (otherwise how will people know this
   is the paper corresponding to the talk?), and I think the original
   title is better anyway.  It's not perfect, because it doesn't get
   across how HoloViews makes everyday research better, but neither
   does the other title (since "effective" doesn't mean anything).

--------------------------------------------------------------------------
HoloViews: Building Complex Visualizations Easily for Reproducible Science
--------------------------------------------------------------------------


..
      * Researchers in prefer interactivity over the edit-compile-run cycle when exploring their data. By nature, research involves exploring hypotheses and different ideas not all of which will work or be worth keeping. Rapid interactivity and feedback allows researchers to quickly explore ideas by trying out different approaches, keeping the ones that work and discarding the ones that don't.

      * REPLs have downside too: capturing a history of interactive commands has typically been very fragile and error-prone and often needs lots of post-editing, causing problems for reproducibility. In addition, REPLs have typically been text-based making them easy to work with simple literals (short strings, integers, floats) but nothing more complicated.

      * Together with IPython Notebook, HoloViews extends the idea of interactive exploration in a REPL to the common data-structures used to do research and publish papers. The notebook format improves the idea of a REPL by making it easy to build a sequence of commands while also supporting rich-display not traditionally supported by REPLs. Until now, complex visualizations have not integrated well with the REPL mode of exploration.

      * In addition to making regular research work more productive and more succinct, HoloViews adopts a declarative style whenever possible and separates concerns: data and semantic information is never mixed with options relating to the display of data. By being far more expressive and concise than traditional approaches, HoloViews makes it far easier to build truly reproducible scientific documents in IPython Notebook.


..


.. class:: abstract

   ..
      OLD ABSTRACT (full)

      Scientific visualization typically requires large amounts of custom
      coding that obscures the underlying principles of the work and makes
      it more difficult to reproduce the results.  Here we describe how the
      new HoloViews Python package, combined with the IPython Notebook,
      provides a rich interface for flexible and nearly code-free
      visualization of your results while storing a full record of the
      process for later reproduction.

      Visualization is one of the most serious bottlenecks in science and
      engineering research.  Highly specialized plotting code often
      outweighs the code implementing the underlying algorithms and data
      structures.  Over time, this inflexible, non-reusable visualization
      code accumulates, making it much more difficult to try new analyses
      and to document the procedure by which results have been turned
      into publication figures.  The result is that very few research
      projects are currently reproducible, even under a very loose
      definition of the term.

      The new HoloViews Python package is designed to make reproducible
      research happen almost as a byproduct of having a much more
      efficient workflow, with flexible visualization of your data at
      every stage of a project from initial exploration to final
      publication.  HoloViews provides a set of general-purpose data
      structures that allow you to pair your data with a small but
      crucial amount of metadata that indicates roughly how you want to
      view the data (e.g. as images, 3D surfaces, curves, etc.).  It
      also provides powerful containers that allow you to organize this
      data for analysis, embedding it whatever multidimensional
      continuous or discrete space best characterizes it.  For each of
      these data structures, there is corresponding (but completely
      separate) highly customizable visualization code that provides
      publication-quality plotting of the data, in any combination
      (alone, sampled, sliced, concatenated as subfigures in a
      complicated final figure, animated over time, etc.).  You can then
      easily and interactively explore your data, letting it display
      itself without providing further instructions except when you wish
      to change plotting options.

      Combined with the optional IPython Notebook interface, HoloViews
      lets you do nearly code-free exploration, analysis, and
      visualization of your data and results, which leads directly to an
      exportable recipe for reproducible research.  Try it!

   Scientific visualization typically requires large amounts of custom
   coding that obscures the underlying principles of the work and
   makes it difficult to reproduce the results.  Here we describe how
   the new HoloViews Python package, when combined with the IPython
   Notebook and Matplotlib, provides a rich, interactive interface for
   flexible and nearly code-free visualization of your results while
   storing a full record of the process for later reproduction.

   HoloViews provides a set of general-purpose data structures that
   allow you to pair your data with a small amount of metadata that
   acts as a declarative specification for how you want to view it.
   These data structures are then used by a separate plotting system
   to render your data interactively, e.g. within the IPython Notebook
   environment, revealing even complex data in publication-quality
   form without requiring custom plotting code for each figure.

   HoloViews also provides powerful containers that allow you to organize
   this data for analysis, embedding it whatever multidimensional
   continuous or discrete space best characterizes it. The resulting
   workflow allows you to focus on exploring, analyzing, and
   understanding your data and results, while leading directly to an
   exportable recipe for reproducible research.

.. class:: keywords

   reproducible, interactive, visualization, notebook

Introduction
------------

Scientific research alternates between stretches of speculative,
exploratory investigation and periods where crucial findings are
distilled and disseminated as publications or reports. The exploratory
phase typically involves interactive plotting and analysis tools, with
many possible views and analyses tested as rapidly as possible before
deciding on the specific aspects of the data that are most important.
Once the data is understood, the results are then typically prepared
as static, non-interactive figures for dissemination, often putting
together many subfigures into a complicated figure that reveals
multiple interrelated aspects of the results.

Current software tools provide relatively poor support for this dual
exploring/reporting nature of scientific research, severely
limiting scientific progress.  On the one hand, developing new
exploratory visualizations typically requires large amounts of custom
software coding, which is slow, error-prone, and distracts from the
actual scientific analysis.  Moreover, this process typically involves
a large amount of trial and error, generating voluminous logs of
transitory code, analyses, and results, making it difficult to later
reproduce the steps that led to any particular result [Repro]_.
Switching to different tools for final, non-interactive,
publication-quality figures exacerbates this problem, further
disconnecting the reported results from the process by which they were
created.  This lack of reproducibility is a serious handicap both for
progress within a single lab and for the community as a whole, making
it nearly impossible for researchers to build on each others' work
even for purely computational projects [Repro]_.

In this paper, we present a new Python software package built to address
these problems directly and definitively, by providing simple tools
for gradually building more elaborate visualizations and analyses
interactively yet reproducibly.  HoloViews supports immediate
exploration of data as it is obtained, without requiring custom
coding, while allowing even complex relationships to be revealed
between datasets in a smooth progression from initial exploration to
final publication of fully reproducible scientific results.  As
outlined in detail below, HoloViews achieves these goals by enforcing
a strict separation between the semantic properties of the data
(expressed declaratively and independently of any visualization),
specification of plotting options (expressed declaratively and
independently of the underlying code), and implementation of specific
visualizations (expressed as general-purpose code reused and curated
over time).


The interactive interpreter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand this approach, we need to briefly consider the history
of how we interact with computational data. The idea of an interactive
programming session originates with the earliest LISP interpreters in
the late 1950s. Since then, high-level programming languages have
become even more dynamic in nature. In recent years, the Python
language has been widely adopted by researchers due to its concise,
readable syntax. Python is well suited to dynamic interaction and
offers an interactive, textual interpreter.

However, like most such command prompts, the standard Python prompt is
a text-only environment. Commands are entered by the user, parsed, and
executed, with results returned back to the user as text.  This
approach offers immediate feedback and works well for data that is
naturally expressed in a concise textual form. Unfortunately, this
approach begins to fail when the data cannot be usefully visualized as
text, as is typical for the large datasets now commonplace.  In such
instances, a separate plotting package 
offering a rich graphical display would normally be used to present
the results outside the environment of the interpreter, controlled
either by custom computer code or through an interactive graphical
user interface (GUI).

This disjointed approach is a reflection of history: text-only
environments, where interactive interpreters were first employed,
appeared long before any rich graphical interfaces and GUI
environments. To this day, text-only interpreters are standard due to
the relative simplicity of working with text. Proprietary
attempts to overcome these limitations, such as the Mathematica
Notebook [Wol03]_, have remained constrained by limited
interoperability and a lack of standardized open formats.  Other
approaches focusing explicitly on reproducibility involve building a
recipe for reproducing results only at the end of the scientific
project [knitr]_, when it is often too late to capture the important
steps involved.  Here we consider how graphical output can be
integrated fully into an interactive workflow, addressing both
exploration and reproducibility simultaneously.


Fixing the disconnect between data and representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the same time as text-based interpreters have failed to overcome the inherent
limitations of working with rich data, the web browser has emerged as
a ubiquitous means of interactively working with rich media
documents. In addition to being universally available, web browsers
have the benefit of being supported by open standards that remain
supported almost indefinitely. Although early versions of the HTML
standard only supported passive page viewing, the widespread adoption
of HTML5 (and websockets) has made it possible for anyone to engage
with complex, dynamic documents in a bi-directional, interactive
manner.

The emergence of the web browser as a platform has been exploited by
the Python community and the scientific community at large with the
development of tools such as the IPython Notebook [Per07]_ and SAGE
MathCloud [Ste05]_. These projects offer interactive computation sessions
in a notebook format instead of a traditional text prompt. Although
similar in design to the traditional text-only interpreters, these
notebooks allow embedded graphics or other media (such as video) while
maintaining a permanent record of useful commands in a rich document
that supports the gradual development of a document with interleaved
code and exposition. 

Yet despite the greatly improved capabilities of these tools for
computational interaction, the spirit of the interactive interpreter
has not been restored: there is an ongoing disconnect between data and
its representation. This artificial distinction is a lingering
consequence of a text-only world and has resulted in a strict split
between how we conceptualize 'simple' and 'complex' data. Simple data
returned from an operation is rendered directly in the interpreter (as
strings, small arrays, etc.), but more complex data requires an
extremely detailed set of explicit steps using external plotting
packages such as Matplotlib [Mpl]_ before results can be revealed
in a graphical representation.

Here we introduce HoloViews, a library of simple classes designed to
provide an immediately available representation for even complex data
in notebooks, making such data be as easy to work with as simple
data always has been in interactive sessions.  HoloViews is not a
plotting package; instead, it offers a set of useful data structures
paired with rich, customizable visual representations that display
effortlessly in the IPython Notebook environment. The result is
research that is more interactive, concise, declarative, and
reproducible. Figure :ref:`layout` shows a self-contained example 
of building a complex visualization using a single line of HoloViews
code (or, more precisely, a single line of declarative specification).

..
  jbednar: I strongly believe in showing the full code.  
  Where does "data" come from in cell 1 of a notebook otherwise?  
  Please add a line "data = np.load('mandelbrot.npy')", 
  and consider adding "load_ext holoviews.ipython"

.. figure:: introductory_layout_example.png
   :scale: 25%
   :align: center
   :figclass: w

   Example of a composite HoloViews data structure and how it is
   displayed in an IPython Notebook session. The array ``data`` is a
   400x400 Numpy array corresponding to a rendering of part of the
   Mandelbrot set. **A.** The ``Raster`` element displays the ``data``
   overlaid via the ``*`` operator with a horizontal line (an
   ``HLine`` element). The ``Histogram`` element constructed with
   ``.hist()`` displays the distribution of values in the data
   array. **B.** A ``Curve`` element showing the values across the
   middle of the ``Raster`` image as indicated by the blue horizontal
   line. The curve is concatenated with the ``Overlay`` in **A** via the
   ``+`` operation. :label:`layout`


..
   Emphasize the importance of reproducibility more? I had this:

   Although this has increased the speed of exploration, this has come
   at the cost of reproducibility, a cornerstone of the scientific
   method. In some fields, the lack of reproducibility is a major
   problem, making it clear that there is still much scope for
   improving the ways in which we use computers to do research.

..
   Principles:

   * Declarative (user API, param for developers)

   * Separating visualization/elements

   * Composable semantics (as a DB or as visualization).

   *  Associating sufficient semantic metadata to the element that sensible, immediate plotting is possible by default.

   *  Declare semantic relationships between elements, e.g shared dimensions across different element. 

   * Work in the natural dimensions of your data i.e the real-world continuous space instead of directly worrying about samples.

   *  Raw data must always be accessible no matter how nested the data is.


Design principles
-----------------

The design principles of HoloViews are an extension of the basic
principles underlying all interactive interpreters, generalized to
handle the more complex data structures commonly used in research. The
goal is to *automatically* and *transparently* return and display
declarative data structures to the user for immediate feedback without
requiring additional code. Although this concept is familiar and
intuitive when interactively working with simple data types, it is
worth reviewing explicitly what is going on so that the appropriate
graphical extension of these ideas is clear.

When executing an addition operation like ``1 + 2.5`` at a Python
prompt, the expression is parsed, converted into bytecode, and then
executed, resulting in the float value ``3.5``. This floating-point
value is immediately returned to the user in the appropriate
displayable representation, giving the user immediate feedback. Of
course, the string representation is not the float itself, but a
sequence of characters ``'3.5'``. Such strings are automatically
generated by the interpreter, via the display object's ``__repr__``
method.

This automatic, immediate feedback also exists in the interpreter for
more complex data types such as large Numpy arrays, but the displayed
string has very little utility because it is either incomplete
(presenting only a few values of the array) or impractical (presenting
an impossible-to-interpret dump of all the values).  In a terminal,
this restriction is a result of the ``__repr__`` method only
supporting a text-based display value. Using HoloViews in the IPython
Notebook, you can give your array a more useful, interpretable visual
representation as an image, curve, or similar plot according to the
following principles:

* It must be easy to assign a useful and understandable default
  representation to your data. The goal is to keep the initial barrier
  to productivity as low as possible -- data should simply reveal itself.
* These atomic data objects (elements) should be almost trivially simple
  wrappers around your data, acting as proxies for the contained
  arrays along with a small amount of semantic metadata (such as whether
  the user thinks of the data as a curve or as a set of points).
* Any metadata included in the element must address issues of *content*
  and not be concerned with *display* issues --  elements should
  hold essential information only.
* There are always numerous aesthetic alternatives associated with
  rich visual representations, but such option settings should be
  stored and implemented entirely separately from the elements, so
  that elements can be generated, archived, and distributed without
  any dependencies on the visualization code.
* As the principles above force the atomic elements to be simple, they
  must then be *compositional* in order to build complex data
  structures that reflect the interrelated plots typical of
  publication figures.

The outcome of these principles is a set of compositional data
structures that contain only the essential information underlying
potentially complex, publication quality figures. These data
structures have an understandable, default visualization that
transparently reveals their contents, making them a useful proxy for
the data itself, just as the text ``3.5`` is a proxy for the
underlying floating-point value.  This default visualization may then
be customized declaratively to achieve the desired aesthetics, without
needing to store these customizations on the objects themselves. This
separation of content and presentation is already a well established
design principle outside of science, and is analogous to the
relationship between HTML content and CSS.


Data Structures
---------------

In this section we discuss the data structures that hold the data and
the essential semantic content of interest. The Elements section
introduces each of the primitives, and the Collections section
explains how they can be combined. Finally, we will discuss working
with Elements embedded in high-dimensional continuous or discrete spaces.


Elements
~~~~~~~~

..
   Call these the Element primitives?

The atomic classes that wrap raw data are the ``Element``
primitives. These classes are named by the natural representation they
suggest for the supplied data, ``Image``, ``Curve`` and ``Scatter``
being some simple examples. These elements are easily constructed as
they only require the raw data (such as a Numpy array) to display.

In Figure :ref:`layout`, we have some examples of the Element
primitives. On the left, in subfigure **A**, we see the ``Raster``
primitive containing a two-dimensional Numpy array. This ``Raster``
was simply declared as ``Raster(data)`` and the corresponding,
automatically generated visual representation of this object shows
that the array is a part of the Mandelbrot set. Our object merely
holds the supplied Numpy array which may be easily accessed via the
``.data`` attribute. In part **B** of Figure :ref:`layout` we have an
example of a ``Curve`` containing a cross section of the
two-dimensional array.

..
   For instance, executing ``c=Curve(range(10))`` will build a simple
   ``Curve`` object and assigned it to the variable ``c``. If in the
   IPython notebook, we look at the value of the object ``c``, we will
   see that the object ``c`` has a rich representation given by a linear
   plot of our supplied *y*-values over the implicit *x*-axis.

Although the names of the ``Elements`` suggest that these objects are
about visualization, they are primarily concerned with content and *not*
display. The visually meaningful class names offer a convenient way to
intuitively understand the dimensionality of the data in terms of an
appropriate visual representation. For instance, in Figure
:ref:`layout` **A**, the name ``Raster`` conveys the notion that the
contained data is in the form of a two-dimensional Numpy array.

The particular ``Raster`` shown in Figure :ref:`layout` **A** is
declared in the simplest possible fashion, allowing the two dimensions
to default to *x* along the x-axis and *y* along the y-axis. This is
fine for describing the visual space but if you wanted to make it
clear that the Mandelbrot is actually computed over the complex plane,
you can associate this semantic information with the array using the
declaration ``hv.Raster(data, kdims=['Re','Im'])``. Similarly, for the
cross-section, we could supply ``kdims=['Re']`` and
``vdims=['Intensity']`` to the ``Curve`` constructor.

Although our dimension labels are used to update the visual output by
setting the axis labels appropriately, this information is semantic
content concerning the dimensionality of the data. This information
exists regardless of the exact visual representation used: for
instance, you can pass the curve with the improved dimension labels
directly to the constructor of the ``Scatter`` or ``Histogram``
element and the dimension labels will be preserved. This type of
operation merely changes the default representation associated with
the supplied data.

In our improved declarations of ``Raster`` and ``Curve``, we declare
two types of dimension: the key dimensions (``kdims``) and the value
dimensions (``vdims``). The key dimension correspond to the
independent dimensions used to index or slice the element whereas the
value dimensions corresponds to dependent dimension. For instance, you
can slice the Mandelbrot element to get a new ``Raster`` element
containing a portion of the original Numpy array by applying the slice
ranges ``[100:200, 0:100]``. This slices the first key dimension (the
real 'Re' axis) from index 100 to 200 and the second key dimension
(the imaginary 'Im' axis) from index 0 to 100. Similarly, you apply
the slice ``[100:200]`` to the cross-section to get a new ``Curve``
containing the profile from index 100 to 200 along the real axis.

..
  Add something about providing an extensible library of Elements as
  primitives to compose complex plots.

To summarize, there are many available element classes as there are
many common visual representations for data. Within the set of all
elements, there are subsets forming equivalence classes according to
the allowed number of key and value dimensions. Within these groups,
you can easily cast your data between equivalent representations
provided that the number and type of dimensions remain unchanged. You
can then index and slice your elements along their respective key
dimensions to get new elements holding the appropriately sliced
data.

..
   From a Curve object, only conversion to Scatter works!!  Should be
   able to also do Histogram(curve) or Bars(curve) as the number of
   key/value dimensions match.


Collections
~~~~~~~~~~~

..
   Place holder for Design Principles introduction.

The elements are simple wrappers that hold the supplied data with a
rich, meaningful representation. An individual element is therefore a
data structure holding the semantic contents corresponding to a simple
visual element of the sort you may see in a publication. Although the
elements are sufficient to cover simple cases such as individual
graphs, raster images or histogram, they are not sufficient to
represent more complex figures.

A typical figure does not present data using a single representation
but allows comparison between data or order to illustrate similarities
or differences between different aspects of the data. In other words,
a typical figure is a single object composed of many visual
representations combined together. HoloViews makes it trivial to
compose elements in the two most common ways: concatenating
representations side-by-side into a single figure or overlaying visual
elements within the same set of axes.

These types of composition are so common that both have already been
used in Figure :ref:`layout` as our very first example. The ``+``
operation implements the first type of composition of concatenation
and ``*`` implements the act of overlaying elements together. When we
refer to subfigures :ref:`layout` **A** and :ref:`layout` **B**, we
are making use of labels generated by HoloViews when representing a
composite data structure called a ``Layout``. Similarly, subfigure
:ref:`layout` **A** is itself a composite data structure called an
``Overlay`` which, in this particular case, consists of a ``Raster``
element overlaid by the ``HLine`` element.

The overall data structure that corresponds to Figure :ref:`layout` is
therefore a ``Layout`` which itself contains another composite
collection in the form of an ``Overlay``. This object is in fact a
highly flexible, compositional tree-based data structure: intermediate
nodes correspond either to ``Layout`` nodes (``+``) or ``Overlay``
nodes (``*``), with element primitives at the leaf nodes. All the raw
data corresponding to every visual element is conveniently accessible
via key or attribute access on the tree by selecting a leaf element
using its path through the tree, and then inspecting the ``.data``
attribute, making it simple to decalare which part of a complex
dataset you want to work with.

..
  jbednar: probably most people won't be able to figure out the
  following description; can we improve it and make it concrete?

As the elements of the tree may be of heterogeneous types there needs
to be an automatic, easy and universal way to select either leaf
elements or subtrees in a way that works across all allowable leaf
nodes.  Such specification is achieved in HoloViews by semantic "group"
and "label" strings that may be explicitly specified in the constructor
to any primitives (otherwise appropriate defaults are used). Using
these two identifiers, the ``+`` and ``*`` operators are able to
generate trees with a useful two-level indexing system by default.

With the ability to overlay or concatenate any element with any other,
there is great flexibility to define complex relationships between
elements. Whereas a single element primitive holds semantic
information about a particular piece of data, trees encode semantic
information between elements. The composition of visual elements into
a single visual representation expresses some underlying semantic
value in grouping these particular chunks of data together. This is
what composite trees capture; they represent the overall *semantic
content* of a figure in a highly composable and flexible way that
always preserves both the raw data and associated metadata for further
interactive analysis and reproduction.


Spaces
~~~~~~

..
   The visual representation of data faces two main bottlenecks, (1) our
   perceptual limitations and (2) the limits forced on us by the flat 2D
   media we use to display it.

A single plot can represent at most a few dimensions before it becomes
visually cluttered. Since real-world datasets often have higher
dimensionality, we face a tradeoff between representing the full
dimensionality of our data, and keeping the visual representation
intelligible and therefore effective. In practice we are limited to
two or at most three spatial axes, in addition to attributes such as
the color, angle and size of the visual elements. To effectively
explore higher dimensional spaces we therefore have to find other
solutions.

One way of dealing with this problem is to lay out multiple plots
spatially.  Some plotting packages [Was14]_ [Wic09]_ have shown how
this can be done easily using various grid-based layouts. Another
solution is to introduce a time dimension, animating values over
time.  A third solution is to provide interactive control, allowing
the user to reveal further dimensionality by interacting with the plots.

In HoloViews, we solve this problem with composable data structures
that embed homogeneous collections of ``Element`` objects in any
arbitrarily dimensioned space. Fundamentally, this set of data
structures (subclasses of ``NdMapping``) are multi-dimensional
dictionaries that allow the user to declare the dimensionality of the
space via a list of key dimensions (``kdims``). The multi-dimensional
location of the items held by the dictionary are defined by tuples,
where the values in the tuple matches the declared key dimension by
position. In addition to regular Python dictionary indexing semantics,
these data structures also support slicing semantics to select
precisely the subregion of the continuous multi-dimensional space that
the user wants to explore.

..
   Explain what key dimensions mean for spaces and what it means to be
   'dimensioned'.


The full list of currently supported ``NdMapping`` classes includes:

* ``HoloMaps``: The most flexible high-dimensional data structure in
  HoloViews, allowing ``Element`` instances to be embedded in an
  arbitrarily high-dimensional space, to be rendered either as a
  video animation or as an interactive plot that allows
  exploration via a set of widgets.

* ``GridSpaces``: A data structure for generating spatial layouts
  with either a single row (1D) or a two-dimensional grid. Each
  overall grid axis corresponds to a key dimension.

* ``NdLayouts``/``NdOverlays``: Unlike ``Layout`` or ``Overlay``
  objects, these spaces only support homogeneous sets of elements, but
  allow you to define the various dimensions over which these items
  vary.

All of the above classes are simply different ways to package and view a
high-dimensional dataset. Just as with ``Elements``, it is possible to
cast between these different spaces via the constructor (although
``GridSpace`` is restricted to a maximum of two dimensions). In
addition, they can all be tabularized into a HoloViews ``Table``
element or a pandas ``DataFrame`` [pandas]_, a feature that is also supported by
the ``Element`` primitives.


To get a sense of how composing data and generating complex figures
works within this framework, we explore some artificial data in Figure
:ref:`spaces`. Here we will vary the frequency and amplitude of sine
and cosine waves, demonstrating how we can quickly embed this data
into a high-dimensional space. The first thing we have to do is to
declare the dimensions of the space we want to explore as the key
dimensions (``kdims``) of the HoloMap. Next we populate the space
iterating over the frequencies, amplitudes, and the two trigonometric
functions, generating each ``Curve`` element individually and
assigning to the HoloMap at the correct position in the declared
multi-dimensional space.

..
  jbednar: it's embarrassing that the subfigure label is missing in A.  
  Maybe suppress both such labels to avoid the problem?  But then
  it's hard to refer to A and B in the caption.
  Should 'xs' be 'ps' or even 'phases', for clarity?
  Isn't even A a HoloMap, covering cos and sin?

.. figure:: spaces_example.png
   :scale: 30%
   :align: center
   :figclass: w

   Example of a HoloViews Spaces object being visualized in two
   different ways. On the left is a ``GridSpace`` providing a condensed
   representation of Curve Elements across 'Frequency' and
   'Amplitude', for the function selected at the right.  In the
   middle (labelled **B**) is one frame of a HoloMap, selected using
   the other sliders, which appear automatically because at least one
   item being selected has more dimensions than can be shown on the 
   screen at a given time.  In this way HoloViews can represent
   arbitrarily high-dimensional data naturally and conveniently.
   :label:`spaces`

We can immediately go ahead and display this HoloMap either as an
animation or using the default widgets. Visualizing individual curves
in isolation is not very useful, of course; instead we probably want
to see how the curves vary across ``Frequency`` and ``Amplitude`` in a
single plot. A ``GridSpace`` provides such a representation and by
using of the space conversion method ``.grid`` we can easily transform
our three-dimensional HoloMap into a two-dimensional GridSpace (which
then allows the remaining dimension (the choice of trigonometric
function) to be varied via the drop-down menu). Finally, after
composing a ``Layout`` together with the original ``HoloMap``, we let
the display system handle the plotting and rendering.

If we decide that a different representation of the data would be more
appropriate, it is trivial to rearrange the dimensions without needing
to write new plotting code. Even very high-dimensional spaces can be
condensed into an individual plot or expressed as an interactive plot
or animation, by simply specifying which part of the data we are
interested in rather than writing new brittle and error-prone custom
plotting code.


Customizing the visual representation
-------------------------------------

In this section we show how HoloViews achieves a total separation of
concerns, keeping the composable data structures introduced above
completely separate from both customization options and the plotting
code. This design is much like the separation of content and
presentation in markup languages such as HTML and CSS, and provides
the same benefits of making the content easily maintainable while the
presentation is easily controllable.

The only connection between the above data structures and the custom
display options is a single, automatically managed integer
identification attribute stored with the data structure.  Using this
ID as a key, we can make the data structures behave as if they were
rich, stateful and customizable objects, without actually storing
anything to do with visualization on the objects. We will show how
this separation is useful and extensible so that the user can quickly
and easily customize almost every aspect of their plot. For instance,
it is easy to change the font size of text, change the subfigure label
format, change the output format (e.g switch from PNG to SVG) and even
alter the plotting backend (currently defaulting to Matplotlib)
without changing anything about the object that is being rendered.

.. HoloViews is enabled by IPython display hooks automatically linking the displayed object type to the code that generates its visual representation.

.. 
   jbednar: I think people will have a hard time understanding what 
   the trees, group, label, etc. are in this section; it deserves
   some thought and rewriting.

The connection between the data structure and the rendered
representation is made according to the object type, the
aforementioned ID integer, and optionally specified group and label
strings. The declarative data structures define what will be plotted,
specifying the arrangements of the plots, e.g. grids, layouts and
overlays, which can then be customized via the options system to tweak
aesthetic details such as tick marks, colors and normalization
options. Finally, the plotting and rendering process occurs
automatically in the background so that the user never needs to worry
about it.

The default display options are held on a global tree structure
similar in structure to the composite trees described in the previous
section, but with nodes now holding custom display options in the
form of arbitrary collections of keywords. In fact, these option trees
also use labels and groups the same way as composite trees except they
additionally support type-specific customization. For instance, you
may specify colormap options on the ``Image`` node of the tree that
will then be applied to all ``Images``. If this chosen colormap is not
always suitable, you can ensure that all ``Image`` elements belonging
to a group (e.g ``group='Fractal'``) make use of a different colormap
by overriding it on the ``Image.Fractal`` node of the tree.

This global default tree is held on the ``Store`` object which can
also hold display settings per object instance via the integer ID
attribute. This provides a highly flexible styling system, allowing
the user to specify display options that apply to all objects of a
particular type or only specific subsets of them. For instance, it is
easy to select a particular colormap that only applies to a specific
object.

A major benefit of separating data and customization options in this
way is that all the options can be gathered in one place. There's no
longer any need to dig deep into the documentation of a particular
plotting package for a particular option as all the options are easily
accessible via a tab-completable IPython magic and are documented via
the ``help`` function. This ease of discovery once again enables a
workflow where the visualization details of a plot can be easily and
quickly iteratively refined after they have determined that some data
is of interest.

This options system is inherently extendable.  New options may be
added at any time, and will immediately become available for
tab-completion. In fact, the plotting code for each Element and
container type may be switched out completely and independently, and
the options system will automatically reflect the changes in the
available customization options. This approach lets the user work with
a variety of plotting backends at the same time, without even having to
worry about the different plotting APIs.

Figure :ref:`schematic` provides an overall summary of how the
different types of components interact. The user first defines the
data as elements and containers.  When the data needs to be displayed,
the rendering system looks up the appropriate plot type for the object
in a global registry, which then processes the object in order to
display it with the applicable display options. Once the plotting
backend has generated the plot instance (as e.g. a Matplotlib figure),
it is converted to an appropriate format for embedding into HTML for
display in the notebook.

..
  jbednar: This figure seems too crammed, with very tiny font size.  I
  think the schematic at left needs to be in a different figure from
  the stuff at right, so that we can understand them.  The figure at
  the left should either be simplified or explained more fully, or
  both.
  
.. figure:: display_system.pdf
   :scale: 25%
   :align: center
   :figclass: w

   Diagram of the HoloViews display and customization system,
   highlighting the complete separation between the actual displayed
   content, the customization options and the plotting and rendering
   system. :label:`schematic`

At no point does the user have to worry about the intermediate
rendering step. We can see this directly if we look at the example in
Figure :ref:`schematic`, which is a customized version of Figure
:ref:`layout`. Using the ``%%opts`` magic we have specified various
display attributes about the plot including aspects, line widths, the
``cmap`` and the ``sublabel_format``. By printing the string
representation of the content and the options separately we can see
immediately how there are two distinct objects, and also how they
correspond, with each entry in the ``OptionsTree`` matching an
applicable object type. Finally, in the output section of Figure
:ref:`schematic` we can see how these options have resulted in the
desired output. Despite all the changes, the data structure will be
identical to the one generated in Figure :ref:`layout`, except the
different ``id`` value.

This three-part design explicitly supports the workflows that are
common in science, repeatedly switching between phases of exploration
and periods of writing up.  Interesting data can be collected and
curated over time, instantly and transparently visualizable without
any custom code cluttering up the notebook.  Visualizations of data
that are worth keeping can be customized through an interactive and
iterative process, and the final set of plotting options can then be
expressed as a single, separate data structure from the actual
displayed data, ready to be applied to the next batch of data from a
subsequent measurement or experiment.  Throughout, the scientist
curates the data of interest, revealed in associated visual
representations, along with the visualization options and a separate
codebase of general-purpose plots (mostly included in HoloViews, but
potentially extended locally for specific domains).  Each of these
three aspects of the process (data, options, and code) can be
developed, maintained, archived, and improved independently, providing
comprehensive support for the natural process of exploration and
dissemination common to all scientific disciplines.

Discussion
----------

This paper has demonstrated a succinct, flexible and interactive approach for data exploration, analysis and visualization.

.. Points we would like Jim to mention

   * One of the most important factors for reproducibility is to get the whole workflow into a notebook in a readable, succinct format.
   * Layouts and overlays increase the density of information delivered in a single plot, which aids in analysis and understanding. This is in contrast to the default Matplotlib inline approach, which wastes a lot of vertical space unless you decide to waste vertical space writing subplot code instead!
   * Animations and interactivity are much, much easier in HoloViews than in any other package including R's shiny, IPython widgets, Matplotlib widgets, spyre and MoviePy etc.
   * Widgets are embeddable unlike IPython and Matplotlib widgets (but also support live mode).
   * Notebook testing: Split between display and data tests. Made possible because data structures are content only.
   * Some mention that because we have data structures you can pickle them.
   * Entire styles can be switched out to rerender the same data (by replacing the OptionsTree)
   * While HoloViews plotting is based on Matplotlib

.. Comment from outline

   * Pandas dataframes have a convenient plot method. This means if you always process your data as   
     dataframes and if the capabilities of the pandas plotting are sufficient with little/no     
     customization, then this has many of the same benefits of HoloViews. The difference is that typical 
     visualizations are complex and compositional which HoloViews handles but the output of pandas plot 
     will not. You can write custom plotting code for pandas but this defeats the point.

   * Reproducibility: Makes notebook format works by capturing all the steps by being compact succinct 
     and holding onto data is always available. Declarative is related to succinct.
     Best practice, random numbers, version control, restart and re-run.

.. Originally from spaces

   Various solutions exist to bring interactivity to scientific
   visualization including IPython notebook widgets, Bokeh and the R
   language's shiny [shiny]_ web application framework. While these tools
   can provide extremely polished interactive graphics, getting them set
   up always requires additional effort and custom code, placing a
   barrier to their primary use case, the interactive exploration of
   data.


Reproducibility
~~~~~~~~~~~~~~~

.. jbednar: shouldn't it mention normalization somewhere?  It's a big deal.

.. jbednar: shouldn't it mention exporting, too?

.. jbednar: Need to cite Topographica and acknowledge NIH support
   (which paid for Param and for the continuous coordinate support).

References
----------

.. jbednar: Some of these citations are just placeholders (with
   XXXXX); they need to be filled out.

.. [Per07] Fernando Perez and Brian E. Granger, 
       IPython: a System for Interactive Scientific Computing, 
       *Computing in Science and Engineering*, 9:21-19, 2007.

.. [Ste05] William Stein and David Joyner. 
       SAGE: System for Algebra and Geometry Experimentation. 
       *ACM SIGSAM Bulletin*, 39:61-64, 2005.

.. [Was14] Michael Waskom et al.. *seaborn: v0.5.0*,
       Zenodo. 10.5281/zenodo.12710, November 2014.

.. [Wic09] Hadley Wickham, *ggplot2: elegant graphics for data analysis*,
       Springer New York, 2009.
		   
.. [shiny] RStudio, Inc, *shiny: Easy web applications in R.*,
       http://shiny.rstudio.com, 2014.

.. [knitr] Foundation for Open Access Statistics, *knitr*,
       http://yihui.name/knitr, 2015.

.. [Repro] XXXXX Crook or some other citation about reproducibility.

.. [Wol03] Stephen Wolfram,
       *The Mathematica Book*, Fifth Edition. Wolfram Media/Cambridge University Press, 2003.

.. [Mpl] Matplotlib XXXXX.

.. [pandas] Pandas XXXXX.
