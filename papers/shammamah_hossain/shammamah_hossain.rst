:author: Shammamah Hossain
:email: shammamah@plot.ly
:institution: Unafilliated
:corresponding:


--------------------------------------------------
Visualization of Bioinformatics Data with Dash Bio
--------------------------------------------------


.. class:: abstract

   Plotly's Dash is a library that empowers data scientists to create
   interactive web applications declaratively in Python. Dash Bio is a
   bioinformatics-oriented suite of components that are compatible
   with Dash. Common data visualizations characteristic to the field
   of bioinformatics can now be integrated into Dash applications. We
   present the Dash Bio suite of components and its auxiliary file
   parsers.

.. class:: keywords

   visualization, bioinformatics

Introduction
------------

The emergent field of bioinformatics is an amalgamation of computer
science, statistics, and biology; it has proven itself revolutionary
in biomedical research.  As scientific techniques in areas such as
genomics and proteomics improve, they can yield larger volumes of
valuable data that require processing in order to efficiently provide
meaningful solutions to biological problems.

Many bioinformaticians have already created analysis and visualization
tools with Dash and plotly.py, but only through significant
workarounds and modifications made to preexisting graph types. In
addition to permitting single-line declarations of charts for complex
datasets such as hierarchical clustering and multiple sequence
alignment, we introduce several new chart types, three-dimensional and
interactive molecule visualization tools, and components that are
specifically related to genomic and proteomic sequences. In addition
to these components, we present a set of simple parsing scripts that
handle some of the most common file types found in
bioinformatics-related databases.

This paper outlines the contents of the Dash Bio package, which
imparts the powerful data-visualization tools and flexibility of Dash
to the flourishing bioinformatics community.

Dash
====

Dash applications are web applications that are written declaratively
with Python. In addition to the main :code:`dash` library, the
:code:`dash-html-components` and :code:`dash-core-components` packages
comprise the building blocks of a Dash
app. :code:`dash-html-components` provides an interface for building
the layout of a Dash application that mimics the process of building
the layout of a website; :code:`dash-core-components` is a suite of
common tools used for interactions with an application (e.g.,
dropdowns, text inputs, and sliders) and additionally provides a
:code:`dcc.Graph` component for interactive graphs made with
plotly.py.

For instance, a simple Dash application might look like
this:

.. code-block:: python

   import dash
   import dash_html_components as html

   app = dash.Dash()
   app.layout = html.Div('Hello, world!')

   app.run_server()

Upon running the above code, a :code:`localhost` address is specified
in the console. Visiting this address in the browser yields a simple
webpage that contains the text "Hello, world!" (see
Fig. :ref:`helloworld`).

.. figure:: helloworld.png

   A simple Dash application. :label:`helloworld`

Interactivity is implemented with callbacks. These allow for reading
the values of inputs in the application (e.g., text inputs, dropdowns,
and sliders), which can subsequently be used to compute the value of a
property of another component in the app. The function that computes
the output is wrapped in a decorator function that connects the
specified inputs and outputs to user interactions with the app.  For
instance, this :code:`dash_core_components.Input()` component controls
the :code:`children` property of a :code:`dash_html_components.Div()`
component:

.. code-block:: python

   import dash
   import dash_html_components as html
   import dash_core_components as dcc

   app = dash.Dash()
   app.layout = html.Div(children=[
	html.Div(id='output-div'),
	dcc.Input(id='text-input')
   ])

   @app.callback(
       dash.dependencies.Output('output-div', 'children'),
       [dash.dependencies.Input('text-input', 'value')]
   )
   def capitalize_user_input(text):
	return text.upper()

   app.run_server()

The output of the code is shown in Fig. :ref:`helloworld`.

.. figure:: helloworld_interactive.png

   A simple Dash application that showcases interactivity. Text that
   is entered into the input component is converted to uppercase and
   displayed in the app. :label:`helloworld`

React.js and Python
===================

Some of the components in the Dash Bio package are wrappers around
pre-existing JavaScript or React libraries. The development process
for JavaScript-based components is fairly straightforward; the only
thing that needs to be added in many cases is an interface for Dash to
access the state of the component and read or write to its
properties. This provides an avenue for interactions with the
components from within a Dash application.

The package also contains three Python-based components: Clustergram,
Manhattan Plot, and Volcano Plot. Unlike the
React components, the Python-based components are essentially
functions that return JSON data that is in the format of the
:code:`figure` argument for a :code:`dash_core_components.Graph`
component.

Dash Bio Components
-------------------

Dash Bio components fall into one of three categories.

Custom chart types
  Specialized chart types that allow for intuitive
  visualizations of complex data.

Three-dimensional visualization tools
  Structural diagrams of biomolecules that support a wide variety of
  user interactions.

Sequence analysis tools
  Interactive and searchable genomic and proteomic sequences, with
  additional features such as multiple sequence alignment.


Circos
======

.. figure:: circos.png
   :scale: 25%
   :figclass: bht

   A simple Dash Bio Circos component with chords connecting pairs of
   data points. :label:`circos`

Circos is a circular graph. It can be used to highlight relationships
between, for example, different genes by drawing chords that connect
the two (see Fig. :ref:`circos`).

The Dash Bio Circos component is a wrapper of the :code:`CircosJS`
[Circos]_ library, which supports additional graph types like
heatmaps, scatter plots, histograms, and stacked charts. Input data to
the component take the form of a dictionary, and are supplied to the
:code:`tracks` property of the component.

Clustergram
===========

.. figure:: clustergram.png
   :figclass: bht

   A Dash Bio clustergram component displaying hierarchical clustering of gene
   expression data from two lung cancer subtypes. Data taken from
   [KR09]_. :label:`clustergram`

A clustergram is a combination heatmap-dendrogram that is commonly used
in gene expression data. The hierarchical clustering that is
represented by the dendrograms can be used to identify groups of genes
with related expression levels.

The Dash Bio Clustergram component is a Python-based component that
uses plotly.py to generate a figure. It takes as input a
two-dimensional numpy array of floating-point values. Imputation of
missing data and computation of hierarchical clustering both occur
within the component itself. Clusters that are past a user-defined
threshold of similarity comprise a single trace in the corresponding
dendrogram, and can be highlighted with annotations (see
Fig. :ref:`clustergram`).

The user can specify additional parameters to customize the metrics
and methods used to compute parts of the clustering, such as the
pairwise distance between observations and the linkage matrix.

Ideogram
========

.. figure:: ideogram.png
   :figclass: bht

   A Dash Bio ideogram component demonstrating the homology feature
   with two human chromosomes. :label:`ideogram`

An ideogram is a schematic representation of genomic data. Chromosomes
are represented as strands, and the location of specific genes is
denoted by bands on the chromosomes.

The Dash Bio Ideogram component is built on top of the
:code:`ideogram.js` library [Ideo]_, and includes features like
annotations, histograms, and homology (see
Fig. :ref:`ideogram`). Annotations can be made to different portions
of each chromosome and displayed in the form of bands, and
relationships between different chromosomes can be highlighted by
using the homology feature to connect a region on one chromosome to a
region on another (see Fig. :ref:`ideogram`). The ideogram component
is based on the

Manhattan Plot
==============

.. figure:: manhattan.png
   :figclass: bht

   A Manhattan plot. The threshold level is denoted by the red line;
   all points of interest are colored red. The purple line is the
   suggestive line. :label:`manhattan`

A Manhattan plot is a plot commonly used in genome-wide association
studies; it can highlight specific nucleotides that, when changed, are
associated with certain genetic conditions.

The Dash Bio ManhatanPlot component is built with plotly.py. Input
data take the form of a pandas dataframe. The two lines on the plot
(see Fig. :ref:`manhattan`) represent, respectively, the threshold
level and the suggestive line. [#]_ The y-values of these lines can be
controlled by the user.

.. [#] Information about the meaning of these two lines can be found
       in [ER15]_.

Needle Plot
===========

.. figure:: needle.png
   :figclass: bht

   A needle plot that shows the properties of mutations in a genomic
   strand. :label:`needle`

A needle plot is a bar plot for which each bar has been replaced with
a marker at the top and a line from the x-axis to the aforementioned
marker. Its primary use-case is visualization of dense datasets that
can look "busy" when represented with a bar plot. In bioinformatics, a
needle plot may be used to annotate the positions on a genome at which
genetic mutations happen (see Fig. :ref:`needle`).

The Dash Bio NeedlePlot component was built using plotly.js. It
receives input data in a dictionary. It can distinguish between
different types of mutations with different colors and marker styles,
and can demarcate the domains of specific genes.

Volcano Plot
============

.. figure:: volcano.png
   :figclass: bht

   A Dash Bio VolcanoPlot component. Points of interest are colored in
   red, and the effect size and statistical significance thresholds
   are represented by dashed lines. :label:`volcano`

A volcano plot is a plot used to concurrently display the statistical
significance and a defined "effect size" (e.g., the fold change [#]_)
of a dataset. This type of plot is incredibly useful when visualizing
a large number of data points that represent replicate data; it
facilitates identification of data that simultaneously have
statistical significance and a large effect.

.. [#] This refers to the ratio of a measurement to its preceding
       measurement.

The Dash Bio VolcanoPlot component was built using plotly.py. It takes
a pandas dataframe as input data. Lines that represent the threshold
for effect size (both positive and negative) and a threshold for
statistical significance can be defined by the user (see
Fig. :ref:`volcano`).

Molecule 3D Viewer
==================

.. figure:: mol3d.png
   :scale: 35%
   :figclass: bht

   A Dash Bio Molecule3DViewer component displaying the ribbon
   structure of a section of DNA. A selected residue is highlighted in
   cyan. :label:`mol3d`

The Dash Bio Molecule3DViewer component was built on top of the
:code:`molecule-3d-for-react` [Mol3D]_ library. Its purpose is to
display molecular structures.  These types of visualizations can be
useful when communicating the mechanics of biomolecular process, as it
can show the shapes of proteins and provide insight into the way that
they bind to other molecules.

Molecule3DViewer receives input data as a dictionary which specifies
the layout and style of each atom in the molecule. It can render
molecules in a variety of styles, such as ribbon diagrams, and allows
for mouse-click selection of specific atoms or residues (see
Fig. :label:`mol3d`) that can be read from or written to within a Dash
app.

Speck
=====

.. figure:: speck.png
   :figclass: bht

   A Dash Bio Speck component displaying the atomic structure of a
   strand of DNA in a ball-and-stick representation. Ambient occlusion
   is used to provide realistic shading on the atoms. :label:`speck`

The Dash Bio Speck component is a WebGL-based 3D renderer that is
built on top of :code:`Speck` [Speck]_. It uses techniques like ambient
occlusion and outlines to provide a rich view of molecular structures
(see Fig. :ref:`speck`).

The Dash Bio Speck component receives input data as a dictionary that
contains, for each atom, the atomic symbol and the position in space
(given as x, y, and z coordinates). Parameters related to the
rendering of the molecule, such as the atom sizes, levels of ambient
occlusion, and outlines, can optionally be specified in another
dictionary supplied as an argument.

Alignment Chart
=======================

.. figure:: alignment.png
   :figclass: bht

   A Dash Bio AlignmentChart component displaying the P53 protein's
   amino acid sequences from different organisms. A conservation
   barplot is displayed on top, and the bottom row of the heatmap
   contains the consensus sequence. :label:`alignment`

An alignment chart is a tool for viewing multiple sequence
alignment. Multiple related sequences of nucleotides or amino acids
(e.g., the amino acid sequences of proteins from different organisms
that appear to serve the same function) are displayed in the chart to
show their similarities.

The Dash Bio AlignmentChart component is built on top of
:code:`react-alignment-viewer` [Align]_. It takes a FASTA file as input
and computes the alignment. It can optionally display a barplot that
represents the level of conservation of a particular amino acid or
nucleotide across each sequence defined in the input file (see
Fig. :ref:`alignment`).

Onco Print
==========

.. figure:: onco.png
   :figclass: bht

   A Dash Bio OncoPrint component that shows mutation events for the
   genomic sequences that encode different proteins. :label:`onco`

Onco Print is a type of heatmap that facilitates the visualization of
multiple genomic alteration events (see Fig. :ref:`onco`).

The Dash Bio OncoPrint component is built on top of
:code:`react-oncoprint` [Onco]_. Input data for the component takes
the form of a list of dictionaries that define a sample, gene,
alteration, and mutation type.

**Sequence Viewer** is a simple tool that allows for annotating
 genomic or proteomic sequences. It allows for highlighting
 subsequences and applying sequence coverages, and supports regex
 search within the sequence.

File Parsers
------------

The `dash-bio-utils` package was developed in tandem with the
`dash-bio` package. It contains parsers for many common bioinformatics
databases that translate the data encoded in those files to inputs
that are compatible with Dash Bio components.

FASTA files
===========

FASTA files are commonly used to represent one or more genomic or
proteomic sequences. Each sequence may be preceded by a line starting
with the :code:`>` character and contains information about the
sequence, such as the name of the gene or organism.

Different databases (e.g., neXtProt, GenBank, and SWISS-PROT) encode
this metadata in different ways.

PDB files
=========

SOFT files
=========

References
----------

.. [Mol3D] Autodesk. *Molecule 3D for React*. URL:
	     `<https://github.com/plotly/molecule-3d-for-react>`_
.. [Circos] Girault, Nic. *circosJS: d3 library to build circular graphs*. URL: `<https://github.com/nicgirault/circosJS>`_
.. [KR09] Kuner R, Muley T, Meister M, Ruschhaupt M et al. *Global gene expression analysis reveals specific patterns of cell junctions in non-small cell lung cancer subtypes.* Lung Cancer 2009 Jan;63(1):32-8. PMID: 18486272
.. [Ideo] Weitz, Eric. *ideogram: Chromosome visualization with JavaScript*. URL: `<https://github.com/eweitz/ideogram>`_
.. [ER15] Reed, E., Nunez, S., Kulp, D., Qian, J., Reilly, M. P., and Foulkes, A. S. (2015) *A guide to genome‐wide association analysis and post‐analytic interrogation.* Statist. Med., 34: 3769– 3792. doi: 10.1002/sim.6605.
.. [Speck] Terrell, Rye. *Speck*. URL: `<https://github.com/wwwtyro/speck>`_
.. [Align] Plotly. *React Alignment Viewer*. URL: `<https://github.com/plotly/react-alignment-viewer>`_
.. [Onco] Plotly. *React OncoPrint*. URL: `<https://github.com/plotly/react-oncoprint>`_
