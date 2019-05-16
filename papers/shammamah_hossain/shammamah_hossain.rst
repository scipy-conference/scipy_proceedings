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
science, statistics, and biology; it has proved revolutionary in
biomedical research.  As scientific techniques in areas such as
genomics and proteomics improve, they can yield larger volumes of
valuable data that require processing in order to provide solutions to
biological problems.

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
----

Dash applications are written declaratively with Python. For instance,
a simple Dash application might look like this:

.. code-block:: python

   import dash
   import dash_html_components as html

   app = dash.Dash()
   app.layout = html.Div('Hello world!')

   app.run_server()

The :code:`dash-html-components` and :code:`dash-core-components`
packages comprise the building blocks of a Dash
app. :code:`dash-html-components` provides an interface for building
the layout of a Dash application that mimics the process of building
the layout of a website; :code:`dash-core-components` is a suite of
common tools used for interactions with an application (e.g.,
dropdowns, text inputs, and sliders) and additionally provides a
:code:`dcc.Graph` component for interactive graphs made with
plotly.py.

Dash Bio Components
-------------------

React.js and Python
###################

Some of the components in the Dash Bio package are wrappers around
pre-existing JavaScript or React components; an example is
:code:`Molecule3DViewer`, which is based on
:code:`molecule-3d-for-react` [Mol3D]_. The development process for
JavaScript-based components is fairly straightforward; the only thing
that needs to be added in many cases is an interface for Dash to
access the state of the component and read or write to it. This
provides an avenue for interactions with the components.

The package also contains three Python-based components: Clustergram,
Manhattan Plot, and Volcano Plot. Clustergram uses the
:code:`plotly.py` library and popular machine-learning libraries such
as :code:`scikit-learn` to compute hierarchical clustering. Unlike the
React components, the Python-based components return JSON data that is
in the format of the :code:`figure` argument for a
:code:`dash_core_components.Graph` component.

Component Categories
####################

Dash Bio components fall into one of three categories: custom chart
types, 3D visualization tools, and sequence analysis tools.

Custom Chart Types
==================

Specialized chart types allow for intuitive visualizations of complex
datasets.

**Circos** is a circular representation of data based on the
 CircosJS library. [TODO add citation] Within its circular layout, it
 can display a multitude of different plot types. Circos can be used
 to denote relationships between, for example, different genes with
 the "chords" property; an organism's genome can be wrapped around the
 circle, and a chord can connect one part the circle, or one gene, to
 another. [TODO add pic of chords] Circos also can display heatmaps,
 scatter plots, histograms, and stacked charts. [TODO add pic of all
 other types of circos plots]

**Clustergram** is a combination heatmap-dendrogram that is commonly
 used in gene expression data. The hierarchical clustering that is
 represented by the dendrograms can be used to identify groups of
 genes with related expression levels. [TODO include image of
 clustergram] It also supports creating annotations for specific
 clusters from within a Dash application. This makes use of the
 :code:`clickData` property of the :code:`dash_core_components.Graph`
 component to determine which cluster has been clicked, and reads data
 from a color picker and a text input to, respectively, color and
 label the annotation. [TODO add image of annotation] Calculation of
 hierarchical clustering for a dataset happens within the component
 itself and uses the :code:`scikit-learn` and :code:`scipy` libraries.

**Ideogram** is a graphical representation of chromosomal
 data. Annotations can be made to different portions of each
 chromosome and displayed in the form of bands.

**Manhattan Plot** is a plot commonly used in genome-wide association
 studies; it can highlight specific nucleotides that, when changed,
 are associated with certain genetic conditions.

**Needle Plot** is essentially a bar plot for which the bars have been
 replaced with a marker at the top of the bar and a line from the
 x-axis to the aforementioned marker. It is useful in dense data sets
 that can look "busy" when represented with a bar plot.

**Volcano Plot** is a plot used to concurrently display the
 statistical significance and the "fold change" (i.e., the ratio of a
 measurement to its preceding measurement) of data.

3D Visualization Tools
======================

Three-dimensional visualizations of biomolecules are beautiful and can
additionally shed insight into certain biological mechanisms (e.g.,
protein binding).

**Molecule 3D Viewer** is a tool that can be used to display molecular
 structures. It can render a variety of styles, including ribbon
 diagrams.

**Speck** is a WebGL-based 3D renderer that uses techniques like
 ambient occlusion and outlines to provide a rich view of molecular
 structures.

Sequence Analysis Tools
=======================

Sequence analysis tools can be used in a multitide of ways to
highlight important genes and proteins, as well as extract meaningful
data about genomic and proteomic sequences.

**Alignment Chart** is a tool for viewing multiple sequence
 alignments. Given an input FASTA file, it can compute and display the
 alignments of the sequences from the file. [TODO add image]

**Onco Print** is a visualization of genomic alteration events that
 can distinguish between different types of alterations that can
 occur.

**Sequence Viewer** is a simple tool that allows for annotating
 genomic or proteomic sequences. It allows for highlighting
 subsequences and applying sequence coverages, and supports regex
 search within the sequence.

File Parsers
------------

The Dash Bio package also includes utilities that can parse common
file types for use with Dash Bio components.

FASTA file are commonly used to represent one or more genomic or
proteomic sequences. Each sequence may be preceded by a line starting
with the :code:`>` character and contains information about the
sequence, such as the name of the gene or organism.

Different databases (e.g., neXtProt, GenBank, and SWISS-PROT) encode this metadata in different ways. Writing a parser

References
----------

.. [Mol3D] Autodesk. *Molecule 3D for React*. GitHub repository:
	     `<https://github.com/plotly/molecule-3d-for-react>`_
