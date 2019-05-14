:author: Shammamah Hossain
:email: shammamah@plot.ly
:institution: Unafilliated
:corresponding:


--------------------------------------------------
Visualization of Bioinformatics Data with Dash Bio
--------------------------------------------------


.. class:: abstract

   Plotly's Dash is a popular tool used by data scientists worldwide
   to create interactive and responsive web applications to suit their
   data-visualization needs. Dash components are used to construct the
   interfaces of these web applications. The most recent addition to
   the set of Dash component libraries is Dash Bio, a suite of several
   components and auxiliary tools related heavily to common
   visualizations and analyses characteristic to the field of
   bioinformatics. This paper provides an overview of the motivations
   behind creating the Dash Bio library, as well as brief descriptions
   of its constituent components.

.. class:: keywords

   visualization, bioinformatics

Introduction
------------

Bioinformatics as a field is experiencing rapid growth as a result of
technological advances that allow for collection of more and more
informative data sets.

Dash Bio brings the flexibility and ease of use provided by Dash to
the bioinformatics community. In addition to single-line declarations
of complex chart types, we provide parsers for some of the most common
genomics and proteomics databases that allow for direct
database-to-visualization workflows.

This paper outlines the contents of the Dash Bio package.

Dash
----

Dash applications are written declaratively with Python. For instance,
a simple Dash application might look like this:

.. code-block:: python

   import dash
   import dash_html_components as html

   app = dash.Dash()
   app.layout = html.Div('Hello world!')

   if __name__ == '__main__':
       app.run_server()

The :code:`dash-html-components` and :code:`dash-core-components`
packages comprise the building blocks of a Dash
app. :code:`dash-html-components` provides an interface for
building the layout of a Dash application that mimics the process of
building the layout of a website; :code:`dash-core-components` is a
suite of common tools used for interactions with an application (e.g.,
dropdowns, text inputs, and sliders).

TODO include an image

Dash Bio Components
-------------------

React.js and Python
###################

Some of the components in the Dash Bio package are wrappers around
pre-existing JavaScript or React components; an example is
:code:`Molecule3DViewer`, which is based on
:code:`molecule-3d-for-react` [Mol3D]_.

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

**Dash Circos** is a circular representation of data based on the
 CircosJS library. [TODO add citation] It supports features such as
 chords, which are used to annotate relationships between different
 data.

**Clustergram** is a combination heatmap-dendrogram that is commonly
 used in gene expression data. The hierarchical clustering that is
 represented by the dendrograms can be used to identify groups of
 genes with related expression levels.

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
 alignments of the sequences from the file.

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
file types for use with Dash Bio components. For instance,

References
----------

.. [Mol3D] Autodesk. *Molecule 3D for React*. GitHub repository:
	     `<https://github.com/plotly/molecule-3d-for-react>`_
