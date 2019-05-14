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

Dash Bio brings the flexibility and ease of use provided by dash to
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
application. :code:`dash-html-components` provides an interface for
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

3D Visualization Tools
======================

Sequence Analysis Tools
=======================

File Parsers
------------

The Dash Bio package also includes utilities that can parse common
file types for use with Dash Bio components.

References
----------

.. [Mol3D] Autodesk. *Molecule 3D for React*. GitHub repository:
	     `<https://github.com/plotly/molecule-3d-for-react>`_
