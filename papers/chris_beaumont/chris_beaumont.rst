:author: Chris Beaumont
:email: cbeaumont@cfa.harvard.edu
:institution: University of Hawaii, Harvard University

:author: Thomas Robitaille
:email: robitaille@mpia.de
:institution: Max Planck Institute for Astronomy

:author: Alyssa Goodman
:email: agoodman@cfa.harvard.edu
:institution: Harvard University

:author: Michelle Borkin
:email: borkin@seas.harvard.edu
:institution: Harvard University


-------------------------------------------
Multidimensional Data Exploration with Glue
-------------------------------------------

.. class:: abstract

    Modern research projects incorporate data from several sources,
    and new insights are increasingly driven by the ability to
    interpret data in the context of other data. Glue
    (http://glueviz.org) is an interactive environment built on top of
    the standard Python science stack to visualize relationships
    within and between data sets. With Glue, users can load and
    visualize multiple related data sets simultaneously. Users specify
    the logical connections that exist between data, and Glue
    transparently uses this information as needed to enable
    visualization across files. This functionality makes it trivial,
    for example, to interactively overplot catalogs on top of images.

    The central philosophy behind Glue is that the structure of
    research data is highly customized and problem-specific. Glue aims
    to accomodate and to simplify the "data munging" process, so that
    researchers can more naturally explore what their data have to
    say. The result is a cleaner scientific workflow, and more rapid
    interaction with data.

.. class:: keywords

   data visualization, eploratory data analysis, python

Introduction
------------

   Moore's law has revolutionized the way data are collected and
   analyzed in many fields of science. Increasingly often, researchers
   supplement their own data collection with analysis of
   legacy data products. In astronomy, for example, researchers
   are often able to complement their data with publically-available
   survey data targeting a different range of the electromagnetic spectrum.
   Because of this, the pace of discovery is increasingly dependent upon
   interpreting data in the context of other data.

   Unfortunately, most of the current interactive tools for data
   exploration focus on analyzing a single data set at a time. It is
   considerably more difficult to interactively explore several
   conceptually related datasets at once. Scientists typically resort
   to non-interactive techniques (e.g., writing scripts to
   produce static visualizations). This slows the pace of
   investigation, and can make it more difficult to uncover
   subtle relationships between datasets.

   To address this shortcoming, we have been developing
   [Glue](http://glue-viz.org). Glue is an interactive data
   visualization environment that focuses on multi-dataset
   exploration. Glue allows users to specify how different data sets
   are related, and uses this information to dynamically link and
   overlay visualizations of several datasets. Glue also aims to
   integrate into python-based analysis workflows, to ease
   the transition between interactive and non-interactive data
   analysis.


The Basic Glue Worfklow
-----------------------

   The central visualization philosophy behind Glue is the idea of
   linked visualizations -- that is, multiple related representations
   of a dataset that are dynamically connected, such that interaction
   with one visualization affects the appearance of another. For example,
   a user might create two different scatter plots of a multi-dimensional
   table, select a particular region of parameter space in one plot,
   and see the points in that region highlighed in both plots. Linked-view
   visualizations are especially effective at exploring high-dimensional
   data. Glue applies this idea to multiple files.

   Let's illustrate the basic Glue workflow with an example. An
   astronomer is studing Infrared Dark Clouds (regions of young star
   formation) in our Galaxy. Her data include a catalog of known
   Infared Dark Clouds, a second catalog of "cores"
   (substructures embedded in these clouds where the stars actually
   form), and a wide-field infrared survey image of a particular cloud.

   She begins by loading the cloud catalog into Glue. She creates a
   scatter plot of the x/y position of each cloud, as well as a histogram
   showing the distribution of masses. She creates each visualization by
   dragging the data entry onto the visualization area. At this point,
   her screen looks like Figure 1.

   She is interested in a particular region of the sky, and thus draws
   a lasso around particular points in the scatter plot. This creates
   a new "subset", which is shown in read on each visualization. If she
   traces a different region on either plot, the subset will update
   on both views automatically.

   Next she loads the infrared image. She would like to see how the
   points in the catalog relate to structures in the image, by
   overplotting the subset on the image. To do this, she first "links"
   the data by defining the logical relationships between the two
   files. She opens the data linking dialog, which displays the
   attributes defined for each dataset. The image has attributes for
   the x and y location of each pixel, and the catalog has columns
   which list the location of each object in the same coordinate
   system. She highlights the attribute describing the x location
   attribute for each dataset, and "links" them (in effect informing
   Glue that the two attributes describe the same quantity). She
   repeats this for the y location attribute, and closes the
   dialog. Now, she can drag the subset onto the image, to overplot
   these points at their proper location (this is possible because
   Glue now has enough information to compute the location of each
   catalog source in the image). All three plots are still linked:
   if the user highlights a new region on the image, this will
   redefine the subset and update each plot.

   The relationship between the catalog and image was very simple,
   each dataset described the same spatial quantities, in the same
   units. In general, connections between datasets are more
   complicated. For example, the core catalog defines the location
   of points in a different coordinate system. Because of this,
   Glue allows users to connect quantities across dataset
   using transformation functions. Glue includes some of these
   functions by default, but the users can also write their own
   functions for arbitrary transformations. Glue uses these functions
   as needed to transform quantities between coordinate systems,
   to overlay visualizations and/or filter data in subsets.

   Our scientist discovers several interesting relationships between
   these datasets -- in particular, that several distinct entries in
   the cloud catalog appear to form a coherent, extended structure in
   the image. Furthermore, the cores embedded in these clouds all have
   similar velocities, strengthening the argument that they are
   related.  At this point, she decides to test this hypothesis more
   rigorously, by comparing to models of structure formation. This
   analysis will happen outside of Glue. She saves all of her subsets
   as masks, to easily analyze them in this followup
   analysis. Furthermore, she saves the entire Glue session, which
   allows her to re-load these datasets, dataset connections, and
   subset definitions at any time.


Glue Architecture
-----------------

   The scenario above outlines the basic workflow that Glue enables -- Glue allows users to create interactive visualizations, and drill down into interesting subsets of these visualizations.
