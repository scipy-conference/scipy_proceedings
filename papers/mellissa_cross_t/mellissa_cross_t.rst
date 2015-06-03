:author: Mellissa Cross
:email: cros0324@umn.edu, mellissa.cross@gmail.com
:institution: Department of Earth Sciences, University of Minnesota

:video: http://www.youtube.com/watch?v=dhRUe-gz690

-------------------------------------------------------------------------------------------------------------------------
TrendVis: an Elegant Interface for dense, sparkline-like, quantitative visualizations of multiple series using matplotlib
-------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   TrendVis is a plotting package that uses matplotlib to create information-dense, sparkline-like visualizations of multiple disparate data sets in a common plot area against a common variable.  This plot type is particularly well-suited for time-series data.  While working through an example, we discuss the rationale behind and the challenges associated with adapting matplotlib to this particular plot style, the TrendVis API and architecture, and various features available for users to customize and enhance the accessiblity of their figures.

.. class:: keywords

   time series visualization, matplotlib, plotting

Introduction
------------

Data visualization and presentation is a key part of scientific communication, and many disciplines depend on the visualization of multiple time-series or other series datasets.  The field of paleoclimatology (the study of past climate and climate change), for example, relies heavily on plots of multiple time-series or "depth series", where data are plotted against depth in a sediment or ice core or stalagmite. These plots are critical for placing new data in regional and global contexts and they facilitate interpretations of the nature, timing, and drivers of climate change.  A well done example of such a plot is given below:

Creating such plots can be difficult, however.  Many scientists depend on expensive software such as SigmaPlot and Adobe Illustrator.  In the scientific Python ecosystem, there is no option to create this specific plot type.  matplotlib offers two options:  display data separately in a grid of separate subplots or overlain with twin axes.  This works for two or three traces, but doesn't scale well and is unsatisfactory for larger datasets.  Instead of a clutter of smaller plots or a mess of overlain curves, the ideal style in cases with larger datsets is the style shown above:  one densely-plotted figure that permits direct comparison of curve features.  TrendVis was created to fulfill the need for an open-source, scientific Python plot type constructor.  Here we discuss how TrendVis uses the matplotlib library to construct figures, and how users can easily customize and improve the accessibility of their TrendVis plots, and discuss several challenges faced in creating this plot type with matplotlib.

Initializing a TrendVis Figure
------------------------------

TrendVis plots come in two flavors:  XGrid and YGrid, which respectively have x and y as the main axis, and y and x as the data axes.  The base class for XGrid and YGrid, simply called Grid, contains the Figure initialization and basic attributes used by both subclasses, as well as functions and attributes that are completely or relatively orientation-agnostic.  Grid should never be directly initialized.  As most time-series data will be plotted such that x is the main axis, we will examine TrendVis from the perspective of XGrid.

Building the plot framework
---------------------------------------------
Although TrendVis plots appear to have a single, common plot space, this is an illusion.  At work is a grid of Axes and systematically hidden Axes spines, ticks, and labels.

The dimensions of the figure are determined by ystack_ratios and xratios.  Each parameter is a list of the relative sizes of the rows and columns, respectively, with length equal to the desired numer of main rows and columns.  ystack_ratios and xratios are not directly comparable.  The sum of ystack_ratios and the sum of xratios form the number of rows and columns (self.gridrows, self.gridcols).  To populate the figure, plt.subplot2grid() is used to initialize axes, moving first across the main dimension and then across the stack dimension.  Axes are stored in a nested list, where the sublists are the axes in the same stack layer (rows in the case of XGrid).   Each Axes instance shares its x or y axis with the first Axes in its row and column to ensure identical responses to application of axes limits and other parameters.

..code-block:: python
   :linenos:
   :linenostart: 49

   # Set initial x and y grid positions (top left)
       xpos = 0
       ypos = 0

   # Create axes row by row
   for rowspan in self.yratios:
       row = []

       for c, colspan in enumerate(self.xratios):
           sharex = None
           sharey = None

           # All axes in a row share y axis with first axis in row
           if xpos > 0:
               sharey = row[0]

           # All axes in a column share x axis with first axis in column
           if ypos > 0:
               sharex = self.axes[0][c]

           ax = plt.subplot2grid((self.gridrows, self.gridcols),
                                 (ypos, xpos), rowspan=rowspan,
                                 colspan=colspan, sharey=sharey,
                                 sharex=sharex)

           ax.patch.set_visible(False)

           row.append(ax)
           xpos += colspan

       self.axes.append(row)

       # Reset x position to left side, move to next y position
       xpos = 0
       ypos += rowspan

At this point, an XGrid instance with ystack_ratios = [] and xratios = [] appears thus:

Such a grid is unattractive and cluttered
After the axes are created, XGrid initializes the attributes that indicate the distribution of visible axis spines and ticks: self.dataside_list, and self.stackpos_list, which respectively indicate the y (stacked) axis spine visibility and x (main) axis spine visibility.  Together with self.spine_begone and self.mainax_ticks, these four attributes make the systematic removal of all uncessary spines possible.  After calling self.cleanup_grid(), the figure framework is thus decluttered:

Creating Axes Twins
-------------------
Overlaying curves using separate axes can improve data visualization.  TrendVis provides the means to easily and systematically create and manage twinned x axes (rows) in an XGrid instances.  In XGrid, self.make_twins() creates twin x axes, one per column, across the rows indicated.  An issue arose with twin rows in figures with a main_ax dimension > 1 (i.e., in XGrid, multiple columns).  The axes in the twinned row share x axes with the original axes, but do not share y axes with each other, as occurs in all original rows.  The twinned row were forced to share y axes via:

..code-block:: python
   twin_row[0].get_shared_y_axes().join(*twin_row)

After creation, twinned axes are stored, one row of twins per list, at the end of the list of main rows.
Many scientific disciplines depend on the visualization of multiple disparate data sets against a common variable- time series data, for example.  There are two choices in matplotlib for displaying this data:  separately in a grid of subplots or on top of each other with twinned axes.  This works for two or three traces, but does not scale well.  Instead of a clutter of separate plots or a mess of overlain curves, the ideal style is a single densely-plotted figure that permits direct comparison of curve features.  In such a plot, each dataset has its own y (or x) axis, and all data are arranged in one cohesive plot area in a vertical (or horizontal) stack against a single x (or y) axis.  This style is critical to some scientific discplines and well-suited to other realms of science and economics, but there are few options available to generate such plots and, until TrendVis, none within the scientific Python ecosystem.

   Here we examine the rationale behind and the challenges associated with adapting matplotlib to this particular plot style.  We discuss the TrendVis API, plot generation, and various features available for users to customize and enhance the accessibility of their plots.


 Data visualization and presentation is a key part of scientific communication, and many disciplines depend on the visualization of multiple time-series or other series datasets.  However, many commonly available plotting tools are severely limited when it comes to adequately displaying this data.  In matplotlib, however, there are two possibilities.  One can plot all data sets separately in a grid of subplots, or on top of each other using twinned axes.  This works for two or three traces, but does not scale well.  Instead of a clutter of separate plots or a mess of overlain curves, the ideal style in cases with larger numbers of curves is a single densely-plotted figure that permits direct comparison of curve features.  In such a plot, each dataset has its own y (or x) axis, and all data are arranged in one cohesive plot area in a vertical (or horizontal) stack against a single x (or y) axis In response to this need, TrendVis was created as a open-source, highly customizable alternative that uses the only matplotlib plotting library to easily create publication-quality, information-dense plots

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }

Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

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


