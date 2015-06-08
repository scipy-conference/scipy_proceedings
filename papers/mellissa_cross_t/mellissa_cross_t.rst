:author: Mellissa Cross
:email: cros0324@umn.edu, mellissa.cross@gmail.com
:institution: Department of Earth Sciences, University of Minnesota

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

.. code-block:: python

   xpos = 0
   ypos = 0

   # Create axes row by row
   for rowspan in self.yratios:
       row = []

       for c, colspan in enumerate(self.xratios):
           sharex = None
           sharey = None

           # All ax in row share y with first ax in row
           if xpos > 0:
               sharey = row[0]

           # All ax in col share x with first ax in col
           if ypos > 0:
               sharex = self.axes[0][c]

           ax = plt.subplot2grid((self.gridrows,
                                  self.gridcols),
                                 (ypos, xpos),
                                 rowspan=rowspan,
                                 colspan=colspan,
                                 sharey=sharey,
                                 sharex=sharex)

           ax.patch.set_visible(False)

           row.append(ax)
           xpos += colspan

       self.axes.append(row)

       # Reset x position to left, move to next y pos
       xpos = 0
       ypos += rowspan

At this point, an XGrid instance with ystack_ratios = [] and xratios = [] appears thus:

After the axes are created, XGrid initializes the attributes that indicate the distribution of visible axis spines and ticks: self.dataside_list, and self.stackpos_list, which respectively indicate the y (stacked) axis spine visibility and x (main) axis spine visibility.  Together with self.spine_begone and self.mainax_ticks, these four attributes make the systematic removal of all uncessary spines possible.  After calling self.cleanup_grid(), the figure framework is thus decluttered:

Creating Axes Twins
-------------------
Overlaying curves on twinned axes can improve data visualization.  TrendVis provides the means to easily and systematically create and manage twinned x axes (rows) in XGrid instances.  In XGrid, self.make_twins() creates twin x axes, one per column, across the rows indicated.  An issue arose with twin rows in figures with a main_ax dimension > 1 (i.e., in XGrid, multiple columns).  The axes in the twinned row share x axes with the original axes, but do not share y axes with each other, as occurs in all original rows.  This is problematic when attempting to change the y axis limits, as only one axis will respond.  As a result, the twinned row are now forced to share y axes via:

.. code-block:: python

   twin_row[0].get_shared_y_axes().join(*twin_row)

After creation, twinned axes are stored, one row of twins per list, at the end of the list of main rows.

Axes Accessibility
------------------
get ax, how to acquire twin
get index of twin row/col
axes storage

Plotting Data
-------------
acquiring axes
gridwrapper- make grid, plot data

Formatting Ticks and Spines
---------------------------
set ticks, set ticknums, ticknum format, limits, labels, reverse ax, autocolor spines/ticks, shifting axes

Visualizing Trends
------------------
Large stacks of curves are overwhelming and inpenetrable to viewers.  In complicated figures, it becomes especially important to  tidy the plot area and draw the viewer's eye to essential features.  TrendVis enables drawing horizontal and vertical bars across the entire plot area, allowing the user to highlight trends or demarcate particular spaces.  This is a simple call:

.. code-block:: python

    draw_bar(self, ll_axis, ur_axis, bar_limits, orientation='vertical',zorder=-1, make_adjustable=True, **kwargs)

The user provides the axes (which of course can be obtained via get_axes()) containing the lower left corner of the bar, the upper right corner of the bar.  In the case of a vertical bar on an XGrid, the vertical limits consist of the upper limit of the upper right axis and the lower limit of the lower left axis.  the horizontal upper and lower limits are provided in data units via the argument bar limits.  The default zorder is -1 in order to place the bar behind the curves, preventing data from being obscured.  Formatting keywords can be provided.

As these bars typically span multiple axes, they must be drawn in Figure space rather than on the Axes.  There are two main challenges associated with this need.  The first is converting data coordinates to figure coordinates.  In the private function _convert_coords(), we transform data coordinates into axes coordinates, and then into figure coordinates:

.. code-block:: python

    ax_coords = axis.transData.transform(coordinates)

    fig_coords = self.fig.transFigure.inverted().transform(ax_coords)

The figure coordinates are then used to determine the size and positioning of the Rectangle in figure space.

Of course, a patch drawn in figure space is completely divorced from the data we would like the patch to highlight.  If axes limits are changed, or the vertical or horizontal spacing of the plot is adjusted, then the bar will no longer be in the correct position relative to the data:

This is where the make_adjustable keyword comes in.  If make_adjustable is True, which is recommended, then the upper and lower horizontal and vertical limits, the upper right and lower left axes, and once the Rectangle patch is drawn, the index of the patch in XGrid.fig.patches will all be stored in XGrid attributes.  When any of TrendVis' wrappers around matplotlib's subplot spacing adjustment, x or y limit settings, etc are used, the user can stipulate that the bars automatically be adjusted to new figure coordinates.  The stored data coordinates and axes are converted to figure space, and the x, y, width, and height of the existing bars are adjusted.

To tidy the plot space and clarify what users are seeing, TrendVis also enables frames to be drawn around each main axis stack, and cutouts to be placed on the main axes, signifying broken axes.
Correlative bars, frames and cutouts, squishing plot space
Challenges:  have to draw bars, frames in figure coordinates, need them to line up with data coordinates- appear to move if axes limits, plot spacing are adjusted
