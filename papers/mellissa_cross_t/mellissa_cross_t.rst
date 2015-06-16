:author: Mellissa Cross
:email: cros0324@umn.edu, mellissa.cross@gmail.com
:institution: Department of Earth Sciences, University of Minnesota

-------------------------------------------------------------------------------------------------------------------------
TrendVis: an Elegant Interface for dense, sparkline-like, quantitative visualizations of multiple series using matplotlib
-------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   TrendVis is a plotting package that uses matplotlib to create information-dense, sparkline-like visualizations of multiple disparate data sets in a common plot area against a common variable.  This plot type is particularly well-suited for time-series data.  We discuss the rationale behind and the challenges associated with adapting matplotlib to this particular plot style, the TrendVis API and architecture, and various features available for users to customize and enhance the accessiblity of their figures.

.. class:: keywords

   time series visualization, matplotlib, plotting

Introduction
------------

Data visualization and presentation is a key part of scientific communication, and many disciplines depend on the visualization of multiple time-series or other series datasets.  The field of paleoclimatology (the study of past climate and climate change), for example, relies heavily on plots of multiple time-series or "depth series", where data are plotted against depth in a sediment or ice core or stalagmite. These plots are critical for placing new data in regional and global contexts and they facilitate interpretations of the nature, timing, and drivers of climate change.  A well done example of such a plot is given below:

Creating such plots can be difficult, however.  Many scientists depend on expensive software such as SigmaPlot and Adobe Illustrator.  With matplotlib, users have two options: display data in a grid of separate subplots or on top of each other using twinned axes. This works for two or three traces, but does not scale well and is unsatisfactory for larger datasets.  Instead of a clutter of smaller plots or a mess of overlain curves, the ideal style in cases with larger datsets is the style shown above:  one densely-plotted figure that permits direct comparison of curve features.  The key aim of TrendVis is to facilitate the creation and accessibility of these plots in the scientific Python ecosystem using a matplotlib-based workflow.  Here we discuss how TrendVis interfaces with the matplotlib library to construct this complicated plot type, and how users can easily customize and improve the accessibility of their TrendVis plots, and discuss several challenges faced in creating this plot type with matplotlib.

The TrendVis Figure Framework
-----------------------------
The backbone of TrendVis is the Grid class, in which the figure, basic attributes, and relatively orientation-agnostic methods are initialized.  The two subclasses of Grid, XGrid and YGrid, respectively have x and y as the main (common) axis and have y and x as the stacked (data) axes, thus determining the orientation and overall look of the figure.  As a common application of these types of plots is time-series data, we will examine TrendVis from the perspective of XGrid.

TrendVis figures appear to consist of a common plot space.  This, however, is an illusion carefully crafted via a framework of axes and a mechanism for  systematically hiding extra axes spines, ticks, and labels.  The dimensions of the XGrid framework are determined by ystack_ratios and xratios.  Respectively, these are lists of the relative sizes of the desired main rows and columns.  The sum of ystack_ratios (self.gridrows) is the height of the plot grid in unit cells, and the sum of xratios (self.gridcols) is the width of the plot grid in unit cells.  Each item in ystack_ratios and xratios therefore becomes the heighth and width span in unit cells of each Axes.

Insert figure here showing what I mean

To populate the figure, plt.subplot2grid() is used to initialize axes, moving first across the x (main) dimension and then down the y stack) dimension.  Axes are stored in a nested list, where the sublists contains axes in the same row (column in YGrid). All axes in a row share a Y axis, and all axes in a column share an X axis.

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

After the axes framework is created, XGrid calls two Grid methods to intialize lists that indicate for each row: 1. where the y axis spine and ticks are visible (self.dataside_list) and 2. where the x axis spine and ticks are visible (self.stackpos_list), if at all, based on the physical location of the axis in the plot.  Each list is exposed and can be user-modified, if desired, to meet the demands of the particular figure.

These two lists serve as keys to TrendVis formatting dictionaries and as arguments to Axes (and Axes child) methods in self.cleanup_grid().  When this method is called, XGrid systematically hides all unnecessary axis spines and ticks, and forces tick labelling to the indicated sides, transforming the mess at below left to a far clearer and more accessible format at right:

Creating and Accessing Axes Twins
---------------------------------
Although for large datasets, using twinned axes as the sole visualization tool is unadvisable, select usage of twinned axes can improve data visualization.  TrendVis provides the means to easily and systematically create and manage twinned rows (XGrid) or columns (YGrid) of axes.

In XGrid, self.make_twins() creates twin x axes, one per column, across the rows indicated.  An issue arose with twin rows in figures with a main_ax dimension > 1 (i.e., in XGrid, multiple columns).  The axes in the twinned row share x axes with the original axes, but do not share y axes with each other, as occurs in all original rows.  This is problematic when attempting to change the y axis limits, as only one axis will respond.  As a result, the axes in the twinned row are now forced to share y axes via:

.. code-block:: python

   twin_row[0].get_shared_y_axes().join(*twin_row)

After creation, self.dataside_list and self.stackpos_list are updated with the twin row information and twinned axes are stored at the end of the list of axes, which previously contained only original rows.  If the user decides against twin rows, self.axes is reduced to the original axes created upon initializing XGrid and these lists are reduced to their original format.

The TrendVis axes storage method can potentially be non-intuitive when trying to retrieve axes for plotting, especially when dealing with twin axes in a figure with many hapazardly created twins.  The following means are available to access individual axes or determine their storage locations in a TrendVis figure:
1. XGrid.axes[row][column]  In an XGrid with only one column, the last index is still necessary, as axes are stored in a nested list no matter the dimensions of XGrid.  The row index corresponds to the storage position in the list, not the actual physical position on the grid- only in original axes are these the same.
2. To find the row index of a twin row, a user can call self.get_twin_rownum() to list the storage row indices of all or a particular twin rows in the given physical row location.  Once the row number is found, the user can use method number 1.
3.  Any axis can be retrieved by providing its physical row number (and if necessary, column position) to self.get_axes().  Twins can be parsed with the keyword argument is_twin.
In the case of YGrid, axes are accessed via self.axes[column][row].

Plotting and Formatting
-----------------------
The original TrendVis contained only a simple, 1-column version of XGrid made procedurally.  As the figure was made in a single function call, all data had to be provided at once in order, and it all had to be line/point data, as only ax.plot() was called.  This version of TrendVis is designed to be a much more flexible wrapper around matplotlib.  Axes are exposed via the methods in the previous section and via native matplotlib methods, and so it is left to the user to choose appropriate plotting functions.  The author has personally used axes.errobar(), axes.fill_betweenx(), and of course axes.plot() on a single figure created using TrendVis and part of a recent manuscript revision.  For easy figure initialization and fast line plotting on all axes, make_grid() and plot_data() are provided, though fewer options are available via this interface.

Since potentially numerous axes are in play in these types of figures, TrendVis contains wrappers designed to expedite these repetitive formatting tasks, including setting major and minor tick locators and dimensions, axis labels, and axis limits.  There are two other formatting features TrendVis facilitates that are particularly useful in this plot style.  The first is the lateral movement of data axis (y axis, in XGrid) spines.  This plot type is meant for displaying data in a compact format, and it is often appropriate that the plot space be compacted vertically further than what TrendVis and matplotlib output.  In this case, data axis spines may overlap with each other, particularly those of twin rows, which break the default alternating spine pattern.  TrendVis accounts for this by providing a means to easily move the data axis spines laterally.  This can be done all at once using self.move_spines(), providing shifts that are in fractions of the figure and are stored.  Alternatively, all TrendVis methods and attributes involved in self.move_spines() all exposed, and the user can edit the axis shifts manually and then see the results via self.execute_spineshift().

In these highly compact figures with a lot of curves and axes, maintaining readability is important.  A problem that often befalls this plot style is a lack of clarity in which curve belongs with which axis, as well as where one axis begins and another ends.  TrendVis draws a visual link between axis and data by providing means to automatically color the data axis spines and ticks- but not tick labels- to match the color of the data plotted on that axis.

set ticks, set ticknums, ticknum format, limits, labels, reverse ax, autocolor spines/ticks, shifting axes

Visualizing Trends
------------------
Large stacks of curves are overwhelming and inpenetrable to viewers.  In complicated figures, it becomes especially important to  tidy the plot area and draw the viewer's eye to essential features.  TrendVis enables drawing horizontal and vertical bars across the entire plot area, allowing the user to highlight trends or demarcate particular spaces.  This is a simple call:

.. code-block:: python

    draw_bar(self, ll_axis, ur_axis, bar_limits, orientation='vertical',zorder=-1, make_adjustable=True, **kwargs)

The user provides the axes (which of course can be obtained via get_axes()) containing the lower left corner of the bar, the upper right corner of the bar.  In the case of a vertical bar on an XGrid, the vertical limits consist of the upper limit of the upper right axis and the lower limit of the lower left axis.  the horizontal upper and lower limits are provided in data units via the argument bar limits.  The default zorder is -1 in order to place the bar behind the curves, preventing data from being obscured.  Formatting keywords can be provided.

As these bars typically span multiple axes, they must be drawn in Figure space rather than on the Axes.  There are two main challenges associated with this need.  The first is converting data coordinates to figure coordinates.  In the private function _convert_coords(), we transform data coordinates into axes coordinates, and then into figure coordinates:

The figure coordinates are then used to determine the size and positioning of the Rectangle in figure space.

Of course, a patch drawn in figure space is completely divorced from the data we would like the patch to highlight.  If axes limits are changed, or the vertical or horizontal spacing of the plot is adjusted, then the bar will no longer be in the correct position relative to the data:

This is where the make_adjustable keyword comes in.  If make_adjustable is True, which is recommended, then the upper and lower horizontal and vertical limits, the upper right and lower left axes, and once the Rectangle patch is drawn, the index of the patch in XGrid.fig.patches will all be stored in XGrid attributes.  When any of TrendVis' wrappers around matplotlib's subplot spacing adjustment, x or y limit settings, etc are used, the user can stipulate that the bars automatically be adjusted to new figure coordinates.  The stored data coordinates and axes are converted to figure space, and the x, y, width, and height of the existing bars are adjusted.  Alternatively, the user can make changes to axes space relative to figure space without adjusting the bar positioning and dimensions each time, and simply perform an adjustment at the end using adjust_bar_frame().

To tidy the plot space and clarify what users are seeing, TrendVis also enables frames to be drawn around each main axis stack.  In the case of one main axis, the frame appears around the entire plot space.  For a softer division of main axes stacks, the user can signify broken axes via cut marks on the broken ends of the main axes.  Frames are similar to bars, in that they are drawn in figure space and that changing axes positions relative to figure space can move frames out of place.  Frames are handled in the same way that bars are.  Cutouts, however, are actual line plots on the axes that live in axes space (rather than data space) and will not be affected by adjustments in axes limits or subplot positioning.
