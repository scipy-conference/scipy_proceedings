:author: Theodore Lindsay
:email: thlindsay1@gmail.com
:institution: Caltech

:author: Peter Weir
:email: peter.weir@gmail.com
:institution: Yelp

:author: Floris van Breugel
:email: florisvb@gmail.com
:institution: University of Washington

:github: http://flyranch.github.io/figurefirst/

-----------------------------------------------------------
FigureFirst: A Layout First Approach for Scientific Figures
-----------------------------------------------------------

.. class:: abstract

One major reason that Python has been widely adopted as a scientific computing platform is the availability of powerful visualization libraries. Although these tools facilitate discovery and data exploration, they are difficult to use when constructing complex figures required to advance the narrative of a scientific manuscript. For this reason, figure creation often follows an inefficient serial process, where simple representations of raw data are constructed in analysis software and then imported into desktop publishing software to construct the final figure. Though the graphical user interface of publishing software is uniquely tailored to the production publication quality layouts, once the data are imported, all edits must be re-applied if the analysis code or raw data changes. 
Here we introduce a new Python package, FigureFirst, that allows users to lay out figures and  analyze data in a parallel fashion, making it easy to generate and continuously update aesthetically pleasing and informative figures directly from raw data. To accomplish this, FigureFirst acts as a bridge between the Scalable Vector Graphics (svg) format and MatPlotLib plotting in Python. 
To use FigureFirst, the user uses a standard svg editor such as Inkscape to specify the layout of their figure by drawing a set of rectangles on a page. In Python, FigureFirst will use this layout file to generate matplotlib figures and axes that the user can use to plot their data. Additionally, FigureFirst will save the populated figures back into the original svg layout file. This means that after making adjustments to the layout in Inkscape, the script can be run again, updating the data layers to match the new layout.
Building on this architecture we have implemented a number of features that make complex tasks remarkably easy including axis templates; changing attributes of standard svg items such as their size, shape, color, and text; and an api for adding JessyInk extensions to matplotlib objects for automatically generating animated slide presentations. In fact, our presentation will be entirely created by using FigureFirst to bring together Inkscape, JessyInk, and Matplotlib, and the templates and Python software will be made available on our github page: http://flyranch.github.io/figurefirst/.

.. class:: keywords

   plotting, figures, svg, matplotlib

Introduction
------------

Visualization has long been a critical element in the iterative process of science. Skill with the pen and pallet allowed the early pioneers of the scientific revolution to share, explain and convince: Galileo was trained in the Florentine Acatamie delle Arti del Disegno; and the intricate drawings of Di Vinci and Vesalius served to overturn Galen’s entrenched theories. 

Although new web-enabled media formats are emerging to provide alternative mechanisms for scientific communication, the static publication remains the centerpiece of scientific study. A well designed sequence of data-rich figures makes it easy for other researchers across disciplines to follow the narrative, assess the quality of the data, criticise the work, and remember the conclusions [1]. In fact, the importance of the narrative in organizing and structuring the logic of research has led some to propose that writing the manuscript should be a more integral part of the original design and execution of experiments [2]. According to this view, the researcher should create a text outline, as well as a visual story-board, long before all of the data have been collected and analyzed. As new results come to light, the story-board is updated with new data and new experiments.  
 
From a practical standpoint, taking this iterative approach with data-rich figures is challenging because desktop publishing and illustration software is not integrated with scientific analysis software. A few of commercial software packages such as matlab and sigma plot provide some graphical tools to assist in figure layout, but these tools are severely limited compared to those available in vector graphics software such as Inkscape or Adobe Illustrator, especially for creating multi-panel figures. For this reason, figure generation usually follows a unidirectional workflow in which authors first write code to analyze and plot the raw data, and then they import the figures into desktop publishing software for final editing and styling for press.
 
We created the open-source FigureFirst library to enable interoperability between open-source plotting and analysis tools available in Python (e.g. Matplotlib) and the graphical user interface provided by svg editors such as the open-source application Inkscape. By drawing a series of boxes in a blank svg document, a researcher may rapidly generate a prototype of a multi-panel figure, and then populate this figure using powerful analysis and plotting functions in Python. The FigureFirst library allows the user to insert these plots back into the prototype svg document, completing the loop between visualization and analysis. As data are collected, individual sub-panels in the figure may be populated, moved, resized or removed as the events of the ongoing study warrant. In this manner, the library should facilitate a more iterative approach to this key aspect of the scientific method. Finally, by embedding information about the scripts used to generate the final figures within the svg document itself, FigureFirst makes it possible to store an automatically updated and complete traceback from raw data to a publication quality figure. Thus, every step of the process may be kept under version control and published along with the manuscript, greatly increasing the transparency and reproducibility of the final publication.

.. figure:: workflow.pdf
   :scale: 100%
   :align: center
   :figclass: w

   Comparison between different methods of figure creation. (A) Making figures without figure first. (B) The itterative layout-based workflow enabled by figurefirst.

Usage
-----

With figurefirst creating a new figure generally involves four steps:

1) **Design the layout file.** Fundamentally this means decorating a specific subset of the objects in the svg files with xml tags that identify what objects are something figurefirst should expose to Python. If using inkscape, we facilitate this step with a number of optional inkscape extensions.

2) **Import the layout into python.** This is accomplished by constructing a :code:`FigureLayout` object with the path to the layout and then calling the :code:`make_mplfigures` method of this object to generate :code:`matplotlib` figures and axes as specified in the layout.

3) **Plot data.** All the newly created figure axes are available within the :code:`axes` dictionary of the :code:`FigureLayout` object.

4) **Save to svg.** This will merge svg graphics with matplotlib figures allow ing complex vector art to be quickly incorporated as overlays or underays to your data presentation.

Figure 2: single axis, svg editor, inkscape extension, data. 

To illustrate some the capabilities of FigureFirst, consider the task of making a more complex figure that describes three behavioral metrics for three different animals. With FigureFirst, one can draw the layout for one of the animals, and then use this layout as a template for the other two (Fig. 3A-B). Thus, if you decide to change the relative sizes of the axes, or add / remove an axis, this only needs to be done once (to the template). In this example, each of the three groups was created using a new MatPlotLib figure, which is then saved to a seperate layer in the SVG file (Fig. 3C). This organization makes it possible to update the three groups with new data independently (saving computational time). Often when working on a scientific figure early in the process, the overall layout and figure size is unknown. Or perhaps the figure needs to be reformatted for a different journal's size, or for a poster or powerpoint format. With FigureFirst these changes are as easy as rearranging the rectangles in Inkscape, and rerunning the same exact code (Fig. 3D-E). This exemplifies the key contribution of FigureFirst: seperating figure layout from the data analysis, so that the software is not cluttered with code to generate the layout, and allowing for quick reorganization of the layout. 

.. figure:: example_templates.png
   :scale: 100%
   :align: center
   :figclass: w

   Creating and rearranging multi-panel figures using FigureFirst's template feature. (A) Layout for a figure. (B) Output. (C) Inkscape screenshot illustrating the layered structure. (D) Rearranged layout. (E) Output for the new layout (code remains identical). The code used to generate these figures is available as a Jupyter Notebook on out github page: https://github.com/FlyRanch/figurefirst/blob/master/examples/figure_groups_and_templates/figure_templates_example.ipynb

When quickly prototyping analysis and figures, it can be easy to lose track of when you may have updated a figure, and what code you used to generate it. FigureFirst makes it easy to embed this information into the SVG file (Fig. 4). In the future, we plan to expand this capability by optionally linking the traceback to a github page so that when a FigureFirst generated SVG file is shared, other viewers can quickly find the code and data used to generate the figure. This option would directly and automatically link the scientific publication with the data and software, thereby facilitating open science with minimal user overhead.     

.. figure:: traceback_and_notes.png
   :scale: 100%
   :align: center
   :figclass: w

   FigureFirst makes it easy to keep track of when, how, and why your figures are created by embedding the time modified, user notes, and full traceback directly into each FigureFirst generated layer. 




Architecture
------------

FigureFirst uses a minimal Document Object Model interface (xml.dom.minidom) to parse and write to an svg file. We use define a set of xml tags that the user may use to decorate a subset of svg objects. Our library then exposes a a programing interface that addsexposesadds plotting functionality toforto these items from the layout document in PpPython.  We use the FigureFirst namespace in our xml to ensure that theseourthese tags will not collide with any other tags in the document in the document. 


Future Directions
-----------------

Thus far, we have focused our development efforts on using FigureFirst in conjunction with Inkscape. Inkscape is convenient in that it is (a) open source, (b) has a strong feature set, (c) uses the open svg standard, (d) is available for all major operating systems, (d) is available for all major operating systems, and (ede) it has a built- -in xml editor. In principle, however, any svg-compatible capable-compatible graphical layout software can be used. In the future we plan to test other user interfaces to help increase our user base. 

