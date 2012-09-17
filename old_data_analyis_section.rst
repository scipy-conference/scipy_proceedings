
Data Analysis
-------------

Our work flow is designed to handle ordered series spectra, output from both experiment and simulation.  The Python packages ``IPython``, ``Traits``, and ``Pandas`` synergistically facilitate swift data processing and visualization.  Biosensing results are information-rich, both in the spectral and temporal dimensions.  Molecular interactions on the AuNP's surface have spectral signatures which are discernable from those of environmental changes.  Likewise, the broad temporal signature of a binding event stands apart from the stepwise behavior of incremental environment changes (Fig. :ref:`glyc`).  

These recognizable temporal and spectral signatures serve as benchmarks and aid in the interpretation of more complex experiments.  When relying on such patterns, visualization tools that retain spectral and temporal transparency prove invaluable.  Indeed, with the flexibility of ``Chaco`` and ``Pandas``, simplistic, exploratory analysis emerges as a predominant means for rapidly interpreting biosensor data, with sophisticated spectral techniques merely providing supportive or ancillary information.

.. figure:: double_fib.png
   :scale: 29

   Temporal evolution (top) and spectral absorbance (bottom) of the light reflectance at the fiber endface due to a protein-protein interaction (left) as opposed to the stepwise addition of glycerin(right). :label:`glyc`

Two-Dimensional Correlation Analysis (2DCA) [9]_ is a popular and specialized way to analyze multi-dimensional spectral series by projecting the entire dataset into its orthogonal synchronous and asynchronous components.  Results are then visualized as contour maps, which consolidate the entirety of the information in the dataset into two plots.  Using the so-called Noda's rules, one can interpret from the plots the order in which events unfold in the system.  Although this technique is powerful and useful, it has two major drawbacks in the context of biosensing:

   1.  Noda's rules change or fail to apply in certain circumstances.
   2.  Valid interpretation becomes exceedingly difficult for multi-stage events.  

In regard to the second point, most non-trivial biosensing experiments evolve in stages (binding, unbinding, purging of the sensor surface, etc.).  It is necessary to decompose a multi-stage dataset into its constituent phases, and because high experimental variability usually requires manual curation.  Indeed, it is advantageous to visualize and manipulate the data simultaneously, as interaction events often commence and culminate inconspicuously.  In ``Pandas``, slicing a set of ordered series data by rows (spectral dimension) and columns (temporal dimension) is quite simple:

.. code-block:: python

   ## Read series data from tab-delimited file
   ## into a pandas DataFrame object
   from pandas import read_csv
   data=read_csv('path to file', sep='\t')  
	
   ## Select data by column index
   data[['time1', 'time2']]  

   ## Slice data by row label (wavelength)
   data.ix[500.0:750.0]

By interfacing to ``Chaco``, and using ``Pandas'`` plotting interface, we can slice, resample and visualize interesting regions in the dataspace almost effortlessly.  

Sequestering the data into event subsets simplifies information extraction.  By applying a sliding reference point and renormalizing the data each time the slice updates (see Fig. :ref:`varplot`), consistent spectral patterns as well as intrinsic event timescales emerge naturally.  

Python's scientific libraries provide practical tools for dynamic visualization.  These aid in the interpretation of intricate static plots, such as the contour maps of 2DCA, and sometimes supplant them altogether.  As biosensing evolves in complexity, these robust tools will continually evolve to meet the growing demand for an accessible and robust data analysis design framework.

.. figure:: varplot.png
   :height: 160
   :width: 250

   Top: Absorbance plot of the real-time deposition of AuNPs onto an optical fiber.  Bottom: Time-slice later in the datasets shows that the signal is dominated by signal at the surface plasmon resonance peak for gold, :math:`\lambda_{\mbox{SPR} } \approx 520 \; \mbox{nm}`.  The exemplifies the correct timescale over which spectral events manifest.  :label:`varplot`
