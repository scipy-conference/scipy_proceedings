:author: Forrest Sheng Bao
:email: forrest.bao@gmail.com
:institution: Texas Tech University, Texas

:author: Christina R. Zhang
:email: christina.zhang@mail.mcgill.ca
:institution: McGill University, Canada

-------------------------------------------------
PyEEG: A Python module for EEG feature extraction
-------------------------------------------------

.. class:: abstract

Computer-aided diagnosis of neural diseases from EEG signals has become an emerging field in past years. A key component in analyzing EEG signal is to extract various features. As Python is gaining more ground in scientific computing, an open source Python module to extract EEG features will save researchers a lot of time. In this paper, we introduce PyEEG, an open source Python module for EEG feature extraction.

Introduction
------------

Over the past decade, computer-aided diagnosis (CAD) systems based on EEG have emerged in the early diagnosis of several neural diseases such as Alzheimer's disease [Dau10]_ and epilepsy [Bao08]_. For example, conventional epilepsy diagnosis may require long-term or repeated EEG recordings to capture seizures or other epileptic activities. 
Recently, researchers have found it promising to use interictal (i.e., non-seizure) EEG records that do not contain particular activities for conventional epilepsy diagnosis [Bao09]_. If such an approach can be clinically verified, suspicious subjects only need to take short EEG recordings and physicians can be relieved from the tedious visual inspection to EEG waveform.

..
	In addition to analyzing existing signals, the computer-based approach can help us model the neurons in the brain and predict their future behavior, e.g., seizure prediction [OSu09]_. 

A key component in CAD systems mentioned above is to characterize EEG signals into certain features, a process known as feature extraction. EEG features can come from different fields that study time series: power spectrum density from signal processing, fractal dimensions from computational geometry, entropies from information theory, etc. An open source tool for EEG feature extraction will benefit the computational neuroscience community since EEG feature extraction is repeatedly invoked in analyzing EEG signals. Because of Python's increasing popularity in scientific computing, especially in computational neuroscience, we have developed PyEEG, a Python module for EEG feature extraction. 

..
	a Python module for EEG feature extraction is highly useful. As we are not aware of such a Python module, 



Main Framework
--------------

.. figure:: pyeegframe.pdf

	PyEEG Framework

PyEEG consists of two sets of functions. EEG pre-processing functions, which do not return any feature values, are in the first set. The second set contains feature extraction functions that return feature values. A feature value can be a vector or a scalar. We follow PEP8 *Style Guide for Python Code* [1]_ to name variables and functions.

.. [1] http://www.python.org/dev/peps/pep-0008/

Besides standard Python functions, PyEEG only uses functions provided by SciPy, a *de facto* Python module for scientific computing. 

PyEEG does not define any new data structure, using standard Python and NumPy ones only. The reason is that we want to simplify the use of  PyEEG, especially for users without much programming background. The inputs of all functions are time series in form of a list of floating-point numbers and a set of optional feature extraction parameters. Parameters have default values. The output of a feature extraction function is a floating-point number if the feature is a scalar or a list of floating-point numbers if it is a vector.

EEG pre-processing functions
----------------------------

PyEEG currently provides two pre-processing functions, ``embed_seq()`` and ``first_order_diff()``. They build new time series from given time series for  further computation. The first is to build embedding sequence (from given lag and embedding dimension) and the second is to compute first-order differential sequence. One can build differential sequences of higher orders by apply first-order differential computing repeatedly. 

.. |doubleS| unicode:: U+00A7 .. section sign


EEG feature extraction functions
--------------------------------

So far PyEEG can extract 10 features, which are listed below with their corresponding function names and return types:

	* Power Spectral Intensity (PSI) and Relative Intensity Ratio (RIR), ``bin_power()``, two 1-D vectors, [Bao09]_
	* Petrosian Fractal Dimension (PFD), ``pdf()``, a scalar, [Pet95]_
	* Higuchi Fractal Dimension (HFD), ``hfd()``, a scalar, [Hig88]_
	* Hjorth mobility and complexity, ``hjorth()``, two scalars, [Hjo70]_ 
	* Spectral Entropy (Shannon's entropy of RIRs), ``spectral_entropy()``, a scalar        
	* SVD Entropy, ``svd_entropy()``, a scalar, [Rob99]_
	* Fisher Information, ``fisher_info()``, a scalar, [Jam03]_
	* Approximate Entropy (ApEn), ``ap_entropy()``, a scalar, [Pin91]_
	* Sample Entropy (SampEn), ``samp_entropy()``, a scalar, [Ric00]_        
	* Detrended Fluctuation Analysis (DFA), ``dfa()``, a scalar, [Pen95]_

They are frequently used in EEG signal processing research. Please refer to PyEEG documents for more details about functions.
More feature extracting functions will be added gradually. 

..
	+----------------------------------------------------------------+---------------+
	| feature                                                        | type          |
	+================================================================+===============+
	| Relative Intensity Ratios (RIRs) [Bao09]_, |doubleS|.III.A     | 1-D vector    |
	+----------------------------------------------------------------+---------------+
	| Petrosian Fractal Dimension (PFD) [Pet95]_                     | scalar        | 
	+----------------------------------------------------------------+---------------+
	| Higuchi Fractal Dimension (HFD) [Hig88]_                       | scalar        |   
	+----------------------------------------------------------------+---------------+
	| Hjorth mobility [Hjo70]_                                       | scalar        |  
	+----------------------------------------------------------------+---------------+
	| Hjorth complexity [Hjo70]_                                     | scalar        |
	+----------------------------------------------------------------+---------------+
	| Spectral Entropy (Shannon's entropy of RIRs)                   | scalar        |
	+----------------------------------------------------------------+---------------+
	| SVD Entropy [Rob99]_                                           | scalar        |
	+----------------------------------------------------------------+---------------+
	| Fisher Information [Jam03]_                                    | scalar        |
	+----------------------------------------------------------------+---------------+
	| Approximate Entropy (ApEn) [Pin91]_                            | scalar        |
	+----------------------------------------------------------------+---------------+
	| Sample Entropy (SampEn) [Ric00]_                               | scalar        |
	+----------------------------------------------------------------+---------------+
	| Detrended Fluctuation Analysis (DFA) [Pen95]_                  | scalar        |
	+----------------------------------------------------------------+---------------+


Basic Usage
-----------

SciPy is required to run PyEEG. The latest PyEEG is released as a single Python script, which includes
all functions. So users only need to download and place it under a directory that is in
Python module search path, such as the working directory.

The first step to use PyEEG is to import it::

	>>>import pyeeg

Then functions in PyEEG can be called. For example, the code below computes
DFA for ten 4096-point white noise signals (theoretical DFA is 0.5) of mean 0 and variance 1::

	>>> from numpy.random import randn
	>>> for i in xrange(0,10):
	...     pyeeg.dfa(randn(4096))
	... 
	0.50473407278667271
	0.53339499445571614
	0.53034354430841246
	0.50844373446375624
	0.5162319368337136
	0.46319279647779976
	0.44515512343867669
	0.4407740703026245
	0.45894672465613884
	0.49135727073171609


Future works
------------

There are several things we plan to do in a near future. First, we need more comprehensive documents, such as examples for all functions. Second, while implementing more feature extraction functions, we need to add unit testing to them. Third, the speed of some functions is not high because they are implemented from definitions. Faster implementation is needed. Last but not least, we will build up interfaces for classifiers such as LIBSVM or MLPY and interfaces for open source EEG data importers.

..
	 , such as exporting features to SVM-Light [1] format or connecting to LIBSVM [2] Python interface. 

.. .. [1] <http://svmlight.joachims.org/>
.. .. [2] <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>

Availability
------------

Source code and documents of PyEEG are  at http://code.google.com/p/pyeeg/. A shorter URL is http://pyeeg.org. We thank Google for providing free hosting.

..
	Acknowledgments
	---------------


..
	FSB is also very thankful to the developers of the following Open Source software: 
	GNU/Linux, Ubuntu, Scipy/Numpy and Eclipse.


References
----------

.. [Bao08] F. S. Bao, D. Y.-C. Lie, and Y. Zhang, *A new approach to automated epileptic
  diagnosis using EEG and probabilistic neural network*, in
  Proc. of 20th IEEE International Conference on Tools with
  Artificial Intelligence (ICTAI 2008), 2008.

.. [Bao09] F. S. Bao, J.-M. Gao, J. Hu, D. Y. C. Lie, Y. Zhang, and K. J. Oommen,
  *Automated epilepsy diagnosis using interictal scalp EEG*, in
  Proc. of 31st International Conference of IEEE Engineering in
  Medicine and Biology Society (EMBC 2009), 2009.

.. .. [Dau09] J. Dauwels, E. Eskandar, and S. Cash, *Localization of seizure onset area from
  intracranial non-seizure {EEG} by exploiting locally enhanced sychrony*, in
  Proc. of 31st International Conference of IEEE Engineering in
  Medicine and Biology Society (EMBC 2009)}, 2009.

.. [Dau10] J. Dauwels, F. Vialatte, and A. Cichocki, *A comparative study of synchrony
  measures for the early detection of Alzheimer’s disease based on EEG*,
  NeuroImage, 49:668-693, 2010.

.. .. [Gar06] A. Gardner, A. Krieger, G. Vachtsevanos, and B. Litt, *One-class novelty
  detection for seizure analysis from intracranial EEG*, Journal of
  Machine Learning Research, 7:1025-1044, 2006.

.. [Hig88] T. Higuchi, *Approach to an irregular time series on the basis of the fractal
  theory*, Physica D, 31(2):277-283, 1988.

.. [Hjo70] B. Hjorth, *EEG analysis based on time domain properties*,
  Electroencephalography and Clinical Neurophysiology, 29:306-310, 1970.

.. [Jam03] C. J. James and D. Lowe, *Extracting multisource brain activity from a single
  electromagnetic channel*, Artificial Intelligence in Medicine, 28(1):89-104, 2003.

.. .. [OSu09] E. O'Sullivan-Greene, I. Mareels, D. Freestone, L. Kuhlmann, and A. Burkitt,
  *A paradigm for epileptic seizure prediction using a coupled oscillator
  model of the brain*, in Proc. of 31st International Conference
  of IEEE Engineering in Medicine and Biology Society (EMBC 2009), 2009.

.. [Pen95] C.-K. Peng, S. Havlin, H. E. Stanley, and A. L. Goldberger, *Quantification of
  scaling exponents and crossover phenomena in nonstationary heartbeat time
  series*, Chaos, 5(1):82-87, 1995.

.. [Pet95] A. Petrosian, *Kolmogorov complexity of finite sequences and recognition of
  different preictal EEG patterns*, in Proc. of 8th IEEE Symposium on
  Computer-Based Medical Systems, 1995.

.. [Pin91] S. Pincus, I. Gladstone, and R. Ehrenkranz, *A regularity statistic for
  medical data analysis*, Journal of Clinical Monitoring and Computing, 7(4):335-345, 1991.

.. [Ric00] J. S. Richman, and J. R. Moorman, *Physiological time-series analysis using 
   approximate entropy and sample entropy*, American Journal of Physiology - Heart and 
   Circulatory Phsiology, 278:H2039-H2049, 2000.

.. [Rob99] S. Roberts, W. Penny, and I. Rezek, *Temporal and spatial complexity measures
  for electroencephalogram based brain-computer interfacing*, Medical
  and Biological Engineering and Computing, 37(1):93-98, 1999.

