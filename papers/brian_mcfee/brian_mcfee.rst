:author: Brian McFee
:email: brian.mcfee@nyu.edu
:institution: Center for Data Science, New York University
:institution: Music and Audio Research Laboratory, New York University

:author: Colin Raffel
:email: craffel@gmail.com
:institution: LabROSA, Columbia University

:author: Dawen Liang
:email: dliang@ee.columbia.edu
:institution: LabROSA, Columbia University

:author: Daniel P.W. Ellis
:email: dpwe@ee.columbia.edu
:institution: LabROSA, Columbia University

:author: Matt McVicar
:email: mm4819@bristol.ac.uk
:institution: Department of Engineering Mathematics, University of Bristol

:author: Eric Battenberg
:email: eric@ericbattenberg.com
:institution: Baidu, Inc.

:author: Oriol Nieto
:email: oriol@nyu.edu
:institution: Music and Audio Research Laboratory, New York University

--------------------------------------------------
LibROSA: Audio and Music Signal Analysis in Python
--------------------------------------------------

.. class:: abstract

   This paper describes the design and implementation of librosa version 0.4.0, 
   and provides an overview and historical background of the project.


.. class:: keywords

   audio, music, signal processing


Introduction
------------

The emergent research field of music information retrieval (MIR) broadly covers topics at
the intersection of musicology, digital signal processing, machine learning, information
retrieval, and library science.  Although the field is relatively young |---| the first
international symposium on music information retrieval (ISMIR) [#]_ was held in October of
2000 |---| it is rapidly developing, thanks in part to the proliferation and practical
scientific needs of digital music services, such as iTunes, Pandora, and Spotify.
While the preponderance of MIR research has been conducted with custom tools and scripts
developed by researchers in a variety of languages such as MATLAB or C++, the stability, 
scalability, and ease of use these tools has often left much to be desired.

Within recent years, interest in (scientific) Python as a viable alternative has grown
in the MIR community.
This has been driven by a confluence of several factors, including the availability of
high-quality machine learning libraries such as ``scikit-learn`` [Pedregosa11]_ and tools based on
``Theano`` [Bergstra11]_, as well as Python's vast catalog of packages for dealing with text data and
web services.
However, without a stable core library to provide the basic
routines upon which many MIR applications are built, adoption of Python has been slow.
To remedy this situation, we have developed ``librosa``:[#]_ a python package for audio
and music signal processing. [#]_
In doing so, we hope to both ease the transition into Python (and modern software
development practices) for MIR researchers, and 
make core MIR techniques readily available to the broader community of scientists and 
Python programmers.

.. [#] http://ismir.net

.. [#] https://github.com/bmcfee/librosa

.. [#] The name `librosa` is borrowed from `LabROSA`: the LABoratory for the Recognition
    and Organization of Speech and Audio at Columbia University, where much of the development
    of librosa took place.


Design principles
=================

In designing librosa, we prioritized a few key concepts.
First, we strive for a low barrier to entry for researchers familiar with MATLAB.
In particular, we opted for a relatively flat package layout, and following ``scipy`` [Jones01]_ 
rely upon ``numpy`` data types and functions [VanDerWalt11]_, rather than abstract class hierarchies.

Second, we expended considerable effort in standardizing interfaces, variable names, 
and (default) parameter settings across the various analysis functions.
This task was complicated by the fact that reference implementations from which 
our implementations are derived come from various authors, and are often designed 
as one-off scripts rather than proper library functions with well-defined interfaces.

Third, wherever possible, we retain backwards compatibility against existing reference
implementations.
This is achieved via regression testing for numerical equivalence of outputs.
All tests are implemented in the ``nose`` framework. [#]_

.. [#] https://nose.readthedocs.org/en/latest/

Fourth, because MIR is a rapidly evolving field, we recognize that the
exact implementations provided by librosa may not represent the state of the art
for any particular task.  Consequently, functions are designed to be `modular`,
allowing practitioners to provide their own functions when appropriate, e.g.,
a custom onset strength estimate may be provided to the beat tracker as a function
argument.
This allows researchers to leverage existing library functions while experimenting with 
improvements to specific components.  Although this seems simple and obvious, from a practical 
standpoint, the monolithic designs and lack of interoperability between different research codebases
has often made this difficult in the past.

Finally, we strive for readable code, thorough documentation, and exhaustive testing.
All development is conducted on GitHub.  
We apply modern software development practices, such as continuous integration testing (via Travis [#]_) and
coverage (via Coveralls [#]_).
All functions are thoroughly documented using Sphinx, and include example code demonstrating usage.
Librosa mostly complies with PEP-8 recommendations, with a small set of exceptions for variable names 
that make the code more concise without sacrificing clarity |---| 
e.g., ``y`` and ``sr`` are preferred over more verbose names such as ``audio_buffer`` and ``sampling_rate`` 
|---| and limits on the number of function parameters in certain cases.

.. [#] https://travis-ci.org
.. [#] https://coveralls.io

Conventions
===========

In general, librosa's functions tend to expose all relevant parameters to the caller.
While this provides a great deal of flexibility to expert users, it can be overwhelming
to novice users who simply need a consistent interface to process audio files.  
To satisfy both needs, we define a set of general conventions and standardized default 
parameter values shared across many functions.

An audio signal is represented as a one-dimensional ``numpy`` array, denoted as ``y`` 
throughout librosa.  Typically the signal ``y`` is accompanied by the `sampling rate` 
(denoted ``sr``) which denotes the frequency (in Hz) at which values of ``y`` are
sampled.  The duration of a signal can then be computed by dividing the number of samples
by the sampling rate: 

.. code-block:: python

    >>> track_duration = float(len(y)) / sr

By default, when loading stereo audio files, the ``librosa.load()`` function 
downmixes to mono by averaging left- and right-channels, and then resamples the
monophonic signal to the default rate ``sr=22050`` Hz.

Most audio analysis methods operate not at the native sampling rate of the signal, 
but over small `frames` of the signal which are spaced by a `hop length` (in samples).
Librosa uses default frame and hop lengths of 2048 and 512 samples, respectively.
At the default sampling rate of 22050 Hz, this corresponds to overlapping frames of 
approximately 93ms spaced by 23ms.
Frames are centered by default, so frame index ``t`` corresponds to the half-open time interval::

    [t - frame_length / 2, t + frame_length /2),

where the boundary conditions are handled by reflection-padding the input.
For analyses that do not use fixed-width frames (such as the constant-Q transform), the
default hop length of 512 is retained to facilitate alignment of results.

The majority of feature analyses implemented by librosa produce two-dimensional outputs
stored as ``numpy.ndarray``, e.g., ``S[f, t]`` might contain the energy within a particular 
frequency band ``f`` at frame index ``t``.
Librosa follows the convention that the final dimension provides the index over time,
e.g., ``S[:,0], S[:,1]`` access features at the first and second frames.
Feature arrays are organized column-major (Fortran style) in memory, so that common
access patterns benefit from cache locality.



Package organization
--------------------

In this section, we give a brief overview of the structure of the librosa software
package.  This overview is not intended as a complete API reference, which can be found at
https://bmcfee.github.io/librosa.


Core
====

The ``librosa.core`` submodule implements a variety of commonly used functions.  Broadly,
``core`` functionality falls into four categories: audio and time-series operations,
spectrogram calculation, time and frequency conversion, and pitch operations.  For
convenience, all functions within the ``core`` submodule are aliased at the top level of
the package hierarchy, e.g., ``librosa.core.load`` is aliased to ``librosa.load``.




Spectral features
=================


Display
=======



Decompositions
==============

Onsets, tempo, and beats
========================

Structural analysis
===================

Effects
=======



Advanced functionality
----------------------

Caching
=======

scikit-learn integration
========================

Filter bank construction
========================

Utilities
=========


.. figure:: tour.pdf
    :scale: 60%
    :figclass: wht

    Top: a waveform plot for a 20-second audio clip ``y``, generated by ``librosa.display.waveplot``.
    Middle: the log-power short-time Fourier transform (STFT) spectrum for ``y`` plotted on a logarithmic
    frequency scale, generated by ``librosa.display.specshow``.
    Bottom: the onset strength function (``librosa.onset.onset_strength``), detected onset events
    (``librosa.onset.onset_detect``), and detected beat events (``librosa.beat.beat_track``) for ``y``.
    :label:`fig:tour`

As you can see in Figure :ref:`fig:tour`, this is how you reference auto-numbered
figures.


Parameter tuning
----------------

Future directions
-----------------

Conclusion
----------

References
----------
.. [Pedregosa11] Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier
                 Grisel, Mathieu Blondel et al. *Scikit-learn: Machine learning in Python.*
                 The Journal of Machine Learning Research 12 (2011): 2825-2830.

.. [Bergstra11] Bergstra, James, Frédéric Bastien, Olivier Breuleux, Pascal Lamblin, Razvan Pascanu, Olivier
                Delalleau, Guillaume Desjardins et al. *Theano: Deep learning on gpus with python.*
                In NIPS 2011, BigLearning Workshop, Granada, Spain. 2011.

.. [Jones01] Jones, Eric, Travis Oliphant, and Pearu Peterson. 
             *SciPy: Open source scientific tools for Python.* 
             http://www.scipy.org/ (2001).

.. [VanDerWalt11] Van Der Walt, Stefan, S. Chris Colbert, and Gael Varoquaux.
                  *The NumPy array: a structure for efficient numerical computation.* 
                  Computing in Science & Engineering 13, no. 2 (2011): 22-30.
