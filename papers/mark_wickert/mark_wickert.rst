:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado


----------------------------------------------------------------------------------
Signal Processing and Communications: Teaching and Research Using IPython Notebook
----------------------------------------------------------------------------------

.. class:: abstract

   This paper will take the audience through the story of how an electrical and computer
   engineering faculty member has come to embrace Python, in particular IPython Notebook
   (IPython kernel for Jupyter),
   as an analysis and simulation tool for both teaching and research in signal processing
   and communications. Legacy tools such as MATLAB are well established (entrenched) in
   this discipline, but engineers need to be aware of alternatives, especially in the case
   of Python where there is such a vibrant community of developers and a seemingly well
   designed language. In this paper case studies will also be used to describe domain
   specific code modules that are being developed to support both lecture and lab oriented
   courses going through the conversion from MATLAB to Python. These modules in particular
   augment ``scipy.signal`` in a very positive way and enable rapid prototyping of
   communications and signal processing algorithms. Both student and industry team
   members in subcontract work, have responded favorably to the use of Python as an
   engineering problem solving platform. In teaching IPython notebooks are used to augment
   lecture material with live calculations and simulations. These same notebooks are then
   placed on the course Web Site so students can download and *tinker* on their own. This
   activity also encourages learning more about the language core and Numpy, relative to
   MATLAB. The students quickly mature and are able to turn in homework solutions and
   complete computer simulation projects, all in the notebook. Rendering notebooks to
   PDF via LaTeX is also quite popular. As a side note, students are also learning how
   to typeset mathematics using LaTeX, as this need follows naturally when writing text
   in markdown cells. The next step in the transformation process is to get other faculty
   involved. Going forward the plan is to teach some tutorial sessions to faculty and their graduate students.

.. class:: keywords

   numerical computing, signal processing, communications systems, system modeling

Introduction
------------

This journey into Python for electrical engineering problem solving began with
the writing of the book *Signals and Systems for Dummies* [Wic2013]_, published summer
2013. This book features the use of Python (``Pylab``) to
bring life to the mathematics behind signals and systems theory. Using Python in the Dummies book was done to make
it easy for all readers of the book to develop their signals and system problem solving skills, without additional
software tools investment. Additionally, the author provides the custom code module ``ssd.py`` [ssd]_, which is built on
top of  ``numpy``,  ``matplotlib``, and ``scipy.signal``, to make it easy to work and extend the examples found in
the book. Engineers love to visualize their work with plots of various types. All of the plots in the book were
created using Python, specifically ``matplotlib``.

The next phase of the journey, focuses on the research and development
side of signals and systems work. During a sabbatical working for a small engineering firm, academic year
2013--2014, Python and IPython notebook (IPython kernel for Jupyter) served as the primary digital signal
processing modeling tools on three different projects. Initially it was not clear which tool whould be best, but
following discussions with co-workers Python seemed to be the right choice. Note for the most part the signals and
systems analysis team was new to Python, all of us having spent many years using MATLAB/Octave. A nice motivating
factor is that Python was already in the workflow with one of the real-time DSP products used by the company, so
prototyping algorithms in Python using ``numpy`` was a already proven.

The third and current phase of the Python
transformation began at the start of the 2014-2015 academic year. The big move was made to push out Python to the
students, via the IPython Notebook, in five courses: digital signal processing, digital communications, analog
communications, statistical signal processing, and real-time signal processing. Four of the courses are traditional
lecture format, while the fifth is very hands-on lab oriented, involving embedded systems programming and hardware
interfacing. IPython Notebook works very well for writing lab reports, and easily allows theoretical and experimental
results to be integrated. A notebook interface is not a new concept in scientific computing tool sets, see for example
*Mathematica* [Mathematica]_ (commercial) and *wxMaxima* [Maxima]_ (open source). Both of these tools are very
powerful and are a joy to use for specific problem classes.
In my view Python with ``pylab`` and ``sympy`` provides a more comprehensive set of capabilities. By
comprehensive I mean Python a one stop language with the ability to satisfy modeling, analysis, simulation, and in some
cases an implementation means.

Getting other faculty on-board is the next step. I am optimistic and
look forward to this challenge as tutorial sessions are planned over summer 2015.

The remainder of this paper is organized into the following sections: arriving at Python for communications and signal
processing modeling, Python for , classroom use, case studies, and conclusions.

Arriving at Python for Communications and Signal Processing Modeling
--------------------------------------------------------------------

About three years ago while working on a study contract for a small business, I started investigating the use of
open-source alternatives over MATLAB. I initially homed in on using Octave [Octave]_ for its syntax compatibility
with MATLAB. Later I started to explore Python and became fascinated by the ease of use offered by the IPython (QT)
console and the high quality of ``matplotlib`` 2D plots. The full power of Python
for engineering and scientific computing became evermore obvious as I learned more about the language and the
engineering problem capabilities offered by ``pylab``.

When I took on the assignment of writing the *Signals and Systems for Dummies* book [Wic2013]_ Python seemed like a
good choice because of the relative ease with which anyone could obtain the tools and then get hands-on experience with
the numerical examples I was writing into the book. The power of ``numpy`` and algorithms available in ``scipy`` are
indeed wonderful, but I immediately recognized that enhancements to ``scipy.signal`` would be needed to make signals
and systems tinkering user friendly. As examples were written for the book, I began to write support functions that
fill in some of the missing details not found in ``scipy``. This is the basis for the module ``ssd.py``.
At present writing additional support modules for signals and systems work (in particular communications and signal
processing) is ongoing. Note *control systems* is traditionally the third and final sub heading under the signals and
systems banner.

Modules Developed or Under Development
======================================

As already briefly mentioned, the first code module developed was ``ssd.py``, which was written to support
[Wic2013]_. The function listing via ``dir()`` is given below:

.. code-block:: python

   In[13]: dir(ssd)
   Out[13]:
   ['BPSK_tx', 'CIC', 'NRZ_bits', 'NRZ_bits2',
    'OA_filter', 'OS_filter', 'PN_gen', 'am_rx',
    'am_rx_BPF', 'am_tx', 'biquad2', 'bit_errors',
    'cascade_filters', 'conv_integral', 'conv_sum',
    'cpx_AWGN', 'cruise_control', 'deci24',
    'delta_eps', 'dimpulse', 'downsample', 'drect',
    'dstep', 'env_det', 'ex6_2', 'eye_plot', 'fft',
    'fir_iir_notch', 'from_wav', 'fs_approx',
    'fs_coeff',  'interp24', 'line_spectra', 'lms_ic',
    'lp_samp', 'lp_tri', 'm_seq', 'mlab', 'my_psd',
    'peaking', 'plot_na', 'plt', 'position_CD',
    'prin_alias', 'pylab', 'rc_imp', 'rect',
    'rect_conv', 'scatter', 'signal', 'simpleQuant',
    'simple_SA', 'sinusoidAWGN', 'soi_snoi_gen',
    'splane', 'sqrt_rc_imp', 'step', 'ten_band_eq_filt',
    'ten_band_eq_resp', 'to_wav', 'tri', 'upsample',
    'zplane']

This collection of functions provides general support for both continuous and discrete-time signals and systems as
well as specific support for examples found in [Wic2013]_. More modules have followed since then.

The second module developed, ``digitalcom.py``, focuses on the special needs of digital communications, both modulation
and demodulation. At present this module contains the following functions:

.. code-block:: python

   In[17]: dir(digitalcom)
   Out[17]:
   ['BPSK_BEP', 'BPSK_tx', 'CIC', 'GMSK_bb', 'MPSK_bb',
    'NRZ_bits', 'NRZ_bits2', 'PN_gen', 'QAM_SEP',
    'QAM_bb', 'QPSK_BEP', 'QPSK_bb', 'QPSK_rx',
    'QPSK_tx', 'Q_fctn', 'RZ_bits', 'bit_errors',
    'cpx_AWGN', 'downsample', 'erfc', 'eye_plot',
    'farrow_resample', 'm_seq', 'my_psd', 'rc_imp',
    'scatter', 'signal', 'sqrt_rc_imp', 'strips',
    'time_delay', 'upsample', 'xcorr']

More functions are under development for this module, particularly in the area of orthogonal frequency division
multiplexing (OFDM), the key modulation type found in the wireless telephony standard long term evolution (LTE).

A third module, ``fec_conv.py``, implements a rate one-half convolutional encoding and decoding class.
Arbitrary constraint length codes can be employed as well as puncturing and depuncturing patterns. For decoding the
soft decision Viterbi algorithm is used. A feature of this
class is a graphical display function which shows the survivor traceback paths through the trellis back to the
decision depth. This gives students insight into the operation of the Viterbi algorithm, which at a high level is a
*dynamic programming* algorithm.

.. code-block:: python

   In[19]: dir(fec_conv)
   Out[19]:
   ['Q_fctn', 'binary', 'conv_Pb_bound',
    'fec_conv', 'hard_Pk', 'soft_Pk',
    'trellis_branches', 'trellis_nodes',
    'trellis_paths']

Besides the class ``fec_conv``, this module also contains functions for computing error probability bounds using
the *weight structure* of the code under both *hard* and *soft* branch metric distance calculations [Zie2015]_.
The key methods found in the class ``fec_conv`` are:

..  code-block:: python

     Class: fec_conv
     Method: viterbi_decoder
     Method: depuncture
     Method: conv_encoder
     Method: bm_calc
     Method: traceback_plot
     Method: puncture
     Method: trellis_plot

Both the encoder and especially the Viterbi decoder are numerically intensive. Speed enhancements, perhaps using
*Cython* are the list of things to do. An example of using the class ``fec_conv`` can be found in the Case
Studies section.

A fourth module, ``synchronization.py``, was developed while teaching a phase-locked loops course Summer 2014. This
module supplies simulation functions for a basic phase-locked loop and both carrier and symbol synchronization
functions for digital communications waveforms. This module was utilized in an analog communications course taught
Spring 2015.

.. code-block:: python

   In[21]: dir(synchronization)
   Out[21]:
   ['DD_carrier_sync', 'MPSK_bb', 'NDA_symb_sync',
    'PLL1', 'PLL_cbb', 'phase_step', 'signal',
    'time_step']

More modules are planned as well enhancements to the existing modules. A great side benefit of using IPython
notebook is algorithms can be prototyped in a notebook and later moved to an existing module or perhaps be the start
of a new module. During the fall of spring semesters many new functions and few classes were developed in notebooks.
Where it makes sense, some of this code can now be migrated into modules. On the flip side, modules are neat and
tidy, but when introducing new concepts to students, placing algorithms inside notebooks has the advantage of making
the code visible, and invites tinkering.

Describing IPython Notebook Use Cases
-------------------------------------

In this section I describe how Python and in particular the IPython notebook has been integrated into teaching,
graduate student research, and industry research and development.

Teaching
========

The present lecturing style for all courses I teach
involves the use of a tablet PC, a data projector, a microphone, and audio/video screen capture software, e.g.
Camtasia Studio [Camtasia]_. Live Python demos are run in the notebook, and in many cases all the code is developed in
real-time as questions come from the class. The notebook is more that just a visual experience.
A case in point is the notebook audio control which adds sound playback capability. A 1D ``ndarray`` can be saved as a
*wave file* for playback. Simply put signals do make sounds and the
action of systems changes what can be heard. Students enjoy hearing as well as seeing results.
By interfacing the tablet *lineout* or *headphone* output to the podium
interface to the classroom speakers, everyone can hear the impact of algorithm tweaks on what is being heard. This
is where the fun starts! Core modules such as ``ssd.py`` described earlier, are imported at the top of each notebook.

For each new chapter of material from the course text, a new notebook is created. Starter content is added to
say the *Chapter x* notebook before the lecture to provide relevant theory right in the notebook. Specifically
text and mathematics are placed in *markdown cells*. The theory is very brief as the course lecture notes, written
using LaTeX, form the core lecture material. Back in the notebook, numerical examples follow the brief mathematical
development. Here some plots will be generated in advance, but the intent is to make parameter changes during the
lecture, so the students can get a feel for how the math model relates to real-word signals and systems.

Computer projects benefit greatly from the use of the notebook, as sample notebooks with starter code can easily be
posted to the course Web Site. The sample notebook serves as a template for the project report document that the
student will work with and ultimately turn in for grading.  The ability to convert the notebook to a LaTeX PDF
document has proven to work well in practice. It is work noting that setting up Pandoc and a LaTeX install takes
some effort on the student's part. From my recent experiences, not all students went to this extreme.
An easy alternative is to take *screenshots* of selected notebook cells and paste them into a word processor document.

Graduate Student Research
=========================

In working with graduate students on their research, it is normal to exchange code developed by fellow graduate
students working on related problems. Explaining how code works with worked examples is a perfect use case for
IPython notebook. The same approach holds for faculty interaction with their graduate students. In this scenario the
faculty member, who is typically short on free time, gains a powerful advantage in that more than one student may need
to brought up to speed on the same code base. Once the notebook is developed it can be shared with many students and
can be demoed in front of students on a lab or office computer. More fundamentally, the markdown cells of the notebook
can be used to refresh your memory as to the mathematical model implemented in code and explain the code interface
beyond what is found in the *docstring*. The ability to include figures means that system block diagrams can also be
placed in the notebook.

As the student makes progress on research tasks the faculty member(s) can be easily briefed on the math models and
simulation results. Since the notebook is live, the inevitable *what if* questions can be asked and hopefully quickly
answered.

Industry Research and Development
---------------------------------

With the notebook engineers working on the same team are able to share analytical models and  development approaches
using markdown cells. The ability to include equations using LaTeX markup is fantastic, as mathematical developments,
including the establishment of notational conventions, is the first step in the development of signal processing
algorithms.

Later, prototype algorithm development can be started using code cells. Initially synthesized signals (waveforms)
can be used to validated the core functionality of an algorithm. Next, signal captures from the actual real-time
hardware can be used as a source of test vectors to verify that performance metrics are being achieved. Notebooks
can again be passed around to team members for further algorithm testing. Soon code cell functions can be moved to
code modules and the code modules distributed to team members via GIT or some other distributed revision control
system. At every step of the way ``matplotlib`` graphics are used to visualize performance of a particular
algorithm versus say a performance bound.

Complete subsystem testing at the Python level may be sufficient in some cases. In a more typical case code will
have to moved to a production environment and recoding may be required. It might also be that the model is simply
an abstraction of real electronic hardware, in which case a hardware implementer uses the notebook (may just a PDF
version) to create a hardware prototype.

Live From the Classroom
-----------------------

Here live from the classroom means responding to questions using on-the-fly IPython notebook demos. This is an excellent
way to show off the power of Python. Sometimes questions come and you feel like building a quick model right then and
there during a lecture. When successful, this hopefully locks in a solid understanding of the concepts involved for
the whole class. The fact that the lecture in being recorded means that students can recreate a same demo at their
leisure when they watch the lecture video. The notebook goes further than a commandline interface live demo. The
notebook can be saved and posted as a supplement/companion to the lecture. As mentioned earlier, I started a new
notebook for each chapter of lecture material. The goal was to re-post the chapter IPython notebook each time a new
leture video was posted. This way the students would have something to play with as they started to work on their
next homework assignment.

Case Studies
------------

In this section several case studies are presented. Each case study details one or more of the IPython notebook
use cases described in the previous sections of this paper. Case studies from industry R&D are not included here due
to the propriety nature of the work.

Digital Signal Processing
=========================

As a simple starting point this first case study deals with the mathematical representation of signals. It is taken
from a notebook used during a lecture. A step function
sequence :math:`u[n]` is defined as

.. math::

   u[n] = \begin{cases} 1, & n \geq 0 \\ 0, & \text{otherwise} \end{cases}

Here I consider the difference between two step sequences starting at :math:`n=0` and the other starting at :math:`n=5`.
I thus construct in Python

.. math::

   x_3[n] = x_1[n] - x_2[n] = u[n] - u[n-5],

which forms a pulse sequence which *turns on* at :math:`n=0` and *turns off* at :math:`n=5`. A screen capture from
the IPython notebook is shown in Fig. :ref:`fig1`.

.. figure:: scipy_2015_fig1.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Discrete-time signal generation and manipulation. :label:`fig1`

Of special note in this case study is how the code syntax for the generation of the sequences follows closely the
mathematical form. Note to save space the details of plotting :math:`x_2[n]` and :math:`x_3[n]` is omitted, but the
code that generates and plots :math:`x_3[n]` is simply:

.. code-block:: python

   stem(n,x1 - x2)

Digital Communications
======================

In this case study the coding theory class contained in ``fec_conv.py`` is exercised. The specific case is taken from
the final exam. Fig. :ref:`fig2` shows the construction of the ``fec_conv`` object and a plot of one stage of the
Viterbi algorithm (dynamic programming) trellis.

.. figure:: scipy_2015_fig2.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Construction of a `fec_conv` object and the corresponding trellis structure. :label:`fig2`

Fig. :ref:`fig3` shows the survivor traceback paths in the 16-state
trellis while sending 1000 random bits through the encoding/decoding processes.
The channel signal-to-noise ratio (SNR) (in the code cell denoted :math:`E_b/N_0`) is 7 dB, but at *decision depth* of 25
code symbols all 16 paths merge to a common path, making it very likely that the probability of a bit error is very
very small. At lower SNR merging take longer and errors bit errors are more likely.

.. figure:: scipy_2015_fig3.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Passing random bits through the encoder/decoder and plotting an instance of the survivor paths. :label:`fig3`

Pulse Train Power Spectral Density
==================================

Fourier analysis is common place in both communications and signal processing problems. This case study considers the
power spectral density (PSD) of a continuous-time *pulse train*. The screenshot of Fig. :ref:`fig4` is taken from a
notebook used during
an analog communications theory course lecture. The mathematical model is contained in the notebook followed by a
numerical example which includes a PSD plot. The function ``ssd.line_spectra`` plots the theoretical spectrum.

.. figure:: scipy_2015_fig4.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Formulating the power spectrum of a pulse train signal and then plotting the line spectrum for a particular
   parameter set. :label:`fig4`

Real-Time Signal Processing
===========================

In the real-time signal processing course C-code is written for an embedded processor. In this case the processor
is an ARM Cortex-M4. The objective of this case study is to implement an equal-ripple lowpass filter of prescribed
amplitude response specifications. Python (`scipy.signal`) is used to design the filter and obtain the filter
coefficients in `float64` precision. The processor is working with `int16` precision so once the filter is design
the coefficients are scaled and rounded to 16 bit signed integers as shown in Fig. :ref:`fig5`. The fixed-point filter
coefficients are written to a C header file using a custom function defined in the notebook (not shown here however).

.. figure:: scipy_2015_fig5.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Designing an equal-ripple lowpass filter using `scipy.signal.remez` for real-time operation. :label:`fig5`

The filter frequency response magnitude is obtained using a noise source to drive the filter the input (first passing
through an analog-to-digital converter) and the filter output (following digital-to-analog conversion) is processed
by instrumentation to obtain a spectral estimate. The spectrum estimate corresponds to the filter frequency response.
The measured frequency reesponse is imported into the notebook using `loadtxt`. Fig. :ref:`fig6` compares the
theoretical frequency response, including quantization errors, with the measured. The results are impressive, and the
IPython notebook has made this a breeze.

.. figure:: scipy_2015_fig6.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   Comparing the theoretical fixed-point frequency response with the measured. :label:`fig6`

Statistical Signal Processing
=============================

This case study is taken from a computer simulation project in a statistical signal processing course. The problem
involves the theoretical calculation of the probability density function of a random variable :math:`\mathbf{w}` where

.. math::

   \mathbf{w} = \mathbf{xy}+\mathbf{z}

The screenshot of Fig. :ref:`fig7` explains the problem details, including the theoretical results written out as the
piecewise function ``pdf_proj1_w(w)``.

.. figure:: scipy_2015_fig7.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   One function of three random variables simulation problem. :label:`fig7`

A simulation is constructed and the results are compared with theory in Fig. :ref:`fig8`.

.. figure:: scipy_2015_fig8.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   The simulation of random variable :math:`\mathbf{w}` and the a comparison plot of theory versus a scaled
   histogram. :label:`fig8`

Conclusions
-----------

Scientific Python and the IPython notebook without a doubt has proven its usefulness in a variety of signals and
systems courses and in a real-world R&D work environment. The enthusiasm of the scientific Python developer
community has a lot to do with making Python truly viable as a *first class* engineering problem solving tool.

Communications and signal processing, as a discipline that sits inside electrical computer engineering, is build on
a strong mathematical modeling foundation. When theoretical expressions need to be evaluated and real-time algorithms
need to be tested, we turn to tools with the power to get the job done. Open source community driven alternatives
should not be overlooked.

References
----------
.. [Wic2013] M.A. Wickert. *Signals and Systems for Dummies*,
           Wiley, 2013.
.. [ssd] ``http://www.eas.uccs.edu/wickert/SSD/``.
.. [Mathematica] ``https://en.wikipedia.org/wiki/Mathematica``.
.. [Maxima] ``http://andrejv.github.io/wxmaxima/``.
.. [Octave] ``https://en.wikipedia.org/wiki/GNU_Octave``.
.. [Zie2015] R.E. Ziemer and W.H. Tranter *Principles of Communications*, seventh edition, Wiley, 2015.
.. [Camtasia] ``https://en.wikipedia.org/wiki/Camtasia_ Studio``.



