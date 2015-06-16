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
software tools investment. Additionally, the author provides the custom code module ``ssd.py``, which is built on
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
results to be integrated.

Getting other faculty on-board is the next step. I am optimistic and
look forward to this challenge as tutorial sessions are planned over summer 2015.

The remainder of this paper is organized into the following sections: arriving at Python for signals and systems
modeling, problem solving needs, classroom use, case studies, and conclusions.

Arriving at Python for Signals and Systems Modeling
---------------------------------------------------

About three years ago while working on a study contract for a small business, I started investigating the use of
open-source alternatives over MATLAB. I initially homed in on using Octave [Octave]_ for its syntax compatibility
with MATLAB. Later I started to explore Python and became fascinated by the ease of use offered by the IPython (QT)
console and the high quality of ``matplotlib`` 2D plots. The full power of Python
for engineering and scientific computing became evermore obvious as I learned more about the language and the
engineering problem capabilities offered by ``pylab``.

When I took on the assignment of writing the *Signals and Systems for Dummies* book [Wic2013]_ Python seemed like a
good choice because of the relative ease with which anyone could obtain the tools and then get hands-on experience with
the numerical examples I was writing into the book. The power of ``numpy`` and algorithms available in ``scipy`` are
indeed wonderful, but I immediately recognized that enhancements to ``scipy.signal`` would needed make signals
and systems tinkering user friendly. As examples were written for the book I began to write support functions that
fill in some of the missing details not found in ``scipy``. This is the basis for the first module, ``ssd.py``.
At present writing additional support modules for signals and systems work is ongoing.

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

Both the encoder and the decoder are in need of speed enhancements, perhaps using *Cython*. An example of using the
class ``fec_conv`` will be presented in the Case Studies section.

A fourth module, ``synchronization.py``, was developed while teaching a phase-locked loops course summer 2014. This
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

Python for Signals and Systems Problem Solving
----------------------------------------------

In this section I describe how Python and in particular the IPython notebook has been integrated into teaching,
graduate student research, and industry research and development.

Teaching
========

The present lecturing style for all courses I teach
involves the use of a tablet PC, a data projector, a microphone, and audio/video screen capture software, e.g.
Camtasia Studio [Camtasia]_. Live Python demos are run in the notebook, and in many cases all the code is developed in
real-time as questions come from the class. The notebook audio control adds sound capability, and is very useful in
signal processing and communications courses. By interfacing the tablet *lineout* or *headphone* output to the podium
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
use cases described in the previous sections of this paper.

Digital Signal Processing
=========================



Digital Communications
======================



Analog Modulation
=================



Real-Time Signal Processing
===========================



Statistical Signal Processing
=============================



Conclusions
-----------

IPython notebook without a doubt has proven its usefulness in a variety of signals and systems courses and in a
real-world R&D work environment.


Paper Formatting Examples
-------------------------

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

It is well known [Wic2013]_ that Spice grows on the planet Dune.  Test
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
.. [Wic2013] M. Wickert. *Signals and Systems for Dummies*,
           Wiley, 2013.
.. [Octave] ``https://en.wikipedia.org/wiki/GNU_Octave``.
.. [Camtasia] ``https://en.wikipedia.org/wiki/Camtasia_ Studio``.
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


