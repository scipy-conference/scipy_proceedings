:author: Antoine Dujardin
:email: dujardin@slac.stanford.edu
:institution: SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, CA 94025, USA

:author: Elliott Slaugther
:institution: SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, CA 94025, USA

:author: Jeffrey Donatelli
:institution: Lawrence Berkeley National Laboratory, Berkeley, CA 94720-8142, USA

:author: Peter Zwart
:institution: Lawrence Berkeley National Laboratory, Berkeley, CA 94720-8142, USA

:author: Amedeo Perazzo
:institution: SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, CA 94025, USA

:author: Chun Hong Yoon
:email: yoon82@slac.stanford.edu
:institution: SLAC National Accelerator Laboratory, 2575 Sand Hill Road, Menlo Park, CA 94025, USA

:bibliography: mybib

:video: <To Be Determined>

------------------------------------------
Fluctuation X-ray Scattering real-time app
------------------------------------------

.. class:: abstract

   The Linac Coherent Light Source (LCLS) at the SLAC National Accelerator Laboratory is an X-ray Free Electron Laser (X-FEL) enabling scientists to take snapshots of single macromolecules to study their structure and dynamics. A major LCLS upgrade, LCLS-II, will bring the repetition rate of the X-ray source from 120 to 1 million pulses per second.  and High Performance Computing capabilities in the exascale will be required for the data analysis to keep up with the future data taking rates.

   We present here a Python application for Fluctuation X-ray Scattering (FXS), an emerging technique for analyzing biomolecular structure from the angular correlations of FEL diffraction snapshots with one or more particles in the beam. This FXS application for experimental data analysis is being developed to run on supercomputers in near real-time while an experiment is taking place.

   We will discuss how we accelerated the most compute intensive parts of the application and how we used Pygion, a Python interface for the Legion task-based programming model, to parallelize and scale the application.

.. class:: keywords

   To Be Determined

Introduction
------------

LCLS-II, an upgrade to LCLS
+++++++++++++++++++++++++++

The Linac Coherent Light Source (LCLS) at the SLAC national accelerator laboratory is an X-ray Free Electron Laser providing femtosecond pulses with an ultrabright beam approximately one billion times brighter than synchrotrons. Such a brightness allows to work with much smaller sample sizes while the shortness allows imaging below the rotational diffusion time of the molecules. With pulses of such an unprecedented brightness and shortness, scientists are able to take snapshots of single macromolecules without the need for crystallization at ambient temperature.

To push the boundaries of the science available at the lightsource, LCLS is currently being upgraded after 10 years of operation. The LCLS-II upgrade will progressively increase the sampling rate from 120 pulses per second to 1 million. At these rates, the LCLS instruments will generate multiple terabytes per second of science data and it will therefore be critical to know what data is worth saving, requiring on-the-fly processing of the data. Earlier, users could classify and preprocess their data after the experiment, but this approach will become either prohibitive or plainly impossible. This leads us to the requirement of performing some parts of the analysis in real time during the experiment.

Quasi real time analysis of the LCLS-II datasets will require High Performance Computing, potentially at the Exascale, which cannot be offered in-house. Therefore, a pipeline to a supercomputing center is required. The Pipeline itself starts with a Data Reduction step to reduce the data size, using vetoing, feature extraction, and compression in real time. We then pass the data over the Energy Sciences Network (ESnet) to the National Energy Research Scientific Computing Center (NERSC). ESNet is a high-speed network with a current capability of 100Gbps, which will have to be increased to the Tbps range. At the end of the pipeline, the actual analysis can take place on NERSC’s supercomputers. This makes the whole process, from the sample to the analysis, quite challenging to change and adapt.

However, LCLS experiments are typically high-risk / high-reward and involve novel setups, varying levels of requirements, and last only for a few days of beam time. The novelty in the science can require adaptations in the algorithms, requiring the data analysis itself to be highly flexible. Furthermore, we want to give users as much freedom as possible in the way they analyze their data without expecting them to have a deep knowledge of large-scale computer programming.

Therefore, we require real time analysis, high performance computing capabilities and a complex pipeline, while at the same time requiring enough flexibility to adapt to novel experimental setups and analysis algorithms. We believe Python helps us in this compromise pretty well.

FXS: an example analysis requiring HPC
++++++++++++++++++++++++++++++++++++++

While a variety of experiments that can be performed at LCLS, we will here focus on one specific example: Fluctuation X-ray Scattering (FXS).

X-ray scattering of particles in a solution is a common technique in the study of the structure and dynamics of macromolecules in biologically-relevant conditions and gives an understanding of their function. However, traditional methods currently used at synchrotrons suffer from the fact that the exposure time is longer than the rotation time of the particle, leading to the capture of angularly-averaged patterns.
FXS techniques fully utilize the femtosecond pulses to measure diffraction patterns from multiple identical macromolecules below the sample rotational diffusion times (Fig. :ref:`fig:fxs`). The patterns are then collected to reconstruct a 3D structure of the macromolecule or measure some of its properties. It has been described in the late 1970s (Kam, 1977; Kam et al., 1981) and has then been performed at LCLS before the upgrade (Pande et al., 2018; Kurta et al., 2017; Medez et al., 2014 & 2016).

.. figure:: FXS-overview.jpg

   Fluctuation X-ray Scattering experiment setup. :label:`fig:fxs`

While a few hundreds of diffraction patterns might be sufficient in good conditions and for a low resolution (Kurta et al., 2017), the number of snapshots required can be dramatically increased when working with low signal-to-noise ratios (e.g. small proteins) or when studying low-probability events. More interestingly, the addition of a fourth dimension, time, to study dynamical processes expands again the amount of data required. At these points, billions or more snapshots could be required.

We present here a Python application for FXS data analysis that is being developed to run on supercomputing facilities at US national laboratories in near real-time while an experiment is taking place. As soon as data is produced, it is passed through a Data Reduction Pipeline on-site and sent to a supercomputer via ESNet, where reconstructions can be performed. It is critical to complete this analysis in near real-time to guide experimental decisions.

In FXS, each diffraction pattern contains several identical particles in random orientations. Information about the structure of the individual particle can be recovered by studying the two-point angular correlation of the data. To do so, the 2D images are expanded in a 3D, orientation-invariant space, where they are aggregated using the following formula:

C2(q,q',DeltaPhi)=1/2piN Sum(j=1->N) Int(0->2pi) Ij(q,Phi) Ij(q',Phi+DeltaPhi) dPhi, (1)

where  Ij(q,) represents the intensity of the j-th image, in polar coordinates. This correlator can then be used as a basis for the actual 3D reconstruction of the data (Fig. 2), using an algorithm described elsewhere (Donatelli et al., 2015; Pande et al., 2018).

Acceleration: getting the best out of numpy
-------------------------------------------

The expansion/aggregation step presented in Equation (1) was originally the most computation intensive part of the application, representing the vast majority of the computation time. The original implementation was processing each Ij(q,)image one after the other and aggregating the results. This resulted in taking 424 milliseconds per image using numpy functions and slightly better performances using numba. As we will illustrate in this section, rewriting this critical step allowed us to gain a factor of 40 in its speed, without any other libraries or tools.

Let us start by simplifying Equation (1). The integral corresponds to the correlation over  of Ij(q,) and Ij(q',). Thanks to the Convolution Theorem, we have
C2(q,q',)=12N j=1NF-1[F[Ij(q,)] F[Ij(q',)]], (2)
where F represents the Fourier transform over . The inverse Fourier transform being linear, we can get it outside of the sum, and on the left side. For the simplicity of the argument, we will also neglect all coefficients.

Using  as the equivalent of  in the Fourier transform and Aj(q,) as a shorthand for F[Ij(q,)], we have:
C2(q,q',)=12N j=1NAj(q,) Aj(q',). (3)
We end up with the naive implementation below:

.. code-block:: python

  C2 = np.zeros(C2_SHAPE, np.complex128)
  for i in range(N_IMGS):
      A = np.fft.fft(images[i], axis=-1)
      for j in range(N_RAD_BINS):
          for k in range(N_RAD_BINS):
              C2[j, k, :] += A[j] * A[k].conj()

taking 42.4 seconds (for 100 images), using the following parameters:

.. code-block:: python

  N_IMGS = 100
  N_RAD_BINS = 300
  N_PHI_BINS = 256
  IMGS_SHAPE = (N_IMGS, N_RAD_BINS, N_PHI_BINS)
  C2_SHAPE = (N_RAD_BINS, N_RAD_BINS, N_PHI_BINS)

and the dataset

.. code-block:: python

  images = np.random.random(IMGS_SHAPE)

We will note that a typical application would be processing millions of images, but let us use 100 for the example.

This naive version can be slightly accelerated using the fact that our matrix is conjugate-symmetric:

.. code-block:: python

  C2 = np.zeros(C2_SHAPE, np.complex128)
  for i in range(N_IMGS):
      A = np.fft.fft(images[i], axis=-1)
      for j in range(N_RAD_BINS):
          C2[j, j, :] += A[j] * A[j].conj()
          for k in range(j+1, N_RAD_BINS):
              tmp = A[j] * A[k].conj()
              C2[j, k, :] += tmp
              C2[k, j, :] += tmp.conj()

which takes 36.0 seconds. Let us note that this is only 1.18 times faster, far from a 2x speed-up.

That naive implementation should not be confused with a pure Python implementation, which would be expected to be slow, since we already operate on numpy arrays along the  axis. Such an implementation could be approximated by:

.. code-block:: python

  A = np.fft.fft(images[i], axis=-1)
  for j in range(N_RAD_BINS):
      for k in range(N_RAD_BINS):
          for l in range(N_PHI_BINS):
              C2[j, k, l] += A[j, l] * A[k, l].conj()

which takes 49.1 seconds per image, i.e. about 100 times slower, in accordance with the stereotype of Python being much slower than other languages.

A common acceleration strategy is to use numba:

.. code-block:: python

  @numba.jit
  def A_to_C2(A):
      C2 = np.zeros(C2_SHAPE, np.complex128)
      for j in range(N_RAD_BINS):
          C2[j, j, :] += A[j] * A[j].conj()
          for k in range(j+1, N_RAD_BINS):
              tmp = A[j] * A[k].conj()
              C2[j, k, :] += tmp
              C2[k, j, :] += tmp.conj()
      return C2
  C2 = np.zeros(C2_SHAPE, np.complex128)
  for i in range(N_IMGS):
      A = np.fft.fft(images[i], axis=-1)
      C2 += A_to_C2(A)

which takes 38.5 seconds, i.e. 1.10 times faster than the naive implementation.

When considering our problem size of up to millions of images, processing images one at a time makes sense. However, focusing on a small batch as we have been doing in these examples, a strategy can be to have numpy and/or numba work on arrays of images, rather than the individual images. We then have the following:

.. code-block:: python

  @numba.jit
  def As_to_C2(As):
      C2 = np.zeros(C2_SHAPE, np.complex128)
      for i in range(N_IMGS):
          A = As[i]
          for j in range(N_RAD_BINS):
              C2[j, j, :] += A[j] * A[j].conj()
              for k in range(j+1, N_RAD_BINS):
                  tmp = A[j] * A[k].conj()
                  C2[j, k, :] += tmp
                  C2[k, j, :] += tmp.conj()
      return C2
  As = np.fft.fft(images, axis=-1)
  C2 = As_to_C2(As)

which takes 11.9 seconds, i.e. 3.56 times faster. We will note also here the batching of the Fast Fourier Transform.

However, such an implementation does not sound trivial using numpy… although one can recognize a nice (generalized) Einstein sum in Equation (3), leading to:

.. code-block:: python

  As = np.fft.fft(images, axis=-1)
  C2 = np.einsum('hik,hjk->ijk', As, As.conj())

This takes 17.9 seconds, which is slower than the version using numba per batch. However, we can realize that, at this batch level, the last axis is independent from the others… and that the underlying alignment of the arrays matters. Thanks to numpy’s `asfortranarray` function, however, that is not an issue. We will use the F-ordered dataset.

.. code-block:: python

  images_F = np.asfortranarray(images)

We observe, for the Einstein sum:

.. code-block:: python

  As = np.fft.fft(images_F, axis=-1)
  C2 = np.einsum('hik,hjk->ijk', As, As.conj())

taking 4.05 seconds, i.e. 4.42 times faster than the C-ordered Einstein sum and 10.5 times faster than the naive implementation.

Further than that, in our precise case, we can actually express it as a more optimized dot product:

.. code-block:: python

  As = np.fft.fft(images, axis=-1)
  C2 = np.zeros(C2_SHAPE, np.complex128)
  for k in range(N_PHI_BINS):
      C2[..., k] += np.dot(As[..., k].T,
                           As[..., k].conj())

which now brings us down to 1.37 seconds, i.e. 30.9 times faster than the naive version.

For the F-ordered case, we have:

.. code-block:: python

  As = np.fft.fft(images_F, axis=-1)
  C2 = np.zeros(C2_SHAPE, np.complex128, order='F')
  for k in range(N_PHI_BINS):
      C2[..., k] += np.dot(As[..., k].T,
                           As[..., k].conj())

taking 1.06 seconds, i.e. 1.29 times faster than the C-ordered case and 40.0 times faster than the naive implementation.
We could note that, at that speed, the main computation gets close to the time required to perform the Fast Fourier Transform, which is, in our case at least, faster on C-ordered (107 ms) than F-ordered (230 ms) data. Removing the FFT computation would yield an even starker contrast (977 ms vs. 499 ms), but would neglect the cost of the re-alignment.

In conclusion, implementing using numpy or numba naively gives significant improvement on computational speed compared to pure Python, but there is still a lot of room for improvement. On the other hand, such improvement does not necessarily require using fancier tools. In our case, we showed that batching our computation helped in the numba case. From there, a batched numpy expression looked interesting. However, it required playing around with the mathematical formulation of the problem to come up with a canonical expression, which could then be handed over to numpy. Last but not least, the memory layout can have a sizable impact on the computation, while being easy to tweak in numpy.

Parallelization: effortless scaling with Pygion
-----------------------------------------------

<Placeholder>

Results
-------

<Placeholder>

Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension. 

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work. 

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references. 

So what would previously have registered as a in text reference ``[Atr03]_`` for 

:: 

     [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

what you actually see will be an empty reference rendered as **[?]**.

E.g., [Atr03]_.


If you wish to have a block quote, you can just indent the text, as in 

    When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication. :cite:`hume48`

Dois in bibliographies
++++++++++++++++++++++

In order to include a doi in your bibliography, add the doi to your bibliography
entry as a string. For example:

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = "10.1017/CBO9780511808432",
   }


If there are errors when adding it due to non-alphanumeric characters, see if
wrapping the doi in ``\detokenize`` works to solve the issue.

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = \detokenize{10.1017/CBO9780511808432},
   }

Source code examples
--------------------

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


