:author: Jaime Arias
:email: jaime.arias@inria.fr
:institution: Inria, MISTIS, Univ. Grenoble Alpes, LJK, F-38000 Grenoble, France
:corresponding:
:equal-contributor:

:author: Philippe Ciuciu
:email: philippe.ciuciu@cea.fr
:institution: CEA/NeuroSpin and Inria Saclay, Parietal, France
:equal-contributor:

:author: Michel Dojat
:email: michel.dojat@univ-grenoble-alpes.fr
:institution: Inserm, U1216, F-38000 Grenoble, France
:institution: Univ. Grenoble Alpes, GIN, F-38000 Grenoble, France
:equal-contributor:

:author: Florence Forbes
:email: florence.forbes@inria.fr
:institution: Inria, MISTIS, Univ. Grenoble Alpes, LJK, F-38000 Grenoble, France
:equal-contributor:

:author: Aina Frau-Pascual
:email: aina.frau-pascual@inria.fr
:institution: Inria, MISTIS, Univ. Grenoble Alpes, LJK, F-38000 Grenoble, France
:equal-contributor:

:author: Thomas Perret
:email: thomas.perret@grenoble-inp.org
:institution: Inria, MISTIS, Univ. Grenoble Alpes, LJK, F-38000 Grenoble, France
:equal-contributor:

:author: Jan M. Warnking
:email: jan.warnking@univ-grenoble-alpes.fr
:institution: Inserm, U1216, F-38000 Grenoble, France
:institution: Univ. Grenoble Alpes, GIN, F-38000 Grenoble, France
:equal-contributor:

:bibliography: biblio

----------------------------------------------------------------------------------------------------------------
PyHRF: A Python Library for the Analysis of fMRI Data Based on Local Estimation of Hemodynamic Response Function
----------------------------------------------------------------------------------------------------------------

.. class:: abstract

   Functional Magnetic Resonance Imaging (fMRI) is a neuroimaging technique
   that allows the non-invasive study of brain functions. It is based on the
   hemodynamic changes induced by cerebral activity following sensory or
   cognitive stimulation. The measured signal depends on the function of blood
   oxygenation level (BOLD signal) which is related to brain activity:
   a decrease in deoxyhemoglobin induces an increase in BOLD signal. In fact,
   this signal is convoluted by the Hemodynamic Response Function (HRF) whose
   exact form is unknown and depends on various parameters (age, brain region,
   physiological conditions).

   In this paper we present PyHRF, a software to analyze fMRI data using
   a joint detection-estimation (JDE) approach. It jointly detects cortical
   activation and estimates the HRF. In contrast to existing tools, PyHRF
   estimates the HRF instead of considering it constant in brain, improving
   thus the reliability of the results. Here, we present an overview of the
   package and ilustrate its use with a real case in order to show that PyHRF
   is a tool suitable for non experts and clinicians.

.. class:: keywords

   BOLD response, fMRI, hemodynamic response function


Introduction
------------

Neuroimaging techniques, as functional Magnetic Resonance Imaging (fMRI), allow
the *in vivo* study of brain functions by measuring the changes induced by
cerebral activity following sensory or cognitive stimulation. For more than
20 years, the blood-oxygen-level-dependent (BOLD) fMRI modality has being the
technique most used by clinicians and neuroscientists to map the main
functional regions of the brain.

BOLD :cite:`Ogawa:1990` reflects the changes in oxygen concentration in the
blood. Briefly, when brain activity occurs, oxygen is locally consumed by
neurons and its concentration in the blood decreases (see Fig.
:ref:`boldchain`).  Therefore, an inflow of oxygenated blood is achieved to
replenish the tissue, increasing blood oxygen concentration. Deoxygenated blood
causes locally magnetic distortions. Thus, BOLD signal is an indirect measure
of cerebral activity based on physiological changes in oxygen consumption,
cerebral blood flow and blood volume.

.. figure:: figures/bold_chain.pdf
   :align: center
   :figclass: htb

   fMRI BOLD signal :cite:`Ogawa:1990`. The BOLD signal measures the local
   changes in blood oxygenation. This ratio changes during brain activity.
   :label:`boldchain`


.. INFO: I put the figure here in order to display it on the top of the second page.

.. figure:: figures/rois_hrfs.pdf
   :align: center
   :scale: 25%
   :figclass: wt

   HRF computed using PyHRF from BOLD data in several parcels belonging to
   visual, auditory and motor regions. :label:`hrfs`


BOLD is non-invasive, non-ionizing, and gives access *in vivo* to brain
activity with a relatively high spatial resolution. It is highly dependent of
the hemodynamic response function (HRF) of the brain. BOLD does not give access
to true physiological parameters such as cerebral blood flow or cerebral blood
volume, but rather measures a mixture of these quantities that is difficult to
untangle. In this regard, BOLD is a very interesting tool in neuroscience, but
in general it is not widely used for clinical applications because the impact
of physiological situation on HRF is unknown, hampering the BOLD signal
interpretation. For instance, it cannot detect chronic changes in the baseline
states :cite:`Buxton:2013`, as it is the case of normal ageing
:cite:`Fabiani:2014` and pathologies like Alzheimer's disease
:cite:`Cantin:2011` or Stroke :cite:`Attye:2014`.

Most used open source libraries for the analysis of fMRI data (*e.g.,* SPM
[#]_, FSL [#]_) consider the HRF of the neuronal activity as
a constant in all the brain and the same for all subjects. However, several
works (see :cite:`Badillo13` for a survey) show that the HRF changes across
different regions of the brain and other individuals, increasing thus the
possibility of obtaining false negatives and decreasing the reliability of the
results. The software PyHRF :cite:`Vincent:2014` was developed to overcome the
above limitation by analyzing fMRI data using a joint detection-estimation
(JDE) approach.


.. [#] SPM official website: http://www.fil.ion.ucl.ac.uk/spm/software/

.. [#] FSL official website: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/


In the JDE approach, the detection of the cortical activation is achieved
together with the estimation of the unknown HRF response by analyzing non
smoothed data.  This detection-estimation is calculated on different parcels of
interest paving the cerebral volume.  Therefore, PyHRF allows to navigate the
brain and to focus on the regions of interest during the experiment in order to
visualize the activations and their temporal behavior through the estimated
HRF. In the last years, efforts have been made in terms of user-friendliness
and usability of the PyHRF package to make it more easy to use by non experts
and clinicians.

Next, we present the PyHRF package. Then, we illustrate its use via a real
example.  Finally, we conclude by discussing directions of current/future work.
An online jupyter notebook containing the results presented here can be found
at http://www.pyhrf.org/scipy2017_notebook.


.. Background
.. ----------
..
.. The development of neuroimaging techniques have allowed neuroscientifics to
.. study brain function *in vivo*, in the healthy and pathological conditions.
.. Since brain function is related to blood oxygen supply, the access to blood
.. perfusion (the arrival of blood supply to a tissue) with neuroimaging is also
.. an important tool for brain research. Different imaging techniques have been
.. developed following different principles. Next, we briefly introduce fMRI and
.. BOLD modality.
..
.. Functional Magnetic Resonance Imaging (fMRI)
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. Magnetic Resonance Imaging (MRI) uses nuclear magnetic resonance (NMR):
.. physical phenomenon in which protons inside a magnetic field align their spin
.. with the magnetic field vector and can absorb and re-emit electromagnetic
.. radiation. In MRI, a large cylindrical magnet creates a magnetic field around
.. the subject, that is place inside (see Figure :ref:`irm3t`). Then, radio waves
.. are sent and their echo signals are collected and used to construct an image.
..
.. .. figure:: figures/irm_3t_neurospin.jpg
..    :align: center
..    :figclass: bht
..
..    3T MRI scanner at Neurospin for clinical research. :label:`irm3t`
..
..
.. Blood-Oxygen-Level-Dependent (BOLD) fMRI
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. In 1990 a Japanese scientist called S. Ogawa :cite:`Ogawa:1990` discovered that
.. the scanner can "see" where the blood goes after brain activity happens. This
.. is due to the fact that hemoglobin works as a natural contrast agent: changes
.. in the local oxygenation of the blood cause magnetic distortions that the
.. scanner can detect. These changes in local oxygenation of the blood happen with
.. brain activity, since oxygen is consumed and a subsequent blood supply causes
.. an over-oxygenation of the local blood. This effect is called the Blood Oxygen
.. Level Dependent (BOLD) effect and it is a popular measure in fMRI because there
.. is no need for the invasive injection of other contrast agents (see Figure
.. :ref:`boldchain`). With the BOLD signal, we can measure the effect of brain
.. activity after a stimulus is given or a specific task is performed.
..
.. .. figure:: figures/bold_chain.pdf
..    :align: center
..    :scale: 50%
..    :figclass: w
..
..    fMRI BOLD signal :cite:`Ogawa:1990`. The BOLD signal measures the local
..    changes in blood oxygenation. This ratio changes during brain activity.
..    :label:`boldchain`
..

PyHRF
-----


PyHRF (http://www.pyhrf.org) is an open source tool implemented in Python that
allows to jointly detect activation and estimate (JDE) the hemodynamic response
function (HRF) :cite:`Makni08`, which gives the temporal changes in the BOLD
effect after brain activity.  This estimation is not easy in a *voxel-wise*
manner :cite:`Ciuciu03`, and a spatial structure was added to JDE
:cite:`Vincent10` in order to have a more robust estimation. In this regard,
HRF estimation in JDE is *parcel-wise* and the input of a parcellation is
needed.  However, this added a huge computational load to the method, leading
to the development of a faster method to deal with the parameter estimation.
Thus, a variational expectation maximization (VEM) solution :cite:`Chaari13`
was implemented.


.. In fact, PyHRF is composed of some C-extensions that handle computationally
.. intensive parts of the algorithms. The package relies on robust scientific
.. libraries such as Numpy [#]_, Scipy [#]_, Nipy [#]_ as well as Nibabel [#]_ to
.. handle data reading/writing in the NIFTI format. Its source code is hosted on
.. Github (https://github.com/pyhrf/pyhrf) and it can be easily installed from the
.. PyPi repository (https://pypi.python.org/pypi/pyhrf). The reader can found the
.. documentation of PyHRF and all the related information at http://www.pyhrf.org.
..
.. .. [#] Numpy official website: http://www.numpy.org/
.. .. [#] Scipy official website: https://www.scipy.org/
.. .. [#] Nipy official website: http://nipy.org/nipy/
.. .. [#] Nibabel official website: http://nipy.org/nibabel/


JDE aims at improving activation detection by capturing the correct
hemodynamics, since using the wrong HRF function could hide existing
activations. The use of a canonical HRF is usually sufficient for activation
detection. However, HRF functions have been found to have different shapes in
different regions :cite:`Handwerker04`, and to have different delays in
specific populations :cite:`Badillo13`. They are also believed to change in
some pathologies as stenosis. Fig. :ref:`hrfs` shows some HRF functions
estimated using PyHRF from BOLD data of a healthy adult acquired in
a block-design setting with visual, auditory and motor experimental conditions.
The parcels correspond to regions of the brain that are known to activate with
these experimental conditions.

Standard methods, as GLM, with the posterior classical statistics applied, give
statistical parametric maps (SPM) that describe the significance of the
activation in each region. JDE is a probabilistic method and estimates, for
each parameter, posterior probability functions. For this reason, we can
compute posterior probability maps (PPMs) from the outputs of PyHRF. These PPMs
are not directly comparable to the classical SPM maps, but give a similar
measure of significance of activation. For instance, in Fig. :ref:`spmvsppm` we
show the SPM and PPM maps for a visual experimental condition in the same data
used for Fig. :ref:`hrfs`. We use the package Nilearn
(http://nilearn.github.io) to generate the beautiful figures presented in this
document.


.. INFO: I use raw latex to display two subfigures

.. raw:: latex

   \begin{figure}[!htb]
   \centering
   \hspace{-12mm} (a) \hspace{36mm} (b) \\[3mm]
   \includegraphics[width=0.2\textwidth]{figures/visual_ppm.pdf}\hspace{5mm}
   \includegraphics[width=0.2\textwidth]{figures/visual_pvalue.pdf}
   \caption{PPM (a) and SPM (b) maps computed with JDE and GLM, respectively. Scale is logarithmic.} \DUrole{label}{spmvsppm}
   \end{figure}


In Fig. :ref:`pyhrf` we present the inputs and the outputs of PyHRF for the
analysis of BOLD data. It needs as inputs the data volume (BOLD), the
experimental paradigm, and a parcellation of the brain. After running the JDE
algorithm, the outputs will consist of HRF functions per parcel, BOLD effect
maps per experimental condition, and posterior probability maps (PPMs) per
condition. In the next section, we will describe in more detail these elements
and how to use PyHRF.

.. figure:: figures/pyhrf4bold.pdf
   :align: center
   :scale: 50%
   :figclass: w

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`pyhrf`


Example of Use
--------------

To illustrate the use of PyHRF, we will describe each step needed for the
analysis of BOLD data. A jupyter notebook containing the complete code is
available at http://www.pyhrf.org/scipy2017_notebook.


Getting fMRI BOLD Data
~~~~~~~~~~~~~~~~~~~~~~

First of all, we need to get BOLD data. In this example, we will analyze the
dataset used in :cite:`Gorgolewski2013`. This dataset (``ds000114``) is open
shared and it can be found at https://openfmri.org/dataset/ds000114/. For that,
we implemented the method ``get_from_openfmri`` that uses the library
``fetchopenfmri`` (https://github.com/wiheto/fetchopenfmri) to download
datasets from the site ``openfmri``.

.. code-block:: python

    >>> dataset_path = get_from_openfmri('114', '~/data')
    Dataset ds000114 already exists
    /home/jariasal/data/openfmri/ds000114_R2.0.1

Briefly, in this dataset ten healthy subjects in their fifties were scanned
twice using an identical protocol. This protocol consists of five task-related
fMRI time series: finger, foot and lip movement; overt verb generation; covert
verb generation; overt word repetition; and landmark tasks. For the sake of
simplicity, we will focus only on motor tasks (*i.e.,* finger, foot and lip
movement). Fig. :ref:`paradigm` shows the protocol containing only the three
tasks mentioned above. As we can see, in the experimental paradigm the tasks do
not overlap each other and the stimuli are presented to the subject during
a certain time (*i.e.,* block paradigm).

.. figure:: figures/paradigm.png
   :align: center
   :figclass: htb

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`paradigm`


fMRI BOLD Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~

Once we have downloaded the BOLD volumes, we need to apply some transformations
to the images in order to correct possible errors induced during the
acquisition. For instance, a BOLD volume (*e.g.,* a whole brain) is usually not
built at once but with a series of successively measured 2D slices. Each slice
take some time to acquire, so slices are observed at different time points,
leading to suboptimal statistical analysis.

We use the library ``Nipype`` (https://github.com/nipy/nipype) to define and
apply our preprocessing pipeline. This library allows to use  robust tools,
such as SPM and FSL, in a easy way. The proposed workflow (see Fig.
:ref:`nipype`) starts by uncompressing (gunzip) the images since they are in
a ``nii.gz`` format. After, it applies a *slice timing* in order to make appear
that all voxels of the BOLD volume have been acquired at the same time. We then
apply a *realignment* in order to correct head movements. Also, we apply
a *coregistration* operation in order to have the anatomical image (high
spatial resolution) in the same space as the BOLD images. Finally, we
*normalize* our images in order to transform them into a standard space (a
template).


.. figure:: figures/nipype_workflow.png
   :align: center
   :figclass: htb

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`nipype`


The pipeline described above is executed for the images of all subjects from
the dataset (*i.e.,* 10 subjects). Our pipeline is executed on multiple
processors since ``Nipype`` uses the library ``joblib``.

.. code-block:: python

    # Number of subjects
    >>> N_SUBJECTS = 10
    >>> SUBJECTS = ['sub-%02d' % i
                    for i in range(1,N_SUBJECTS+1)]


We use the acquisition parameters in :cite:`Gorgolewski2013` to parameterize
each preprocessing task. For instance, the number of slices for the volume, the
time for acquiring all slices (TR), and the order in which they were acquired
(*e.g.,* interleaved). In the following code, we show a snippet of how to
define a slice timing task with ``Nipype``.


.. code-block:: python

    # Acquisition parameters
    >>> TR = 2.5
    >>> NUM_SLICES = 30
    >>> TA = TR - (TR / NUM_SLICES)
    >>> REF_SLICE = 1

    # interleaved slice order
    >>> SLICE_ORDER = list(range(1, NUM_SLICES+1, 2) +
                           range(2, NUM_SLICES+1, 2))

    # slice timing
    >>> slice_timing = Node(
          spm.SliceTiming(num_slices=NUM_SLICES,
                          time_repetition=TR,
                          time_acquisition=TA,
                          slice_order=SLICE_ORDER,
                          ref_slice=REF_SLICE),
          name='slice_timing_node')

PyHRF Analysis
~~~~~~~~~~~~~~

So far, we have prepared our functional and structural images for BOLD
analysis. It is important to note that PyHRF receives as input *non-smoothed*
images, thus we excluded this operation from our preprocessing pipeline.

For the sake of simplicity, in this paper we only analyze the 4th subject from
our dataset. Moreover, we use the package ``Nilearn``
(http://nilearn.github.io/) to load and visualize neuroimaging volumes. Fig
:ref:`bold` shows the mean of the functional images of this subject after
preprocessing.


.. figure:: figures/bold.png
   :align: center
   :figclass: htb

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`bold`


The JDE framework estimates HRF parcels-wide. This means that PyHRF needs
a parcellation mask to compute the estimation-detection. The package provides
a Willard atlas :cite:`Richiardi2015` which was created from the files
distributed by Stanford (http://findlab.stanford.edu/functional_ROIs.html) with
a voxel resolution of 3x3x3mm and a volume shape of 53x63x52 voxels.

We use the method ``get_willard_mask`` to resize the original mask to match the
shape of the BOLD images. Moreover, it saves the resampled mask in a specified
path. For instance, Fig. :ref:`willard` shows the Willard parcellation resized
to the shape of the functional image in Fig. :ref:`bold`.


.. code-block:: python

    >>> willard = get_willard_mask('~/pyhrf',
                                   '~/data/bold.nii')
    /home/jariasal/pyhrf/mask_parcellation/willard_2mm.nii


.. figure:: figures/willard.png
   :align: center
   :figclass: htb

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`willard`


PyHRF also needs the experimental paradigm as input. It must be a ``csv`` file
following a specific convention which is described at
https://pyhrf.github.io/manual/paradigm.html. For that, we use the method
``convert_to_pyhrf_csv`` which reads the paradigm file provided by the dataset
(a ``tsv`` file) and rewrites it using the PyHRF format. Since each dataset has
its own organization, we give it as an input to the method.

.. code-block:: python

    >>> paradigm = convert_to_pyhrf_csv(
          '~/data/paradigm.tsv', 0,
          ['onset', 'duration', 'weight', 'trial_type'])
    /tmp/tmpM3zBD5


Table :ref:`csv` shows the paradigm experiment using the PyHRF format. Note
that it only contains motor stimuli since we are only interested in motor tasks
for our BOLD analysis. As we will show below, this paradigm is not optimized
for the underlying model of PyHRF. This causes that some brain regions that are
expected to be active, *e.g.,* the supplementary motor area (SMA), have not
significant values in the PPMs generated by PyHRF.

.. table:: This is the caption for the materials table. :label:`csv`

    +---------+-----------+-------+----------+-----------+
    | session | condition | onset | duration | amplitude |
    +=========+===========+=======+==========+===========+
    | 0       | Finger    | 10    | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Foot      | 40    | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Lips      | 70    | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Finger    | 100   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Foot      | 130   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Lips      | 160   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Finger    | 190   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Foot      | 220   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Lips      | 250   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Finger    | 280   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Foot      | 310   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Lips      | 340   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Finger    | 370   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Foot      | 400   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+
    | 0       | Lips      | 430   | 15.0     | 1         |
    +---------+-----------+-------+----------+-----------+



PyHRF Analysis
~~~~~~~~~~~~~~

Now we are ready to start our BOLD analysis using PyHRF. For that, we need to
specify some important parameters of the underlying JDE model. Moreover, we
need to defined if we want to estimate the HRF signal or use, for example, its
canonical form. The reader can found more details about the parameters in
http://www.pyhrf.org.

Once the parameters of the model have been defined, we run our analysis by
using the command-line tool ``pyhrf_jde_vem_analysis`` provided by PyHRF. We
can specify to execute this analysis using several processors since PyHRF uses
the library ``joblib`` (https://github.com/joblib/joblib). For instance,


.. code-block:: bash

    pyhrf_jde_vem_analysis \
      --dt 1.25 \
      --hrf-duration 25.0 \
      --output /home/jariasal/pyhrf \
      --beta 1.0 \
      --hrf-hyperprior 1000 \
      --sigma-h 0.1 \
      --estimate-hrf \
      --zero-constraint \
      --drifts-type cos \
      --parallel \
      --log-level WARNING \
      2.5 \
      {$HOME}/pyhrf/mask_parcellation/willard_2mm.nii \
      /tmp/tmpM3zBD5
      {$HOME}/data/bold.nii

Fig. :ref:`output` shows the PPMs (upper left) and the estimated HRFs (right)
generated by PyHRF for the motor task ``Finger``. Reading the description in
:cite:`Gorgolewski2013`, this task corresponds to finger tapping. Recall that
PyHRF estimates a HRF for each active parcel.

We compared the output of PyHRF with the T-maps found on the site *Neuovault*
(http://www.neurovault.org/images/307/) for the same dataset. As we can
observe, at cut *z=60* both results (Fig. :ref:`output` and Fig.
:ref:`neurovalt`) are quite similar, showing an activation in the
*supplementary motor area* and the *left primary sensorimotor cortex*.


.. figure:: figures/neurovault.png
   :align: center
   :figclass: htb

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`neurovalt`


.. figure:: figures/pyhrf_output.png
   :align: center
   :scale: 35%
   :figclass: w

   Inputs and outputs of PyHRF when analyzing BOLD data. :label:`output`



Concluding Remarks
------------------

In this paper we presented PyHRF, a software to analyze fMRI data using a joint
detection-estimation (JDE) approach of the cerebral activity. Roughly, it
jointly detects cortical activation and estimates the hemodynamic response
function (HRF). Contrary to existing tools, PyHRF estimates the HRF instead of
considering it constant in all the brain and for all subjects, improving thus
the reliability of the results.

PyHRF is an open source software, which has allowed it to evolve rapidly over
the last few years. As we showed, it allows to generate posteriori probability
maps (PPM) to describe the significance of the activation in each region of the
brain. Moreover, PyHRF uses efficient estimation methods in order to provide
a fast and reliable tool. In fact, in 2013, a similar solution based on the
BOLD JDE was developed in PyHRF for the Functional Arterial Spin Labelling
(fASL) :cite:`Vincent13` method, with the inclusion of a physiological prior to
make the perfusion estimation more robust :cite:`Frau14b` :cite:`Frau15a`.
A fast solution for fASL based on VEM was proposed in :cite:`Frau15b`, with
similar results to the classical solution based on stochastic simulation
techniques :cite:`Frau15c`.

In the last years, many efforts are made in terms of user-friendliness and
usability of the PyHRF tool to make it more easy to use by non experts and
clinicians.  Moreover, since PyHRF is able to analyze both BOLD and ASL data,
it has begun to emerge as a tool suitable for use in a clinical environment.
