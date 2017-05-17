:author: Jaime Arias
:email: jaime.arias@inria.fr
:institution: Inria, MISTIS, Grenoble University, LJK, Grenoble, France
:corresponding:
:equal-contributor:

:author: Philippe Ciuciu
:email: philippe.ciuciu@cea.fr
:institution: CEA/NeuroSpin and INRIA Saclay, Parietal, France
:equal-contributor:

:author: Michel Dojat
:email: michel.dojat@univ-grenoble-alpes.fr
:institution: INSERM, U1216, F-38000 Grenoble, France
:institution: Univ. Grenoble Alpes, Grenoble Institut des Neurosciences, GIN, F-38000 Grenoble, France
:equal-contributor:

:author: Florence Forbes
:email: florence.forbes@inria.fr
:institution: Inria, MISTIS, Grenoble University, LJK, Grenoble, France
:equal-contributor:

:author: Aina Frau-Pascual
:email: aina.frau-pascual@inria.fr
:institution: Inria, MISTIS, Grenoble University, LJK, Grenoble, France
:equal-contributor:

:author: Thomas Perret
:email: thomas.perret@grenoble-inp.org
:institution: Inria, MISTIS, Grenoble University, LJK, Grenoble, France
:equal-contributor:

:author: Jan Warnking
:email: jan.warnking@univ-grenoble-alpes.fr
:institution: INSERM, U1216, F-38000 Grenoble, France
:institution: Univ. Grenoble Alpes, Grenoble Institut des Neurosciences, GIN, F-38000 Grenoble, France
:equal-contributor:

:bibliography: biblio

----------------------------------------------------------------------------------------------------------------
PyHRF: A Python Library for the Analysis of fMRI Data Based on Local Estimation of Hemodynamic Response Function
----------------------------------------------------------------------------------------------------------------

.. class:: abstract

   Functional Magnetic Resonance Imaging (fMRI) is a neuroimaging technique
   that allows the non-invasive study of brain functions. It is based on the
   hemodynamic changes induced by cerebral activity following sensory or
   cognitive stimulation. The measured MR signal depends on the function of
   blood oxygenation level (BOLD signal) which is related to brain activity:
   a decrease in deoxyhemoglobin induces an increase in BOLD signal. Indeed,
   this signal is convoluted by the Hemodynamic Response Function (HRF) whose
   exact form is unknown and depends on various parameters (age, brain region,
   physiological conditions).

   Most used open source libraries for the analysis of fMRI data (*e.g.,* SPM,
   FSL) consider a priori the HRF as constant in brain and the same for all
   subjects. However, several studies show that the HRF changes across
   different regions of the brain and between subjects. Assuming a constant HRF
   can therefore be responsible for a large number of false negatives and
   degrade the reliability of the results.

   We will present PyHRF (http://www.pyhrf.org), a software to analyze fMRI
   data using a joint detection-estimation (JDE) approach. It jointly detects
   cortical activation and estimates the HRF. In contrast to existing tools,
   PyHRF estimates the HRF instead of considering it constant, improving thus
   the reliability of the results. Here, we lay out the architecture, concept
   and implementation of the package and present some use cases to show that
   PyHRF is a tool suitable for non experts and clinicians.

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

BOLD reflects the changes in oxygen concentration in the blood. Briefly, when
brain activity occurs, oxygen is locally consumed by neurons and its
concentration in the blood decreases. Therefore, an inflow of oxygenated blood
is achieved to replenish the tissue, increasing blood oxygen concentration.
Deoxygenated blood causes locally magnetic distortions. Thus, BOLD signal is an
indirect measure of cerebral activity based on physiological changes in oxygen
consumption, cerebral blood flow and blood volume.

BOLD is non-invasive, non-ionizing, and gives access *in vivo* to brain
activity with a relatively high spatial resolution. It is highly dependent of
the hemodynamic response function (HRF) of the brain. It does not give access
to true physiological parameters such as cerebral blood flow or cerebral blood
volume, but rather measures a mixture of these quantities that is difficult to
untangle. In this regard, BOLD is a very interesting tool in neuroscience, but
in general it is not widely used for clinical applications because the impact
of physiological situation on HRF is unknown, hampering the BOLD signal
interpretation. For example, it cannot detect chronic changes in the baseline
states :cite:`Buxton:2013`, as it is the case of normal ageing
:cite:`Fabiani:2014` and pathologies like Alzheimer's disease
:cite:`Cantin:2011` or Stroke :cite:`Attye:2014`.

Most used open source libraries for the analysis of fMRI data (*e.g.,* SPM,
FSL, AFNI) consider the HRF of the neuronal activity as a constant in all the
brain and the same for all subjects. However, several works (see
:cite:`Badillo13`) show that the HRF changes across different regions of the
brain and other individuals, increasing thus the probability of obtaining false
negative results and decreasing the reliability of the results. The software
PYHRF :cite:`Vincent:2014` has been developed to overcome the above limitation
by analyzing fMRI data using a joint detection-estimation (JDE) approach. In
the JDE approach, the detection of the cortical activation is achieved together
with the estimation of the unknown HRF response by analyzing non smoothed data.
This detection-estimation is calculated on different parcels of interest paving
the cerebral volume. Therefore, PYHRF allows to navigate the brain and to focus
on the regions of interest during the experiment in order to visualize the
activations and their temporal behavior through the estimated HRF. In the last
years, efforts are made in terms of user-friendliness and usability of the
PYHRF package to make it more easy to use by non experts and clinicians.

PYHRF is an open source tool implemented in Pythonwith some C-extensions that
handle computationally intensive parts of the algorithms. The package relies on
robust scientific libraries such as Numpy, Scipy, Sympy, as well as Nibabel to
handle data reading/writing in the NIFTI format. Its source code is hosted on
Github (https://github.com/pyhrf/pyhrf) and it can be easily installed from the
PyPi repository (https://pypi.python.org/pypi/pyhrf). The reader can found the
documentation of PYHRF and all the related information at http://www.pyhrf.org.
