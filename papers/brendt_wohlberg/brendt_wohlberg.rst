:author: Brendt Wohlberg
:email: brendt@ieee.org
:institution: Los Alamos National Laboratory
:corresponding:

:bibliography: references

------------------------------------------------------------------------------
SPORCO: A Python package for standard and convolutional sparse representations
------------------------------------------------------------------------------

.. class:: abstract

   SParse Optimization Research COde (SPORCO) is an open-source Python package for solving optimisation problems with sparsity-inducing regularisation, consisting primarily of sparse coding and dictionary learning, for both standard and convolutional forms of sparse representation. In the current version, all optimization problems are solved within the Alternating Direction Method of Multipliers (ADMM) framework. SPORCO was developed for applications in signal and image processing, but is also expected to be useful for problems in computer vision, statistics, and machine learning.

.. class:: keywords

   sparse representations, convolutional sparse representations, sparse coding, convolutional sparse coding, dictionary learning, convolutional dictionary learning, alternating direction method of multipliers, convex optimization


Introduction
------------

SPORCO is an open-source Python package for solving inverse problems with sparsity-inducing regularization :cite:`mairal-2014-sparse`. It was developed for applications in signal and image processing, but is also expected to be useful for problems in computer vision, statistics, and machine learning.

SPORCO was initially a Matlab library, but the implementation language was switched to Python for a number of reasons, including (i) the substantial cost of Matlab licenses within an environment that does not qualify for an academic discount, and the difficulty of running large scale experiments on multiple hosts with a limited supply of toolbox licenses, (ii) the greater maintainability and flexibility of the object-oriented design possible in Python, (iii) the flexibility provided by NumPy in indexing arrays of arbitrary numbers of dimensions (essentially impossible in Matlab), and (iv) the vast superiority of Python as a general-purpose programming language.

SPORCO supports a variety of inverse problems, including Total Variation :cite:`rudin-1992-nonlinear` :cite:`alliney-1992-digital` denoising and deconvolution, and Robust PCA :cite:`cai-2010-singular`, but the primary focus is on sparse coding and dictionary learning, for solving problems with sparse representations :cite:`mairal-2014-sparse`. Both standard and convolutional forms of sparse representations are supported. In the standard form the dictionary is a matrix, which limits the sizes of signals, images, etc. that can be directly represented; the usual strategy is to compute independent representations for a set of overlapping blocks. In the convolutional form :cite:`zeiler-2010-deconvolutional`:cite:`wohlberg-2016-efficient`, which is more recent, the dictionary is a set of linear filters, making it feasible to directly represent an entire signal or image. The support for the convolutional form is one of the major strengths of SPORCO since it is the only Python package to provide such a breadth of options for convolutional sparse coding and dictionary learning. Some features are not available in any other open-source package, including support for representation of multi-channel images (e.g. RGB color images) :cite:`wohlberg-2016-convolutional`, and representation of arrays of arbitrary numbers of dimensions, allowing application to one-dimensional signals, images, and video and volumetric data.

In the current version, all optimization problems are solved within the Alternating Direction Method of Multipliers (ADMM) :cite:`boyd-2010-distributed` framework, which is implemented as flexible class hierarchy designed to minimize the additional code that has to be written to solve a specific problem. This design also simplifies the process of deriving algorithms for solving variants of existing problems, in some cases only requiring overriding one or two methods, involving a few additional lines of code.


ADMM
----

The ADMM :cite:`boyd-2010-distributed` framework addresses problems of the form

.. math::
   :label: eq:admmform

    \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;\;
    f(\mathbf{x}) + g(\mathbf{y}) \;\;\mathrm{such\;that}\;\;
    A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.

This general problem is solved by iterating over the following three update steps:

.. math::
    :type: align

     \mathbf{x}^{(j+1)} &= \mathrm{argmin}_{\mathbf{x}} \;\;
     f(\mathbf{x}) + \frac{\rho}{2} \left\| A\mathbf{x} -
     \left( -B\mathbf{y}^{(j)} + \mathbf{c} - \mathbf{u}^{(j)} \right)
     \right\|_2^2 \\
     \mathbf{y}^{(j+1)} &= \mathrm{argmin}_{\mathbf{y}} \;\;
     g(\mathbf{y}) + \frac{\rho}{2} \left\| B\mathbf{y} - \left(
     -A\mathbf{x}^{(j+1)} + \mathbf{c} - \mathbf{u}^{(j)} \right)
     \right\|_2^2 \\
     \mathbf{u}^{(j+1)} &= \mathbf{u}^{(j)} + A\mathbf{x}^{(j+1)} +
     B\mathbf{y}^{(j+1)} - \mathbf{c}

which we will refer to as the :math:`\mathbf{x}`, :math:`\mathbf{y}`, and :math:`\mathbf{u}`, steps respectively.

The feasibility conditions (see Sec. 3.3 :cite:`boyd-2010-distributed`) for the ADMM problem are

.. math::
    :type: align

     & A\mathbf{x}^* + B\mathbf{y}^* - \mathbf{c} = 0 \\
     & 0 \in \partial f(\mathbf{x}^*) + \rho^{-1} A^T \mathbf{u}^* \\
     & 0 \in \partial g(\mathbf{u}^*) + \rho^{-1} B^T \mathbf{u}^* \;\;,

where :math:`\partial` denotes the subdifferential operator. It can be shown that the last feasibility condition is always satisfied by the solution of the :math:`\mathbf{y}` step. The primal and dual residuals :cite:`boyd-2010-distributed`

.. math::
    :type: align

     \mathbf{r} &= A\mathbf{x}^{(j+1)} + B\mathbf{y}^{(j+1)} - \mathbf{c}\\
     \mathbf{s} &= \rho A^T B (\mathbf{y}^{(j+1)} - \mathbf{y}^{(j)}) \;\;,

which can be derived from the feasibility conditions, provide a convenient measure of convergence, and can be used to define algorithm stopping criteria. The :math:`\mathbf{u}` step can be written in terms of the primal residual as

.. math::

     \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + \mathbf{r}^{(j+1)} \;.

It is often preferable to use normalised versions of these residuals :cite:`wohlberg-2015-adaptive`, obtained by dividing the definitions above by their corresponding normalisation factors

.. math::
    :type: align

    r_{\mathrm{n}} &= \mathrm{max}(\|A\mathbf{x}^{(j+1)}\|_2,
     \|B\mathbf{y}^{(j+1)}\|_2, \|\mathbf{c}\|_2) \\
    s_{\mathrm{n}} &= \rho \|A^T \mathbf{u}^{(j+1)} \|_2 \;.

These residuals can also be used in a heuristic scheme :cite:`wohlberg-2015-adaptive` for selecting the critical *penalty parameter* :math:`\rho`.


SPORCO ADMM Classes
===================

SPORCO provides a flexible set of classes for solving problems within the ADMM framework. All ADMM algorithms are derived from class ``admm.admm.ADMM``, which provides much of the infrastructure required for solving a problem, so that the user need only override methods that define the constraint components :math:`A`, :math:`B`, and :math:`\mathbf{c}`, and that compute the :math:`\mathbf{x}` and :math:`\mathbf{y}` steps. This infrastructure includes the computation of the primal and dual residuals, which are used as convergence measures on which termination of the iterations can be based, and are also used within an optional scheme for automatically setting the penalty parameter. Additional class attributes and methods can be defined to customize the calculation of diagnostic information, such as the functional value, at each iteration. The SPORCO documentation includes a `detailed description <http://sporco.rtfd.io/en/latest/admm/admm.html>`_ of the required and optional methods to be overridden in defining a class for solving a specific optimisation problem.

The ``admm.admm`` module also includes classes that are derived from ``admm.admm.ADMM`` to specialise to less general cases; for example, class ``admm.admm.ADMMEqual`` assumes that :math:`A = I`, :math:`B = -I`, and :math:`\mathbf{c} = \mathbf{0}`, which is a very frequently occurring case, allowing derived classes to avoid overriding methods that specify the constraint. The most complex partial specialisation is ``admm.admm.ADMMTwoBlockCnstrnt``, which implements the commonly-occurring ADMM problem form with a block-structured :math:`\mathbf{y}` variable,

.. math::
   :type: align

   \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
   f(\mathbf{x}) + g_0(\mathbf{y}_0) + g_0(\mathbf{y}_1)
   \\ \;\text{such that}\;
   \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
   \right) = \left( \begin{array}{c} \mathbf{c}_0 \\
   \mathbf{c}_1 \end{array} \right) \;\;,

for solving problems that have the form

.. math::
   \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g_0(A_0 \mathbf{x}) +
   g_1(A_1 \mathbf{x})

prior to variable splitting. The block components of the :math:`\mathbf{y}` variable are concatenated into a single NumPy array, with access to the individual components provided by methods ``block_sep0`` and ``block_sep1``.


Defining new classes derived from ``admm.admm.ADMM`` or one of its partial specialisations provides complete flexibility in constructing a new ADMM algorithm, while reducing the amount of code that has to be written compared with implementing the entire ADMM algorithm from scratch. When a new ADMM algorithm to be implemented is closely related to an existing algorithm, it is often much easier to derived the new class from that of the existing algorithm, as described in the section *Extending SPORCO*.


Sparse Coding
-------------

Sparse coding in SPORCO is based on the Basis Pursuit DeNoising (BPDN) problem :cite:`chen-1998-atomic`

.. math::
   \mathrm{argmin}_X \;
   (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1 \;,

which is implemented by class ``admm.bpdn.BPDN``. A number of variations on this problem are supported by other classes in module ``admm.bpdn``. BPDN is solved via the equivalent ADMM problem

.. math::
   \mathrm{argmin}_X \;
   (1/2) \| D X - S \|_F^2 + \lambda \| Y \|_1
   \quad \text{such that} \quad X = Y \;\;.

This algorithm is effective because the :math:`Y` step can be solved in closed form, and is computationally relatively cheap.  The main computational cost is in solving the :math:`X` step, which involves solving the potentially-large linear system

.. math::
   (D^T D + \rho I) X = D^T S + \rho (Y - U) \;\;.

SPORCO solves this system efficiently by pre-computing an LU factorisation of :math:`(D^T D + \rho I)` which enables a rapid direct-method solution at every iteration (see Sec. 4.2.3 in :cite:`boyd-2010-distributed`). In addition, if :math:`(D D^T + \rho I)` is smaller than :math:`(D^T D + \rho I)`, the matrix inversion lemma is used to reduce the size of the system that is actually solved (see Sec. 4.2.4 in :cite:`boyd-2010-distributed`).



Dictionary Learning
-------------------

Dictionary learning is based on the problem

.. math::
   \mathrm{argmin}_{D, X} \;
   (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1 \; \text{ s.t }
   \; \|\mathbf{d}_m\|_2 = 1 \;,

which is solved by alternating between a sparse coding stage, as above, and a constrained dictionary update obtained by solving the problem

.. math::
   \mathrm{argmin}_D (1/2) \| D X - S \|_2^2 \; \text{ s.t }
   \; \|\mathbf{d}_m\|_2 = 1 \;.

This approach is implemented by class ``admm.bpdndl.DictLearn``. An unusual feature of this dictionary learning algorithm is the adoption from convolutional dictionary learning :cite:`bristow-2013-fast` :cite:`wohlberg-2016-efficient` :cite:`garcia-2017-subproblem` of the very effective strategy of alternating between a single step of each of the sparse coding and dictionary update algorithms. To the best of this author's knowledge, this strategy has not previously been applied to standard (non-convolutional) dictionary learning.



Convolutional Sparse Coding
---------------------------

Convolutional sparse coding (CSC) is based on a convolutional form of BPDN, which we will be referred to as Convolutional BPDN (CBPDN) :cite:`wohlberg-2016-efficient`

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   \frac{1}{2} \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
   \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;\;,

which is implemented by class ``admm.cbpdn.ConvBPDN``. Module ``admm.cbpdn`` also contains a number of other classes implementing variations on this basic form. As in the case of standard BPDN, the main computational cost of this algorithm is in solving the :math:`\mathbf{x}` step, which can be solved very efficiently by exploiting the Sherman-Morrison formula :cite:`wohlberg-2014-efficient`. SPORCO provides support for solving the basic form above, as well as a number of variants, including one with a gradient penalty, and two different approaches for solving a variant with a spatial mask :math:`W` :cite:`heide-2015-fast`:cite:`wohlberg-2016-boundary`

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   \frac{1}{2} \left \|  W \left( \sum_m \mathbf{d}_m * \mathbf{x}_m -
   \mathbf{s} \right) \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;\;.

SPORCO also supports two different methods for convolutional sparse coding of multi-channel (e.g. colour) images :cite:`wohlberg-2016-convolutional`. The one represents a multi-channel input with channels :math:`\mathbf{s}_c` with single-channel dictionary filters :math:`\mathbf{d}_m` and multi-channel coefficient maps :math:`\mathbf{x}_{c,m}`,

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
   \mathbf{s}_c \right\|_2^2 +
   \lambda \sum_c \sum_m \| \mathbf{x}_{c,m} \|_1 \;\;,

and the other uses multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and single-channel coefficient maps :math:`\mathbf{x}_m`,

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -
   \mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;\;.

In the former case the representation of each channel is completely independent unless they are coupled via an :math:`\ell_{2,1}` norm term :cite:`wohlberg-2016-convolutional`, which is supported by class ``admm.cbpdn.ConvBPDNJoint``.

An important issue that has received surprisingly little attention in the literature is the need to explicitly consider the representation of the smooth/low frequency image component when constructing convolutional sparse representations. If this component is not properly taken into account, convolutional sparse representations tend to give poor results. As briefly mentioned in :cite:`wohlberg-2016-efficient` (Sec. I), the simplest approach is to lowpass filter the image to be represented, computing the sparse representation on the highpass residual. In this approach the lowpass component forms part of the complete image representation, and should, of course, be added to the reconstruction from the sparse representation in order to reconstruct the image being represented. SPORCO supports this separation of an image into lowpass/highpass components via the function ``util.tikhonov_filter``, which computes the lowpass component of :math:`\mathbf{s}` as the solution of the problem

.. math::
   \mathrm{argmin}_\mathbf{x} \; \frac{1}{2} \left\|\mathbf{x} - \mathbf{s}
   \right\|_2^2 + \frac{\lambda}{2} \sum_i \| G_i \mathbf{x} \|_2^2 \;\;,

where :math:`G_i` is an operator computing the derivative along axis :math:`i` of the array represented as vector :math:`\mathbf{x}`, and :math:`\lambda` is a parameter controlling the amount of smoothing.
In some cases it is not feasible to handle the lowpass component via such a pre-processing strategy, making it necessary to include the lowpass component in the CSC optimization problem itself. The simplest approach to doing so is to append an impulse filter to the dictionary and include a gradient regularisation term on corresponding coefficient map in the functional (Sec. 3) :cite:`wohlberg-2016-convolutional2`. This approach is supported by class ``admm.cbpdn.ConvBPDNGradReg``, the use of which is demonstrated in section *Removal of Impulse Noise via CSC*.


Convolutional Dictionary Learning
---------------------------------

Convolutional dictionary learning is based on the problem

.. math::
   :type: align

   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \; &
   \frac{1}{2} \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
   \\ & \; \text{ s.t } \; \mathbf{d}_m \in C \;\;,

which is solved by alternating between a convolutional sparse coding stage, as above, and a constrained dictionary update obtained by solving the problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 \; \text{ s.t. } \; \mathbf{d}_m
   \in C \;\;,

where :math:`\iota_C(\cdot)` is the indicator function of feasible set :math:`C`, consisting of filters with unit norm and constrained support :cite:`wohlberg-2016-efficient`. This approach is implemented by class ``admm.cbpdndl.ConvBPDNDictLearn``. Dictionary learning with a spatial mask :math:`W`,

.. math::
   :type: align

   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \; &
   \frac{1}{2} \sum_k \left \|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right) \right \|_2^2 + \lambda \sum_k \sum_m \|
   \mathbf{x}_{k,m} \|_1 \\ & \; \text{ s.t } \; \mathbf{d}_m \in C

is also supported by class ``ConvBPDNMaskDcplDictLearn`` in module ``admm.cbpdndl``.


Convolutional Representations
-----------------------------

SPORCO convolutional representations are stored within NumPy arrays of ``dimN`` + 3 dimensions, where ``dimN`` is the number of spatial/temporal dimensions in the data to be represented. This value defaults to 2 (i.e. images), but can be set to any other reasonable value, such as 1 (i.e. one-dimensional signals) or 3 (video or volumetric data). The roles of the axes in these multi-dimensional arrays are required to follow a fixed order: first spatial/temporal axes, then an axis for multiple channels (singleton in the case of single-channel data), then an axis for multiple input signals (singleton in the case of only one input signal), and finally the axis corresponding to the index of the filters in the dictionary.


Sparse Coding
=============

For the convenience of the user, the ``D`` (dictionary) and ``S`` (signal) arrays provided to the convolutional sparse coding classes need not follow this strict format, but they are internally reshaped to this format for computational efficiency. This internal reshaping is largely transparent to the user, but must be taken into account when passing weighting arrays to optimization classes (e.g. option ``L1Weight`` for class ``admm.cbpdn.ConvBPDN``). When performing the reshaping into internal array layout, it is necessary to infer the intended roles of the axes of the input arrays, which is performed by class ``admm.cbpdn.ConvRepIndexing`` (note that this class is expected to be moved to a different module in a future version of SPORCO). The inference rules are relatively complex, depending on both the number of dimensions in the ``D`` and ``S`` arrays, and on parameters ``dimK`` and ``dimN``. The most fundamental parameter is ``dimN``, which should be common to *input* ``S`` and ``D``, and is also common to *internal* ``S``, ``D``, and ``X`` (convolutional representation). The remaining dimensions of input ``S`` can correspond to multiple channels (e.g. for RGB images) and/or multiple signals (e.g. the array contains multiple independent images). If input ``S`` contains two dimensions in addition to the ``dimN`` spatial dimensions, then those are considered to correspond, in order, to channel and signal indices. If there is only a single additional dimension, then determination whether it represents a channel or signal index is more complicated. The rule for making this determination is as follows:

* if ``dimK`` is set to 0 or 1 instead of the default ``None``, then that value is taken as the number of signal indices in input `S` and any remaining indices are taken as channel indices (i.e. if ``dimK`` = 0 then ``dimC`` = 1 and if ``dimK`` = 1 then ``dimC`` = 0).
* if ``dimK`` is ``None`` then the number of channel dimensions is determined from the number of dimensions in the input dictionary ``D``. Input ``D`` should have at least ``dimN`` + 1 dimensions, with the final dimension indexing dictionary filters. If it has exactly ``dimN`` + 1 dimensions then it is a single-channel dictionary, and input ``S`` is also assumed to be single-channel, with the additional index in ``S`` assigned as a signal index (i.e. ``dimK`` = 1).  Conversely, if input ``D`` has ``dimN`` + 2 dimensions it is a multi-channel dictionary, and the additional index in ``S`` is assigned as a channel index (i.e. ``dimC`` = 1).

It is an error to specify ``dimK`` = 1 if input ``S`` has ``dimN`` + 1 dimensions and input ``D`` has ``dimN`` + 2 dimensions since a multi-channel dictionary requires a multi-channel signal. (The converse is not true: a multi-channel signal can be decomposed using a single-channel dictionary.)


Dictionary Update
=================

The handling of convolutional representations by the dictionary update classes in module ``admm.ccmod`` are similar to those for sparse coding, the primary difference being the the dictionary update classes expect that the sparse representation inputs ``X`` are already in the standard layout as described above since they are usually obtained as the output of one of the sparse coding classes, and therefore already have the required layout. The inference of internal dimensions for these classes is handled by class ``admm.ccmod.ConvRepIndexing`` (which is also expected to be moved to a different module in a future version of SPORCO).



Installing SPORCO
-----------------

The primary requirements for SPORCO are Python itself (version 2.7 or 3.x), and modules `numpy <http://www.numpy.org>`_, `scipy <https://www.scipy.org>`_, `future <http://python-future.org>`_, `pyfftw <https://hgomersall.github.io/pyFFTW>`_, and `matplotlib <http://matplotlib.org>`_. Module `numexpr <https://github.com/pydata/numexpr>`_ is not required, but some functions will be faster if it is installed. If module `mpldatacursor <https://github.com/joferkington/mpldatacursor>`_ is installed, ``plot.plot`` and ``plot.imview`` will support the data cursor that it provides. Additional information on the requirements are provided in the `installation instructions <http://sporco.rtfd.io/en/latest/install.html>`_.


SPORCO is available on `GitHub <https://github.com/bwohlberg/sporco>`_ and can be installed via ``pip``:

::

   pip install sporco

SPORCO can also be installed from source, either from the development
version from `GitHub <https://github.com/bwohlberg/sporco>`_, or from
a release source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco/>`_.

To install the development version from `GitHub
<https://github.com/bwohlberg/sporco>`_ do

::

    git clone git://github.com/bwohlberg/sporco.git

followed by

::

   cd sporco
   python setup.py build
   python setup.py install

The install command will usually have to be performed with root
permissions, e.g. on Ubuntu Linux

::

   sudo python setup.py install

The procedure for installing from a source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco/>`_ is similar.

A summary of the most significant changes between SPORCO releases can
be found in the ``CHANGES.rst`` file. It is strongly recommended to
consult this summary when updating from a previous version.

SPORCO includes a large number of usage examples, some of which make use of a set of standard test images, which can be installed using the ``sporco_get_images`` script. To download these images from the root directory of the source distribution (i.e. prior to installation) do

::

   bin/sporco_get_images --libdest

after setting the ``PYTHONPATH`` environment variable to point to the root directory of the source distribution; for example, in a ``bash``
shell

::

   export PYTHONPATH=$PYTHONPATH:`pwd`


from the root directory of the package. To download the images as part of a
package that has already been installed, do

::

  sporco_get_images --libdest

which will usually have to be performed with root privileges.



Using SPORCO
------------

The simplest way to use SPORCO is to make use of one of the many existing classes for solving problems that are already supported, but SPORCO is also designed to be easy to extend to solve custom problems, in some cases requiring only a few lines of additional code to extend an existing class to solve an extended problem. This latter, more advanced usage is described in the section *Extending SPORCO*.

Detailed `documentation <http://sporco.rtfd.io>`_ is available. The distribution includes a large number of example scripts and a selection of Jupyter notebook demos, which can be viewed online via `nbviewer <https://nbviewer.jupyter.org/github/bwohlberg/sporco/blob/master/index.ipynb>`_, or run interactively via `mybinder <http://mybinder.org/repo/bwohlberg/sporco>`_.


A Simple Usage Example
======================

Each optimization algorithm is implemented as a separate class. Solving a problem is straightforward, as illustrated in the following example, which assumes that we wish to solve the BPDN problem

.. math::
   \mathrm{argmin}_{\mathbf{x}} \;
   (1/2) \| D \mathbf{x} - \mathbf{s} \|_F^2 + \lambda \| \mathbf{x} \|_1

for a given dictionary :math:`D` and signal vector :math:`\mathbf{s}`, represented by NumPy arrays ``D`` and ``s`` respectively. After importing the appropriate module

.. code-block:: python

   from sporco.admm import bpdn

create an object representing the desired algorithm options

.. code-block:: python

  opt = bpdn.BPDN.Options({'Verbose' : True,
			   'MaxMainIter' : 500,
			   'RelStopTol' : 1e-6})

then initialise the solver object

.. code-block:: python

  lmbda = 25.0
  b = bpdn.BPDN(D, s, lmbda, opt)

and call the ``solve`` method

.. code-block:: python

  x = b.solve()

leaving the result in NumPy array ``x``. Since the optimizer objects retain algorithm state, calling ``solve`` again gives a warm start on an additional set of iterations for solving the same problem (e.g. if the first solve terminated because it reached the maximum number of iterations, but the desired solution accuracy was not reached).


Removal of Impulse Noise via CSC
================================

We now consider a more detailed and realistic usage example, based on using CSC to remove impulse noise from a colour image. First we need to import some modules, including ``print_function`` for Python 2/3 compatibility, numpy, and a number of modules from SPORCO:

.. code-block:: python

  from __future__ import print_function

  import numpy as np
  from scipy.misc import imsave

  from sporco import util
  from sporco import plot
  from sporco import metric
  from sporco.admm import cbpdn


Boundary artifacts are handled by performing a symmetric extension on the image to be denoised and then cropping the result to the original image support. This approach is simpler than the boundary handling strategies described in :cite:`heide-2015-fast` and :cite:`wohlberg-2016-boundary`, and for many problems gives results of comparable quality. The functions defined here implement symmetric extension and cropping of images.

.. code-block:: python

  def pad(x, n=8):

    if x.ndim == 2:
	return np.pad(x, n, mode='symmetric')
    else:
	return np.pad(x, ((n, n), (n, n), (0, 0)),
		      mode='symmetric')


  def crop(x, n=8):

    return x[n:-n, n:-n]


Now we load a reference image (see the discussion on the script for downloading standard test images in section *Installing SPORCO*), and corrupt it with 33% salt and pepper noise. (The call to ``np.random.seed`` ensures that the pseudo-random noise is reproducible.)

.. code-block:: python

   img = util.ExampleImages().image('standard',
	 'monarch.png', zoom=0.5, scaled=True,
	 idxexp=np.s_[:, 160:672])
   np.random.seed(12345)
   imgn = util.spnoise(img, 0.33)


We use a colour dictionary, as described in :cite:`wohlberg-2016-convolutional`. The impulse denoising problem is solved by appending some additional filters to the learned dictionary ``D0``, which is one of those distributed with SPORCO. The first of these additional components is a set of three impulse filters, one per colour channel, that will represent the impulse noise, and the second is an identical set of impulse filters that will represent the low frequency image components when used together with a gradient penalty on the coefficient maps, as discussed below.

.. code-block:: python

  D0 = util.convdicts()['RGB:8x8x3x64']
  Di = np.zeros(D0.shape[0:2] + (3, 3))
  np.fill_diagonal(Di[0, 0], 1.0)
  D = np.concatenate((Di, Di, D0), axis=3)


The problem is solved using class ``admm.cbpdn.ConvBPDNGradReg``, which implements the form of CBPDN with an additional gradient regularization term,

.. math::

   \mathrm{argmin}_\mathbf{x} \;
   \frac{1}{2} \left \| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
   \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
   \frac{\mu}{2} \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2

where :math:`G_i` is an operator computing the derivative along index :math:`i`, as described in :cite:`wohlberg-2016-convolutional2`. The regularization parameters for the :math:`\ell_1` and gradient terms are ``lmbda`` and ``mu`` respectively. Setting correct weighting arrays for these regularization terms is critical to obtaining good performance. For the :math:`\ell_1` norm, the weights on the filters that are intended to represent the impulse noise are tuned to an appropriate value for the impulse noise density (this value sets the relative cost of representing an image feature by one of the impulses or by one of the filters in the learned dictionary), the weights on the filters that are intended to represent low frequency components are set to zero (we only want them penalised by the gradient term), and the weights of the remaining filters are set to zero. For the gradient penalty, all weights are set to zero except for those corresponding to the filters intended to represent low frequency components, which are set to unity.

.. code-block:: python

  lmbda = 2.8e-2
  mu = 3e-1
  w1 = np.ones((1, 1, 1, 1, D.shape[-1]))
  w1[..., 0:3] = 0.33
  w1[..., 3:6] = 0.0
  wg = np.zeros((D.shape[-1]))
  wg[..., 3:6] = 1.0
  opt = cbpdn.ConvBPDNGradReg.Options(
	 {'Verbose': True, 'MaxMainIter': 100,
	  'RelStopTol': 5e-3, 'AuxVarObj': False,
	  'L1Weight': w1, 'GradWeight': wg})

Now we initialise the ``cbpdn.ConvBPDNGradReg`` object and call the ``solve`` method.

.. code-block:: python

  b = cbpdn.ConvBPDNGradReg(D, pad(imgn), lmbda, mu,
			    opt=opt, dimK=0)
  X = b.solve()


The denoised estimate of the image is just the reconstruction from all coefficient maps except those that represent the impulse noise, which is why we subtract the slice of ``X`` corresponding the impulse noise representing filters from the result of ``reconstruct``.

.. code-block:: python

  imgdp = b.reconstruct().squeeze() \
	  - X[..., 0, 0:3].squeeze()
  imgd = crop(imgdp)


Now we print the PSNR of the noisy and denoised images, and display the reference, noisy, and denoised images. These images are shown in Figures :ref:`fig:idref`, :ref:`fig:idnse`, and :ref:`fig:idden` respectively.

.. code-block:: python

  print('%.3f dB   %.3f dB' % (sm.psnr(img, imgn),
	sm.psnr(img, imgd)))

  fig = plot.figure(figsize=(21, 7))
  plot.subplot(1,3,1)
  plot.imview(img, fgrf=fig, title='Reference')
  plot.subplot(1,3,2)
  plot.imview(imgn, fgrf=fig, title='Noisy')
  plot.subplot(1,3,3)
  plot.imview(imgd, fgrf=fig, title='CSC Result')
  fig.show()

Finally, we save the low frequency image component estimate as an NPZ file, for use in a subsequent example.

.. code-block:: python

  imglp = X[..., 0, 3:6].squeeze()
  np.savez('implslpc.npz', imglp=imglp)


.. figure:: example_gndtrth.png
   :scale: 75%
   :align: center

   Reference image :label:`fig:idref`


.. figure:: example_implsns.png
   :scale: 75%
   :align: center

   Noisy image :label:`fig:idnse`


.. figure:: example_denoise1.png
   :scale: 75%
   :align: center

   Denoised image (first method) :label:`fig:idden`



Extending SPORCO
----------------

We illustrate the ease of extending of modifying existing algorithms in SPORCO by constructing an alternative approach to removing impulse noise via CSC. The previous method gave good results, but the weight on the filter representing the impulse noise is an additional parameter that has to be tuned. This parameter can be avoided by switching to an :math:`\ell_1` data fidelity term instead of including dictionary filters to represent the impulse noise, as in the problem :cite:`wohlberg-2016-convolutional2`

.. math::
   :label: eq:l1cbpdn

   \mathrm{argmin}_\mathbf{x} \;
   \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
   \right \|_1 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;.

Ideally we would also include a gradient penalty term to assist in the representation of the low frequency image component. While this relatively straightforward, it is a bit more complex to implement, and is omitted from this example. Instead of including a representation of the low frequency image component within the optimization, we use the low frequency component estimated by the previous example, subtracting it from the signal passed to the CSC algorithm, and adding it back to the solution of this algorithm.

An algorithm for the problem in Equation (:ref:`eq:l1cbpdn`) is not included in SPORCO, but there is an existing algorithm that can easily be adapted. CBPDN with mask decoupling, with mask array :math:`W`,

.. math::
   :label: eq:mskdcpl

   \mathrm{argmin}_\mathbf{x} \;
   (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
   \mathbf{s}\right) \right\|_2^2 + \lambda \sum_m
   \| \mathbf{x}_m \|_1 \;\;,

is solved via the ADMM problem

.. math::
   :type: align
   :label: eq:mskdcpladmm

   & \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
   (1/2) \| W \mathbf{y}_0 \|_2^2 + \lambda \| \mathbf{y}_1 \|_1 \nonumber \\
   & \;\text{such that}\;
   \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
     \right) = \left( \begin{array}{c} \mathbf{s} \\
     \mathbf{0} \end{array} \right) \;\;,

where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`. We can express Equation (:ref:`eq:l1cbpdn`) using the same variable splitting, as

.. math::
   :type: align
   :label: eq:l1cbpdnadmm

   & \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
   \| W \mathbf{y}_0 \|_1 + \lambda \| \mathbf{y}_1 \|_1 \nonumber \\
   & \;\text{such that}\;
   \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
     \right) = \left( \begin{array}{c} \mathbf{s} \\
     \mathbf{0} \end{array} \right) \;\;.

(We don't need the :math:`W` for the immediate problem at hand, but there isn't any reason for discarding it.) Since Equation (:ref:`eq:l1cbpdnadmm`) has no :math:`f(\mathbf{x})` term (see Equation (:ref:`eq:admmform`)), and has the same constraint as Equation (:ref:`eq:mskdcpladmm`), the :math:`\mathbf{x}` and :math:`\mathbf{u}` steps for these two problems are the same.  The :math:`\mathbf{y}` step for Equation (:ref:`eq:mskdcpladmm`) decomposes into the two independent subproblems

.. math::
   :type: align

   \mathbf{y}_0^{(j+1)} &= \mathrm{argmin}_{\mathbf{y}_0} \frac{1}{2}
    \left\| W \mathbf{y}_0 \right\|_2^2 + \frac{\rho}{2}
    \left\| \mathbf{y}_0 \!-\! (D \mathbf{x}^{(j+1)}  - \mathbf{s}
    + \mathbf{u}_0^{(j)}) \right\|_2^2 \\
   \mathbf{y}_1^{(j+1)} &= \mathrm{argmin}_{\mathbf{y}_1}  \lambda
   \| \mathbf{y}_1 \|_1 + \frac{\rho}{2} \left\| \mathbf{y}_1 -
    (\mathbf{x}^{(j+1)}   + \mathbf{u}_1^{(j)}) \right\|_2^2 \;.

The only difference between the ADMM algorithms for Equations (:ref:`eq:mskdcpladmm`) and (:ref:`eq:l1cbpdnadmm`) is in the :math:`\mathbf{y}_0` subproblem, which becomes

.. math::

   \mathbf{y}_0^{(j+1)} = \mathrm{argmin}_{\mathbf{y}_0}
    \left\| W \mathbf{y}_0 \right\|_1 + \frac{\rho}{2}
    \left\| \mathbf{y}_0 \!-\! (D \mathbf{x}^{(j+1)}  - \mathbf{s}
    + \mathbf{u}_0^{(j)}) \right\|_2^2 \;.

Therefore, the only modifications we expect to make to the class implementing the problem in Equation (:ref:`eq:mskdcpl`) are changing the computation of the functional value, and part of the :math:`\mathbf{y}` step.

We turn now to the implementation for this example. The module import statements and definitions of functions ``pad`` and ``crop`` are the same as for the example in section *Removal of Impulse Noise via CSC*, and are not repeated here. Our main task is to modify ``cbpdn.ConvBPDNMaskDcpl``, the class for solving the problem in Equation (:ref:`eq:mskdcpl`), to replace the :math:`\ell_2` norm data fidelity term with an :math:`\ell_1` norm. The :math:`\mathbf{y}` step of this class is

.. code-block:: python

   def ystep(self):
	AXU = self.AX + self.U
	Y0 = (self.rho*(self.block_sep0(AXU) - self.S)) \
	     / (self.W**2 + self.rho)
	Y1 = sl.shrink1(self.block_sep1(AXU),
			(self.lmbda/self.rho)*self.wl1)
	self.Y = self.block_cat(Y0, Y1)

	super(ConvBPDNMaskDcpl, self).ystep()

where the ``Y0`` block of ``Y`` represents the variable in the data fidelity term, and the ``Y1`` block represents the variable in the regularization term. All we need do to change the data fidelity term to an :math:`\ell_1` norm is to modify the calculation of ``Y0`` to be a soft thresholding instead of the calculation derived from the existing :math:`\ell_2` norm. We also need to override method ``obfn_g0`` so that the functional values are calculated correctly, taking into account the change of the data fidelity term. We end up with a definition of our class solving Equation (:ref:`eq:l1cbpdn`) consisting of only a few lines of additional code

.. code-block:: python

   class ConvRepL1L1(cbpdn.ConvBPDNMaskDcpl):

     def ystep(self):

	AXU = self.AX + self.U
	Y0 = sl.shrink1(self.block_sep0(AXU) - self.S,
			(1.0/self.rho)*self.W)
	Y1 = sl.shrink1(self.block_sep1(AXU),
			(self.lmbda/self.rho)*self.wl1)
	self.Y = self.block_cat(Y0, Y1)

	super(cbpdn.ConvBPDNMaskDcpl, self).ystep()


     def obfn_g0(self, Y0):

	return np.sum(np.abs(self.W *
			     self.obfn_g0var()))


To solve the impulse denoising problem we load the reference image and dictionary, and construct the test image as before. We also need to load the low frequency component saved by the previous example

.. code-block:: python

   imglp = np.load('implslpc.npz')['imglp']


Now we initialise an instance of our new class, solve, and reconstruct the denoised estimate

.. code-block:: python

   lmbda = 3.0
   b = ConvRepL1L1(D, pad(imgn) - imglp, lmbda,
		   opt=opt, dimK=0)
   X = b.solve()
   imgdp = b.reconstruct().squeeze() + imglp
   imgd = crop(imgdp)


The resulting denoised image is displayed in Figure :ref:`fig:idden2`.


.. figure:: example_denoise2.png
   :scale: 75%
   :align: center

   Denoised image (second method) :label:`fig:idden2`



Support Functions and Classes
-----------------------------

In addition to the main set of classes for solving inverse problems,
SPORCO provides a number of supporting functions and classes, within
the following modules:

* ``util``: Various utility functions and classes, including a parallel-processing grid search for parameter optimisation, access to a set of pre-learned convolutional dictionaries, and access to a set of example images.

* ``plot``: Functions for plotting graphs or 3D surfaces and visualising images, providing simplified access to matplotlib functionality.

* ``linalg``: Various linear algebra and related functions, including solvers for specific forms of linear system and filters for computing image gradients.

* ``metric``: Various image quality metrics including standard metrics such as MSE, SNR, and PSNR.

* ``cdict``: A constrained dictionary class that constrains the allowed dict keys, and also initialises the dict with default content on instantiation. All of the inverse problem algorithm options classes are derived from this class.


Conclusion
----------

SPORCO is an actively maintained and thoroughly documented open source Python package for computing with sparse representations. While standard sparse representations are supported, the focus is on convolutional sparse representations for which SPORCO provides a wider range of features than any other publicly available library. The set of ADMM classes on which the optimization algorithms are based is also potentially useful for a much broader range of convex optimization problems.



Acknowledgement
---------------

Development of SPORCO was supported by the U.S. Department of Energy through the LANL/LDRD Program.
