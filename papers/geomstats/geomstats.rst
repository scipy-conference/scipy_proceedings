:author: Nina Miolane
:email: nmiolane@stanford.edu
:institution: Stanford University
:corresponding:

:author: Nicolas Guigui
:email: nicolas.guigui@inria.fr

:author: Hadi Zaatiti
:email: hadi.zaatiti@irt-systemx.fr

:author: Christian Shewmake
:email: cshewmake2@gmail.com

:author: Daniel Brookes:
:email:

:author: Alice Le Brigant
:email: alice.le-brigant@univ-paris1.fr

:author: Johan Mathe
:email: johan@froglabs.ai

:author: Benjamin Hou
:email: benjamin.hou11@imperial.ac.uk

:author: Yann Thanwerdas
:email: yann.thanwerdas@inria.fr

:author: Stefan Heyder
:email: stefan.heyder@tu-ilmenau.de

:author: Olivier Peltre
:email: opeltre@gmail.com

:author: Niklas Koep
:email: niklas.koep@gmail.com

:author: Hatem Hajri
:email: hatem.hajri@irt-systemx.fr

:author: Yann Cabanes
:email: yann.cabanes@gmail.com

:author: Thomas Gerald
:email: thomas.gerald@lip6.fr

:author: Paul Chauchat
:email: pchauchat@gmail.com

:author: Bernhard Kainz
:email: b.kainz@imperial.ac.uk

:author: Claire Donnat
:email: cdonnat@stanford.edu

:author: Susan Holmes
:email: susan@stat.stanford.edu

:author: Xavier Pennec
:email: xavier.pennec@inria.fr%

:video: http://www.youtube.com/watch?v=dhRUe-gz690

:bibliography: main

-------------------------------------------------------------
Introduction to Geometric Statistics in Python with Geomstats
-------------------------------------------------------------

.. class:: abstract


We introduce `geomstats`, an open-source Python package for computations and statistics for data on non-linear manifolds such as hyperbolic spaces, spaces of symmetric positive definite matrices, Lie groups of transformations, etc. We provide object-oriented and extensively unit-tested implementations. The manifolds come with families of Riemannian metrics, with associated Exponential/Logarithm maps, geodesics, and parallel transport. The learning algorithms follow scikit-learn API and provide methods for estimation, clustering and dimension reduction on manifolds. The operations are vectorized for batch computations and available with NumPy, PyTorch, and TensorFlow backends, which allows GPU acceleration. This talk will present the package, compare it with related libraries, and show relevant examples. Code and documentation: www.geomstats.ai.


:cite:`Evans1993`

.. class:: keywords

   different geometry, statistics, manifold, machine learning

Introduction
------------

Data on manifolds naturally arise in different fields. Hyperspheres model directional data in molecular and protein biology, and some aspects of 3D shapes. Density estimation on hyperbolic spaces arises for electrical impedance, networks or reflection coefficients extracted from a radar signal. Symmetric Positive Definite (SPD) matrices are used to characterize data from Diffusion Tensor Imaging (DTI) and functional Magnetic Resonance Imaging (fMRI). Examples of manifold data are numerous: as a result, there has been a growing interest in leveraging differential geometry in the machine learning community.

Yet, the adoption of differential geometry computations has been inhibited by the lack of a reference implementation. Code sequences are often custom-tailored for specific problems and are not easily reused. Some python packages do exist, but focus on optimization (Pymanopt, Geoopt, and McTorch), or are dedicated to a single manifold (PyRiemann, PyQuaternion, PyGeometry), or lack unit-tests and continuous integration (TheanoGeometry). There is a need for an open-source low-level implementation of differential geometry, and associated learning algorithms, for manifold-valued data.

We present `geomstats`, an open-source Python package of computations and statistics for data on non-linear manifolds such as hyperbolic spaces, spaces of symmetric positive definite matrices, Lie groups of transformations, etc: a field called “geometric statistics”. We provide object-oriented and extensively unit-tested implementations. Geomstats has three main objectives: (i) support research in differential geometry and geometric statistics, by providing code to get intuition or test a theorem (ii) democratize the use of geometric statistics, by implementing user-friendly geometric learning algorithms using scikit-learn API (iii) provide educational support to learn "hands-on" differential geometry and geometric statistics, through its examples and visualizations.


Presentation of Geomstats
-------------------------

In Geomstats, the module `geometry` implements low-level differential geometry with an object-oriented approach and two main parent classes: Manifold and RiemannianMetric. Standard manifolds inherit from Manifold, space-specific attributes and methods can then be added. The class RiemannianMetric provides methods such as the inner product of two tangent vectors at a base point, the geodesic distance between two points, the Exponential and Logarithm maps at a base point, etc. Going beyond Riemannian geometry, the class Connection implements affine connections using automatic differentiation with `autograd` to provide computations when closed-form formulae do not exist.

The module `learning` implements statistics and machine learning algorithms for data on manifolds. The code is object-oriented and classes inherit from scikit-learn base classes and mixin: BaseEstimator, ClassifierMixin, RegressorMixin, etc. This module provides implementations of Frechet mean estimators, K-means and principal component analysis (PCA) designed for manifold data. These algorithms can be applied seamlessly to the different manifolds implemented in the library.

The code follows international standards for readability and ease of collaboration, is vectorized for batch computations, undergoes unit-testing with continuous integration, relies on TensorFlow/PyTorch backend allowing GPU acceleration, and is partially ported to R. The package comes with a `visualization` module that enables users to develop an intuition on differential geometry.


Tutorials of Geometric Statistics with Geomstats
------------------------------------------------

Computing with data on manifolds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    -> Notebooks 01 and 02 here: https://github.com/geomstats/geomstats/tree/master/notebooks

Classification of SPD matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    -> Application here to be converted to a notebook: https://github.com/geomstats/applications/tree/master/brain_connectome

Learning graph representations with Hyperbolic spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    -> Example here to be converted to a notebook: https://github.com/geomstats/geomstats/blob/master/examples/learning_graph_structured_data_h2.py


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


