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

:author: Daniel Brooks
:email: daniel.brooks@lip6.fr

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

There is a growing interest in leveraging differential geometry in the machine learning community. Yet, the adoption of the associated geometric computations has been inhibited by the lack of reference implementation. Such implementation should typically allow its users: (i) to get intuition on concepts from differential geometry through a hands-on approach, often not provided by traditional textbooks; and (ii) to run geometric statistical learning algorithms seemlessly, without delving into the mathematical details. To adress this gap, we introduce the open-source Python package :code:`geomstats` and present hands-on tutorials of differential geometry in statistics - Geometric Statistics - that rely on it. Code and documentation: :code:`www.geomstats.ai`.


.. class:: keywords

   different geometry, statistics, manifold, machine learning

Introduction
------------

Data on manifolds naturally arise in different fields. Hyperspheres model directional data in molecular and protein biology :cite:`Kent2005UsingStructure`, and some aspects of 3D shapes :cite:`Jung2012AnalysisSpheres, Hong2016`. Density estimation on hyperbolic spaces arises to model electrical impedances :cite:`Huckemann2010MobiusEstimation`, networks :cite:`Asta2014GeometricComparison` or reflection coefficients extracted from a radar signal :cite:`Chevallier2015ProbabilityProcessing`. Symmetric Positive Definite (SPD) matrices are used to characterize data from Diffusion Tensor Imaging (DTI) :cite:`Pennec2006b, Yuan2012` and functional Magnetic Resonance Imaging (fMRI) :cite:`Sporns2005TheBrain`. Examples of manifold data are numerous: as a result, there is a growing interest in leveraging differential geometry in the machine learning community.

Yet, the adoption of differential geometry computations has been inhibited by the lack of a reference implementation. Code sequences are often custom-tailored for specific problems and are not easily reused. Some python packages do exist, but focus on optimization (Pymanopt :cite:`Townsend2016Pymanopt:Differentiation`, Geoopt :cite:`Becigneul2018RiemannianMethods, Kochurov2019Geoopt:Optim`, and McTorch :cite:`Meghwanshi2018McTorchLearning`), are dedicated to a single manifold (PyRiemann :cite:`Barachant2015PyRiemann:Interface`, PyQuaternion :cite:`Wynn2014PyQuaternions:Quaternions`, PyGeometry :cite:`Censi2012PyGeometry:Manifolds.`), or lack unit-tests and continuous integration (TheanoGeometry :cite:`Kuhnel2017ComputationalTheano`). An open-source low-level implementation of differential geometry, and associated learning algorithms, for manifold-valued data is thus thoroughly welcome.

We present :code:`geomstats`, an open-source Python package of computations and statistics for data on non-linear manifolds: a field called Geometric Statistics. We provide object-oriented and extensively unit-tested implementations, supported for different execution backends -- namely NumPy, PyTorch, and TensorFlow. This paper illustrates the use of :code:`geomstats` through hands-on introductory tutorials of geometric statistics. The tutorials enable users: (i) to get intuition on concepts from differential geometry through a hands-on approach, often not provided by traditional textbooks; and (ii) to run geometric statistical learning algorithms seemlessly, without delving into the mathematical details.


Presentation of Geomstats
-------------------------

The package :code:`geomstats` is organized into two main modules: :code:`geometry` and :code:`learning`. The module `geometry` implements low-level differential geometry with an object-oriented approach and two main parent classes: :code:`Manifold` and :code:`RiemannianMetric`. Standard manifolds like the hypersphere or the hyperbolic space are classes that inherit from :code:`Manifold`. The class :code:`RiemannianMetric` provides computations related to Riemannian geometry, such as the inner product of two tangent vectors at a base point, the geodesic distance between two points, the Exponential and Logarithm maps at a base point, etc.

The module `learning` implements statistics and machine learning algorithms for data on manifolds. The code is object-oriented and classes inherit from :code:`scikit-learn` base classes and mixin: :code:`BaseEstimator`, :code:`ClassifierMixin`, :code:`RegressorMixin`, etc. This module provides implementations of Frechet mean estimators, k-means and principal component analysis (PCA) designed for manifold data. The algorithms can be applied seamlessly to the different manifolds implemented in the library.

The code follows international standards for readability and ease of collaboration, is vectorized for batch computations, undergoes unit-testing with continuous integration, relies on TensorFlow/PyTorch backend allowing GPU acceleration. The package comes with a :code:`visualization` module that enables users to develop an intuition on differential geometry.


First steps
-----------

Before starting any tutorial, we need to download and set-up geomstats. We choose the backend of interest: :code:`numpy`, :code:`pytorch`, or :code:`tensorflow` and import it, together with the visualization module. In the command line::

    export GEOMSTATS_BACKEND=numpy

then, in the python script:

.. code:: ipython3

    import geomstats.backend as gs
    import geomstats.visualization as visualization

    visualization.tutorial_matplotlib()

.. parsed-literal::

    INFO: Using numpy backend

Modules related to :code:`matplotlib` should be imported during setup too.

Tutorial: Statistics and Geometric Statistics
---------------------------------------------

This tutorial illustrates how Geometric Statistics differ from traditional Statistics. Statistical theory is usually defined
for data belonging to vector spaces, which are linear spaces. For
example, we know how to compute the mean of a data set of numbers or of multidimensional
arrays.

Now consider a non-linear space: a manifold. A manifold
:math:`M` of dimension :math:`m` is a space that is allowed to be
curved but that looks like an :math:`m`-dimensional vector space in the
neighborhood of every point. A sphere, like the earth, is a good example of a manifold.
What happens to the usual statistical theory when the data does not
naturally belong to a linear space. For example, if we want to perform
statistics on the coordinates of world cities, which lie on the earth: a
sphere?


.. code:: ipython3

    from geomstats.geometry.hypersphere import \
        Hypersphere

    sphere = Hypersphere(dim=2)
    points_in_manifold = sphere.random_uniform(
         n_samples=2)

    linear_mean = gs.sum(
        points_in_manifold, axis=0) / n_samples


.. figure:: 01_data_on_manifolds_files/01_data_on_manifolds_16_0.png
   :align: center
   :scale: 50%

   Linear mean of two points on a manifold, the sphere :label:`fig:linearmean`.

|

The result in shown on Figure :ref:`fig:linearmean`.What happened? The mean of two points on a manifold (the sphere) is not
on the manifold. In our example, the mean city is not on the earth. This
leads to errors in statistical computations. The line:

.. code:: ipython3

    sphere.belongs(linear_mean)

returns :code:`False`. For this reason, researchers aim to build a theory of statistics that is
by construction compatible with any structure we equip the manifold
with. This theory is called Geometric Statistics.

In this specific example of mean computation, we use a generalization of
the definition of “mean” to manifolds: the Fréchet mean.

.. code:: ipython3

    from geomstats.learning.frechet_mean import \
        FrechetMean

    estimator = FrechetMean(metric=sphere.metric)
    estimator.fit(points_in_manifold)
    frechet_mean = estimator.estimate_


We plot the result on Figure :ref:`fig:frechetmean`. We observe that the Fréchet mean belongs to
the sphere!

.. figure:: 01_data_on_manifolds_files/01_data_on_manifolds_22_0.png
   :align: center
   :scale: 50%

   Frechet mean of two points on a manifold, the sphere :label:`fig:frechetmean`.

Beyond the computation of the mean, Geometric Statistics is a theory of statistics on manifolds, that takes into account their geometric structures. Geometric Statistics is therefore the child of two major pillars of Mathematics: Geometry and Statistics.

Tutorial: Elementary Operations for Data on Manifolds
-----------------------------------------------------

The previous tutorial showed why we need to generalize traditional statistics for data on manifold. This tutorial shows how to perform the elementary operations that allow to "translate" learning algorithms from linear spaces to manifolds.

We import the dataset :code:`cities` of the coordinates of cities on the earth, and visualize it on Figure :ref:`fig:cities`.

.. code:: ipython3

    import geomstats.datasets.utils as data_utils

    data, names = data_utils.load_cities()


.. figure:: 01_data_on_manifolds_files/01_data_on_manifolds_33_0.png
   :align: center
   :scale: 50%

   World cities as data on a manifold, the sphere :label:`fig:cities`.


How can we compute with data that lie on such a manifold? The elementary operations on a vector space are: addition and substraction. We can add a vector to a point,
substract two points to get a vector. Can we generalize these operations to compute on manifolds?

For points on a manifold, like the sphere, the same operations are not permitted. Indeed, adding a vector to a point will not give a point that belongs to the manifold. The exponential map is the operation that generalizes the addition of a vector to a point, on manifolds.

The exponential map takes a point and a tangent vector as inputs, and outputs the point on the manifold that is reached by “shooting” with the tangent vector. “Shooting” means taking the path of shortest length. This path is called a “geodesic”. Figure :ref:`fig:operations` illustrates this operation and plots the corresponding geodesic.


.. code:: ipython3

    from geomstats.geometry.hypersphere import \
        Hypersphere

    sphere = Hypersphere(dim=2)

    paris = data[19]
    vector = gs.array([1, 0, 0.8])
    tangent_vector = sphere.to_tangent(
         vector, base_point=paris)

    result = sphere.metric.exp(
        tangent_vector, base_point=paris)

    geodesic = sphere.metric.geodesic(
        initial_point=paris,
        initial_tangent_vec=tangent_vector)


.. figure:: 02_from_vector_spaces_to_manifolds_files/02_from_vector_spaces_to_manifolds_19_0.png
   :align: center
   :scale: 50%

   Exponential map, Logarithm map and geodesic on a manifold: the sphere :label:`fig:operations`.


The logarithm map is the operation that generalizes the substraction of two points, that gives a vector.

The logarithm map takes two points on the manifold as inputs, and
outputs the tangent vector that is required to “shoot” from one point to the other.

.. code:: ipython3

    paris = data[19]
    beijing = data[15]

    log = sphere.metric.log(
        point=beijing, base_point=paris)


Using the Riemannian exponential and logarithm instead of the linear addition and substraction, allows to generalize many learning algorithms to manifolds. The next tutorials show more involved examples of learning algorithms on manifold, that use these elementary operations.

Tutorial: Classification of SPD matrices
----------------------------------------


SPD matrices in the literature
******************************

Before going into this tutorial, let us recall a few applications of symmetric positive definite (SPD) matrices
in the machine learning literature.
SPD matrices are ubiquitous in machine learning across many fields :cite:`Cherian2016`, either as input or output to the problem. In diffusion tensor imaging (DTI) for instance, voxels are represented by "diffusion tensors" which are 3x3 SPD matrices. These ellipsoids spatially characterize the diffusion of water molecules in the tissues. Each DTI thus consists in a field of SPD matrices, which are inputs to regression models. In :cite:`Yuan2012` for example, the authors use an intrinsic local polynomial regression applied to comparison of fiber tracts between HIV subjects and a control group. Similarly, in functional magnetic resonance imaging (fMRI), it is possible to extract connectivity graphs from a set of patients' resting-state images' time series :cite:`wang2013disruptedDisease` --a framework known as brain connectomics. The regularized graph Laplacians of the graphs form a dataset of SPD matrices. They represent a compact summary of the brain's connectivity patterns which is used to assess neurological responses to a variety of stimuli (drug, pathology, patient's activity, etc.).

More generally speaking, covariance matrices are also SPD matrices which appear in many settings. We find covariance clustering used for sound compression in acoustic models of automatic speech recognition (ASR) systems :cite:`Shinohara2010` or for material classification :cite:`Faraki2015` among others. Covariance descriptors are also popular image or video descriptors :cite:`Harandi2014`.

Lastly, SPD matrices have found applications in deeep learning, where they are used as features extracted by a neural network. The authors of :cite:`Gao2017` show that an aggregation of learned deep convolutional features into a SPD matrix creates a robust representation of images that enables to outperform state-of-the-art methods on visual classification.


Tutorial context and description
********************************

We demonstrate how any standard machine learning
algorithm can be used on data that live on a manifold yet respecting its
geometry. In the previous tutorials we saw that linear operations (mean, linear weighting) do not work on manifold. However, to each point on a manifold, is associated a tangent space, which is a vector space, where all our off-the-shelf machine learning operations are well defined!

We will use the logarithm map (Figure :ref:`fig:operations`) to go from points of the manifolds to vectors in the tangent space at a reference point. This will enable to use a simple logistic regression to classify our data.


Manifold of SPD matrices
************************

Let us recall the definition of manifold of SPD matrices. The manifold of symmetric positive definite (SPD) matrices in :math:`n` dimensions is defined as:

.. math::
    SPD = \left\{
    S \in \mathbb{R}_{n \times n}: S^T = S, \forall z \in \mathbb{R}^n, z \neq 0, z^TSz > 0
    \right\}.

The class :code:`SPDMatricesSpace` inherits from the class :code:`EmbeddedManifold` and has an :code:`embedding_manifold` attribute which stores an object of the class :code:`GeneralLinearGroup`. We equip the manifold of SPD matrices with an object of the class :code:`SPDMetric` that implements the affine-invariant Riemannian metric of :cite:`Pennec2006b` and inherits from the class :code:`RiemannianMetric`.

Classifying brain connectomes in Geomstats
******************************************

We use data from the `MSLP 2014 Schizophrenia
Challenge <https://www.kaggle.com/c/mlsp-2014-mri/data>`__. The dataset correponds to the Functional Connectivity Networks (FCN) extracted from resting-state fMRIs of 86 patients at 28 Regions Of Interest (ROIs). Roughly, an FCN corresponds to a correlation matrix and can be seen as a point on the manifold of Symmetric Positive-Definite (SPD) matrices. Patients are separated in two classes: schizophrenic and control. The goal will be to classify them.

First we load the data (reshaped as matrices):

.. code:: ipython3

    import geomstats.datasets.utils as data_utils

    data, patient_ids, labels = \
        data_utils.load_connectomes()

As mentionned above, correlation matrices are SPD matrices. Because
multiple metrics could be used on SPD matrices, we also import two of
the most commonly used ones: the Log-Euclidean metric and the
Affine-Invariant metric :cite:`Pennec2006b`. We can use the SPD module from
``geomstats`` to handle all the geometry, and check that our data indeed
belongs to the manifold of SPD matrices:

.. code:: ipython3

    import geomstats.geometry.spd_matrices as spd

    manifold = spd.SPDMatrices(28)
    ai_metric = spd.SPDMetricAffine(28)
    le_metric = spd.SPDMetricLogEuclidean(28)
    print(gs.all(manifold.belongs(data)))


.. parsed-literal::

    True


Great! Now, although the sum of two SPD matrices is an SPD matrix, their
difference or their linear combination with non-positive weights are not
necessarily! Therefore we need to work in a tangent space to perform
simple machine learning. But worry not, all the geometry is handled by
geomstats, thanks to the preprocessing module.

.. code:: ipython3

    from geomstats.learning.preprocessing import \
        ToTangentSpace

What ``ToTangentSpace`` does is simple: it computes the Frechet Mean of
the data set (covered in the previous tutorial), then takes the log of
each data point from the mean. This results in a set of tangent vectors,
and in the case of the SPD manifold, these are simply symmetric
matrices. It then squeezes them to a 1d-vector of size
``dim = 28 * (28 + 1) / 2``, and thus outputs an array of shape
``[n_patients, dim]``, which can be fed to your favorite scikit-learn
algorithm.

Because the mean of the input data is computed, ``ToTangentSpace``
should be used in a pipeline (as e.g. scikit-learn’s ``StandardScaler``)
not to leak information from the test set at train time.

.. code:: ipython3

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    pipeline = Pipeline(
        steps=[
            ('feature_ext',
             ToTangentSpace(geometry=ai_metric)),
            ('classifier',
             LogisticRegression(C=2))])

We now have all the material to classify connectomes, and we evaluate
the model with cross validation. With the affine-invariant metric we
obtain:

.. code:: ipython3

    result = cross_validate(pipeline, data, labels)
    print(result['test_score'].mean())


.. parsed-literal::

    0.71


And with the log-Euclidean metric:

.. code:: ipython3

    pipeline = Pipeline(
        steps=[
            ('feature_ext',
             ToTangentSpace(geometry=le_metric)),
            ('classifier',
             LogisticRegression(C=2))])

    result = cross_validate(pipeline, data, labels)
    print(result['test_score'].mean())


.. parsed-literal::

    0.67

But wait, why do the results depend on the metric used? The Riemannian metric defines the notion of geodesics and distance on the manifold. Both notions are used to compute the Frechet Mean and the logarithms, so changing the metric changes the results, and some metrics may be more suitable than others for different applications.


In this example using Riemannian geometry, we observe that the choice of metric has an impact on the classification accuracy.
There are published results that show how useful geometry can be
with this type of data (e.g :cite:`Wong2018`, :cite:`Ng2014`). We saw how to use the representation of points on the manifold as tangent vectors at a reference point to fit any machine learning algorithm, and compared the effect of different metrics on the space of symmetric positive-definite matrices.


Tutorial: Learning graph representations with Hyperbolic spaces
---------------------------------------------------------------

Hyperbolic spaces and machine learning applications
***************************************************

Before going into this tutorial, let us recall a few applications of hyperbolic spaces
in the machine learning literature. Hyperbolic spaces arise in information and
learning theory. Indeed, the space of univariate Gaussians endowed with the Fisher
metric densities is a hyperbolic space :cite:`1531851`. This characterization
is used in various fields, such as in image processing, where each image pixel is
represented by a Gaussian distribution :cite:`Angulo2014`, or in radar signal
processing where the corresponding echo is represented by a stationary Gaussian process :cite:`Arnaudon2013`.

The hyperbolic spaces can also be stanfordeen as continuous versions of trees and are
therefore interesting when learning hierarchical representations of data
:cite:`Nickel2017`. Hyperbolic geometric graphs (HGG) have also been suggested
as a promising model for social networks, where the hyperbolicity appears through
a competition between similarity and popularity of an individual :cite:`papadopoulos2012popularity`.

Tutorial context and description
********************************

Recently, the embedding of Graph Structured Data (GSD) on manifolds has
received considerable attention. Learning GSD has known major achievements in recent years thanks to the
discovery of hyperbolic embeddings. Although it has been speculated since
several years that hyperbolic spaces would better represent GSD than
Euclidean spaces :cite:`Gromov1987` :cite:`PhysRevE` :cite:`hhh` :cite:`6729484`, it is only recently
that these speculations have been proven effective through concrete studies
and applications :cite:`Nickel2017` :cite:`DBLP:journals/corr/ChamberlainCD17` :cite:`DBLP:conf/icml/SalaSGR18` :cite:`gerald2019node`.
As outlined by :cite:`Nickel2017`, Euclidean embeddings require large
dimensions to capture certain complex relations such as the Wordnet
noun hierarchy. On the other hand, this complexity can be captured by
a simple model of hyperbolic geometry such as the Poincaré disc of two
dimensions :cite:`DBLP:conf/icml/SalaSGR18`.
Additionally, hyperbolic embeddings provide better visualisation of
clusters on graphs than Euclidean embeddings
:cite:`DBLP:journals/corr/ChamberlainCD17`.

In the scope of these recent
discoveries, this tutorial shows how to learn such embeddings in :code:`geomstats`
using the Poincaré Ball manifold applied to the well-known ‘Karate Club’ dataset.
Please note that in the sequel we omit details regarding resizing the data arrays.
A full working code is available in the ``examples`` directory and additionally a detailed notebook under ``notebooks``.

We will first recall a few properties of hyperbolic spaces. Then show how to
import the necessary modules from :code:`geomstats` and initialize embedding parameters.
The embedding method is then presented formally while showing how it is
implemented in :code:`geomstats`. Finally the resulting embedding is plotted.

Hyperbolic space
****************

The :math:`n`-dimensional hyperbolic space :math:`H_n` is defined by its embedding in the :math:`(n+1)`-dimensional Minkowski space, which is a flat pseudo-Riemannian manifold, as:

.. math::
   :label: hyperbolic

   H_{n} = \left\{
        x \in \mathbb{R}^{n+1}: - x_1^2 + ... + x_1{n+1}^2 = -1
    \right\}.


In :code:`geomstats`, the hyperbolic space is implemented in the classes :code:`Hyperboloid` and :code:`PoincareBall` depending on the coordinate system used to represent the points. These classes  inherit from the class :code:`EmbeddedManifold`. They implement methods such as: conversion functions from intrinsic $n$-dimensional coordinates to extrinsic :math:`(n+1)`-dimensional coordinates in the embedding space (and vice-versa); projection of a point in the embedding space to the embedded manifold; projection of a vector in the embedding space to a tangent space at the embedded manifold.

The Riemannian metric defined on :math:`H_n` is derived from the Minkowski metric in the embedding space and is implemented in the class :code:`HyperbolicMetric`.


Learning graph representations with hyperbolic spaces in `Geomstats`
********************************************************************

`Setup`
~~~~~~~

We start by importing standard tools for logging and visualization,
allowing us to draw the embedding of the GSD on the manifold. Next, we
import the manifold of interest, visualization tools, and other methods
from :code:`geomstats`.

.. code:: ipython3

    import logging

    import matplotlib.pyplot as plt

    import geomstats
    import geomstats.backend as gs
    import geomstats.visualization as visualization
    from geomstats.datasets
        import graph_data_preparation as gdp
    from geomstats.geometry.poincare_ball
        import PoincareBall


`Parameters and Initialization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Table :ref:`tabparam` defines the parameters needed for embedding.
Let us discuss a few things about these parameters. The
number of dimensions should be high (i.e., 10+) for large datasets
(i.e., where the number of nodes/edges is significantly large). In this
tutorial we consider a dataset that is quite small with only 34 nodes.
The Poincaré disk of only two dimensions is therefore sufficient to
capture the complexity of the graph and provide a faithful
representation. Some parameters are hard to know in advance, such as
``max_epochs`` and ``lr``. These should be tuned specifically for each
dataset. Visualization can help with tuning the parameters. Also, one
can perform a grid search to find values of these parameters which
maximize some performance function. In learning embeddings, one can
consider performance metrics such as a measure for cluster seperability
or normalized mutual information (NMI) or others. Similarly, the number
of negative samples and context size can also be thought of as
hyperparameters and will be further discussed in the sequel. An instance
of the ``Graph`` class is created and set to the Karate club dataset.
The latter and several others can be found in the ``datasets.data`` module.

.. table:: Embedding parameters :label:`tabparam`

    +--------------+------------------------------------------------+
    | Parameter    | Description                                    |
    +==============+================================================+
    | random.seed  | An initial manually set number                 |
    |              | for generating pseudorandom                    |
    |              | numbers                                        |
    +--------------+------------------------------------------------+
    | dim          | Dimensions of the manifold used for embedding  |
    +--------------+------------------------------------------------+
    | max_epochs   | Number of iterations for learning the embedding|
    +--------------+------------------------------------------------+
    | lr           | Learning rate                                  |
    +--------------+------------------------------------------------+
    | n_negative   | Number of negative samples                     |
    +--------------+------------------------------------------------+
    | context_size | Size of the considered context                 |
    |              | for each node of the graph                     |
    +--------------+------------------------------------------------+


.. code:: ipython3

    gs.random.seed(1234)
    dim = 2
    max_epochs = 15
    lr = .05
    n_negative = 2
    context_size = 1
    karate_graph = gdp.Graph(
        graph_matrix_path=
            geomstats.datasets.utils.KARATE_PATH,
        labels_path=
            geomstats.datasets.utils.KARATE_LABELS_PATH)

The Zachary karate club network was collected from the members of a
university karate club by Wayne Zachary in 1977. Each node represents a
member of the club, and each edge represents an undirected relation
between two members. An often discussed problem using this dataset is to
find the two groups of people into which the karate club split after an
argument between two teachers. Figure :ref:`karafig` displays the dataset graph.
Further information about the dataset is
displayed to provide insight into its complexity.

.. figure:: learning_graph_structured_data_h2_files/karate_graph.png
    :scale: 30%
    :align: center

    Karate club dataset graph. :label:`karafig`


.. code:: ipython3

    nb_vertices_by_edges =\
        [len(e_2) for _, e_2 in
            karate_graph.edges.items()]
    logging.info('
        Number of edges: %s', len(karate_graph.edges))
    logging.info(
        'Mean vertices by edges: %s',
        (sum(nb_vertices_by_edges, 0) /
            len(karate_graph.edges)))

.. parsed-literal::

    INFO: Number of edges: 34
    INFO: Mean vertices by edges: 4.588235294117647


Let us now prepare the hyperbolic space for embedding.
Recall that :math:`H_2` is the Poincaré disk equipped with the distance function
:math:`d`. Declaring an instance of the ``PoincareBall`` manifold of two dimensions
in :code:`geomstats` is straightforward:

.. code:: ipython3

    hyperbolic_manifold = PoincareBall(2)


`Learning embedding by optimizing a loss function`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Denote :math:`V` as the set of nodes and :math:`E \subset V\times V` the
set of edges of the graph. The goal of embedding GSD is to provide a faithful and
exploitable representation of the graph structure. It is mainly achieved
by preserving first-order proximity that enforces nodes sharing edges
to be close to each other. It can additionally preserve second-order
proximity that enforces two nodes sharing the same context (i.e., nodes
that are neighbours but not necessarily directly connected) to be close.
To preserve first and second-order proximities we adopt the following loss function
similar to :cite:`NIPS2017_7213` and consider the negative sampling
approach as in :cite:`NIPS2013_5021`:

.. math::      \mathcal{L} = - \sum_{v_i\in V} \sum_{v_j \in C_i} \bigg[ log(\sigma(-d^2(\phi_i, \phi_j'))) + \sum_{v_k\sim \mathcal{P}_n} log(\sigma(d^2(\phi_i, \phi_k')))  \bigg]

where :math:`\sigma(x)=(1+e^{-x})^{-1}` is the sigmoid function and
:math:`\phi_i \in H_2` is the embedding of the :math:`i`-th
node of :math:`V`, :math:`C_i` the nodes in the context of the
:math:`i`-th node, :math:`\phi_j'\in H_2` the embedding of
:math:`v_j\in C_i`. Negatively sampled nodes :math:`v_k` are chosen according to
the distribution :math:`\mathcal{P}_n` such that
:math:`\mathcal{P}_n(v)=(deg(v)^{3/4}).(\sum_{v_i\in V}deg(v_i)^{3/4})^{-1}`.

Intuitively one can see on Figure :ref:`fignotation` that to minimizing :math:`\mathcal{L}`, the distance
between :math:`\phi_i` and :math:`\phi_j` should get smaller, while the one
between :math:`\phi_i` and :math:`\phi_k` would get larger. Therefore
by minimizing :math:`\mathcal{L}`, one obtains representative embeddings.

.. figure:: learning_graph_structured_data_h2_files/Notations.png
    :scale: 40%
    :align: center

    Distances between node embeddings after applying one optimization iteration :label:`fignotation`.

`Riemannian optimization`
~~~~~~~~~~~~~~~~~~~~~~~~~

Following the idea of :cite:`ganea2018hyperbolic` we use the following formula
to optimize :math:`\mathcal{L}`:

.. math::  \phi^{t+1} = \text{Exp}_{\phi^t} \left( -lr \frac{\partial \mathcal{L}}{\partial \phi} \right)

where :math:`\phi` is a parameter of :math:`\mathcal{L}`,
:math:`t\in\{1,2,\cdots\}` is the epoch iteration number and :math:`lr`
is the learning rate. The formula consists of first computing the usual
gradient of the loss function giving the direction in which the
parameter should move. The Riemannian exponential map :math:`\text{Exp}`
is a function that takes a base point :math:`\phi^t` and some direction
vector :math:`T` and returns the point :math:`\phi^{t+1}` such that
:math:`\phi^{t+1}` belongs to the geodesic initiated from
:math:`\phi{t}` in the direction of :math:`T` and the length of the
geoedesic curve between :math:`\phi^t` and :math:`\phi^{t+1}` is of 1
unit. The Riemannian exponential map is implemented as a method of the
``PoincareBallMetric`` class in the ``geometry`` module of
:code:`geomstats`.

As a summary to minimize :math:`\mathcal{L}`, we will need to compute its gradient.
To do so, we will need the gradient of:


1. the squared distance :math:`d^2(x,y)`
2. the log sigmoid :math:`log(\sigma(x))`
3. the composition of 1. with 2.


For 1., we use the formula proposed by :cite:`Arnaudon2013` which uses the Riemannian
logarithmic map to compute the gradient of the distance implemented below. Similarly as the exponential
:math:`\text{Exp}`, the logarithmic map is implemented under the ``PoincareBallMetric``.

.. code:: ipython3

    def grad_squared_distance(point_a, point_b):
        hyperbolic_metric = PoincareBall(2).metric
        log_map = hyperbolic_metric.log(point_b, point_a)
        return -2 * log_map

For 2. define the ``log_sigmoid`` as below. Note that the used `log` here is
the usual function and not the Riemannian logarithmic map.

.. code:: ipython3

    def log_sigmoid(vector):
        return gs.log((1 / (1 + gs.exp(-vector))))

The gradient of the logarithm of sigmoid function is implemented as:

.. code:: ipython3

    def grad_log_sigmoid(vector):
        return 1 / (1 + gs.exp(vector))

For 3., apply the composition rule to obtain the gradient of :math:`\mathcal{L}`.
To obtain the value of :math:`\mathcal{L}` the loss function
formula is simply applied. For the gradient of :math:`\mathcal{L}`, we apply the composition of
``grad_log_sigmoid`` with ``grad_squared_distance`` while paying
attention to the signs. For simplicity, the following function computes the loss function and gradient of
:math:`\mathcal{L}` while ignoring the part dealing with the negative samples (The code
implementing the whole loss function is available in in the `examples` directory).

.. code:: ipython3

    def context_loss(
        example_embedding, context_embedding, manifold):

        dim = example_embedding.shape[-1]

        context_distance =\
            manifold.metric.squared_dist(
                example_embedding,
                context_embedding)
        context_loss =\
            log_sigmoid(-context_distance)

        context_log_sigmoid_grad =\
            -grad_log_sigmoid(-context_distance)

        context_distance_grad =\
            grad_squared_distance(example_embedding,
            context_embedding)

        context_grad =\
            context_log_sigmoid_grad,
            * context_distance_grad

        example_grad = -context_grad
        return context_loss, example_grad


`Capturing the graph structure`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point we have the necessary bricks to compute the resulting
gradient of :math:`\mathcal{L}`. We are ready to prepare the nodes :math:`v_i`,
:math:`v_j` and :math:`v_k` and initialise their embeddings
:math:`\phi_i`, :math:`\phi^{'}_j` and :math:`\phi^{'}_k`. First,
initialize an array that will hold embeddings :math:`\phi_i` of each
node :math:`v_i\in V` with random points belonging to the Poincaré disk.

.. code:: ipython3

    embeddings = gs.random.normal(
        size=(karate_graph.n_nodes, dim)) * 0.2

Next, to prepare the context nodes :math:`v_j` for each node
:math:`v_i`, we compute random walks initialised from each :math:`v_i`
up to some length (5 by default). The latter is done via a special
function within the ``Graph`` class. The nodes :math:`v_j` will be later
picked from the random walk of :math:`v_i`.

.. code:: ipython3

    random_walks = karate_graph.random_walk()

Negatively sampled nodes :math:`v_k` are chosen according to the
previously defined probability distribution function
:math:`\mathcal{P}_n(v_k)` implemented as

.. code:: ipython3

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table +=\
            ([i] * int((nb_v**(3. / 4.)))
                * negative_table_parameter)


Numerically optimizing the loss function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimising the loss function is performed numerically over the number of
epochs. At each iteration, we will compute the gradient of :math:`\mathcal{L}`.
Then the graph nodes are moved in the direction pointed by the gradient.
The movement of the nodes is performed by following geodesics in the
gradient direction. The key to obtain an embedding representing
accurately the dataset, is to move the nodes smoothly rather than brutal
movements. This is done by tuning the learning rate, such as at each
epoch all the nodes made small movements.

A first level loop iterates over the epochs, the table ``total_loss``
will record the value of :math:`\mathcal{L}` at each iteration and help us track
the minimization of :math:`\mathcal{L}`.
A second level nested loop iterates over each path in the previously
computed random walks. Observing these walks, notice that nodes having
many edges appear more often. Such nodes can be considered as important
crossroads and will therefore be subject to a greater number of
embedding updates. This is one of the main reasons why random walks have
proven to be effective in capturing the structure of graphs. The context
of each :math:`v_i` will be the set of nodes :math:`v_j` belonging to
the random walk from :math:`v_i`. The ``context_size`` specified earlier
will limit the length of the walk to be considered. Similarly, we use
the same ``context_size`` to limit the number of negative samples. We
find :math:`\phi_i` from the ``embeddings`` array.

A third and fourth level nested loops will iterate on each :math:`v_j` and
:math:`v_k`. From within, we find :math:`\phi'_j` and :math:`\phi'_k`
then call the ``loss`` function to compute the gradient. Then the
Riemannian exponential map is applied to find the new value of
:math:`\phi_i` as we mentioned before.

.. code:: ipython3

    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:

            for example_index, one_path in enumerate(path):
                context_index = path[max(
                    0, example_index - context_size):
                    min(example_index + context_size,
                    len(path))]
                negative_index =\
                    gs.random.randint(
                        negative_sampling_table.shape[0],
                        size=(len(context_index),
                        n_negative))
                negative_index =
                    negative_sampling_table[negative_index]

                example_embedding =
                    embeddings[one_path]
                for one_context_i, one_negative_i in
                    zip(context_index, negative_index):
                    context_embedding =
                        embeddings[one_context_i]
                    negative_embedding =
                        embeddings[one_negative_i]
                    l, g_ex = loss(
                        example_embedding,
                        context_embedding,
                        negative_embedding,
                        hyperbolic_manifold)
                    total_loss.append(l)

                    example_to_update =
                        embeddings[one_path]
                    embeddings[one_path] =
                        hyperbolic_manifold.metric.exp(
                        -lr * g_ex, example_to_update)
        logging.info(
            'iteration %d loss_value %f',
            epoch, sum(total_loss, 0) / len(total_loss))

.. parsed-literal::

    INFO: iteration 0 loss_value 1.819844
    INFO: iteration 14 loss_value 1.363593

Figure :ref:`embeddingiterations` shows how the node embeddings move at different iterations.

.. figure:: learning_graph_structured_data_h2_files/embedding_iterations.png
    :align: center
    :scale: 60%

    Embedding at different `epoch` iterations. :label:`embeddingiterations`

Conclusion
----------

This paper demonstrated the use of :code:`geomstats` to perform geometric learning on data that belong to manifolds. These tutorials, as well as many other examples, can be found at :code:`geomstats.ai`.

Acknowledgements
----------------

This work is partially supported by the National Science Foundation, grant NSF
DMS RTG 1501767, the Inria-Stanford associated team GeomStats, and the European
Research Council (ERC) under the European Union's Horizon 2020 research and
innovation program (grant agreement G-Statistics No. 786854).
