:author: Nina Miolane
:email: nmiolane@stanford.edu
:institution: Stanford University
:orcid: 0000-0002-1200-9024
:corresponding:

:author: Nicolas Guigui
:orcid: 0000-0002-7901-0732
:institution: Université Côte d'Azur, Inria
:email: nicolas.guigui@inria.fr

:author: Hadi Zaatiti
:email: hadi.zaatiti@irt-systemx.fr

:author: Christian Shewmake
:email: cshewmake2@gmail.com

:author: Hatem Hajri
:email: hatem.hajri@irt-systemx.fr

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

:video: https://youtu.be/Ju-Wsd84uG0

:bibliography: main

-----------------------------------------------------------
Introduction to Geometric Learning in Python with Geomstats
-----------------------------------------------------------

.. class:: abstract

There is a growing interest in leveraging differential geometry in the machine learning community. Yet, the adoption of the associated geometric computations has been inhibited by the lack of a reference implementation. Such an implementation should typically allow its users: (i) to get intuition on concepts from differential geometry through a hands-on approach, often not provided by traditional textbooks; and (ii) to run geometric machine learning algorithms seamlessly, without delving into the mathematical details. To address this gap, we present the open-source Python package :code:`geomstats` and introduce hands-on tutorials for differential geometry and geometric machine learning algorithms - Geometric Learning - that rely on it. Code and documentation: :code:`github.com/geomstats/geomstats` and :code:`geomstats.ai`.


.. class:: keywords

   differential geometry, statistics, manifold, machine learning

Introduction
------------

Data on manifolds arise naturally in different fields. Hyperspheres model directional data in molecular and protein biology :cite:`Kent2005UsingStructure` and some aspects of 3D shapes :cite:`Jung2012AnalysisSpheres, Hong2016`. Density estimation on hyperbolic spaces arises to model electrical impedances :cite:`Huckemann2010MobiusEstimation`, networks :cite:`Asta2014GeometricComparison`, or reflection coefficients extracted from a radar signal :cite:`Chevallier2015ProbabilityProcessing`. Symmetric Positive Definite (SPD) matrices are used to characterize data from Diffusion Tensor Imaging (DTI) :cite:`Pennec2006b, Yuan2012` and functional Magnetic Resonance Imaging (fMRI) :cite:`Sporns2005TheBrainArxiv`. These manifolds are curved, differentiable generalizations of vector spaces. Learning from data on manifolds thus requires techniques from the mathematical discipline of differential geometry. As a result, there is a growing interest in leveraging differential geometry in the machine learning community, supported by the fields of Geometric Learning and Geometric Deep Learning :cite:`Bronstein2017`.

Despite this need, the adoption of differential geometric computations has been inhibited by the lack of a reference implementation. Projects implementing code for geometric tools are often custom-built for specific problems and are not easily reused. Some Python packages do exist, but they mainly focus on optimization (Pymanopt :cite:`Townsend2016Pymanopt:DifferentiationArxiv`, Geoopt :cite:`Becigneul2018RiemannianMethods, Kochurov2019Geoopt:Optim`, McTorch :cite:`Meghwanshi2018McTorchLearning`), are dedicated to a single manifold (PyRiemann :cite:`Barachant2015PyRiemann:Interface`, PyQuaternion :cite:`Wynn2014PyQuaternions:Quaternions`, PyGeometry :cite:`Censi2012PyGeometry:Manifolds.`), or lack unit-tests and continuous integration (TheanoGeometry :cite:`Kuhnel2017ComputationalTheano`). An open-source, low-level implementation of differential geometry and associated learning algorithms for manifold-valued data is thus thoroughly welcome.

:code:`Geomstats` is an open-source Python package built for machine learning with data on non-linear manifolds :cite:`MiolaneGeomstatsLearning`: a field called Geometric Learning. The library provides object-oriented and extensively unit-tested implementations of essential manifolds, operations, and learning methods with support for different execution backends - namely NumPy, PyTorch, and TensorFlow. This paper illustrates the use of :code:`geomstats` through hands-on introductory tutorials of Geometric Learning. These tutorials enable users: (i) to build intuition for differential geometry through a hands-on approach, often not provided by traditional textbooks; and (ii) to run geometric machine learning algorithms seamlessly without delving into the lower-level computational or mathematical details. We emphasize that the tutorials are not meant to replace theoretical expositions of differential geometry and geometric learning :cite:`Postnikov2001,Pennec2019RiemannianAnalysis`. Rather, they will complement them with an intuitive, didactic, and engineering-oriented approach.


Presentation of Geomstats
-------------------------

The package `geomstats <https://github.com/geomstats/geomstats>`__ is organized into two main modules: `geometry <https://github.com/geomstats/geomstats/tree/master/geomstats/geometry>`__ and `learning <https://github.com/geomstats/geomstats/tree/master/geomstats/learning>`__. The module :code:`geometry` implements low-level differential geometry with an object-oriented paradigm and two main parent classes: :code:`Manifold` and :code:`RiemannianMetric`. Standard manifolds like the :code:`Hypersphere` or the :code:`Hyperbolic` space are classes that inherit from :code:`Manifold`. At the time of writing, there are over 15 manifolds implemented in :code:`geomstats`. The class :code:`RiemannianMetric` provides computations related to Riemannian geometry on such manifolds such as the inner product of two tangent vectors at a base point, the geodesic distance between two points, the Exponential and Logarithm maps at a base point, and many others.

The module :code:`learning` implements statistics and machine learning algorithms for data on manifolds. The code is object-oriented and classes inherit from :code:`scikit-learn` base classes and mixins such as :code:`BaseEstimator`, :code:`ClassifierMixin`, or :code:`RegressorMixin`. This module provides implementations of Fréchet mean estimators, :math:`K`-means, and principal component analysis (PCA) designed for manifold data. The algorithms can be applied seamlessly to the different manifolds implemented in the library.

The code follows international standards for readability and ease of collaboration, is vectorized for batch computations, undergoes unit-testing with continuous integration, and incorporates both TensorFlow and PyTorch backends to allow for GPU acceleration. The package comes with a `visualization <https://github.com/geomstats/geomstats/blob/master/geomstats/visualization.py>`__ module that enables users to visualize and further develop an intuition for differential geometry. In addition, the `datasets <https://github.com/geomstats/geomstats/tree/master/geomstats/datasets>`__ module provides instructive toy datasets on manifolds. The repositories `examples <https://github.com/geomstats/geomstats/tree/master/examples>`__ and `notebooks <https://github.com/geomstats/geomstats/tree/master/notebooks>`__ provide convenient starting points to get familiar with :code:`geomstats`.


First Steps
-----------

To begin, we need to install :code:`geomstats`. We follow the installation procedure described in the `first steps <https://geomstats.github.io/first-steps.html>`__ of the online documentation. Next, in the command line, we choose the backend of interest: NumPy, PyTorch or TensorFlow. Then, we open the iPython notebook and import the backend together with the visualization module. In the command line::

    export GEOMSTATS_BACKEND=numpy

then, in the notebook:

.. code:: python

    import geomstats.backend as gs
    import geomstats.visualization as visualization

    visualization.tutorial_matplotlib()

.. parsed-literal::

    INFO: Using numpy backend

Modules related to :code:`matplotlib` and :code:`logging` should be imported during setup too. More details on setup can be found on the documentation website: :code:`geomstats.ai`. All standard NumPy functions should be called using the :code:`gs.` prefix - e.g. :code:`gs.exp, gs.log` - in order to automatically use the backend of interest.

Tutorial: Statistics and Geometric Statistics
---------------------------------------------

This tutorial illustrates how Geometric Statistics and Learning differ from traditional Statistics. Statistical theory is usually defined
for data belonging to vector spaces, which are linear spaces. For
example, we know how to compute the mean of a set of numbers or of multidimensional
arrays.

Now consider a non-linear space: a manifold. A manifold
:math:`M` of dimension :math:`m` is a space that is possibly
curved but that looks like an :math:`m`-dimensional vector space in a small
neighborhood of every point. A sphere, like the earth, is a good example of a manifold. What happens when we apply statistical theory defined for linear vector spaces to data that does not naturally belong to a linear space? For example, what happens if we want to perform
statistics on the coordinates of world cities lying on the earth's surface: a
sphere? Let us compute the mean of two data points on the sphere using the traditional definition of the mean.


.. code:: python

    from geomstats.geometry.hypersphere import \
        Hypersphere

    n_samples = 2
    sphere = Hypersphere(dim=2)
    points_in_manifold = sphere.random_uniform(
         n_samples=n_samples)

    linear_mean = gs.sum(
        points_in_manifold, axis=0) / n_samples


.. figure:: 01_data_on_manifolds_files/all_means_paper.pdf
   :align: center
   :scale: 50%

   Left: Linear mean of two points on the sphere. Right: Fréchet mean of two points on the sphere. The linear mean does not belong to the sphere, while the Fréchet mean does. This illustrates how linear statistics can be generalized to data on manifolds, such as points on the sphere. :label:`fig:linearmean`


The result is shown in Figure :ref:`fig:linearmean` (left). What happened? The mean of two points on a manifold (the sphere) is not
on the manifold. In our example, the mean of these cities is not on the earth's surface. This
leads to errors in statistical computations. The line :code:`sphere.belongs(linear_mean)` returns :code:`False`. For this reason, researchers aim to build a theory of statistics that is - by construction - compatible with any structure with which we equip the manifold. This theory is called Geometric Statistics, and the associated learning algorithms: Geometric Learning.

In this specific example of mean computation, Geometric Statistics provides a generalization of
the definition of “mean” to manifolds: the Fréchet mean.

.. code:: python

    from geomstats.learning.frechet_mean import \
        FrechetMean

    estimator = FrechetMean(metric=sphere.metric)
    estimator.fit(points_in_manifold)
    frechet_mean = estimator.estimate_


Notice in this code snippet that :code:`geomstats` provides classes and methods whose API will be instantly familiar to users of the widely-adopted :code:`scikit-learn`. We plot the result in Figure :ref:`fig:linearmean` (right). Observe that the Fréchet mean now belongs to the surface of the sphere!

Beyond the computation of the mean, :code:`geomstats` provides statistics and learning algorithms on manifolds that leverage their specific geometric structure. Such algorithms rely on elementary operations that are introduced in the next tutorial.

Tutorial: Elementary Operations for Data on Manifolds
-----------------------------------------------------

The previous tutorial showed why we need to generalize traditional statistics for data on manifolds. This tutorial shows how to perform the elementary operations that allow us to “translate” learning algorithms from linear spaces to manifolds.

We import data that lie on a manifold: the `world cities <https://simplemaps.com/data/world-cities>`__ dataset, that contains coordinates of cities on the earth's surface. We visualize it in Figure :ref:`fig:cities`.

.. code:: python

    import geomstats.datasets.utils as data_utils

    data, names = data_utils.load_cities()


.. figure:: 01_data_on_manifolds_files/cities_on_manifolds_paper.pdf
   :align: center
   :scale: 50%

   Subset of the world cities dataset, available in :code:`geomstats` with the function :code:`load_cities` from the module :code:`datasets.utils`. Cities' coordinates are data on the sphere, which is an example of a manifold. :label:`fig:cities`


How can we compute with data that lie on such a manifold? The elementary operations on a vector space are addition and subtraction. In a vector space (in fact seen as an affine space), we can add a vector to a point and subtract two points to get a vector. Can we generalize these operations in order to compute on manifolds?

For points on a manifold, such as the sphere, the same operations are not permitted. Indeed, adding a vector to a point will not give a point that belongs to the manifold: in Figure :ref:`fig:operations`, adding the black tangent vector to the blue point gives a point that is outside the surface of the sphere. So, we need to generalize to manifolds the operations of addition and subtraction.

On manifolds, the exponential map is the operation that generalizes the addition of a vector to a point. The exponential map takes the following inputs: a point and a tangent vector to the manifold at that point. These are shown in Figure :ref:`fig:operations` using the blue point and its tangent vector, respectively. The exponential map returns the point on the manifold that is reached by “shooting” with the tangent vector from the point. “Shooting” means following a “geodesic” on the manifold, which is the dotted path in Figure :ref:`fig:operations`. A geodesic, roughly, is the analog of a straight line for general manifolds - the path whose, length, or energy, is minimal between two points, where the notions of length and energy are defined by the Riemannian metric. This code snippet shows how to compute the exponential map and the geodesic with :code:`geomstats`.


.. code:: python

    from geomstats.geometry.hypersphere import \
        Hypersphere

    sphere = Hypersphere(dim=2)

    initial_point = paris = data[19]
    vector = gs.array([1, 0, 0.8])
    tangent_vector = sphere.to_tangent(
         vector, base_point=initial_point)

    end_point = sphere.metric.exp(
        tangent_vector, base_point=initial_point)

    geodesic = sphere.metric.geodesic(
        initial_point=initial_point,
        initial_tangent_vec=tangent_vector)


.. figure:: 02_from_vector_spaces_to_manifolds_files/manifold_operations_paper.pdf
   :align: center
   :scale: 50%

   Elementary operations on manifolds illustrated on the sphere. The exponential map at the initial point (blue point) shoots the black tangent vector along the geodesic, and gives the end point (orange point). Conversely, the logarithm map at the initial point (blue point) takes the end point (orange point) as input, and outputs the black tangent vector. The geodesic between the blue point and the orange point represents the path of shortest length between the two points. :label:`fig:operations`


Similarly, on manifolds, the logarithm map is the operation that generalizes the subtraction of two points on vector spaces. The logarithm map takes two points on the manifold as inputs and returns the tangent vector required to “shoot” from one point to the other. At any point, it is the inverse of the exponential map. In Figure :ref:`fig:operations`, the logarithm of the orange point at the blue point returns the tangent vector in black. This code snippet shows how to compute the logarithm map with :code:`geomstats`.

.. code:: python

    log = sphere.metric.log(
        point=end_point, base_point=initial_point)


We emphasize that the exponential and logarithm maps depend on the “Riemannian metric” chosen for a given manifold: observe in the code snippets that they are not methods of the :code:`sphere` object, but rather of its :code:`metric` attribute. The Riemannian metric defines the notion of exponential, logarithm, geodesic and distance between points on the manifold. We could have chosen a different metric on the sphere that would have changed the distance between the points: with a different metric, the “sphere” could, for example, look like an ellipsoid.

Using the exponential and logarithm maps instead of linear addition and subtraction, many learning algorithms can be generalized to manifolds. We illustrated the use of the exponential and logarithm maps on the sphere only; yet, :code:`geomstats` provides their implementation for over 15 different manifolds in its :code:`geometry` module with support for a variety of Riemannian metrics. Consequently, :code:`geomstats` also implements learning algorithms on manifolds, taking into account their specific geometric structure by relying on the operations we just introduced. The next tutorials show more involved examples of such geometric learning algorithms.

Tutorial: Classification of SPD Matrices
----------------------------------------

Tutorial context and description
********************************

We demonstrate that any standard machine learning
algorithm can be applied to data on manifolds while respecting their geometry. In the previous tutorials, we saw that linear operations (mean, linear weighting, addition and subtraction) are not defined on manifolds. However, each point on a manifold has an associated tangent space which is a vector space. As such, in the tangent space, these operations are well defined! Therefore, we can use the logarithm map (see Figure :ref:`fig:operations` from the previous tutorial) to go from points on manifolds to vectors in the tangent space at a reference point. This first strategy enables the use of traditional learning algorithms on manifolds.

A second strategy can be designed for learning algorithms, such as :math:`K`-Nearest Neighbors classification, that rely only on distances or dissimilarity metrics. In this case, we can compute the pairwise distances between the data points on the manifold, using the method :code:`metric.dist`, and feed them to the chosen algorithm.

Both strategies can be applied to any manifold-valued data. In this tutorial, we consider symmetric positive definite (SPD) matrices from brain connectomics data and perform logistic regression and :math:`K`-Nearest Neighbors classification.

SPD matrices in the literature
******************************

Before diving into the tutorial, let us recall a few applications of SPD matrices
in the machine learning literature. SPD matrices are ubiquitous across many fields :cite:`Cherian2016`, either as input of or output to a given problem. In DTI for instance, voxels are represented by "diffusion tensors" which are 3x3 SPD matrices representing ellipsoids in their structure. These ellipsoids spatially characterize the diffusion of water molecules in various tissues. Each DTI thus consists of a field of SPD matrices, where each point in space corresponds to an SPD matrix. These matrices then serve as inputs to regression models. In :cite:`Yuan2012` for example, the authors use an intrinsic local polynomial regression to compare fiber tracts between HIV subjects and a control group. Similarly, in fMRI, it is possible to extract connectivity graphs from time series of patients' resting-state images :cite:`wang2013disruptedDisease`. The regularized graph Laplacians of these graphs form a dataset of SPD matrices. This provides a compact summary of brain connectivity patterns which is useful for assessing neurological responses to a variety of stimuli, such as drugs or patient's activities.

More generally speaking, covariance matrices are also SPD matrices which appear in many settings. Covariance clustering can be used for various applications such as sound compression in acoustic models of automatic speech recognition (ASR) systems :cite:`Shinohara2010` or for material classification :cite:`Faraki2015`, among others. Covariance descriptors are also popular image or video descriptors :cite:`Harandi2014`.

Lastly, SPD matrices have found applications in deep learning. The authors of :cite:`Gao2017` show that an aggregation of learned deep convolutional features into an SPD matrix creates a robust representation of images which outperforms state-of-the-art methods for visual classification.


Manifold of SPD matrices
************************

Let us recall the mathematical definition of the manifold of SPD matrices. The manifold of SPD matrices in :math:`n` dimensions is embedded in the General Linear group of invertible matrices and defined as:

.. math::
    SPD = \left\{
    S \in \mathbb{R}_{n \times n}: S^T = S, \forall z \in \mathbb{R}^n, z \neq 0, z^TSz > 0
    \right\}.

The class :code:`SPDMatricesSpace` inherits from the class :code:`EmbeddedManifold` and has an :code:`embedding_manifold` attribute which stores an object of the class :code:`GeneralLinear`. SPD matrices in 2 dimensions can be visualized as ellipses with principal axes given by the eigenvectors of the SPD matrix, and the length of each axis proportional to the square-root of the corresponding eigenvalue. This is implemented in the :code:`visualization` module of :code:`geomstats`. We generate a toy data-set and plot it in Figure :ref:`fig:spd` with the following code snippet.

.. code:: python

    import geomstats.datasets.sample_sdp_2d as sampler

    n_samples = 100
    dataset_generator = sampler.DatasetSPD2D(
        n_samples, n_features=2, n_classes=3)

    ellipsis = visualization.Ellipsis2D()
    for i,x in enumerate(data):
        y = sampler.get_label_at_index(i, labels)
        ellipsis.draw(
            x, color=ellipsis.colors[y], alpha=.1)

.. figure:: samples_spd_paper.pdf
   :align: center
   :scale: 50%

   Simulated dataset of SPD matrices in 2 dimensions. We observe 3 classes of SPD matrices, illustrated with the colors red, green, and blue. The centroid of each class is represented by an ellipse of larger width. :label:`fig:spd`

Figure :ref:`fig:spd` shows a dataset of SPD matrices in 2 dimensions organized into 3 classes. This visualization helps in developing an intuition on the connectomes dataset that is used in the upcoming tutorial, where we will classify SPD matrices in 28 dimensions into 2 classes.

Classifying brain connectomes in Geomstats
******************************************

We now delve into the tutorial in order to illustrate the use of traditional learning algorithms on the tangent spaces of manifolds implemented in :code:`geomstats`. We use brain connectome data from the `MSLP 2014 Schizophrenia
Challenge <https://www.kaggle.com/c/mlsp-2014-mri/data>`__. The connectomes are correlation matrices extracted from the time-series of resting-state fMRIs of 86 patients at 28 brain regions of interest: they are points on the manifold of SPD matrices in :math:`n=28` dimensions. Our goal is to use the connectomes to classify patients into two classes: schizophrenic and control. First we load the connectomes and display two of them as heatmaps in Figure :ref:`fig:conn`.

.. code:: python

    import geomstats.datasets.utils as data_utils

    data, patient_ids, labels = \
        data_utils.load_connectomes()

.. figure:: connectomes_paper.pdf
   :align: center
   :scale: 50%

   Subset of the connectomes dataset, available in :code:`geomstats` with the function :code:`load_connectomes` from the module :code:`datasets.utils`. Connectomes are correlation matrices of 28 time-series extracted from fMRI data: they are elements of the manifold of SPD matrices in 28 dimensions. Left: connectome of a schizophrenic subject. Right: connectome of a healthy control. :label:`fig:conn`

Multiple metrics can be used to compute on the manifold of SPD matrices :cite:`dryden_non-euclidean_2009`. As mentionned in the previous tutorial, different metrics define different geodesics, exponential and logarithm maps and therefore different algorithms on a given manifold. Here, we import two of the most commonly used metrics on the SPD matrices, the log-Euclidean metric and the
affine-invariant metric :cite:`Pennec2006b`, but we highlight that :code:`geomstats` contains many more. We also check that our connectome data indeed belongs to the manifold of SPD matrices:

.. code:: python

    import geomstats.geometry.spd_matrices as spd

    manifold = spd.SPDMatrices(n=28)
    le_metric = spd.SPDMetricLogEuclidean(n=28)
    ai_metric = spd.SPDMetricAffine(n=28)
    logging.info(gs.all(manifold.belongs(data)))


.. parsed-literal::

    INFO: True


Great! Now, although the sum of two SPD matrices is an SPD matrix, their
difference or their linear combination with non-positive weights are not
necessarily. Therefore we need to work in a tangent space of the SPD manifold to perform
simple machine learning that relies on linear operations. The :code:`preprocessing` module with its :code:`ToTangentSpace` class allows to do exactly this.

.. code:: python

    from geomstats.learning.preprocessing import \
        ToTangentSpace

``ToTangentSpace`` has a simple purpose: it computes the Fréchet Mean of
the data set, and takes the logarithm map of
each data point from the mean. This results in a data set of tangent vectors at the mean. In the case of the SPD manifold, these are simply symmetric
matrices. ``ToTangentSpace`` then squeezes each symmetric matrix into a 1d-vector of size
``dim = 28 * (28 + 1) / 2``, and outputs an array of shape
``[n_connectomes, dim]``, which can be fed to your favorite :code:`scikit-learn`
algorithm.

We emphasize that ``ToTangentSpace`` computes the mean of the input data, and thus
should be used in a pipeline (as e.g. :code:`scikit-learn`’s ``StandardScaler``)
to avoid leaking information from the test set at train time.

.. code:: python

    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    pipeline = make_pipeline(
        ToTangentSpace(le_metric), LogisticRegression(C=2))

We use a logistic regression on the tangent space at the Fréchet mean to classify connectomes, and evaluate the model with cross-validation. With the log-Euclidean metric we
obtain:

.. code:: python

    result = cross_validate(pipeline, data, labels)
    logging.info(result['test_score'].mean())


.. parsed-literal::

    INFO: 0.67


And with the affine-invariant metric, replacing :code:`le_metric` by :code:`ai_metric` in the above snippet:

.. parsed-literal::

    INFO: 0.71

We observe that the result depends on the metric. The Riemannian metric indeed defines the notion of the logarithm map, which is used to compute the Fréchet Mean and the tangent vectors corresponding to the input data points. Thus, changing the metric changes the result. Furthermore, some metrics may be more suitable than others for different applications. Indeed, we find published results that show how useful geometry can be with data on the SPD manifold (e.g :cite:`Wong2018`, :cite:`Ng2014`).

We saw how to use the representation of points on the manifold as tangent vectors at a reference point to fit any machine learning algorithm, and we compared the effect of different metrics on the manifold of SPD matrices. Another class of machine learning algorithms can be used very easily on manifolds with ``geomstats``: those relying on dissimilarity matrices. We can compute the matrix of pairwise Riemannian distances, using the `dist` method of the Riemannian metric object. In the following code-snippet, we use :code:`ai_metric.dist` and pass the corresponding matrix :code:`pairwise_dist` of pairwise distances to ``scikit-learn``'s :math:`K`-Nearest-Neighbors (KNN) classification algorithm:

.. code:: python

    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(
        metric='precomputed')

    result = cross_validate(
        classifier, pairwise_dist, labels)
    logging.info(result['test_score'].mean())

.. parsed-literal::

    INFO: 0.72


This tutorial showed how to leverage :code:`geomstats` to use standard learning algorithms for data on a manifold. In the next tutorial, we see a more complicated situation: the data points are not provided by default as elements of a manifold. We will need to use the low-level :code:`geomstats` operations to design a method that embeds the dataset in the manifold of interest. Only then, we can use a learning algorithm.

Tutorial: Learning Graph Representations with Hyperbolic Spaces
---------------------------------------------------------------

Tutorial context and description
********************************

This tutorial demonstrates how to make use of the low-level geometric operations in :code:`geomstats` to implement a method that embeds graph data into the hyperbolic space. Thanks to the discovery of hyperbolic embeddings, learning on Graph-Structured Data (GSD) has seen major achievements in recent years. It had been speculated for years that hyperbolic spaces may better represent GSD than Euclidean spaces :cite:`Gromov1987` :cite:`PhysRevE` :cite:`hhh` :cite:`6729484`.
These speculations have recently been shown effective through concrete studies
and applications :cite:`Nickel2017` :cite:`DBLP:journals/corr/ChamberlainCD17` :cite:`DBLP:conf/icml/SalaSGR18` :cite:`gerald2019node`.
As outlined by :cite:`Nickel2017`, Euclidean embeddings require large
dimensions to capture certain complex relations such as the Wordnet
noun hierarchy. On the other hand, this complexity can be captured by
a lower-dimensional model of hyperbolic geometry such as the hyperbolic space of two
dimensions :cite:`DBLP:conf/icml/SalaSGR18`, also called the hyperbolic plane. Additionally, hyperbolic embeddings provide
better visualizations of clusters on graphs than their Euclidean counterparts
:cite:`DBLP:journals/corr/ChamberlainCD17`.

This tutorial illustrates how to learn hyperbolic embeddings in :code:`geomstats`. Specifically, we will embed
the `Karate Club graph <http://konect.cc/networks/ucidata-zachary/>`__ dataset, representing the social interactions of the members of a university Karate club, into the Poincaré ball. Note that we will omit implementation details but an unabridged example and detailed notebook can be found on GitHub in the ``examples`` and ``notebooks`` directories of :code:`geomstats`.

Hyperbolic spaces and machine learning applications
***************************************************

Before going into this tutorial, we review a few applications of hyperbolic spaces
in the machine learning literature. First, Hyperbolic spaces arise in information and
learning theory. Indeed, the space of univariate Gaussians endowed with the Fisher
metric densities is a hyperbolic space :cite:`1531851`. This characterization
is used in various fields, for example in image processing, where each image pixel can be
represented by a Gaussian distribution :cite:`Angulo2014`, or in radar signal
processing where the corresponding echo is represented by a stationary Gaussian process :cite:`Arnaudon2013`. Hyperbolic spaces can
also be seen as continuous versions of trees and are
therefore interesting when learning representations of hierarchical data
:cite:`Nickel2017`. Hyperbolic Geometric Graphs (HGG) have also been suggested
as a promising model for social networks - where the hyperbolicity appears through
a competition between similarity and popularity of an individual :cite:`papadopoulos2012popularity`
and in learning communities on large graphs :cite:`gerald2019node`.


Hyperbolic space
****************

Let us recall the mathematical definition of the hyperbolic space. The :math:`n`-dimensional hyperbolic space :math:`H_n` is defined by its embedding in the :math:`(n+1)`-dimensional Minkowski space as:

.. math::
   :label: hyperbolic

   H_{n} = \left\{
        x \in \mathbb{R}^{n+1}: - x_1^2 + ... + x_{n+1}^2 = -1
    \right\}.


In :code:`geomstats`, the hyperbolic space is implemented in the class :code:`Hyperboloid` and :code:`PoincareBall`, which use different coordinate systems to represent points. These classes inherit from the class :code:`EmbeddedManifold` and have an :code:`embedding_manifold` attribute which stores an object of the class :code:`Minkowski`. The 2-dimensional hyperbolic space is called the hyperbolic plane or Poincaré disk.


Learning graph representations with hyperbolic spaces in :code:`geomstats`
**************************************************************************


`Parameters and Initialization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We now proceed with the tutorial embedding the Karate club graph in a hyperbolic space. In the Karate club graph, each node represents a member of the club, and each edge represents an undirected relation between two members. We first load the Karate club dataset, display it in Figure :ref:`karafig` and print information regarding its nodes and vertices to provide insights into the graph's complexity.

.. code:: python

    karate_graph = data_utils.load_karate_graph()
    nb_vertices_by_edges = (
        [len(e_2) for _, e_2 in
            karate_graph.edges.items()])
    logging.info(
        'Number of vertices: %s', len(karate_graph.edges))
    logging.info(
        'Mean edge-vertex ratio: %s',
        (sum(nb_vertices_by_edges, 0) /
            len(karate_graph.edges)))

.. parsed-literal::

    INFO: Number of vertices: 34
    INFO: Mean edge-vertex ratio: 4.588235294117647


.. figure:: learning_graph_structured_data_h2_files/karate_club.pdf
    :scale: 48%
    :align: center

    Karate club dataset, available in :code:`geomstats` with the function :code:`load_karate_graph` from the module :code:`datasets.utils`. This dataset is a graph, where each node represents a member of the club and each edge represents a tie between two members of the club. :label:`karafig`


Table :ref:`tabparam` defines the parameters needed to embed this graph into a hyperbolic space. The number of hyperbolic dimensions should be high (:math:`n > 10`) only for graph datasets with a large number of nodes and edges. In this tutorial we consider a dataset with only 34 nodes, which are the 34 members of the Karate club.
The Poincaré ball of two dimensions is therefore sufficient to capture the complexity of the graph. We instantiate an object of the class :code:`PoincareBall` in :code:`geomstats`.

.. code:: python

    from geomstats.geometry.poincare_ball
        import PoincareBall

    hyperbolic_manifold = PoincareBall(dim=2)

Other parameters such as
``max_epochs`` and ``lr`` will be tuned specifically for each
dataset, either manually leveraging visualization functions or through a grid/random search that looks for parameter values maximizing some performance function (a measure for cluster separability, normalized mutual information (NMI), or others). Similarly, the number
of negative samples and context size are hyperparameters and will be further discussed below.

.. table:: Hyperparameters used to embed the Karate Club Graph into a hyperbolic space. :label:`tabparam`

    +--------------+---------------------------------------------------+-------+
    | Parameter    | Description                                       | Value |
    +==============+===================================================+=======+
    | dim          | Dimension of the hyperbolic space                 |   2   |
    +--------------+---------------------------------------------------+-------+
    | max_epochs   | Number of embedding iterations                    |  15   |
    +--------------+---------------------------------------------------+-------+
    | lr           | Learning rate                                     |  0.05 |
    +--------------+---------------------------------------------------+-------+
    | n_negative   | Number of negative samples                        |   2   |
    +--------------+---------------------------------------------------+-------+
    | context_size | Size of the context for each node                 |   1   |
    +--------------+---------------------------------------------------+-------+
    | karate_graph | Instance of the ``Graph`` class returned by the           |
    |              | function ``load_karate_graph`` in ``datasets.utils``      |
    +--------------+---------------------------------------------------+-------+




`Learning the embedding by optimizing a loss function`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Denote :math:`V` as the set of nodes and :math:`E \subset V\times V` the
set of edges of the graph. The goal of hyperbolic embedding is to provide a faithful and
exploitable representation of the graph. This goal is mainly achieved
by preserving first-order proximity that encourages nodes sharing edges
to be close to each other. We can additionally preserve second-order
proximity by encouraging two nodes sharing the “same context”, i.e. not necessarily directly connected but sharing a neighbor, to be close. We define a context size (here equal to 1) and call two nodes “context samples” if they share a neighbor, and “negative samples” otherwise. To preserve first and second-order proximities, we adopt the following loss function
similar to :cite:`Nickel2017` and consider the “negative sampling” approach from :cite:`NIPS2013_5021`:

.. math::
   :label: loss

   \mathcal{L} = - \sum_{v_i\in V} \sum_{v_j \in C_i} \bigg[ \log(\sigma(-d^2(\phi_i, \phi_j'))) + \sum_{v_k\sim \mathcal{P}_n} \log(\sigma(d^2(\phi_i, \phi_k')))  \bigg]

where :math:`\sigma(x)=(1+e^{-x})^{-1}` is the sigmoid function and
:math:`\phi_i \in H_2` is the embedding of the :math:`i`-th
node of :math:`V`, :math:`C_i` the nodes in the context of the
:math:`i`-th node, :math:`\phi_j'\in H_2` the embedding of
:math:`v_j\in C_i`. Negatively sampled nodes :math:`v_k` are chosen according to
the distribution :math:`\mathcal{P}_n` such that
:math:`\mathcal{P}_n(v)=(\mathrm{deg}(v)^{3/4}).(\sum_{v_i\in V}\mathrm{deg}(v_i)^{3/4})^{-1}`.

Intuitively one can see in Figure :ref:`fig:notation` that minimizing :math:`\mathcal{L}` makes the distance
between :math:`\phi_i` and :math:`\phi_j` smaller, and the distance
between :math:`\phi_i` and :math:`\phi_k` larger. Therefore
by minimizing :math:`\mathcal{L}`, one obtains representative embeddings.

.. figure:: learning_graph_structured_data_h2_files/notations_pdf2.pdf
    :scale: 60%
    :align: center

    Embedding of the graph's nodes :math:`\{v_i\}_i` as points :math:`\{\phi_i\}_i` of the hyperbolic plane :math:`H_2`, also called the Poincaré ball of 2 dimensions. The blue and red arrows represent the direction of the gradient of the loss function :math:`\mathcal{L}` from Equation :ref:`loss`. This brings context samples closer and separates negative samples. :label:`fig:notation`


`Riemannian optimization`
~~~~~~~~~~~~~~~~~~~~~~~~~

Following the literature on optimization on manifolds :cite:`ganea2018hyperbolic`, we use the following gradient updates
to optimize :math:`\mathcal{L}`:

.. math::  \phi^{t+1} = \text{Exp}_{\phi^t} \left( -lr \frac{\partial \mathcal{L}}{\partial \phi} \right)

where :math:`\phi` is a parameter of :math:`\mathcal{L}`,
:math:`t\in\{1,2,\cdots\}` is the iteration number, and :math:`lr`
is the learning rate. The formula consists of first computing the usual
gradient of the loss function for the direction in which the
parameter should move. The Riemannian exponential map :math:`\text{Exp}` is the operation introduced in the second tutorial: it takes a base point :math:`\phi^t` and a tangent vector :math:`T` and returns the point :math:`\phi^{t+1}`. The Riemannian exponential map is a method of the
``PoincareBallMetric`` class in the ``geometry`` module of
:code:`geomstats`. It allows us to implement a straightforward generalization of standard gradient update in the Euclidean case. To compute the gradient of :math:`\mathcal{L}`, we need to compute the gradients of: (i) the squared distance :math:`d^2(x,y)` on the hyperbolic space, (ii) the log sigmoid :math:`\log(\sigma(x))`, and (iii) the composition of (i) with (ii).


For (i), we use the formula proposed by :cite:`Arnaudon2013` which uses the Riemannian
logarithmic map. Like the exponential
:math:`\text{Exp}`, the logarithmic map is implemented under the ``PoincareBallMetric``.

.. code:: python

    def grad_squared_distance(point_a, point_b, manifold):
        log = manifold.metric.log(point_b, point_a)
        return -2 * log

For (ii), we compute the well-known gradient of the logarithm of the sigmoid function as: :math:`(\log \sigma)'(x) = (1 + \exp(x))^{-1}`. For (iii), we apply the composition rule to obtain the gradient of :math:`\mathcal{L}`. The following function computes :math:`\mathcal{L}` and its gradient on the context samples, while ignoring the part dealing with the negative samples for simplicity of exposition. The code
implementing the whole :code:`loss` function is available on GitHub.

.. code:: python

    def loss(example, context_embedding, manifold):

        context_distance = manifold.metric.squared_dist(
            example, context_embedding)
        context_loss = log_sigmoid(-context_distance)
        context_log_sigmoid_grad = -grad_log_sigmoid(
            -context_distance)

        context_distance_grad = grad_squared_distance(
            example, context_embedding, manifold)

        context_grad = (context_log_sigmoid_grad
            * context_distance_grad)

        return context_loss, -context_grad


`Capturing the graph structure`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We perform initialization computations that capture the graph structure. We compute random walks initialized from each :math:`v_i`
up to some length (five by default). The context nodes :math:`v_j` will be later
picked from the random walk of :math:`v_i`.

.. code:: python

    random_walks = karate_graph.random_walk()

Negatively sampled nodes :math:`v_k` are chosen according to the
previously defined probability distribution function
:math:`\mathcal{P}_n(v_k)` implemented as

.. code:: python

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table += (
            [i] * int((nb_v**(3. / 4.)))
                * negative_table_parameter)


`Numerically optimizing the loss function`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now embed the Karate club graph into the Poincaré disk. The details of the initialization are provided on GitHub. The array :code:`embeddings` contains the embeddings :math:`\phi_i`'s of the nodes :code:`v_i`'s of the current iteration. At each iteration, we compute the gradient of :math:`\mathcal{L}`.
The graph nodes are then moved in the direction pointed by the gradient.
The movement of the nodes is performed by following geodesics in the Poincaré disk in the
gradient direction. In practice, the key to obtaining a representative embedding is to carefully tune the learning rate so that all of the nodes make small movements at each iteration.

A first level loop iterates over the epochs while the table ``total_loss``
records the value of :math:`\mathcal{L}` at each iteration.
A second level nested loop iterates over each path in the previously
computed random walks. Observing these walks, note that nodes having
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
and call the ``loss`` function to compute the gradient. Then the
Riemannian exponential map is applied to find the new value of
:math:`\phi_i` as we mentioned before.

.. code:: python

    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:
            for example_index,
                    one_path in enumerate(path):
                context_index = path[max(
                    0, example_index - context_size):
                    min(example_index + context_size,
                    len(path))]
                negative_index = gs.random.randint(
                    negative_sampling_table.shape[0],
                    size=(len(context_index), n_negative))
                negative_index = (
                    negative_sampling_table[negative_index])
                example_embedding = embeddings[one_path]
                for one_context_i, one_negative_i in \
                    zip(context_index, negative_index):
                    context_embedding = (
                        embeddings[one_context_i])
                    negative_embedding = (
                        embeddings[one_negative_i])
                    l, g_ex = loss(
                        example_embedding,
                        context_embedding,
                        negative_embedding,
                        hyperbolic_manifold)
                    total_loss.append(l)

                    example_to_update = (
                        embeddings[one_path])
                    embeddings[one_path] = (
                        hyperbolic_metric.exp(
                        -lr * g_ex, example_to_update))
        logging.info(
            'iteration %d loss_value %f',
            epoch, sum(total_loss, 0) / len(total_loss))

.. parsed-literal::

    INFO: iteration 0 loss_value 1.819844
    INFO: iteration 14 loss_value 1.363593

Figure :ref:`embeddingiterations` shows the graph embedding at different iterations with the true labels of each node represented with color. Notice how the embedding at convergence separates well the two clusters. Thus, it seems that we have found a useful representation of the graph.

.. figure:: learning_graph_structured_data_h2_files/embedding_iterations.pdf
    :align: center
    :scale: 64%

    Embedding of the Karate club graph into the hyperbolic plane at different iterations. The colors represent the true label of each node. :label:`embeddingiterations`

To demonstrate the usefulness of the embedding learned, we show how to apply a :math:`K`-means algorithm in the hyperbolic plane to predict the label of each node in an unsupervised approach. We use the :code:`learning` module of :code:`geomstats` and instantiate an object of the class :code:`RiemannianKMeans`. Observe again how :code:`geomstats` classes follow :code:`scikit-learn`'s API. We set the number of clusters and plot the results.

.. code:: python

    from geomstats.learning.kmeans import RiemannianKMeans

    kmeans = RiemannianKMeans(
        hyperbolic_manifold.metric, n_clusters=2,
        mean_method='frechet-poincare-ball')
    centroids = kmeans.fit(X=embeddings, max_iter=100)
    labels = kmeans.predict(X=embeddings)

Figure :ref:`fig:kmeans` shows the true labels versus the predicted ones: the two groups of the karate club members have been well separated!

.. figure:: learning_graph_structured_data_h2_files/kmeans_paper.pdf
    :align: center
    :scale: 35%

    Results of the Riemannian :math:`K`-means algorithm on the Karate graph dataset embedded in the hyperbolic plane. Left: True labels associated to the club members. Right: Predicted labels via Riemannian :math:`K`-means on the hyperbolic plane. The centroids of the clusters are shown with a star marker. :label:`fig:kmeans`

Conclusion
----------

This paper demonstrates the use of :code:`geomstats` in performing geometric learning on data belonging to manifolds. These tutorials, as well as many other learning examples on a variety of manifolds, can be found at :code:`geomstats.ai`. We hope that this hands-on presentation of Geometric Learning will help to further democratize the use of differential geometry in the machine learning community.

Acknowledgements
----------------

This work is partially supported by the National Science Foundation, grant NSF DMS RTG 1501767, the Inria-Stanford associated team GeomStats, and the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement G-Statistics No. 786854).
