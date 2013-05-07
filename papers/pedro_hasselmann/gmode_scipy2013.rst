:author: Pedro Henrique A. Hasselmann
:email: hasselmann@on.br
:institution: Observatorio Nacional

:author: Jorge Márcio Carvano
:email: carvano@on.br
:institution: Observatorio Nacional

:author: Daniela Lazzaro
:email: lazzaro@on.br
:institution: Observatorio Nacional

.. include:: <isogrk1.txt>

------------------------------------------------
G-mode Clustering Method applied to Asteroid Taxonomy
------------------------------------------------

.. class:: abstract

G-mode is a clustering method developed by A. I. Gavrishin in the 60's for geochemical classification of rocks, 
but was also applied to asteroid photometry, cosmic rays, lunar sample and planetary science spectroscopy data. 
In this work, we used it to classify the asteroids from SDSS Moving Objects Catalog. 
The method works identifying normal distributions in a multidimensional space of variables. 
The identification starts with finding the closest points in the sample, which is a consumable problem when the data is not planar. 
Therefore, to achieve satisfying results in a human bearable time, we implemented the method, 
which was previously written in FORTRAN 77, in PYTHON 2.7 and using NumPy, SciPy and Matplotlib packages. 
The NumPy was used for array and matrix manipulation and Matplotlib for plot control. 
The Scipy had a import role in speeding up G-mode, Scipy.cKD-Tree and Numpy.histogramdd were applied to find the initial seeds 
from which the clusters are going to evolve. Scipy was also used to quickly produce dendograms showing the distances between the clusters.

Finally, results for Asteroids Taxonomy and tests for different sample sizes and implementations are going to be presented.

.. class:: keywords

   clustering, taxonomy, asteroids, multivariate data 

Introduction
------------
The classes were defined using the G-mode multivariate clustering method, designed by A. I. Gavrishin [Cor76]_ and 
previously implemented to FORTRAN V by Cor77_ to classify geochemical samples, but applicable to a wide range of research fields, 
as planetary sciences [REF], disk-resolved remote sensing [REF] and cosmology [REF]. 
The G-mode classifies *N* elements into *Nc* unimodal clusters containing *Na* elements each. Elements are described by M variables. 

This method is unsupervised, which allows an automatic identification of clusters without any a priori knowledge of the sample distribution. 
For that, user must control only two critical parameters for the classification, the confidence levels *q1* and *q2*, that may be equated 
for simplification. Smaller these parameters get, more clusters are resolved and lower their variances are.

The G-mode used here follows a adapted version of the original method published by Gav92_, briefly described by Ful00_ and reviewed by Tos05_ and Ley10_. 
Robust central tendency and absolute deviation estimators, a faster initial seed finder and statistical whitenning were introduced to produce a more 
robust set of clusters and optimize the processing time. The coding was performed in Python 2.7 with help of Matplotlib, Numpy and Scipy packages. 
The method procedure can be briefly summarized in two parts: the first one is the cluster recognition and 
the second evaluates each variable in the classification process. Each are described in the following sections. 

 
Recognition Of The Unimodal Clusters
------------------------------------

The first procedure can be summarized on the following topics:

- *The data is arranged according to the code*:

.. code-block:: python

   """ from module: Gmode.py """

   def LoadData(filename):

       from operator import getitem, itemgetter
       from numpy import genfromtxt

       data = map(list,genfromtxt(filename, dtype=None))

       design   = map(itemgetter(0),data)
       unique_id  = map(itemgetter(1),data)

       elements  = [array(item[2::2], dtype=float64) for item in data]
       errors   = [array(item[3::2], dtype=float64) for item in data]

       indexes = range(len(self.design))

All variables are whiten (:scipy.cluster.vq.whiten:), which means they are divided by their absolute deviation to scale all them up. 
This is a important measure when dealing with percentage variables, such as geometric albedos.

- *Initial seed of a forming cluster is identified*. 
  At the original implementation, the G-mode relied on a force-brute algorithm to find the three closest elements as initial seed, 
  which required long processing time. Therefore, in our version, the initial seeds are searched recursively using ``numpy.histogramdd``, which
  produced a faster result:

.. code-block:: python

   """  From module: barycenter.py """

   def boolist(index, values, lim):
        if all([boo(item[0],item[1]) for item in izip(values,lim)]):
           return index

   def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

   def volume(lst):
       p = 1
       for i in lst: p *= i[1] - i[0]
       return p
    
   def barycenter_density(data, grid, upper, lower, dens = 0e0, nmin = 6):
   
       from numpy import histogramdd, array, unravel_index, amax
   
       rng   = range(data.shape[1])
       nbin = map(int,array([grid]*data.shape[1]))
       hist, edges = histogramdd(data,bins=nbin,range=tuple(zip(lower, upper)))
       limits = array([list(pairwise(edges[i])) for i in rng])
       ind = unravel_index(argmax(hist), hist.shape) 
       zone = array([limits[i,j] for i, j in izip(rng, ind)])
       density = amax(hist) / volume(zone)
       
       """ Recurvise until the density goes lower or the minimum number is reached. """
       if density > dens and amax(hist) > nmin:
          zone = zone.T
          return barycenter_density(data, grid, zone[1], zone[0], density, nmin)
       else:
          return filter(lambda x: x != None, \
                 imap(lambda i, y: boolist(i,y,zone), xrange(data.shape[0]), data))

The function above divides the variable hyperspace into large sectors, and just in the most crowded sector the initial seed is searched for. 
Recursively, the most crowded sector is once divided as long as the density grows up. 
When density decreases or the minimal number of points set by the user is reached, the procedure stops. 
The initial seed is chosen from the elements of the most crowded sector before ending the procedure. 
In the end, starting central tendency :math:`|mgr|_{i}` and absolute deviation :math:`|sfgr|_{i}` are estimated from the initial seed. 
If any absolute deviation is zeroth, the value is replaced by the median error of the variable.                 

- *Z² criterion*. In the next step, the mahalanobis distance (``scipy.spatial.distance.mahalanobis``) between the tested cluster and all elements are computed.

- *Hypothesis Testing*. The Z² estimator follows a *X* ² distribution, but for sake of simplification, Z² can be transformed to gaussian estimator ``G``
  if the degree of freedom is larger enough, which is satisfied for most of samples. Now, the critical value :math:`G_{q1}`
  in hypothesis testing are given as multiples of |sfgr|, simplifying its interpretation. 
  Therefore, the vectorized transformation [Abra72]_ can be put into the following line of code:

.. code-block:: python

   """ From module: gmode_module.py """

   from numpy import all, sqrt, ravel, sum

    """ R is the correlation matrix """
    f = lambda R: ravel( R.shape[1]/sum(R, axis=0) )

   """ G  --> Gaussian vector Estimator """
   """ Z2 --> Chi-squared vector Estimator """
   """ N is the number of elements """
   def G(N, f, Z2):
       from numpy import all, sqrt

       if all(N*f > 100e0):
          return  sqrt(2e0*z2) -  sqrt(2e0*f - 1e0)

       elif all(N*f >= 30e0) and aall(N*f <= 100e0):
            return ((z2/f)**(1e0/3) - (1e0 - (2e0/9)*f))/sqrt((2e0/9)*f)
    
       elif all(N*f < 30e0):
            return None

   # Hypothesis Testing
   """ Gq1 --> critical values related to confidence level q1 """
   def hyp_test(N, Gq1, f, index, Z2):
       if all(G(N, f, Z2) < Gq1):
          return index

- *|mgr|_{i} and |sfgr|_{i} are redefined in each iterative run*. The iteration is executed until the *Na*
  and R become unchanged over successive runs. Once the first unimodal cluster is formed, its members are removed from the sample and 
  the above procedure is applied again until all the sample is depleted, no more initial seed is found or the condition ``N > M-1``
  is not satisfied anymore. If a initial seed fails to produce a cluster, its elements are also excluded from the sample.
| 
As soon as all unimodal clusters are found and its central tendency and absolute deviation are computed, the method goes to the next stage: 
to measure the hyperdimension distance between classes and evaluate the variable relevance to the classification.

Variable Evaluation and Distance Matrix
---------------------------------------

This part of the method is also based on Z² criterion, but now the objects of evaluation are the clusters identified on the previous stage. 
The variables are tested for their power to discriminate clusters against each other. For this purpose, the elements of the ``Nc X Nc``
(*Nc*, the number of classes) symmetric matrices of G estimators are computed for each variable i as follows:

.. code-block:: python

   """ From module: eval_variables.py """

   from numpy import sum

   """ a, b --> a pair of evaluated clusters. """
   """ member[a], member[b] --> array of members of a cluster. """
   """ ct[a], ct[b]   --> Central Tendencies. """
   """ dev[a], dev[b] --> Median Absolute Deviations. """
   """ R[a], R[b]     --> Inverted Correlation Matrix. """
   
   # Degrees of freedom:
   fab = (Nb - 1e0)*(M**2)/sum(R_a)
   fba = (Na - 1e0)*(M**2)/sum(R_b)
          
   # Calculating Z²i(a,b) e Z²i(b,a) --> Normalized Eucliadean Distance. 
   Z2iab = sum( ( (member[b] - ct[a])/dev[a] )**2, axis=0 )
   Z2iba = sum( ( (member[a] - ct[b])/dev[b] )**2, axis=0 )

   # Calculating Z²(a,b) e Z2(b,a):
   Z2ab = sum( dot(iR[a], Z2iab) )
   Z2ba = sum( dot(iR[b], Z2iba) )

   # Calculating G matrix :
   for i in xrange(M):
       G[i][a][b] = sqrt(2e0*(Z2iab[i] + Z2iba[i])) - sqrt(2e0*(Na + Nb) - 1e0)

   # Absolute Distance Matrix.
   D2[a][b] = (Z2ab + Z2ba)/(fab + fba - 1e0)

The :math:`G_{i}` matrix gives the efficiency of variable i to resolve the clusters, thus the smaller are its element values, less separated are the classes. 
To discriminate the redundant variables, all the elements of :math:`G_{i}` matrix are tested against the null hypothesis :math:`|mgr|_{i,a} = |mgr|_{i,b}` , 
and if all of them does not satisfies :math:`G_{i}(a,b) < G_{q_{2}}`, the method is iterated again without the variable *i*. 
The method is repeated until stability is found on the most suitable set of meaningful variables for the sample.

The ``Nc X Nc`` symmetric Distance Matrix between clusters with respect to all meaningful variables is also calculated. 
The same interpretation given to :math:`G_{i}`  matrices can be used here: higher D²(a,b) elements, more distinct are the clusters from each other.
D²(a,b) matrix is used to produce a ``scipy.cluster.hierarchy.dendogram``, which graphically shows the relation among all clusters.

Robust Median Statistics
------------------------

Robust Statistics seeks alternative estimators which are not excessively affected by outliers or departures from an assumed sample distribution. 
For central tendency estimator : math:`|mgr|_{i}`, the median was chosen over mean due to its breakdown point of 50 % against 0% for mean. 
Higher the breakdown point, the estimator is more resistant to variations due to errors or outliers. 
Following a median-based statistics, the Median of Absolute Deviation (MAD) was selected to represent the deviation estimator |sfgr|. 
The MAD is said to be conceived by Gauss in 1816 [Ham74]_ and can be expressed in function below:

.. code-block:: python
   
    from numpy import fabs, median
    
    # X is a array and ct is the median of tested cluster.
    def mad(X, ct, K=1.4826):
        return K*median(fabs(X - ct), axis=0)
   
To be used as a estimator of standard deviation, the MAD must be multiplied by a scaling factor K, which adjusts the value for a assumed distribution. 
For Gaussian distribution, which is the distribution assumed for clusters in the G-mode, ``K = 1.426``.

To compute the mahalanobis distance is necessary to estimate the covariance matrix.
MAD is expanded to calculate its terms:

.. code-block:: python

    """ From module: gmode_module.py """

    from numpy import matrix, median

    # X is a array and ct is the median of tested cluster.
    def cov(X, ct, K=1.4826):
        X = X - ct
        return matrix( [median(X.T*X[:,i], axis=1)*K**2 for i in xrange(X.shape[1])] )

The correlation coefficient :math:`r_{s,k}` used in this G-mode version was proposed by [Shev97]_ to be a median counterpart to 
pearson correlation coefficient, with breakpoint of 50%, similar to MAD versus standard deviation. 
The coefficient is based on linear data transformation and depends on MAD and the deviation of each element from the median:        

.. code-block:: python

   from numpy import median, matrix, isnan, fabs
   from collections import deque

   # X is a array
   # ct is the median of tested cluster.
   # dev is the MAD of tested cluster.
   def Robust_R(X, ct, dev):

       X = (X - ct)/dev
       r2 = deque()
    
       for i in xrange(X.shape[1]):
           u  = median(fabs(X.T + X[:,i]), axis=1)**2
           v  = median(fabs(X.T - X[:,i]), axis=1)**2
           ri = (u - v)/(u + v)
           r2.append(ri**2)

       r2 = matrix(r2)

       if aall(isnan(r2)) : 
          r2 = matrix(eye(X.shape[1]))
       else:
          whereNaN = isnan(r2)
          r2[whereNaN] = 1e0

       return r2

The application of median statistics on G-mode is a departure from the original concept of the method. 
The goal is producing more stable classes and save processing time from unnecessary sucessive iterations.

Code Testing
------------

For testing the efficiency of the Adapted G-mode version, a bidimensional sample of 2000 points was simulated using ``numpy.random``. 
The points filled a range of 0 to 10. Three random Gaussian distributions containing 500 points each (``numpy.random.normal``), 
plus 500 complete random points (``numpy.random.rand``) composed the final sample (Figure [fig:SIMUL-Sample]a). 
These gaussians were the aim for the recognition ability of clustering method, while the random points worked as background noise.
Then, simulated sample was classified using the Original [Gav92]_ and Adapted G-mode version. 
The results are presented in :ref:`Table 1` and figures below.

.. table::
   :label:`Table 1`
   =========   =====          =========   ===   ===========   ============
   Gaussians   C. T. [*]_     S. D. [*]_      N     N-Original     N-Adapted
   =========   =====          =========   ===   ===========   ============
       1       (3,3)          (0.5,0.25)  500    471 (5.8%)    512 (2.4%)
       2       (3,8)          (0.7,0.7)   500    538 (7.6%)    461 (7.8%)
       3       (7,5)          (0.7,0.7)   500    585 (17%)     346 (30.8%)
   =========   =====          =========   ===   ===========   ============

.. [*] Central Tendency.
.. [*] Standard Deviation.

Comparing results from both versions is noticeable the differences of how each version identify clusters. 
Since the initial seed in the Original G-mode starts from just the closest points, 
there is no guarantee that initial seeds will start close or inside clusters. 
The Original version is also limited for misaligned-axis clusters, due to the use of normalized distance estimator, 
that does not have correction for covariance. This limitation turn impossible the identification of misaligned clusters without including 
random elements in, as seen in :ref:`Figure 1.2`.

.. figure:: simulated.png
   :label:`Figure 1.1`
   :scale: 50%

.. figure:: Classic_Gmode_Indentification.png
   :label:`Figure 1.2`
   :scale: 50%

.. figure:: Vectorized_Gmode_Indentification.png
   :label:`Figure 1.3`
   :scale: 50%

The Adapted version, otherwise, seeks the initial seed through densest regions, thus ensuring its start inside or close to clusters. 
Moreover, by using the mahalonobis distance as estimator, the covariance matrix is taken into account, which makes a more precise 
identification of cluster boundaries (:ref:`Figure 1.3`). Nevertheless, Adapted G-mode has tendency to undersize the number of elements on 
the misaligned clusters. For cluster number 3 in :ref:`Table 1`, a anti-correlated gaussian distribution, the undersizing reaches 30.8%. 
If the undersizing becomes too large, its possible that “lost elements” are identified as new cluster. 
Therefore, may be necessary to group clusters according to its d²(a,b) distances.

References
----------
.. [Cor76] Coradini et al. 1976
.. [Cor77] Coradini et al. 1977
.. [Gav92] Gavrishin et al. 1992
.. [Ful00] Fulchignoni et al. 2000
.. [Tos05] Tosi et al. 2005
.. [Ley10] Leyrat et al. 2010
.. [Abra72] Abramowitz and Stegun 1972
.. [Ham74] Hampel 1974
.. [Shev97] Shevlyakov 1997
