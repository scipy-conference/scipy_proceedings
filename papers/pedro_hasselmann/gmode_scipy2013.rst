:author: Pedro Henrique Hasselmann
:email: hasselmann@on.br
:institution: Observatorio Nacional

:author: Jorge Márcio Carvano
:email: carvano@on.br
:institution: Observatorio Nacional

:author: Daniela Lazzaro
:email: lazzaro@on.br
:institution: Observatorio Nacional

-------------------------------------------------------------
Adapted G-mode Clustering Method applied to Asteroid Taxonomy
-------------------------------------------------------------

.. class:: abstract

   The original G-mode was a clustering method developed by A. I. Gavrishin in the 60's for geochemical classification of rocks, 
   but was also applied to asteroid photometry, cosmic rays, lunar sample and planetary science spectroscopy data. 
   In this work, we used a adapted version to classify the asteroid photometry from SDSS Moving Objects Catalog. 
   The method works identifying normal distributions in a multidimensional space of variables. 
   The identification starts with finding the closest points in the sample, which is a consumable problem when the data is not planar. 
   Therefore, to achieve satisfying results in a human bearable time, we implemented the method, 
   which was previously written in FORTRAN 77, in Python 2.7 and using NumPy, SciPy and Matplotlib packages. 
   The NumPy was used for array and matrix manipulation and Matplotlib for plot control. 
   The Scipy had a import role in speeding up G-mode, Scipy.cKD-Tree and Numpy.histogramdd were applied to find the initial seeds 
   from which clusters are going to evolve. Scipy was also used to quickly produce dendrograms showing the distances among clusters.

   Finally, results for Asteroids Taxonomy and tests for different sample sizes and implementations are going to be presented.

.. class:: keywords

   clustering, taxonomy, asteroids, multivariate data, scipy, numpy

Introduction
------------

The clusters are identified using the G-mode multivariate clustering method, designed by A. I. Gavrishin and Coradini [Cor76]_, 
originally written in FORTRAN V by Cor77_ to classify geochemical samples, but applicable to a wide range of research fields, 
as planetary sciences [REF], disk-resolved remote sensing [REF] and cosmology [REF]. 
In 1987, Bar87_ used Original G-mode to classify asteroids intersected by the Eight-Color Asteroid Survey catalog [Zel85]_ and 
IRAS geometric albedos [Mat86]_. Helding a sample of 442 asteroids with 8 variables, they recognized 18 valid classes using a confidence level
of 97.7 %. Thoses classes were grouped to represent the asteroid taxonomic types. G-mode also identified that just 3 variables
were enough to characterize the asteroid taxonomy.

The G-mode classifies *N* elements into *Nc* unimodal clusters containing *Na* elements each. Elements are described by *M* variables. 
This method is unsupervised, which allows an automatic identification of clusters without any *a priori* knowledge of sample distribution. 
For that, user must control only one critical parameter for the classification, the confidence levels *q1*. 
Smaller this parameter get, more clusters are resolved and lower their spreads are.

The G-mode used here follows a adapted version of the original method published by Gav92_ , briefly described by Ful00_ and reviewed by Tos05_ and Ley10_  . 
median central tendency and absolute deviation estimators, a faster initial seed finder and statistical whitenning were introduced to produce a more 
robust set of clusters and optimize the processing time. The coding was performed using Python 2.7 with help of Matplotlib, NumPy and SciPy packages [*]_. 
The method procedure can be briefly summarized in two parts: the first one is the cluster recognition and 
the second evaluates each variable in the classification process. Each one are going to be described in the following sections. 

.. [*] The codebase_ is hosted through GitHub_ .

.. _codebase: http://pedrohasselmann.github.com/GmodeClass
.. _GitHub: http://github.com
 
Recognition Of The Unimodal Clusters
------------------------------------

The first procedure can be summarized with following topics and code snippets:

- *The data is arranged in N X M matrix*. All variables are ``scipy.cluster.vq.whiten`` , 
  which means they are divided by their absolute deviation to scale all them up. 
  This is a important measure when dealing with percentage variables, such as geometric albedos.

- *Initial seed of a forming cluster is identified*. 
  At the original implementation, the G-mode relied on a force-brute algorithm to find the three closest elements as initial seed, 
  which required long processing time. Therefore, in our version, the initial seeds are searched recursively using ``numpy.histogramdd`` , which
  speeds up the output:

.. code-block:: python

   ###### barycenter.py ######

   def boolist(index, values, lim):
       if all([boo(item[0],item[1]) \
          for item in izip(values,lim)]):
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
    
   def barycenter_density(data, grid, upper, \
       lower, dens, nmin):
   
       from numpy import histogramdd, array, \
       unravel_index, amax
   
       rng   = range(data.shape[1])
       
       nbin = map(int,array([grid]*data.shape[1]))
       
       hist, edges = histogramdd( \
       data,bins=nbin,range=tuple(zip(lower, upper)))
       
       limits = array( \ 
       [list(pairwise(edges[i])) for i in rng])
       
       ind = unravel_index(argmax(hist), hist.shape) 

       zone = array([limits[i,j] \
              for i, j in izip(rng, ind)])
       
       density = amax(hist) / volume(zone)
       
       if density > dens and amax(hist) > nmin:
          zone = zone.T
          return barycenter_density(data, grid, \
                 zone[1], zone[0], density, nmin)
       else:
          return filter(lambda x: x != None, \
                 imap(lambda i, y: boolist(i,y,zone), \
                 xrange(data.shape[0]), data))

The function above divides the variable hyperspace into large sectors, and just in the most crowded sector the initial seed is searched for. 
Recursively, the most crowded sector is once divided as long as the density grows up. 
When density decreases or the minimal number of points set by the user is reached, the procedure stops. 
The initial seed is chosen from the elements of the most crowded sector before ending the procedure. 
In the end, starting central tendency :math:`\mu_{i}` and standard deviation :math:`\sigma_{i}` are estimated from the initial seed. 
If any standard deviation is zeroth, the value is replaced by the median error of the variable.                 

- *Z² criterion*. In the next step, the mahalanobis distance (``scipy.spatial.distance.mahalanobis``) between 
  the tested cluster and all elements are computed.

- *Hypothesis Testing*. The Z² estimator follows a :math:`\chi^{2}` distribution, but for sake of simplification, 
  Z² can be transformed to gaussian estimator ``G`` if the degree of freedom :math:`\vec{f}` is larger enough, which is satisfied for most of samples. 
  Now, the critical value :math:`G_{q1}` in hypothesis testing are given as multiples of :math:`\sigma` , simplifying its interpretation. 
  Therefore, the vectorized transformation [Abr72]_ can be written:

.. math:: 

   \vec{G_{j}}=\sqrt{2\cdot\vec{Z^{2}}}-\sqrt{2\cdot\vec{f}-1}

.. math::

   f_{k}=N\cdot\frac{M}{\sum_{s=1}^{M}r_{ks}^{2}}
 
for :math:`\vec{f} > 100` , where :math:`r_{ks}^{2}` is the correlation coefficient. For :math:`30 < \vec{f} < 100` , the ``G`` parameter becomes: 

.. math::

   \vec{G_{j}}=\frac{\left(\frac{Z^{2}}{\vec{f}}\right)^{1/3}-(1-\frac{2}{9}\cdot\vec{f})}{\sqrt{\frac{2}{9}\cdot\vec{f}}}
 
Then the null hypothesis :math:`\chi_{ij} = \mu_{i}` is tested with a statistical significance of :math:`P(G_{j} \leq G_{q_{1},f})` for a :math:`\chi_{j}`
element to belong to a tested class, i.e., a class contains the :math:`\chi_{j}` sample if its estimator :math:`G_{j}` satisfies :math:`G_{j} \leq G_{q_{1}}` .

- :math:`\mu_{i}` *and* :math:`\sigma_{i}` *are redefined in each iterative run*. The iteration is executed until the *Na*
  and *R* become unchanged over successive runs. Once the first unimodal cluster is formed, its members are removed from the sample and 
  the above procedure is applied again until all the sample is depleted, no more initial seed is found or the condition ``N > M-1``
  is not satisfied anymore. If a initial seed fails to produce a cluster, its elements are also excluded from the sample.


As soon as all unimodal clusters are found and its central tendency and absolute deviation are computed, the method goes to the next stage: 
to measure the hyperdimension distance between classes and evaluate the variable relevance to the classification.

Variable Evaluation and Distance Matrix
---------------------------------------
 
This part of the method is also based on Z² criterion, but now the objects of evaluation are the clusters identified on the previous stage. 
The variables are tested for their power to discriminate clusters against each other. For this purpose, the elements of the :math:`Nc \times Nc`
(*Nc*, the number of classes) symmetric matrices of G estimators are computed for each variable i as follows:

.. math::

   G_{i}(a,b)=\sqrt{2\left[Z_{i}^{2}(a,b)+Z_{i}^{2}(b,a)\right]}-\sqrt{2\left(N_{a}+N_{b}\right)-1}
 
where *Na* and *Nb* are respectively the number of members in the a-th and b-th class, while :math:`Z_{i}^{2}(a,b)` and :math:`Z_{i}^{2}(b,a)` 
are a reformulation of Z² estimator, now given by:

.. math::

   Z_{i}^{2}(a,b)=\sum_{j=1}^{N_{b}}Z_{ijb}^{2}=\sum_{j=1}^{N_{b}}\frac{\left(\chi_{ijb}-\mu_{i,a}\right)^{2}}{\sigma_{i,a}^{2}}
 
:math:`Z_{i}^{2}(b,a)` can be found just by  permuting the equation indices.

The :math:`G_{i}` matrix gives the efficiency of variable i to resolve the clusters, thus the smaller are its element values, less separated are the classes. 
To discriminate the redundant variables, all the elements of :math:`G_{i}` matrix are tested against the null hypothesis :math:`\mu_{i,a} = \mu_{i,b}` , 
and if all of them does not satisfies :math:`G_{i}(a,b) < G_{q_{2}}`, the method is iterated again without the variable *i*. 
The method is repeated until stability is found on the most suitable set of meaningful variables for the sample.

The :math:`Nc \times Nc` symmetric Distance Matrix between clusters with respect to all meaningful variables is also calculated. 
The same interpretation given to :math:`G_{i}`  matrices can be used here: higher D²(a,b) elements, more distinct are the clusters from each other.
D²(a,b) matrix is used to produce a ``scipy.cluster.hierarchy.dendrogram`` , which graphically shows the relation among all clusters.

Robust Median Statistics
------------------------

Robust Statistics seeks alternative estimators which are not excessively affected by outliers or departures from an assumed sample distribution. 
For central tendency estimator : math:`\mu_{i}`, the median was chosen over mean due to its breakdown point of 50 % against 0% for mean. 
Higher the breakdown point, the estimator is more resistant to variations due to errors or outliers. 
Following a median-based statistics, the Median of Absolute Deviation (MAD) was selected to represent the standard deviation estimator :math:`\sigma`. 
The MAD is said to be conceived by Gauss in 1816 [Ham74]_ and can be expressed as:

.. math::
 
   MAD(\chi_{i})=med\left\{ |\chi_{ji}-med\left(\chi_{i}\right)|\right\} 
 
To be used as a estimator of standard deviation, the MAD must be multiplied by a scaling factor K, which adjusts the value for a assumed distribution. 
For Gaussian distribution, which is the distribution assumed for clusters in the G-mode, ``K = 1.426`` . Therefore:

.. math::

   \sigma_{i}=K\cdot MAD
 
To compute the mahalanobis distance is necessary to estimate the covariance matrix.
MAD is expanded to calculate its terms:

.. math::

   S_{ik}=K^{2}\cdot med\left\{ |\left(\chi_{ji}-med\left(\chi_{i}\right)\right)\cdot\left(\chi_{jk}-med\left(\chi_{k}\right)\right)|\right\} 
 
The correlation coefficient :math:`r_{s,k}` used in this G-mode version was proposed by She97_ to be a median counterpart to 
pearson correlation coefficient, with breakpoint of 50%, similar to MAD versus standard deviation. 
The coefficient is based on linear data transformation and depends on MAD and the deviation of each element from the median:        

.. math::

   r_{i,k}=\frac{med^{2}|u|-med^{2}|v|}{med^{2}|u|+med^{2}|v|}

where

.. math::

   u=\frac{\chi_{ij}-med\left(\chi_{s}\right)}{\sigma_{i}}+\frac{\chi_{kj}-med\left(\chi_{k}\right)}{\sigma_{k}}

.. math::

   v=\frac{\chi_{ij}-med\left(\chi_{m}\right)}{\sigma_{i}}-\frac{\chi_{kj}-med\left(\chi_{n}\right)}{\sigma_{k}}
 
The application of median statistics on G-mode is a departure from the original concept of the method. 
The goal is producing more stable classes and save processing time from unnecessary sucessive iterations.

Code Structure, Input And Output
--------------------------------

The ``GmodeClass`` package, hosted in GitHub_ ,  is organized in a object-oriented structure. The code snippets
below show how main class and its objects are implemented, explaining what each one does, 
and also highlighting its dependences:

.. code-block:: python

   ################# Gmode.py #################

   # modules: kernel.py, eval_variables.py, 
   # plot_module.py, file_module.py, gmode_module.py
   
   def main():
       # dependencies: optparse
       # Import shell commands
   
   class Gmode:
         
         def __init__(self):
         # Make directory where tests are hosted.
         
         def Load(self):     
         # Make directory in /TESTS/ where test's plots, 
         # lists and logs are kept.
         # This object is run when 
         # __init__() or Run() is called. 
         
         def LoadData(self, file):
         # dependencies: operator
         # Load data to be classified.
         
         def Run(self, q1, sector, ulim, minlim):
         # dependencies: kernel.py
         # Actually run the recognition procedure.
         
         def Evaluate(self, q2):
         # dependencies: eval_variables.py
         # Evaluate the significance of each variable and
         # produce the distance matrices.
         
         def Extension(self, q1):
         # dependencies: itertools
         # Classify data elements excluded 
         # from the main classification. 
         # Optional feature. 
         
         def Classification(self):
         # Write Classification into a list.
         
         def ClassificationPerID(self):
         # dependencies: gmode_module.py
         # If the data elements are 
         # measurements of group of objects, 
         # organize the classification into 
         # a list per Unique Identification.
         
         def WriteLog(self):
         # dependencies: file_module.py
         # Write the procedure log with informations about 
         # each cluster recognition,
         # variable evaluation and distance matrices.
         
         def Plot(self, lim, norm, axis):
         # dependencies: plot_module.py
         # Save spectral plots for each cluster.
         
         def Dendrogram(self):
         # dependencies: plot_module.py
         # Save scipy.cluster.hierarchy.dendrogram figure.
         
         def TimeIt(self):
         # dependencies time.time
         # Time, in minutes, the whole procedure 
         # and save into the log.

   if __name__ == '__main__':
  
      gmode  = Gmode()
      load   = gmode.LoadData()
      run    = gmode.Run()
      ev     = gmode.Evaluate()
      ex     = gmode.Extension()   # Optional.
      col    = gmode.ClassificationPerID()
      end    = gmode.TimeIt()
      classf = gmode.Classification()
      log    = gmode.WriteLog()
      plot   = gmode.Plot()
      dendro = gmode.Dendrogram()

Originally, G-mode relied on a single parameter, the confidence level *q1*, to resolve cluster from a sample. 
However, tests on simulated sample and asteroid catalogues (More in next sections), plus changes on initial seed finder, 
revealed that three more parameters were necessary for high quality classification.
Thus, the last code version ended up with the following input parameters:

- :math:`q_{1}` or :math:`G_{q_{1}}` ( ``--q1``, ``self.q1``) : Confidence level or critical value. Must be inserted in multiple of :math:`\sigma` .
  Usually it assumes values between 1.5 and 3.0 .

- ``Grid`` (``--grid``, ``-g``, ``self.grid``) : Number of times which ``barycenter.barycenter_density()`` will divide each variable up on each iteration,
  according to the borders of the sample. Values between 2 and 4 are preferable.

- ``Minimum Deviation Limit`` (``--mlim``, ``-m``, ``self.mlim``) : Sometimes the initial seeds starts with zeroth deviation, thus this singularity is corrected
  replacing all deviation lower than minimum limit by this own value. This number is given in percent of median error of each variable.
  
- ``Upper Deviation Limit`` (``--ulim``, ``-u``, ``self.ulim``) : This parameter is important when the clusters have high degree of superposition. 
  The upper limit is a barrier which determines how much a cluster can grow up. 
  This value is given in percent of total standard deviation of each variable.

The output is contained in a directory created in ``/TESTS/`` and organized in a series of lists and plots. 
On the directory ``/TESTS/.../maps/`` , there are on-the-fly density distribution plots showing the *locus* of each cluster in sample.
On ``/TESTS/.../plots/`` , a series of variable plots permits the user to verify each cluster profile.
On lists ``clump_xxx.dat`` , ``gmode1_xxx.dat`` , ``gmode2_xxx.dat`` and ``log_xxx.dat`` the informations about the cluster statistics, 
classification of each data element, classification per unique ID and report on the formation of clusters and distance matrices are gathered.

Users must be aware that input data should be formatted on columns in this order: measurement designation, unique identificator, variables, errors.
If errors are not available, its values should be replaced by ``0.0`` and ``mlim`` parameter might not be used. There is no limit on data size, however
the processing time is very sensitive to the number of identified cluster, which may slow down the method larger its number.
For 20,000 elements and 41 clusters, the G-mode takes around to 2 minutes for whole procedure (plots creation not included).

Our implementation also allows to ``import Gmode`` and use it in ``Python IDLE`` or through shell command, like the example::

   python Gmode.py --in path/to/file \
   --q1 2.0 -g 3 -u 0.5 -m 0.5

Finally, since the plot limits, normalization and axis are optimized to asteroid photometry, 
users using the method on shell are invited to directly change this parameters in ``Gmode.Plot()``. 
A easier way to control the method aesthetics is going to be put to work on future versions.


Code Testing
------------

.. table:: Gaussian Distributions in Simulated Sample. :label:`tabgauss`

   +-----------+-----------+------------+-----+------------+------------+
   | Gaussians | C.T. [*]_ |  S.D. [*]_ |  N  | N-Original | N-Adapted  |
   +-----------+-----------+------------+-----+------------+------------+
   |     1     |    (3,3)  | (0.5,0.25) | 500 | 471 (5.8%) | 512 (2.4%) |
   +-----------+-----------+------------+-----+------------+------------+
   |     2     |    (3,8)  | (0.7,0.7)  | 500 | 538 (7.6%) | 461 (7.8%) |
   +-----------+-----------+------------+-----+------------+------------+
   |     3     |    (7,5)  | (0.7,0.7)  | 500 | 585 (17%)  | 346 (30.8%)|
   +-----------+-----------+------------+-----+------------+------------+

.. [*] Central Tendency.
.. [*] Standard Deviation.


.. figure:: simulated.png
   :scale: 40%
   
   Simulated Sample of 2000 points. 
   Blue dots represent the bidimensional elements and the clusters are three gaussian distributions composed of random points. :label:`figsimul`

.. figure:: Classic_Gmode_Identification.png 
   :scale: 50%
   
   Red filled circles are the elements of clusters identified by Original G-mode. The green filled circles represent the initial seed. :label:`figorig`

.. figure:: Vectorized_Gmode_Identification.png
   :scale: 50%
 
   Clusters identified by Adapted G-mode. Labels are the same as previous graphics. :label:`figadapted`

   
For testing the efficiency of the Adapted G-mode version, a bidimensional sample of 2000 points was simulated using ``numpy.random``. 
The points filled a range of 0 to 10. Three random Gaussian distributions containing 500 points each ( ``numpy.random.normal`` ), 
plus 500 random points ( ``numpy.random.rand`` ) composed the final sample (Figure :ref:`figsimul` ). 
These gaussians were the aim for the recognition ability of clustering method, while the random points worked as background noise.
Then, simulated sample was classified using the Original [Gav92]_ and Adapted G-mode version. 
The results are presented in Table :ref:`tabgauss` and figures below.

Comparing results from both versions is noticeable the differences of how each version identify clusters. 
Since the initial seed in the Original G-mode starts from just the closest points, 
there is no guarantee that initial seeds will start close or inside clusters. 
The Original version is also limited for misaligned-axis clusters, due to the use of normalized euclidean distance estimator, 
that does not have correction for covariance. This limitation turn impossible the identification of misaligned clusters without including 
random elements in, as seen in Figure :ref:`figorig` .

The Adapted version, otherwise, seeks the initial seed through densest regions, thus ensuring its start inside or close to clusters. 
Moreover, by using the mahalonobis distance as estimator, the covariance matrix is taken into account, which makes a more precise 
identification of cluster boundaries (Figure :ref:`figadapted` ). Nevertheless, Adapted G-mode has tendency to undersize the number of elements on 
the misaligned clusters. For cluster number 3 in Table :ref:`tabgauss` , a anti-correlated gaussian distribution, the undersizing reaches 30.8%. 
If the undersizing becomes too large, its possible that “lost elements” are identified as new cluster. 
Therefore, may be necessary to group clusters according to its d²(a,b) distances.

Adapted G-mode Applied to Sloan Digital Sky Survey Moving Objects Catalog 4
---------------------------------------------------------------------------

jhjgjhgjfytf.

Preliminary Results on Asteroid Photometric Classification
----------------------------------------------------------

jhfhgfhgdtrdt.

Conclusions
-----------

khgjhfhgcgfd.

References
----------
.. [Abr72] Abramowitz and Stegun. 1972.
.. [Ham74] Hampel. 1974.
.. [Cor76] Coradini et al. 1976.
.. [Cor77] Coradini et al. 1977.
.. [Zel85] Zellner et al. 1985.
.. [Mat86] Matson et al. 1986.
.. [Bar87] Barucci et al. 1987
.. [Gav92] Gavrishin et al. 1992.
.. [She97] Shevlyakov. 1997.
.. [Ful00] Fulchignoni et al. 2000.
.. [Tos05] Tosi et al. 2005.
.. [Ley10] Leyrat et al. 2010.

