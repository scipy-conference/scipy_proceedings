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
which was previously written in FORTRAN 77, in PYTHON 2.7 and using Numpy, Scipy and Matplotlib packages. 
The Numpy was used for array and matrix manipulation and Matplotlib for plot control. 
The Scipy had a import role in speeding up G-mode, Scipy.cKD-Tree and Numpy.histogramdd were applied to find the initial seeds 
from which the clusters are going to evolve. Scipy was also used to quickly produce dendograms showing the distances between the clusters.

Finally, results for Asteroids Taxonomy and tests for different sample sizes and implementations are going to be presented.

.. class:: keywords

   clustering, taxonomy, asteroids, multivariate data 

Introduction
------------
The classes were defined using the G-mode multivariate clustering method, designed by A. I. Gavrishin (Coradini et al., 1976) and 
previously implemented to FORTRAN V by Coradini et al. (1977) to classify geochemical samples, but applicable to a wide range of research fields, 
as planetary sciences [REF], disk-resolved remote sensing [REF] and cosmology [REF]. 
The G-mode classifies *N* elements into *Nc* unimodal clusters containing *Na* elements each. Elements are described by M variables. 

This method is unsupervised, which allows an automatic identification of clusters without any a priori knowledge of the sample distribution. 
For that, user must control only two critical parameters for the classification, the confidence levels *q1* and *q2*, that may be equated, 
for simplification. The smaller these parameters get, more clusters are resolved and lower their variances are.

The G-mode procedure used here follows a adapted version of the method published by Gavrishin et al. (1992), briefly described by Fulchignoni et al. 
(2000) and reviewed by Tosi et al. (2005) and Leyrat et al. (2010). 
Robust estimators, a faster initial seed finder and statistical whitenning were introduced to produce a more robust set of clusters and 
optimize the processing time. The coding was performed in Python 2.7 with help of Matplotlib, Numpy and Scipy packages. 
The method procedure can be briefly summarized in two parts: the first one is the cluster recognition and 
the second evaluates each variable in the classification process. Each are described in the following subsections. 
In the last subsection, the central tendency and absolute deviation estimators, based on robust statistics, are then presented.

 
Recognition of the unimodal clusters
--------------

The first procedure can be summarized on the following topics:

- *The data is arranged according to the code*:

.. code-block:: python

   def LoadData(self,filename):

       from operator import getitem, itemgetter
       from numpy import genfromtxt

       data = map(list,genfromtxt(filename, dtype=None))

       self.design   = map(itemgetter(0),data)
       self.uniq_id  = map(itemgetter(1),data)

       self.elems  = [array(item[2::2], dtype=float64) for item in data]
       self.errs   = [array(item[3::2], dtype=float64) for item in data]

       self.indexs = range(len(self.design))

  All variables are whiten (:math: scipy.cluster.vq.whiten), which means they are divided by their absolute deviation to scale all them up. 
  This is a important measure when dealing with percentage variables, such as geometric albedos.

- *Initial seed of a forming cluster is identified*. 
  At the original implementation, the G-mode relied on a force-brute algorithm to find the three closest elements as initial seed, 
  which required long processing time. Therefore, in our version, the initial seeds are searched recursively using `numpy.histogramdd`, which
  produced a faster result:

.. code-block:: python

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
   
       from numpy import histogramdd, array
   
       rng   = range(data.shape[1])

       nbin = map(int,array([grid]*data.shape[1]))

       hist, edges = histogramdd(data,bins=nbin,range=tuple(zip(lower, upper)))

       limits = array([list(pairwise(edges[i])) for i in rng])

       ind = unravel_index(argmax(hist), hist.shape) 

       zone = array([limits[i,j] for i, j in izip(rng, ind)])

       density = amax(hist) / volume(zone)
    
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
  In the end, starting central tendency and absolute deviation are estimated from the initial seed. 
  If any absolute deviation is zeroth, the value is replaced by the median error of the variable.                 

- *Z² criterion*. In the next step, all elements are replaced to a single variable given by the equation:

- *Hypothesis Testing*. The Z² estimator follows a \chi^{2} distribution, but for sake of simplification, Z^{2} can be transformed to gaussian 
  estimator G for large degree of fredom. Now, the critical value G_{q_{1}} in hypothesis testing are given as multiples of \sigma, 
  which simplify its interpretation.

- *|mgr| and |sfgr| are redefined in each iterative run*. The iteration is executed until the N_{a}
  and R become unchanged over successive runs. Once the first unimodal cluster is formed, its members are removed from the sample and 
  the above procedure is applied again until all the sample is depleted, no more initial seed is found or the condition N>M-1
  is not satisfied anymore. If a initial seed fails to produce a cluster, its elements are also excluded from the sample.

As soon as all unimodal clusters are found and its central tendency and absolute deviation are computed, the method goes to the next stage: 
to measure the hyperdimension distance between classes and evaluate the variable relevance to the classification.

Perhaps we want to end off with a quote by Lao Tse:

  *Muddy water, let stand, becomes clear.*


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
.. [Atr03] 


