:author: Gustavo Braganca
:email: ga.braganca@gmail.com
:institution: Observatorio Nacional, Brazil

:author: Simone Daflon
:email: daflon@on.br
:institution: Observatorio Nacional, Brazil

:author: Katia Cunha
:email: cunha@email.noao.edu
:institution: Observatorio Nacional, Brazil; National Optical Astronomy Observatory, University of Arizona, U. S. A.

:author: Thomas Bensby
:email: tbensby@astro.lu.se
:institution: Lund Observatory, Sweden

:author: Sally Oey
:email: msoey@umich.edu
:institution: University of Michigan, U. S. A.

:author: Gregory Walth
:email: gwalth@email.arizona.edu
:institution: Steward Observatory, U. S. A.

--------------------------------------------------------------------
Using Python to Study Rotational Velocity Distributions of Hot Stars
--------------------------------------------------------------------

.. class:: abstract

   Stars are a fundamental pieces that compose our Universe. By studying them we can better comprehend the enviroment in which we live. On this work, we have studied a sample of almost 400 nearby OB stars and have  characterized them in aspects of their temperature and projected rotational velocity.
   
   Python is a robust language with a steep learning curve, i.e. one can make rapid progress with it. In this proceeding, we will be discussing our  progress in learning Python at the same time in which the research were being made.

.. class:: keywords

   Astronomy, Stars, Galactic Disk
   
Introduction
------------

The study of O and B stars are an important key to understand how star formation occurs. When these stars born, they have the greatest mass, temperature and rotation. Their mass can achieve **.....**, their temperatures, **....**, and rotation up to 400 km/s. 

By definition, a star is born when it start synthetizing Hydrogen into Helium through nuclear fusion. The star perform this nucleosynthesis during somewhat 90% of their life. When stars are at this stage, they are called dwarfs. Most of the studied stars of this work are dwarfs. Due to their young age, dwarf stars do not have lost too much of their mass, and so, the most of their stellar properties are kept unchanged. This help us understand how this stars formed.

Stars are born inside molecular clouds, and, usually, a molecular cloud can generate several stars. After their formation, these stars compose a stellar association, that, in its infancy, is still gravitationally bounded. With their unchanged properties, it is possible to trace the membership of these stars and then verify if some stars are from the same association.

The Python programming language is very well suited to scientific studies. The scipy, numpy and matplotlib are the basic packages to start doing scientific research using python. Also, on the last years, it has been widely adopted on the Astronomic community. Because this, several packages are being translated to python or just being created. The existence of these packages are one of the reasons that attracted us to use python on our research. Its easy learning and its script nature are other reasons as well. The script nature allows the researcher to have a dynamic workflow and not to loose too much time with debugging and compiling.

On this proceedings, we relate how we used python on our research. A more profound scientific analysis can be found at [Brag12]_.

Research development
--------------------

Initial stages
~~~~~~~~~~~~~~

As we have said before, stars usually are born in groups. Because of that, a great majority of them are binaries or belongs to multiple systems. For a spectroscopic study, as was this, the only problem occurs when the spectrum of one observation have two or more objects. Since the study of these stars were outside the scope of our project, we selected those stars on our sample to further discard them. But before discarding them, we used Python to visualize our sample and the distribution of these objects. We used the matplotlib package to do a polar plot of our objects in Galactic coordinates:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Distance projected on the Galactic plane
   proj_dist = distance_vector * np.cos(latitude_vector)
   
   plt.polar(longitude_vector, proj_dist, 'k.')
   for i in binary_list:
       for j in range(len(coordinate_list)):
           #Compare stellar IDs
           if i == coordinate_list[j, 0]:     
               plt.plot(longitude_vector[j], 
                        proj_dist[j], 
                        'wo', ms=3, mec='r')  
                        
               
And the resulting plot is showed in Figure :ref:`coords`.                 
                        
.. figure:: f1.png

   Polar plot showing the positions of the sample stars projected onto the Galactic plane. 
   The plot is centered on the Sun. The open red circles
   are spectroscopic binaries/multiple systems identified in our sample. :label:`coords`
   
   
To analyse our observation spectra images we have used `IRAF <http://iraf.noao.edu/>`__ (Image and Reduction Analysis Facility), which is a suite of softwares to handle astronomic images developed by the NOAO [1]_. 
We had to do several tasks on our spectra (e.g. cut it in a certain wavelength and normalization) to prepare our sample to further analysis. Some of these tsaks had to be done manully on a one-by-on basis, but some other were automated. The automation ould have bnn done using the IRAF scripting, but fortunately, 
the STSCI [2]_ has developed a python wrapper for IRAF called `PyRAF <http://www.stsci.edu/institute/software_hardware/pyraf>`__.
For example, we show how we used IRAF task SCOPY to cut images from a list using pyRAF:

.. [1] National Optical Astronomy Observatory
.. [2] Space Telescope Science Institute

.. code-block:: python

   from pyraf import iraf
   
   iraf.noao.onedspec.scopy.w1 = 4050  # Starting wavelength
   iraf.noao.onedspec.scopy.w2 = 4090  # Ending wavelength
   
   for name in list_of_stars:
       # Spectrum to be cut
       iraf.noao.onedspec.scopy.input = name
       # Nanme of resulting spectrum
       result = name.split('.fits')[0] + '_cut.fits'
       iraf.noao.onedspec.scopy.output = result
       # Execute
       iraf.noao.onedspec.scopy(mode = 'h')



We also have performed a spectral classification on the stars and, since this was not done using Python, more information can be obtained on the original paper. 

Effective temperature through photometric calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have obtained effective temperature (Teff) from a calibration presentend in [Mass89]_ that uses the photometric reddening-free parameter index  :math:`Q` ([John58]_). 

A histogram showing the distribution of effective temperatures for OB stars with available photometry is shown in Figure :ref:`TqHist`.
The effective temperatures of the target sample peak around 17,000 K, with most stars being cooler than 28,000 K.
                        
.. figure:: f6.png

   Histogram showing the distribution of effective temperatures for the studied sample. :label:`TqHist`
    
Projected rotational velocities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have obtained projected rotational velocities (:math:`v\sin i`) for 266 stars of our sample. 
We do not used Python to obtain :math:`v\sin i`, so, for more information, we suggest the reader to look in the original paper. 
However, for the analysis we used Python, specially the matplotlib package for visualization analysis and the Scipy.stats package to statistics analysis.

The boxplot is a great plot to compare several distributions side by side. 
On this work, we used a boxplot to analyze the :math:`v\sin i` for each spectral type subset, as can be seen on Figure :ref:`boxplot`. 

.. figure:: f7.png

   Box plot for the studied stars in terms of the spectral type. 
   The average :math:`v\sin i` for the stars in each spectral type bin 
   is roughly constant, even considering the least populated bins. 
   :label:`boxplot`
   
The code used to plot it was:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   #Start boxplot
   bp = plt.boxplot(box, notch=0)
   # Define color of medians
   plt.setp(bp['medians'], color='red')
   # Add small box on the mean values
   plt.scatter(range(1,9), mean_vector, 
               c='w', marker='s', edgecolor='r')
   # Set labl for the axis
   plt.xlabel(u'Spectral Type')
   plt.ylabel(r'$v\sin i$ (km s$^{-1}$)')
   # Set limit for the axis
   plt.axis([0, 9, 0, 420])
   # Set spectral types on the x-axis 
   plt.xticks(range(1,9), ['O9', 'B0', 'B1', 
              'B2', 'B3', 'B4', 'B5', 'B6'])
   # Put a text with the number of objects on each bin
   [plt.text(i+1, 395, WSint(length[i]), fontsize=12,
    horizontalalignment='center') for i in range(0,8)]
   # Save figure
   plt.savefig('boxplot.eps')   
      

Results
~~~~~~~

Conclusions
-----------

blahblah

References
----------

.. [Brag12] Braganca, G. A, et al., Astronomical Journal, 144:130, November 2012. 
.. [John58] Johnson, H. L., Lowell Obs. Bull., 4:37, 1958
.. [Mass89] Massey, P., Silkey, M., Garmany, C. D., Degioia-Eastwood, K., Aastronomical Journal, 97, 107, 1989,
