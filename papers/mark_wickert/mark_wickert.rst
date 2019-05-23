:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs

:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------------------
A Real-Time 3D Audio Simulator for Cognitive Hearing Science
------------------------------------------------------------

.. class:: abstract

   This paper describes the development of a 3D audio simulator for use in cognitive hearing science 
   studies and also for general 3D audio experimentation. The framework that the simulator is build 
   upon is :code:`pyaudio_helper`, which is a module of the package :code:`scikit-dsp-comm`. The simulator runs in 
   a Jupyter notebook and makes use of Jupyter widgets for interactive control of audio source 
   positioning in 3D space. 3D audio has application in virtual reality and in hearing assistive 
   devices (HAD) research and development. At its core the simulator uses digital filters to represent the 
   sound pressure wave propagation path from the sound source to each ear canal of a human subject. 
   Digital filters of 200 coefficients each for left and right ears are stored in a look-up table 
   as a function of azimuth and elevation angles of the impinging sounds source.


.. class:: keywords

   Head-related impulse response (HRIR), Head-related transfer function (HRTF), binaural hearing, 
   virtual reality, audiology, hearing assistive devices (HAD), 

Introduction
------------

This paper describes the development of a 3D audio simulator for use in cognitive hearing science 
studies and also for general 3D audio experimentation. The framework that the simulator is build 
upon is :code:`pyaudio_helper`, which is a module of the package :code:`scikit-dsp-comm`. The simulator runs in 
a Jupyter notebook and makes use of Jupyter widgets for interactive control of audio source 
positioning in 3D space. 3D audio has application in virtual reality and in hearing assistive 
devices (HAD) research and development.  At its core the simulator uses digital filters to represent the 
sound pressure wave propagation path from the sound source to each ear canal of a human subject. 
Digital filters of 200 coefficients each for left and right ears are stored in a look-up table (LUT) 
as a function of azimuth and elevation angles of the impinging sounds source.

In cognitive hearing science binaural hearing models how sound pressure waves arrive at either 
the ear drum, at the end of the ear canal, or in the case of typical measurements, at the entry 
to the ear canal, both as a function of the arrival angle in 3D (azimuth and elevation) and 
radial distance. This leads to the need for the head related impulse response (HRIR) 
(time-domain) or head-related transfer function (HRTF) (frequency domain) for a particular 
human subject. Traditionally human subjects are placed in an anechoic chamber with a sounder 
source placed at say one meter from the head and then the subject is moved over a range of 
azimuth and elevation angles, with the HRIR measured at each angle. The 3D simulator described 
here uses a database of HRIR's to describe a given subject [CIPICHRTF]_. In the 
:code:`pyaudio_helper` application 
the HRIR at a given angle is represented by two (left and right ear)  200 coefficient digital 
filters that the sound source audio is passed through. Here the data base for each subject  
holds 25 azimuth and 50 elevation angles to approximate continuous sound source 3D locations. 

Obtaining individual HRTFs is a challenge in itself and the subject of much research. In a related 
research project *deep learning* is being investigated as a means to fit a human subject to a large HRTF 
data base of subjects based on anthropometrics of the subject. As a simple solution, 
we can also consider using a simple spherical head model, and its corresponding HRTF, which 
make use of spherical harmonics to solve for the sound pressure at any location on the sphere 
surface. To that end this simulator is integral to planned human subject testing where headphone 
testing will be used to blind test subjects for their perception of where a sound source is located. 

3D Geometry
===========


Real-Time Signal Processing
===========================



Mapping to the CIPIC Interaural Polar Coordinates
-------------------------------------------------

The class 


3D Audio Simulator Notebook Apps
--------------------------------

Static Sound Source
===================


Dynamic Sound Source Along a Trajectory
=======================================


Spherical Head Model as a Simple Reference
------------------------------------------

Bring in references [Bolein]_, [Duda]_, and [Beranic]_ to discuss a 3D sound source simulator app which use a 
simple spherical head model in place of the subject's HRTF. The HRTF data filter coefficients can be obtained using known expressions for sound pressure wave scattering from a rigid sphere. The sphere radius can for example be set to match the mean radius of the subjects head. Ultimately we wish to evaluate human subjects with multiple HRTF's, and use the ...



Conclusions and Future Work
---------------------------

TBD


References
----------

.. [Fitzpatrick] Fitzpatrick, W., Wickert, M., and Semwal, S. (2013) 3D Sound Imaging with Head Tracking, *Proceedings IEEE 15th Digital Signal Processing Workshop/7th Signal Processing Education Workshop*.
.. [CIPIC] *The CIPIC Interface Laboratory Home Page*, (2019, May 22). Retrieved May 22, 2019, from `https://www.ece.ucdavis.edu/cipic`_.
.. [CIPICHRTF] *The CIPIC HRTF Database*, (2019, May 22). Retrieved May 22, 2019, from `https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data`_.
.. [Wickert] *Real-Time Digital Signal Processing Using pyaudio_helper and the ipywidgets*, (2018, July 15). Retrieved May 22, 2019, from DOI `10.25080/Majora-4af1f417-00e`_.
.. [Beranek] Beranek, L. and Mellow, T (2012). *Acoustics: Sound Fields and Transducers*. London: Elsevier.
.. [Duda] Duda, R. and Martens, W. (1998). Range dependence of the response of a spherical head model, *J. Acoust. Soc. Am. 104 (5)*.
.. [Boelein]  Bogelein, S., Brinkmann, F.,  Ackermann, D., and Weinzierl, S. (2018). Localization Cues of a Spherical Head Model. *DAGA Conference 2018 Munich*. 

.. _`https://www.ece.ucdavis.edu/cipic`: https://www.ece.ucdavis.edu/cipic
.. _`https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data`: https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data
.. _`https://github.com/mwickert/scikit-dsp-comm`: https://github.com/mwickert/scikit-dsp-comm
.. _`10.25080/Majora-4af1f417-00e`: http://conference.scipy.org/proceedings/scipy2018/mark_wickert_250.html
