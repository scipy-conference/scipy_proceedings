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
research project *deep learning* is being investigated as a means to fit a human subject to the CIPIC HRTF 
data base of subjects, based on 27 upper torso anthropometrics (measurements) of the subject. As a simple solution, 
we can also consider using a simple spherical head model, and its corresponding HRTF, which 
make use of spherical harmonics to solve for the sound pressure magnitude and phase at any location on the sphere 
surface, and then the inverse Fourier transform to ultimately obtain the HRIR. To that end this simulator is integral to planned human subject testing where headphone 
testing will be used to blind test subjects for their perception of where a sound source is located. 

3D Geometry
===========

To produce a synthesized 3D audio sound field starts with a geometry. For a given source location 
:math:`(x,y,z) = (x_1,-x_2,x_3)`, we transform to the cylindrical coordinates shown in Figure :ref:`CYLIND`. Later we discuss 
how we transform between the cylindrical coordinates and the CIPIC interaural-polar coordinate system (IPCS) 
in order to use the HRIR filter sets in the simulator.

.. figure:: 3D_Coordinates.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The cylindrical coordinate system used in the 3D audio simulator compared with the CIPIC IPCS. :label:`CYLIND`

The 3D audio rendering provided by the simulator developed in this paper, relies on the 1250 
HRIR measurements were taken using the geometrical configuration shown in Figure :ref:`CIPICLOC`. 
A total of 45 subjects are contained in the CIPIC HRIR database.

.. figure:: CIPIC_Source_Locations.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The CIPIC audio source locations, effectively on a 1 m radius sphere, used to obtain 1250 HRIR measurements 
   for each of 45 subjects (only the right hemisphere locations shown). :label:`CIPICLOC`

For subject 165 in particular, the left-right channel HRIR is shown in Figure :ref:`HRIR`, for a particular cylindrical coordinate 
system triple :math:`(r_{xz},h_y,\phi_{az})`. 

.. figure:: HRIR_example.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Example HRIR for a particular arrival angle pulled from CIPIC for subject 165. :label:`HRIR`


Real-Time Signal Processing
===========================

The cylindrical coordinates of the source point to an LUT entry of filter coefficients for the 
left and right channels. To implement the filtering action we use the :code:`pyaudio_helper` framework 
[Wickert]_ of Figure :ref:`PAH`, which interfaces to the audio subsystem of a personal computer. The 
framework supports real-time signal processing, in particular filtering using core signal 
processing functions of :code:`scipy.signal` [ScipySignal]_. 

.. figure:: pyaudio_helper_BlockDiagram.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   The `pyaudio_helper` framework for real-time DSP in the Jupyter notebook. :label:`PAH`

A top level block diagram of the 3D audio simulator 
is shown in Figure :ref:`FILTERING`.

.. figure:: Filtering_BlockDiagram.pdf
   :scale: 65%
   :align: center
   :figclass: htb

   Real-time DSP filtering with coefficients determined by the audio source :math:`(x,y,z)` location. :label:`FILTERING`

More writing TBD.

Mapping to the CIPIC Interaural Polar Coordinates
-------------------------------------------------

CIPIC uses a special *interaural polar coordinate system* (IPCS) that needs to be addressed in order to make a 3D audio demo. Two other aspects to be consider are:

- CIPIC assumes the sound source lies on a sphere of radius 1m, so due to sound wave divergence, the amplitude needs to be scaled inversely with radial distance (inverse-squared in the sound intensity sense).

- To properly represent a sound source closer than 1m there is a parallax error that must be dealt with as explained in [Fitzpatrick]_.

The ultimate goal is to represent an audio source arriving from any set of coordinates, in this case $(x_1,y_1,z_1$). The class :code:`ss_mapping2CIPIChrif()` manages this:

.. code-block:: python

   class ss_mapping2CIPIChrir(object):
      """
      A class for sound source mapping to the CIPIC 
      HRIR database
      
      CIPIC uses the interaural polar coordinate 
      system (IPCS). The reference sphere for the 
      head-related transfer function (HRTF) 
      measurements/head-related impulse response 
      (HRIR) measurements has a 1m radius.
      
      
      Mark Wickert June 2018
      """


3D Audio Simulator Notebook Apps
--------------------------------

Two applications (apps) that run in the Jupyter notebook at present are a *static* 
location audio and time-varying motion audio source. For human subject test the static 
source is of primary interest.

Static Sound Source
===================

The Jupyter Widgets slider interface is shown in Figure :ref:`STATICAPP` 

.. figure:: Static_3D_AudioApp.pdf
   :scale: 60%
   :align: center
   :figclass: htb

   Jupyter notebook for static positioning of the audio test source. :label:`STATICAPP`


Dynamic Sound Source Along a Trajectory
=======================================

The Jupyter Widgets slider interface is shown in Figure :ref:`DYNAMICAPP`

.. figure:: Dynamic_3D_AudioApp.pdf
   :scale: 60%
   :align: center
   :figclass: htb

   Jupyter notebook for setting the parameters of a sound source moving along a trajectory with 
   prescribed motion characteristics. :label:`DYNAMICAPP`


The trajectory used in this app, shown in Figure :ref:`TRAJECTORY`, is a circular orbit  with parameters of roll, pitch, and hight, relative to the ear canal centerline.


.. figure:: SoundSource_Trajectory.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The sound source trajectory utilized by the dynamic app. :label:`TRAJECTORY`


Spherical Head Model as a Simple Reference
------------------------------------------

Bring in references [Boelein]_, [Duda]_, and [Beranek]_ to discuss a 3D sound source simulator app which use a 
simple spherical head model in place of the subject's HRTF. The HRTF data filter coefficients can be obtained using known expressions for sound pressure wave scattering from a rigid sphere. The sphere radius can for example be set to match the mean radius of the subjects head. Ultimately we wish to evaluate human subjects with multiple HRTF's, and use the ...

Using a spherical harmonics-based solution the incident plus scattered sound pressure, :math:`\tilde{P}`, as a magnitude 
and phase is calculated. For an example shown below a very large sphere is for preliminary calculations at 
an audio frequency of 2 kHz is shown in Figure :ref:`SCATTER`

.. figure:: SphericalHeadScattering.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Using spherical harmonics, [Beranek]_, to calculate the pressure wave magnitude (shown) here and 
   phase with a plane wave audio source arriving from the bottom of the figure. :label:`SCATTER`


Conclusions and Future Work
---------------------------

Applications development is relatively easy on the real-time signal processing side. Getting all of the coordinate transformations together is more complex. 

More writing TBD.


References
----------

.. [Fitzpatrick] Fitzpatrick, W., Wickert, M., and Semwal, S. (2013) 3D Sound Imaging with Head Tracking, *Proceedings IEEE 15th Digital Signal Processing Workshop/7th Signal Processing Education Workshop*.
.. [CIPIC] *The CIPIC Interface Laboratory Home Page*, (2019, May 22). Retrieved May 22, 2019, from `https://www.ece.ucdavis.edu/cipic`_.
.. [CIPICHRTF] *The CIPIC HRTF Database*, (2019, May 22). Retrieved May 22, 2019, from `https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data`_.
.. [Wickert] *Real-Time Digital Signal Processing Using pyaudio_helper and the ipywidgets*, (2018, July 15). Retrieved May 22, 2019, from DOI `10.25080/Majora-4af1f417-00e`_.
.. [ScipySignal] *Signal processing (scipy.signal)*, (2019, May 22). Retrieved May 22, 2019, from `https://docs.scipy.org/doc/scipy/reference/signal.html`_.
.. [Beranek] Beranek, L. and Mellow, T (2012). *Acoustics: Sound Fields and Transducers*. London: Elsevier.
.. [Duda] Duda, R. and Martens, W. (1998). Range dependence of the response of a spherical head model, *J. Acoust. Soc. Am. 104 (5)*.
.. [Boelein]  Bogelein, S., Brinkmann, F.,  Ackermann, D., and Weinzierl, S. (2018). Localization Cues of a Spherical Head Model. *DAGA Conference 2018 Munich*. 

.. _`https://www.ece.ucdavis.edu/cipic`: https://www.ece.ucdavis.edu/cipic
.. _`https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data`: https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data
.. _`https://github.com/mwickert/scikit-dsp-comm`: https://github.com/mwickert/scikit-dsp-comm
.. _`10.25080/Majora-4af1f417-00e`: http://conference.scipy.org/proceedings/scipy2018/mark_wickert_250.html
.. _`https://docs.scipy.org/doc/scipy/reference/signal.html`: https://docs.scipy.org/doc/scipy/reference/signal.html