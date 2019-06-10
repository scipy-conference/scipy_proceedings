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
surface, and then the inverse Fourier transform to ultimately obtain the HRIR. To that end this simulator is 
integral to planned human subject testing where headphone testing will be used to blind test subjects for 
their perception of where a sound source is located. 

3D Geometry
===========

To produce a synthesized 3D audio sound field starts with a geometry. The center of the frame is the subjects head 
where the *mid-sagittal* or vertical *median plane* intersects the line connecting the left and right ear canals. 
For a given source location :math:`(x,y,z) = (x_1,-x_2,x_3)` pointing at the the origin, we transform from rectangular 
coordinates to cylindrical coordinates as shown in Figure :ref:`CYLIND`. This transformation is motivated by 
[Fitzpatrick]_, and will be explained more fully in a later section.
Note also the second rectangular coordinate frame mention above is the notation used by CIPIC. Later we discuss 
how we transform between the cylindrical coordinates and the CIPIC interaural-polar coordinate system (IPCS), 
in order to use the HRIR filter sets in the simulator.


.. figure:: 3D_Coordinates.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The head-centered cylindrical coordinate system used in the 3D audio simulator compared with the 
   CIPIC IPCS. :label:`CYLIND`

The 3D audio rendering provided by the simulator developed in this paper, relies on the 1250 
HRIR measurements were taken using the geometrical configuration shown in Figure :ref:`CIPICLOC`. 
A total of 45 subjects are contained in the CIPIC HRIR database, both human and the mannequin *Kemar* [CIPICHRTF]_. 
For subject 165 in particular, the left-right channel HRIR is shown in Figure :ref:`HRIR`, for a particular 
cylindrical coordinate system triple :math:`(r_{xz},h_y,\phi_{az})`. Figure :ref:`HRIR` in particular illustrates 
two binaural cues, *interaural level difference* ILD and *interaural time difference* ITD, that are used for 
accurate localization of a sound source. With :math:`\phi_{az} = 130^\circ` we see as expected, the impulse 
response for the right ear arriving ahead of the left ear response, and with greater amplitude.

.. figure:: CIPIC_Source_Locations.pdf
   :scale: 60%
   :align: center
   :figclass: htb

   The CIPIC audio source locations, effectively on a 1 m radius sphere, used to obtain 1250 HRIR measurements 
   for each of 45 subjects (only the right hemisphere locations shown). :label:`CIPICLOC`
 

.. figure:: HRIR_example.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Example right/left HRIR plots for a particular arrival angle pulled from CIPIC for subject 165. :label:`HRIR`



Real-Time Signal Processing
===========================

In this section we briefly describe the role real-time digital signal processing (DSP) plays in implementing 
the 3D audio simulator. A top level block diagram of the 3D audio simulator is shown in Figure :ref:`FILTERING`. 
From the block diagram we see that for audio source is positioned at :math:`(x,y,z)` relative to the head 
center, the appropriate HRIR right and left channel digital filter coefficients are utilized along with gain scaling 
to account for radial distance relative to 1 m and a parallax correction factor. Gain scaling and parallax 
correction, are taken from [Fitzpatrick]_, and are explained in more detail in the following section of this paper.

To implement the filtering action we use the :code:`pyaudio_helper` framework 
[Wickert]_ of Figure :ref:`PAH`, which interfaces to the audio subsystem of a personal computer. The 
framework supports real-time signal processing, in particular filtering using core signal 
processing functions of :code:`scipy.signal` [ScipySignal]_. The 200 coefficients of the right and left HRIR 
are equivalent to the coefficients in a finite impulse response (FIR) digital filter which produce a discrete-time 
output signal or sequence :math:`y_R[n]/y_L[n]` from a single audio source signal :math:`x[n]`. All of the signals 
are processed with at a sampling rate of :math:`f_s = 44.1` ksps, as this is rate used in forming the CIPIC 
database. In mathematical terms we have the output signals that drive 

.. math::
   :label: LCCDE
   :type: eqnarray

   y_R[n] &=& G_R \sum_{m=0}^M b_R x[n-m] \\
   y_L[n] &=& G_L \sum_{m=0}^M b_L x[n-m]

where :math:`G_R` and :math:`G_L` are right/left gain scaling factors that take into account the source distance relative 
to the 1 m distance used in the CIPIC database and :math:`b_R` and :math:`b_L` are the right/left HRIR coefficient sets 
appropriate for the source location.

.. figure:: Filtering_BlockDiagram.pdf
   :scale: 65%
   :align: center
   :figclass: htb

   Real-time DSP filtering with coefficients determined by the audio source :math:`(x,y,z)` location. 
   :label:`FILTERING`


.. figure:: pyaudio_helper_BlockDiagram.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   The `pyaudio_helper` framework for real-time DSP in the Jupyter notebook. :label:`PAH`


To produce real-time filtering with :code:`pyaudio_helper` requires [Wickert]_ (i) create an instance of the 
:code:`DSP_io_stream` class by assigning valid PC audio input and output device ports to it, (ii) define 
a :code:`callback` function to process the 
input signal sample frames into right/left output sample frames according to (:ref:`LCCDE`), and (iii) call the 
method :code:`interactive_stream()` to start streaming. All of the code for the 3D simulator is developed in a 
Jupyter notebook for prototyping ease. Since [Wickert]_ details steps (i)-(iii), in the code snippet below 
we focus on the key filtering expressions in the callback and  
describes the playback of a *noise* via headphones:

.. code-block:: python

   def callback(in_data, frame_length, time_info, 
                status):
       global ...
       ...    
       #***********************************************
       # DSP operations here:
       # Apply Kemar HRIR left and right channel filters 
       # at the sound source location in cylindrical 
       # coordinates mapped to cartesian coordinates
       # from GUI sliders
       # The input to both filters comes by first 
       # combining x_left & x_right channels or here
       # input white noise
       x_mono = Gain.value*5000*randn(frame_length) 
       subject.cart2ipcs(r_xz_plane.value*sin(pi/180* \
                         azimuth.value), #x
                         y_axis.value,   #y
                         r_xz_plane.value* \
                         cos(pi/180* \
                         azimuth.value)) #z 
       # Filter a frame of samples and save initial 
       # conditions for the next frame
       y_left, zi_left = signal.lfilter(subject.coeffL,
                                   1,subject.tL*x_mono,
                                   zi=zi_left) 
       y_right, zi_right = signal.lfilter(subject.coeffR,
                                   1,subject.tR*x_mono,
                                   zi=zi_right)
       #***********************************************
       ...
       # Convert ndarray back to bytes
       return y.tobytes(), pah.pyaudio.paContinue

   # Create a ss_mapping2CIPIChrir object
   # SUBJECT 20, 21 (KEMAR SM ears), 
   # & 165 (KEMAR LG ears)
   # subject_200, 201 is 8.75 cm, 10 cm sphere
   subject = ss_mapping2CIPIChrir('subject_165')
   # Initialize L/R filter initial conditions
   zi_left = signal.lfiltic(subject.coeffL,1,[0])
   zi_right = signal.lfiltic(subject.coeffR,1,[0])
   # Create a IO stream object and start streaming
   DSP_IO = pah.DSP_io_stream(callback,0,1,
                              frame_length=1024, 
                              fs=44100,Tcapture=0)
   DSP_IO.interactive_stream(0,2)
   # Show Jupyter widgets
   widgets.HBox([Gain,r_xz_plane,azimuth,y_axis])



Mapping to the CIPIC and Source Range Correction
------------------------------------------------

The real-time signal processing just described requires coordinate transformations to obtain the properly 
CIPIC database filter coefficients as well as range corrections, as the source may be less than or 
greater than 1 m away. The Jupyter notebook apps described in the next section are driven by source position 
using the cyclindrical coordinates described in Figure :ref:`CYLIND`. To allow extensibility to future 
applications it was decided that access to CIPIC is made from :math:`(x,y,z)` and as needed apps  
convert from cylindrical coordinates to cartesian :math:`(x,y,z)`. This decision was strongly motivated by 
the fact that [Fitzpatrick]_ uses :math:`(x,yz)`, as defined in :ref:`CYLIND`, to additionally perform 
the import task of *parallax* correction and source range amplitude/gain correction. 
The main points of amplitude and parallax correction are:

- CIPIC assumes the sound source lies on a sphere of radius 1m, so due to sound wave divergence, the amplitude needs to be scaled inversely with radial distance (inverse-squared in the sound intensity sense).

- To properly represent a sound source closer than 1m there is a parallax error that must be dealt with as explained in [Fitzpatrick]_.

- For a source on the 1 m reference sphere, or further away, the there is no parallax error and the CIPIC HRIR coefficients are those of the corresponding azimuth and elevation for both right and left ears

- When the source is inside is the unit sphere sound parallax [Fitzpatrick]_ requires an adjustment in the HRIR coefficients, unique to the right and left ears. If we extend rays from the left and right ears that pass through the sound source location and then touch the unit sphere, the required azimuth values will be shifted to locations either side of the true source azimuth. The corresponding HRIR values where these rays contact the unit sphere, respectively, perform parallax correction.  


The ultimate goal is to represent an audio source arriving from any set of coordinates, in this 
case :math:`(x,y,z)`. The simple class :code:`ss_mapping2CIPIChrif()`, in a Jupyter notebook, manages this with the single 
method :code:`cart2ipcs(self,x1,y1,z1)`, following object instantiation. The code is listed below:

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
 



.. code-block:: python

   def __init__(self,sub_foldername,
                head_radius_cm = 8.75):
      """
      Object instantiation
      
      The default head radius is 8.75 cm
      """
      # Store the head radius in meters
      self.head_radius = head_radius_cm/100
      
      # Store the HRIR 200 tap FIR filter coefficient sets
      self.subject = sub_foldername
      hrir_LR = io.loadmat( self.subject + '/hrir_final.mat')
      self.hrirL = hrir_LR['hrir_l']
      self.hrirR = hrir_LR['hrir_r']
      
      # Create LUTs for the azimuth and elevation 
      # values. This will make it easy to quantize
      # a given source location to one of the 
      # available HRIRs in the database.
      self.Az_LUT = hstack(([-80,-65,-55],
                     arange(-45,45+5,5.0),[55,65,80]))
      self.El_LUT = -45 + 5.625*arange(0,50)
      
      # Initialize parameters
      self.tR = 1 # place source on unit sphere
      self.tL = 1 # directly in front of listener
      self.elRL = 0
      self.azR = 0
      self.azL = 0
      self.AzR_idx = 0
      self.AzL_idx = 0
      self.ElRL_idx = 0
      
      # Store corresponding right and left ear FIR 
      # filter coefficients
      self.coeffR = self.hrirR[0,0,:]
      self.coeffL = self.hrirL[0,0,:]
        
    

   def cart2ipcs(self,x1,y1,z1):
      """
      Map cartesian source coordinates (x1,y1,z1) to 
      the CIPIC interaural polar coordinate system 
      (IPCS) for easy access to CIPIC HRIR. Parallax 
      error is also dealt with so two azimuth values 
      are found. To fit IPCS the cartesian 
      coordinates are defined as follows:

      (0,0,0) <--> center of head.
      (1,0,0) <--> unit vector pointing outward from 
                   the right on a line passing from 
                   left to right through the left 
                   and right ear (pinna) ear canals
      (0,1,0) <--> unit vector pointing out through 
                   the top of the head.
      (0,0,1) <--> unit vector straight out through 
                   the back of the head, such that 
                   a right-handed coordinate system is 
                   formed.

      Mark Wickert June 2018
      """
      # First solve for the parameter t, which is used
      # to describe parametrically the location of the 
      # source at (x1,y1,z1) on a line connecting the
      # right or left ear canal entry point to the 
      # unit sphere.

      # The right ear (pinna) solution
      aR = (x1-self.head_radius)**2 + y1**2 + z1**2
      bR = 2*self.head_radius*(x1-self.head_radius)
      cRL = self.head_radius**2 - 1
      # The left ear (pinna) solution
      aL = (x1+self.head_radius)**2 + y1**2 + z1**2
      bL = -2*self.head_radius*(x1+self.head_radius)

      # Find the t values which are also the gain 
      # values to be applied to the filter.
      self.tR = max((-bR+sqrt(bR**2-4*aR*cRL))/(2*aR),
               (-bR-sqrt(bR**2-4*aR*cRL))/(2*aR))
      self.tL = max((-bL+sqrt(bL**2-4*aL*cRL))/(2*aL),
               (-bL-sqrt(bL**2-4*aL*cRL))/(2*aL))
      # Find the IPCS elevation angle and mod it
      elRL = 180/pi*arctan2(y1,-z1)
      if elRL < -90:
            elRL += 360
      self.elRL = elRL
      self.azR = 180/pi*arcsin(clip(self.head_radius\
                  + self.tR*(x1-self.head_radius),
                  -1,1))
      self.azL = 180/pi*arcsin(clip(-self.head_radius\
                  + self.tL*(x1+self.head_radius),
                  -1,1))
      
      self.AzR_idx = argmin((self.Az_LUT \
                             - self.azR)**2)
      self.AzL_idx = argmin((self.Az_LUT \
                             - self.azL)**2)
      self.ElRL_idx = argmin((self.El_LUT \
                             - self.elRL)**2)
      self.coeffR = self.hrirR[self.AzR_idx,
                               self.ElRL_idx,:]
      self.coeffL = self.hrirL[self.AzL_idx,
                               self.ElRL_idx,:]


The main take-away is that the coordinate conversion method fills class attributes with the proper 
right and left filter coefficients and the sound wave amplitude correction factors :code:`self.tR` and 
:code:`tL`. The variable name :code:`t` comes from the parallax correction expression in [Fitzpatrick]_ 
as a distance scale factor. This distance scale factor is conveniently also the same as the 
required range scale factors, :math:`G_R` and :math:`G_L` in (1) and (2). 


3D Audio Simulator Notebook Apps
--------------------------------

For human subject testing and general audio virtual reality experiments two applications (apps), that 
run in the Jupyter notebook, have been created. The first is a *static* 
location audio and the the second is a *time-varying motion* audio source. For human subject tests the static 
source is of primary interest.

Static Sound Source
===================

The first and foremost purpose the 3D audio simulator is to to be able statically position the audio source 
and then ask a human subject where the source is located. This is a cognitive experiment, and can serve many 
purposes. One purpose in the present research is to to see how well the HRIR utilized in the simulator 
matches the subjects true HRIR. As mentioned in the introduction an ongoing study is form an *individualized 
HRIR* using say deep machine learning/deep learning. The Jupyter Widgets slider interface is for this 
app is shown in Figure :ref:`STATICAPP` 

.. figure:: Static_3D_AudioApp.pdf
   :scale: 60%
   :align: center
   :figclass: htb

   Jupyter notebook for static positioning of the audio test source. :label:`STATICAPP`


Dynamic Sound Source Along a Trajectory
=======================================

From a virtual reality perspective we were also interested in giving a subject a moving sound source experience. 
In this case we consider an *orbit like* sound source trajectory. The trajectory as shown in Figure 
:ref:`TRAJECTORY`, is a circular orbit  with parameters of roll, pitch, and hight, relative to 
the ear canal centerline. The Jupyter Widgets slider interface is shown in Figure :ref:`DYNAMICAPP`.

.. figure:: SoundSource_Trajectory.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The sound source trajectory utilized in the dynamic sound source app. :label:`TRAJECTORY`


.. figure:: Dynamic_3D_AudioApp.pdf
   :scale: 60%
   :align: center
   :figclass: htb

   Jupyter notebook for setting the parameters of a sound source moving along a trajectory with 
   prescribed motion characteristics. :label:`DYNAMICAPP`



Spherical Head Model as a Simple Reference
------------------------------------------

Bring in references [Boelein]_, [Duda]_, and [Beranek]_ to discuss a 3D sound source simulator app which use a 
simple spherical head model in place of the subject's HRTF. The HRTF data filter coefficients can be obtained 
using known expressions for sound pressure wave scattering from a rigid sphere. The sphere radius can for example be set to match the mean radius of the subjects head. Ultimately we wish to evaluate human subjects with multiple HRTF's, and use the ...

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