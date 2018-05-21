:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs

:video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------------------------------------
Real-Time Digital Signal Processing Using pyaudio_helper and the ipywidgets
---------------------------------------------------------------------------

.. class:: abstract

  The focus of this poster is on teaching real-time digital signal processing to 
  electrical and computer engineers using the Jupyter notebook and the code 
  module ``pyaudio_helper``, which is a component of the package 
  scikit-dsp-comm_. Specifically, we show how easy it is to prototype real-time DSP 
  algorithms for processing analog signal inputs and returning analog signal outputs, 
  all within the Jupyter notebook. Real-time control of running code is provided 
  by ipywidgets. Audio applications and in a reduced bandwidth fashion, 
  software defined radio applications can be developed.

.. _scikit-dsp-comm: https://github.com/mwickert/scikit-dsp-comm

.. class:: keywords

   digital signal processing, pyaudio, real-time, scikit-dsp-comm

Introduction
------------

As the power of personal computer has increased, the dream of rapid prototyping of real-time 
signal processing, without the need to use dedicated DSP-microprocessors or digital signal 
processing (DSP) enhanced 
microcontrollers, such as the ARM Cortex-M4 [cortexM4]_, can be set aside. Students can focus on the 
powerful capability of :code:`numpy` and :code:`scipy`, along with packages such as 
:code:`scipy.signal` [Scipysignal]_ and :code:`scikit-dsp-comm` [DSPComm]_ to explore 
real-time signals and systems computing. 

The focus of this paper is on teaching real-time DSP to electrical 
and computer engineers using the Jupyter notebook and the code module :code:`pyaudio_helper`, 
which is a component of the package :code:`scikit-dsp-comm`. To be clear,  
:code:`pyaudio_helper` is built upon the well known package [pyaudio]_, which has 
its roots in *Port Audio* [portaudio]_. We will see that to set up an audio input/output (I/O) 
stream requires: (1) create and instance of the :code:`DSP_io_stream` class by assigning valid
input and output device ports to it, (2) define a callback function to process the input signal 
sample frames into output sample frames, and (3) call the method :code:`interactive_stream()` 
to start streaming. All of this is done within the Jupyter notebook. Real-time control of 
running code is provided by [ipywidgets]_. Audio applications and in a reduced bandwidth 
fashion, software defined radio (SDR) applications can also be developed.

Analog Input/Output Using DSP Algorithms
----------------------------------------

A class text to learn the theory of digital signal processing is [Opp2010]_. This book is heavy on 
the underlying theoretical concepts of DSP, including the mathematical modeling of analog I/O systems 
as shown in Figure :ref:`basicDSPio`. An analog or continuous-time signal :math:`x(t)` enters the 
system on the left and is converted to the discrete-time signal :math:`x[n]` by the analog to 
digital block. The sampling rate :math:`f_s` which is the inverse of the sampling period, :math:`T` 
establishes 

.. figure:: basic_DSP_IO.pdf
   :scale: 58%
   :align: center
   :figclass: htb

   Analog signal processing implemented using real-time DSP. :label:`basicDSPio`



.. figure:: pyaudio_DSP_IO.pdf
   :scale: 58%
   :align: center
   :figclass: htb

   Two channel analog signal processing implemented using frame-based real-time DSP. :label:`pyaudioDSPio`


.. figure:: pyaudio_helper_classes.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   The major classes and functions of the module :code:`sk_dsp_comm.pyaudio_helper.py`. :label:`pyaudioHelperclasses`

.. figure:: USB_audio_AD2_measure.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Hardware interfaces: (a) iMic stereo USB audio device and the Digilent Analog Discovery 2 and (b) the 
   low-cost Sabrent mono input stereo output USB audio device. :label:`USBAudioAD2`
   
.. code-block:: python

   # Set up two sliders
   L_gain = widgets.FloatSlider(description = 'L Gain', 
                continuous_update = True,
                value = 1.0,
                min = 0.0, 
                max = 2.0, 
                step = 0.01, 
                orientation = 'vertical')
   R_gain = widgets.FloatSlider(description = 'R Gain', 
                continuous_update = True,
                value = 1.0,
                min = 0.0, 
                max = 2.0, 
                step = 0.01, 
                orientation = 'vertical')

   # L and Right Gain Sliders callback
   def callback(in_data, frame_count, time_info, 
                status):  
       DSP_IO.DSP_callback_tic()
       # convert byte data to ndarray
       in_data_nda = np.frombuffer(in_data, 
                                   dtype=np.int16)
       # separate left and right data
       x_left,x_right = DSP_IO.get_LR(in_data_nda.\
                                      astype(float32))
       #*********************************************
       # DSP operations here
       y_left = x_left*L_gain.value
       y_right = x_right*R_gain.value
      
       #*********************************************
       # Pack left and right data together
       y = DSP_IO.pack_LR(y_left,y_right)
       # Typically more DSP code here     
       #*********************************************
       # Save data for later analysis
       # accumulate a new frame of samples
       DSP_IO.DSP_capture_add_samples_stereo(y_left,
                                             y_right)
       #*********************************************
       # Convert from float back to int16
       y = y.astype(int16)
       DSP_IO.DSP_callback_toc()
       # Convert ndarray back to bytes
       return y.tobytes(), pah.pyaudio.paContinue


.. figure:: Left_Right_Gain_Slider_app.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   A simple stereo gain slider app: (a) Jupyter notebook interface and (b) testing using the 
   AD2. :label:`LeftRightGainSlider`

Conclusions and Future Work
---------------------------

What to say?

In the future we hope to easily develop algorithms that can demodulate software-defined radio (SDR) 
streams and send the recovered modulation signal out the computer's audio interface via PyAudio. 
Environments such as GNURadio companion already support this, but being able to do this right in the  
Jupyter notebook is our desire.


References
----------
.. [cortexM4] `Thomas Lorenser, "The DSP capabilities of ARM®  Cortex®-M4 and Cortex-M7 Processors", ARM, November 2016.`_
.. [Scipysignal] `https://docs.scipy.org/doc/scipy/reference/signal.html`_
.. [DSPComm] `https://github.com/mwickert/scikit-dsp-comm`_
.. [pyaudio] `https://people.csail.mit.edu/hubert/pyaudio/`_
.. [portaudio] `http://www.portaudio.com/`_
.. [ipywidgets] `https://github.com/jupyter-widgets/ipywidgets`_
.. [Opp2010] Alan V. Oppenheim and Ronald W. Schafer, *Discrete-Time Signal Processing* (3rd ed.), Prentice Hall, 2010.
.. [AD2] `https://store.digilentinc.com/analog-discovery-2-100msps-usb-oscilloscope-logic-analyzer-and-variable-power-supply/`_

.. _`Thomas Lorenser, "The DSP capabilities of ARM®  Cortex®-M4 and Cortex-M7 Processors", ARM, November 2016.`: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjRhqvO25XbAhUDx2MKHRiJBeYQFggnMAA&url=https%3A%2F%2Fcommunity.arm.com%2Fcfs-file%2F__key%2Ftelligent-evolution-components-attachments%2F01-2142-00-00-00-00-73-48%2FARM-white-paper-_2D00_-DSP-capabilities-of-Cortex_2D00_M4-and-Cortex_2D00_M7.pdf&usg=AOvVaw1jyK7ExAE-2YqmaEzRSx8z
.. _`https://docs.scipy.org/doc/scipy/reference/signal.html`: https://docs.scipy.org/doc/scipy/reference/signal.html
.. _`https://github.com/mwickert/scikit-dsp-comm`: https://github.com/mwickert/scikit-dsp-comm
.. _`https://people.csail.mit.edu/hubert/pyaudio/`: https://people.csail.mit.edu/hubert/pyaudio/
.. _`http://www.portaudio.com/`: http://www.portaudio.com/
.. _`https://github.com/jupyter-widgets/ipywidgets`: https://github.com/jupyter-widgets/ipywidgets
.. _`https://store.digilentinc.com/analog-discovery-2-100msps-usb-oscilloscope-logic-analyzer-and-variable-power-supply/`: https://store.digilentinc.com/analog-discovery-2-100msps-usb-oscilloscope-logic-analyzer-and-variable-power-supply/
