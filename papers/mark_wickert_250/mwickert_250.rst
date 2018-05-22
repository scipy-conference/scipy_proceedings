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
digital block. In practice this block is known as the analog-to-digital converter (ADC). 
The sampling rate :math:`f_s`, which is the inverse of the sampling period, :math:`T`, 
leads to :math:`x[n] = x(nT)`. The sampling theorem tells us [Opp2010]_ tells us that the sampling 
rate :math:`f_s` must be greater than twice the highest frequency we wish to represent in the 
discrete-time domain. Violating this condition results in *aliasing*, which means a signal centered 
on frequency :math:`f_0 > f_s/2` will land inside the band of frequencies :math:`[0, f_s/2]`. Fortunately, 
most audio ADCs limit the signal bandwidth of :math:`x(t)` in such a way that signals with 
frequency content greater than :math:`f_s/2` are eliminated from passing through the ADC. Another 
practical matter is in reality :math:`x[n]` is a scaled and finite precision version 
of :math:`x(t)`.  In real-time DSP environments the ADC maps the analog signal samples to signed 
integers, most likely :code:`int16`. As we shall see in pyaudio, this is indeed the case.

.. figure:: basic_DSP_IO.pdf
   :scale: 58%
   :align: center
   :figclass: htb

   Analog signal processing implemented using real-time DSP. :label:`basicDSPio`

The DSP algorithms block can be any operation on the signals samples :math:`x[n]` that makes sense. 
At this beginning stage, the notion is that the samples flow through the algorithm one at a time, 
that is one input results in one output sample. The output samples are converted back to analog 
signal :math:`y(t)` by placing the samples into a digital-to-analog converter (DAC). The DAC does 
not simply set :math:`y(nT) = y[n]`, as :math:`y(t)` is a continuous function time :math:`t`. A 
*reconstruction* operation takes place inside the DAC which *interpolates* the :math:`y[n]` 
signal samples over continuous time. In most DACs this is accomplished with a combination of 
digital and analog filters, the details of which is outside the scope of this paper.

In a DSP theory class the algorithm for producing :math:`y[n]` from :math:`x[n]` is typically a 
*causal* linear time-invariant (LTI) system/filter, implemented via a difference equation, i.e.,

.. math::
   :label: LCCDE

   y[n] = -\sum_{k=1}^N a_k y[n-k] + \sum_{m=0}^M b_m x[n-m]

where :math:`a_k, k=1,2,\ldots,N` and :math:`b_k, m=0,1,\ldots,M` are the filter coefficients. The 
filter coefficients that implement a particular filter design can be obtained using design tools in [DSPComm]_.

Other algorithms of course are possible. We might have a two channel system and perform operations on 
both signals, say combining them, filtering, and locally generating time varying periodic signals 
to create audio special effects. When first learning about real-time DSP it is important to start 
with simple algorithm configurations, so that external measurements can be used to characterize 
the systems and verify that the intended results are realized. So the process follows along the lines 
of, design, implement, and test using external test equipment. The Jupyter notebook allows all of 
this to happen in one place, particularly if the test instrumentation is also PC-based, since 
PC-based instrument results can be exported as :code:`csv` and then imported in Jupyter notebook 
using :code:`loadtxt`. Here we advocate the use of PC-based instruments, so that all parties, 
student/instructor/tinkerer, can explore real-time DSP from most anywhere at any time. 
In this paper we use the Analog Discovery 2 
[AD2]_ for signal generation (two function generator channels), signal measurement (two scope channels, 
with fast Fourier transform (FFT) spectrum analysis included). It is also helpful to have a signal 
generator cellphone app available, and of course music from the cell phone or PC. All of the cabling 
is done using 3.5mm stereo patch cables and small pin header adapters [3p5mm]_ to interface to the AD2.

Frame-based Real-Time DSP Using the :code:`io_stream` class
-----------------------------------------------------------

The block diagram of Figure :ref:`pyaudioDSPio` illustrates the essence of this paper. 
Implementing the stucture of this figure relies upon the class :code:`DSP_io_stream` which is housed
in :code:`sk_dsp_comm.pyaudio_helper.py`. To make use of this requires the scipy stack 
(numpy, scipy, and matplotlib), as well as [DSPComm]_ and [pyaudio]_. PyAudio is supported 
supported on all majors OSs, e.g., Windows, macOS, and Linux. The configuration varies, 
but the set-up is documented at [pyaudio]_ and SPCommTutorial_. The classes and functions 
of :code:`pyaudio_helper` are detailed in Figure :ref:`pyaudioHelperclasses`.

.. _SPCommTutorial: https://github.com/mwickert/SP-Comm-Tutorial-using-scikit-dsp-comm/wiki

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

With :code:`DSP_io_stream` one or two channel streaming is possible, as shown in :ref:`pyaudioDSPio`. The ADCs 
and DACs can be internal to the PC or external, say using a USB interface. In a modern PC the audio 
subsystem has a microphone hardwired to the ADCs and the DACs are connected to the speakers and 3.5mm 
headphone jack. To provide more flexibility in doing real-time DSP, an external USB audio interface 
is essential. Two worthy options are the Sabrent_ at less than $10 and the iMic_ at under $40. You get 
what you pay for. The iMic is ideal for full two channel audio I/O processing and also has a line-in/mic 
switch setting, while the Sabrent offers a single channel input and two channel output. Both are 
very capable for their intended purposes. A photograph of the AD2 with the iMic interface, 3.5mm 
splitters and the pin header interfaces mentioned earlier, is shown in Figure :ref:`USBAudioAD2`. 
The 3.5mm audio splitters are optional, but allow headphones to be plugged into the output 
while leaving the AD2 scope connected, and the ability to input music/function generator from 
a cellphone while leaving the AD2 input cable connected (pins wires may need to be pulled off the 
AD2 to avoid interaction between the two devices in parallel).

.. _Sabrent: https://www.sabrent.com/product/AU-MMSA/usb-external-stereo-3d-sound-adapter-black/
.. _iMic: https://griffintechnology.com/us/imic

.. figure:: USB_audio_AD2_measure2.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Hardware interfaces: (a) iMic stereo USB audio device and the Digilent Analog Discovery 2 and (b) the 
   low-cost Sabrent mono input stereo output USB audio device. :label:`USBAudioAD2`
   
To list the internal/external devices available on a given PC 
we use the function :code:`available_devices()` found in :ref:`pyaudioHelperclasses`:

.. code-block:: python

   import sk_dsp_comm.pyaudio_helper as pah
   In[3]: pah.available_devices()
   Out[3]:
   Index 0 device name = Built-in Microphone, 
           inputs = 2, outputs = 0
   Index 1 device name = Built-in Output, 
           inputs = 0, outputs = 2
   Index 2 device name = iMic USB audio system, 
           inputs = 2, outputs = 2

The output list can be viewed as a look-up table (LUT) for how to patch physical devices into 
the block diagram of :ref:`pyaudioDSPio`. 

We now shift the focus to the interior of :ref:`pyaudioDSPio` to discuss frame-based DSP and 
the *Frame-Based DSP Callback*. When a DSP micro controller, is configured for real-time DSP, it 
can focus on just this one task very well. Sample-by-sample processing is possible with low 
I/O latency and overall reasonable audio sample throughput. On a PC, with its multitasking OS, 
there is a lot going on. To get reasonable audio sample throughput with a PC, *frames* or 
*chucks* of audio samples must be dealt with. The pack and unpack blocks sitting next to the ADCs 
and DACs of :ref:`pyaudioDSPio`, respectively, are there to make it clear that processing 
takes place one frame at a time rather than one sample at a time. The central block, 
Frame-Based DSP Callback, is where the real-time DSP code resides. Global variables are needed 
inside the call back, as the callback input/output signature is fixed by PyAudio. The globals 
allow algorithm parameters to be available inside the callback, e.g., filter coefficients, 
and in the case of digital filter, the filter state must be maintained from frame-to-frame. 
We will see in a later example how :code:`scipy.signal.lfilter()` conveniently supports 
frame-based digital filtering. To allow interactive control of parameters of the DSP 
algorithm we can use :code:`ipywidgets`. We will see later the sliders widgets are 
particularly nice.

Anatomy of a PyAudio Callback function
======================================

Before writing the callback we first need to instantiate a :code:`DSP_io_stream` object:

.. code-block:: python

   DSP_IO = pah.DSP_io_stream(callback, #callback name
                  2,2, # set I/O device indices
                  fs=48000, # sampling rate
                  Tcapture=0) # capture buffer length

A basic loop through callback function takes the following form in the Jupyter notebook:

.. code-block:: python

   # define a pass through, y = x, callback
   def callback(in_data, frame_count, 
                 time_info, status):
       global b, a, zi # typical globals for a filter
       DSP_IO.DSP_callback_tic() #log entering time
       # convert audio byte data to an int16 ndarray
       in_data_nda = np.frombuffer(in_data, 
                                   dtype=np.int16)
       #***********************************************
       # Begin DSP operations here
       # for this app cast int16 to float32
       x = in_data_nda.astype(float32)
       y = x # pass input to output
       # Typically more DSP code here
       # Optionally apply a linear filter to the input
       #y, zi = signal.lfilter(b,a,x,zi=zi)
       #***********************************************
       # Save data for later analysis
       # accumulate a new frame of samples if enabled
       # with Tcapture
       DSP_IO.DSP_capture_add_samples(y) 
       #***********************************************
       # Convert from float back to int16
       y = y.astype(int16)
       DSP_IO.DSP_callback_toc() #log departure time
       # Convert ndarray back to bytes
       return y.tobytes(), pah.pyaudio.paContinue

In this simple callback example the input sample array of length 1024, is cast to 
:code:`float32` and then passed to the output array, where it ultimately is cast 
back to :code:`int16` signed integers. To start streaming we need to call the method 
:code:`interactive_stream()`, which display :code:`ipywidgets` start/stop buttons 
below the code cell as shown in Figure :ref:`LoopThrough`.

.. figure:: Loop_through_app.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Setting up an interactive stream for the simple :code:`y = x` loop through,  
   using a run time of 0, which implies run forever. :label:`LoopThrough`

Performance Measurements
===============================

The loop through example is good place to explore some performance metrics of 
:ref:`pyaudioDSPio`, and take a look at some of the instrumentation that is part of the 
:code:`DSP_io_stream` class. The methods :code:`DSP_callback_tic()` and 
:code:`DSP_callback_toc()` store time stamps in attributes of the class. Another attribute 
stores samples in the attribute :code:`data_capture`. For the instrumentation to 
collect operating data we need to set :code:`Tcapture` greater than zero. We will also set 
the total run time to 2s:

.. code-block:: python

   DSP_IO = pah.DSP_io_stream(callback,2,2,fs=48000,Tcapture=2)
   DSP_IO.interactive_stream(2,1)

Running the above in Jupyter notebook cell will capture 2s of data. The method 
:code:`stream_stats()` displays the following:

.. code-block:: python

   Ideal Callback period = 21.33 (ms)
   Average Callback Period = 21.33 (ms)
   Average Callback process time = 0.40 (ms)

which tells us that as expected for a sampling rate of 48 kHz, and a frame length of 1024 is simply

.. math::
   :label: callbackPeriod

   T_\text{callback period} = 1024 \times \frac{1}{48000} = 21.33\ \text{ms}

The time spent in the callback should be very small, as very little processing is being done. 
We can also examine the callback latency by having the AD2 input a low duty cycle pulse train 
have a 2 Hz rate. The scope then measures the time difference between the input and output waveforms. 
The resulting plot is shown in Figure :ref:`CBlatency`. We sees that PyAudio and 
and the PC audio subsystem introduces about 70.7ms of latency.  

.. figure:: 48kHz_latency.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Callback latency measurement using the AD2 with a 2 Hz pulse train in the 
   loop through app. :label:`CBlatency`

The frequency response magnitude of an LTI system can be measured using the fact that 
[Opp2010]_ at the output of a system driven by white noise, the measured power output spectrum 
is a scaled version of the underlying system frequency response magnitude squared, i.e., 

.. math::
   :label: HfromNoisePSD

   S_{y,\text{measured}}(f) = \sigma_x^2 |H_\text{LTI system}(f)|^2

where :math:`\sigma_x^2` is the variance of the input white noise signal. Here we use this 
technique to first estimate the frequency response magnitude of the input path (ADC only) 
using the attribute :code:`DSP_IO.capture_buffer`, and then take end-to-end (ADC-DAC) 
measurements using the AD2 spectrum analyzer in dB average mode (500 records). In both 
cases the white noise input is provided by the AD2 function generator.
Finally, the AD2 measurement is saved to a CSV file 
and imported into the Jupyter notebook to overlay the ADC only measurement, which is made 
entirely in the Jupyter notebook. The results are compared in Figure 
:ref:`LoopThroughiMicGainFlatnes`.

.. code-block:: python

   f_AD,Mag_AD = loadtxt('Loop_through_noise_SA.csv',
                        delimiter=',',skiprows=6,unpack=True)
   plot(f_AD,Mag_AD-Mag_AD[100])
   Pxx, F = ss.my_psd(DSP_IO.data_capture,2**11,48000);
   plot(F,10*log10(Pxx/Pxx[20]))
   ylim([-10,5])
   xlim([0,20e3])
   ylabel(r'ADC Gain Flatness (dB)')
   xlabel(r'Frequency (Hz)')
   legend((r'ADC only from DSP_IO.capture_buffer',r
           'ADC-DAC from AD2 SA dB Avg'))
   title(r'Loop Through Gain Flatness using iMic at $f_s = 48$ kHz')
   grid();
   savefig('Loop_through_iMic_gain_flatness.pdf')

.. figure:: Loop_through_iMic_gain_flatness.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Gain flatness of the loop through app of just the ADC path via the :code:`DSP_IO.capture_buffer` 
   and then the ADC-DAC path using the AD2 spectrum analyzer to average the noise 
   spectrum. :label:`LoopThroughiMicGainFlatness`

The results show considerable roll-off in just the ADC path, but then gain peaking above 
17 kHz. As a practical matter, humans do not hear sound much above 16 kHz, so the peaking 
is not much of an issue. The 3dB roll-off out to 15 kHz is not great, but perhaps testing 
other audio I/O devices would reveal better results. For example the native PC audio output 
can easily be tested.

Examples
--------

In this section we consider a collection of examples.

Left and Right Gain Sliders
===========================

In this first example the signal processing is again minimal, but now two-channel (stereo) 
processing is untilized, and left and right channel gain slider using :code:`ipywidgets` 
are introduced. Since the audio stream is running in a thread, the :code:`ipywidgets` can freely 
run and interactively coedntrol parameters inside the callback function. The two slider widgets 
are created below, followed by the callback, and finally calling the
:code:`interactive_stream` method to run without limit in two channel mode. A 1 kHz sinusoid 
test signal is input to the lelft channel and a 5 kHz sinusoid is input to the right channel. 
While viewing the AD2 scope output in real-time, the gain sliders are adjusted and the signal 
levels move up and down. A screen shot taken from the Jupyter notebook is combined with a 
screenshot of the scope output to verify the correlation between the observed signal amplitudes 
and the slider positions is given in Figure :ref:`LeftRightGainSlider`.

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
   AD2 with generators and scope channels. :label:`LeftRightGainSlider`

The results are as expected, especially when listening.

Cross Left-Right Channel Panning
================================

This example again works with a two channel signal flow. The application is to implement a cross 
channel panning system. Ordinarily panning moves a single channel of audio from 100% left to 
100% right as a slider moves from 0% to 100% of its range. At 50% the single channel should 
have equal amplitude in both channels. In cross channel panning two input channels are super 
imposed, but such that at 0% the left and right channels are fully in their own channel. At 
50% the left and right outputs are equally mixed. At 100% the input channels are now swapped. 
AAssuming that :math:`a` represents the panning values on the interval :math:`[0,100]`, a 
mathematical model of the cross panning app is

.. math::
   :type: eqnarray
   :label: crosspanning

   L_\text{out} &=& (100-a)/100 \times L_\text{in} + a/100\times R_\text{in} \\
   R_\text{out} &=& a/100\times L_\text{in} + (100-a)/100 \times R_\text{in}

In code we have:

.. code-block:: python

   panning = widgets.FloatSlider(description = \
                  'Panning (%)', 
                  continuous_update = True,
                  value = 50.0,
                  min = 0.0, 
                  max = 100.0, 
                  step = 0.1, 
                  orientation = 'horizontal')
   #display(panning)

   # Cross Panning
   def callback(in_data, frame_count, time_info, 
                status):  
       DSP_IO.DSP_callback_tic()
       # convert byte data to ndarray
       in_data_nda = np.frombuffer(in_data, 
                                   dtype=np.int16)
       # separate left and right data
       x_left,x_right = DSP_IO.get_LR(in_data_nda.\
                                      astype(float32))
       #***********************************************
       # DSP operations here
       y_left = (100-panning.value)/100*x_left \
                + panning.value/100*x_right
       y_right = panning.value/100*x_left \
                + (100-panning.value)/100*x_right
      
       #***********************************************
       # Pack left and right data together
       y = DSP_IO.pack_LR(y_left,y_right)
       # Typically more DSP code here     
       #***********************************************
       # Save data for later analysis
       # accumulate a new frame of samples
       DSP_IO.DSP_capture_add_samples_stereo(y_left,
                                             y_right)
       #***********************************************
       # Convert from float back to int16
       y = y.astype(int16)
       DSP_IO.DSP_callback_toc()
       # Convert ndarray back to bytes
       return y.tobytes(), pah.pyaudio.paContinue

This app is best experienced by listening, but in picture form Figure :ref:`CrossLeftRightPanning` shows a 
series of scope captures.

.. figure:: Cross_Left_Right_Panning_app.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Cross left/right panning control: (a) launching the app in the Jupyter notebook and (b) 
   a sequence of scope screen shots as the panning slider is from 0% to 50% and then 
   to 100%. :label:`CrossLeftRightPanning`

For dissimilar left and right audio channels, the action of the slider creates a spinning effect 
when listening. It is possible to extend this app with a automation, so that a low frequency 
sinusoid or other waveform changes the panning value at a rate controlled by a slider.

FIR and IIR Filters
===================

In this example we design a high-order FIR bandpass filter using 
:code:`sk_dsp_comm.fir_design_helper` and then implement the design to operate at :math:`f_s = 48` kHz. 
Theory is compared with AD2 measurements using, again using noise excitation. When implementing 
a digital filter using frame-based processing, :code:`scipy.signal.lfilter` works nicely. The key 
is to first create a zero initial condition array :code:`zi` and hold this in a global variable. 
Each time :code:`lfilter` is used in the callback the old initial condition :code:`zi` is passed 
in, then the returned :code:`zi` is held until the next time through the callback.

.. code-block:: python

   import sk_dsp_comm.fir_design_helper as fir_d
   import scipy.signal as signal
   b = fir_d.fir_remez_bpf(2700,3200,4800,5300,
                          .5,50,48000,18)
   a = [1]
   # Set up a zero initial condition to start
   zi = signal.lfiltic(b,a,[0])

   # define callback (#2)
   def callback2(in_data, frame_count, time_info, 
                 status):
       global b, a, zi
       DSP_IO.DSP_callback_tic()
       # convert byte data to ndarray
       in_data_nda = np.frombuffer(in_data, 
                                   dtype=np.int16)
       #***********************************************
       # DSP operations here
       # Here we apply a linear filter to the input
       x = 5*in_data_nda.astype(float32)
       #y = x
       # The filter state/(memory), zi, 
       # must be maintained from frame-to-frame,
       # so hold it in a global 
       # for FIR or simple IIR use:
       y, zi = signal.lfilter(b,a,x,zi=zi) 
       # for IIR use second-order sections:
       #y, zi = signal.sosfilt(sos,x,zi=zi)     
       #***********************************************
       # Save data for later analysis
       # accumulate a new frame of samples
       DSP_IO.DSP_capture_add_samples(y) 
       #***********************************************
       # Convert from float back to int16
       y = y.astype(int16)
       DSP_IO.DSP_callback_toc()
       return y.tobytes(), pah.pyaudio.paContinue

   DSP_IO = pah.DSP_io_stream(callback2,2,2,
                              fs=48000,Tcapture=0)
   DSP_IO.interactive_stream(Tsec=0,numChan=1)

Following the call to :code:`DSP_io.intercative_stream()` the *start* button 
is clicked and the AD2 spectrum analyzer estimates the power spectrum. The estimate 
is saved as a CSV file and brought into the Jupyter notebook to overlay the 
theoretical design. The comparison results are given in Figure :ref:`FIRBPFDesignCompare`.

.. figure:: FIR_BPF_design_compare.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Cross left/right panning control: (a) launching the app in the Jupyter notebook and (b) 
   a sequence of scope screen shots as the panning slider is from 0% to 50% and then 
   to 100%. :label:`FIRBPFDesignCompare`

Excellent agreement is achieved, making the end-to-end design, implement, test very satisfying.

Three Band Equalizer
====================

Here we consider the second-order peaking filter and place three of them in cascade with a 
:code:`ipywidgets` slider used to control the gain of each filter. The peaking filter is 
used in the design of audio equalizer, where perhaps each filter is centered on on octave 
frequency spacings running from from 10 Hz Hz up to 16 kHz, or so. Each peaking filter can 
be implemented as a 2nd-order difference equation, i.e., :math:`N=2` in equation 
(:ref:`LCCDE`). The design equations for a single peaking filter are given below using 
z-transform [Opp2010]_ notation:

.. math::
   :label: peaking1

   H_{pk}(z) = C_\text{pk}\frac{1 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}

which has coefficients

.. math::
   :type: eqnarray
   :label: peaking2

   C_\text{pk} &=& \frac{1+k_q\mu}{1+k_q}\\
   k_q &=& \frac{4}{1+\mu} \tan\left(\frac{2\pi f_c/f_s}{2Q}\right) \\
   b_1 &=& \frac{-2\cos(2\pi f_c/f_s)}{1+k_q\mu} \\
   b_2 &=& \frac{1-k_q\mu}{1+k_q\mu} \\
   a_1 &=& \frac{-2\cos(2\pi f_c/f_s)}{1+k_q} \\
   a_2 &=& \frac{1 - k_q}{1+k_q}

where

.. math:: 
   :label: peaking3

   \mu = 10^{G_\text{dB}/20},\ \ Q \in [2, 10]

and :math:`f_c` is the center frequency in Hz relative to sampling rate :math:`f_s` in Hz, 
and :math:`G_\text{dB}` is the peaking filter gain in dB. Conveniently, the function 
:code:`peaking` is available in the module :code:`sk_dsp_comm.sigsys`. The app code is given 
below starting with the slider creation:

.. code-block:: python

   band1 = widgets.FloatSlider(description \
                     = '100 Hz', 
                     continuous_update = True, 
                     value = 2.0,
                     min = -20.0, 
                     max = 20.0, 
                     step = 1, 
                     orientation = 'vertical')
   band2 = widgets.FloatSlider(description \
                     = '1000 Hz', 
                     continuous_update = True, 
                     value = 10.0,
                     min = -20.0, 
                     max = 20.0, 
                     step = 1, 
                     orientation = 'vertical')
   band3 = widgets.FloatSlider(description \
                     = '8000 Hz', 
                     continuous_update = True,
                     value = -1.0,
                     min = -20.0, 
                     max = 20.0, 
                     step = 1, 
                     orientation = 'vertical')

   import sk_dsp_comm.sigsys as ss
   import scipy.signal as signal
   b_b1,a_b1 = ss.peaking(band1.value,100,Q=3.5,fs=48000)
   zi_b1 = signal.lfiltic(b_b1,a_b1,[0])
   b_b2,a_b2 = ss.peaking(band2.value,1000,Q=3.5,fs=48000)
   zi_b2 = signal.lfiltic(b_b2,a_b2,[0])
   b_b3,a_b3 = ss.peaking(band3.value,8000,Q=3.5,fs=48000)
   zi_b3 = signal.lfiltic(b_b3,a_b3,[0])
   b_12,a_12 = ss.cascade_filters(b_b1,a_b1,b_b2,a_b2)
   b_123,a_123 = ss.cascade_filters(b_12,a_12,b_b3,a_b3)
   f = logspace(log10(50),log10(10000),100)
   w,H_123 = signal.freqz(b_123,a_123,2*pi*f/48000)
   semilogx(f,20*log10(abs(H_123)))
   grid();

   # define a pass through, y = x, callback
   def callback(in_data, frame_count, time_info, 
                status):
       global zi_b1,zi_b2,zi_b3
       DSP_IO.DSP_callback_tic()
       # convert byte data to ndarray
       in_data_nda = np.frombuffer(in_data, 
                                   dtype=np.int16)
       #***********************************************
       # DSP operations here
       # Here we apply a linear filter to the input
       x = in_data_nda.astype(float32)
       #y = x
       # Design the peaking filters on-the-fly
       # and then cascade them
       b_b1,a_b1 = ss.peaking(band1.value,100,
                              Q=3.5,fs=48000)
       z1, zi_b1 = signal.lfilter(b_b1,a_b1,x,
                                  zi=zi_b1) 
       b_b2,a_b2 = ss.peaking(band2.value,1000,
                              Q=3.5,fs=48000)
       z2, zi_b2 = signal.lfilter(b_b2,a_b2,z1,
                                  zi=zi_b2)
       b_b3,a_b3 = ss.peaking(band3.value,8000,
                              Q=3.5,fs=48000)
       y, zi_b3 = signal.lfilter(b_b3,a_b3,z2,
                                 zi=zi_b3)
       #***********************************************
       # Save data for later analysis
       # accumulate a new frame of samples
       DSP_IO.DSP_capture_add_samples(y) 
       #***********************************************
       # Convert from float back to int16
       y = y.astype(int16)
       DSP_IO.DSP_callback_toc()
       # Convert ndarray back to bytes
       return y.tobytes(), pah.pyaudio.paContinue

Following the call to :code:`DSP_io.intercative_stream()` the *start* button 
is clicked and the FFT spectrum analyzer estimates the power spectrum. The estimate 
is saved as a CSV file and brought into the Jupyter notebook to overlay the 
theoretical design. The comparison results are given in Figure :ref:`ThreeBandDesignCompare`.

.. figure:: Three_Band_design_compare.pdf
   :scale: 50%
   :align: center
   :figclass: htb

   Cross left/right panning control: (a) launching the app in the Jupyter notebook and (b) 
   a sequence of scope screen shots as the panning slider is from 0% to 50% and then 
   to 100%. :label:`ThreeBandDesignCompare`

Reasonable agreement is achieved, but a listening to music is a more effective way of evaluating 
the end result. To complete the design more peaking filters should be added. 

Conclusions and Future Work
---------------------------

In this paper we have described an approach to implement real-time DSP in the Jupyter notebook. 
This real-time capability rests on top of PyAudio and the wrapper class :code:`DSP_streaming_io` 
contained in :code:`sk_dsp_comm.pyaudio_helper`. The :code:`ipywidgets` allow for interactivity 
while real-time DSP code is running. The *callback* function does the work using frame-based 
algorithms, which takes some getting used to. By working through examples we have shown that much 
can be accomplished with little coding.

A limitation of using PyAudio is the input-to-output latency. At a 48 kHz sampling rate a simple 
loop though app has around 70 ms of delay. For the application discussed in the paper latency is 
not a show stopper. 

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
.. [3p5mm] `http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ARM/3_5mm_adapter.pdf`_

.. _`Thomas Lorenser, "The DSP capabilities of ARM®  Cortex®-M4 and Cortex-M7 Processors", ARM, November 2016.`: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjRhqvO25XbAhUDx2MKHRiJBeYQFggnMAA&url=https%3A%2F%2Fcommunity.arm.com%2Fcfs-file%2F__key%2Ftelligent-evolution-components-attachments%2F01-2142-00-00-00-00-73-48%2FARM-white-paper-_2D00_-DSP-capabilities-of-Cortex_2D00_M4-and-Cortex_2D00_M7.pdf&usg=AOvVaw1jyK7ExAE-2YqmaEzRSx8z
.. _`https://docs.scipy.org/doc/scipy/reference/signal.html`: https://docs.scipy.org/doc/scipy/reference/signal.html
.. _`https://github.com/mwickert/scikit-dsp-comm`: https://github.com/mwickert/scikit-dsp-comm
.. _`https://people.csail.mit.edu/hubert/pyaudio/`: https://people.csail.mit.edu/hubert/pyaudio/
.. _`http://www.portaudio.com/`: http://www.portaudio.com/
.. _`https://github.com/jupyter-widgets/ipywidgets`: https://github.com/jupyter-widgets/ipywidgets
.. _`https://store.digilentinc.com/analog-discovery-2-100msps-usb-oscilloscope-logic-analyzer-and-variable-power-supply/`: https://store.digilentinc.com/analog-discovery-2-100msps-usb-oscilloscope-logic-analyzer-and-variable-power-supply/
.. _`http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ARM/3_5mm_adapter.pdf`: http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ARM/3_5mm_adapter.pdf
