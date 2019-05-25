:author: Chiranth Siddappa
:email: csiddapp@uccs.edu
:institution: University of Colorado Colorado Springs

:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs
:bibliography: caf

:video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------
CAF Implementation on FPGA Using Python Tools
---------------------------------------------

.. class:: abstract

.. include:: ./papers/chiranth_siddappa/abstract.txt

.. class:: keywords

.. include:: ./papers/chiranth_siddappa/keywords.txt

Introduction
============

The basic algorithm for calculating the complex ambiguity function for time difference of arrival and frequency offset
has been well known since the early 1980's :cite:`stein-orig`. In many radio frequency applications, there is a need to
find a time lag of the signal or the frequency offset of a signal.
The reader would be familiar with a form of frequency offset known as Doppler as a common example.
However, the CAF is the joint time offset and frequency offset generalization.
The CAF was mainly used first for radar and sonar type processing for locating objects in a method known as active echo
location :cite:`knight-sonar`.
In this scenario, a matched filter design would be used to ensure that the signals match :cite:`weiss-waves`.
More commonly with newer radio frequency systems such as GPS, similar but orthogonal signals are transmitted in the same
frequency range.
Because of the property of orthogonal signals not cross correlating they do not collide with one another, and they are
an optimal signal type for testing this application :cite:`ZiemerComm`.

Motivation
==========
The CAF has many practical applications, the more traditional being the before-mentioned radar and sonar type systems.
Another common use case is in image processing.
In image processing, cross-correlations are used to determine similarity between images, and dot products are used as a
means of filtering.
This helped in my understanding of the use of the dot product in cross-correlations which is the basis of this project.
The main motivator for this project comes from the use of the CAF at my workplace where we work on satellite
communications systems.
In the particular case of geolocation systems, the use of collectors and reference emitters are used to create
geometries that will allow for the detection of Doppler and movement in the signal.
This method of calculation proves to be very computationally challenging due to the high mathematical intensity and
cannot be reduced.
Currently GPU's have been employed as the main workhorse but the use of the FPGA has always been an attractive and
feasible option due to the hardware options that are available.
To geolocate a signal's location the Doppler is used to calculate a frequency difference of arrival (FDOA) which
represents a satellite's drift.
Then, cross correlations can be used to determine the time delay by denoting the peak of the resulting output as a time
delay of arrival (TDOA).
The characteristics of the captured signals will change based on the type of signal, which motivates the need to ensure
that the resulting HDL can also be produced to match necessary configurations.
This became a project goal motivated off work done by other projects to be able to produce code in other languages
:cite:`codegen`.
Thus, the solution provided must be able to be reconfigured based off of different needs.
The processing for this system will be targeted to a PYNQ board by Xilinx, but has been designed such that it can be
synthesized to any target device.
All Verilog HDL modules that are produced by the Python classes conform to the AXI bus standards of interfacing
:cite:`axi4`. This allows for a streamlined plug and play connection between all the modules and is the basis of the
templating that is implemented with the help of Jinja.

Starting Point
==============
The main concepts necessary for the understanding of the CAF are topics that are covered in Modern Digital Signal
Processing, Communication Systems, and a digital design course. These concepts would be the Fast Fourier Transform
(FFT), integration in both infinite and discrete forms, frequency shifting, and digital design.
The main basis of this project then is to show a working implementation of digital design HDL modules implementing the
logic accurately with this given knowledge. Given the mathematical basis of this project, it is crucial to have a way to
test implementations against theory. This is the motivation for the discussion of using Python to help generate code
and test benches.

Project Overview
================
The goal of this project was to implement the CAF in an HDL such that the end product can be targeted to any device.
The execution of this goal was taken as a bottom up design approach, and as such the discussion starts from small
elements to larger ones. The steps taken were in the following order:

#. Obtain and generate a working CAF simulation
#. Break simulation into workable modules
#. Design modules
#. Verify and generate with test benches
#. Assemble larger modules
#. Synthesize and Implement using Vivado for the PYNQ-Z1 board

Complex Ambiguity Function
==========================

An example of the signal path in the satellite receiver scenario is described by Fig. :ref:`satellite-diagram-example`.
In this case, an emitted signal is sent to a satellite, and then captured by an RF receiver.
Some amount of offset is expected to have happened during the physical relay of the signal back to a receiver within the
broadcast area of the satellite.
The signal is then downconverted and filtered, and then sent to the CAF via a capture buffer.
While a signal is sent through an upconverter and relayed to the satellite, a copy of the same signal must be stored
away as a reference to compute the TDOA and FDOA.
Both the reference and capture blocks are abstractions, and have individual modules written in Verilog to handle the
storage of these signals.

Another very specific example of the satellite receiver scenario is described by Fig. :ref:`satellite-diagram-example-gps`.
In this scenario, we see that no emitter exists, yet a reference signal is able to be sent to the CAF for TDOA and FDOA
calculations. This is because GPS signals use a PRN sequence as ranging codes, and the taps for the signals are provided
to the user :cite:`gpsgov`.
This provides a significant processing gain as the expected sequence can be computed in real time or stored locally.
This project takes advantage of these signals through the use of gps-helper :cite:`gps-helper`.

.. figure:: satellite_diagram_example.png

   Satellite Block Diagram for Emitter and Receiver. :label:`satellite-diagram-example`

.. figure:: satellite_diagram_example_gps.png

   Satellite Block Diagram for CAF with GPS Signal. :label:`satellite-diagram-example-gps`

As a basis for what the rest of this paper is describing, an overview of the CAF and the various forms of computing are provided.

The general form of the CAF is:

.. math::

   \chi (\tau ,f)=\int _{{-\infty }}^{\infty }s(t)s^{*}(t-\tau )e^{{i2\pi (f/f_s)t}}\,dt,\ \frac{-f_s}{2} < f < \frac{f_s}{2}

The equation describes both a time offset :math:`\tau` and a frequency offset :math:`f` that are used to create a
surface. The frequency shift :math:`f` is bounded by half the sampling rate.
The discrete form is a little simpler, and lends itself to the direct implementation :cite:`hartwellcaf`:

.. math::

   \chi(k, f) = \sum _{n=0}^{N-1}s[n]s^{*}[n - k]e^{i2\pi (f/f_s)(n/N)},\ \frac{-f_s}{2} < f < \frac{f_s}{2}

where :math:`N` is the signal capture window length, :math:`f_s` is the sampling rate in Hz making :math:`f` have units
of Hz and :math:`kD` is a discrete time offset in samples with sample period :math:`1/f_s`. In both the continuous and
discrete-time domains, :math:`\chi` is a function of both time offset and frequency offset.
The symbol :math:`s` represents the signal in question, generally considered to be the reference signal.
The accompanying :math:`s^{*}` is the complex conjugate and time shifted signal. As an example, a signal that was not time
shifted would simply be the autocorrelation :cite:`ZiemerComm`. It is referred to as the captured signal in this context,
and it is the signal that is used to determine both the time and frequency offset. To determine this offset, we are
attempting to shift the signal as close as possible to the original reference signal.
The time offset is what allows for the computation of a TDOA, and the frequency offset is what allows for the
computation of the FDOA.
In this implementation, the frequency offset is created by a signal generator and a complex multiply module that are
both configurable.
Once this offset has been applied, a cross-correlation is applied directly in the form of the dot product.
This eliminates the costly implementation case where an FFT and an inverse FFT are used to produce a result.
The signal generator can supply a specified frequency step and accuracy with configuration of the signal generator class
:cite:`caf-verilog`. An example of the signal generator is shown in Fig. :ref:`dds-one`.
The resulting spectrum is shown in Fig. :ref:`sig-gen`. This satisfies the frequency (:math:`f`) portion of the equation.
The complex multiply module is similarly configurable for different bit widths through the complex multiply generator
class :cite:`caf-verilog`.
An example CAF surface is provided in Fig. :ref:`caf-surface-example` showing how the energy of the signal is spread over
both frequency and time. This type of visualization is very useful for real-world signals with associated noise. In
this project, care was taken in truncation choices to ensure that the correlation summation ensures signal energy
retention.
In this project, the CAF module that has been implemented will return a time offset index and frequency offset index
back to the user based off provided build parameters shown in Listing :ref:`caf-listing`, described in a later
section for the CAF Module.
When writing the module, all simulation and testing was done at the sample by sample level to ensure validity so the CAF
surface was not used in testing. A method for computing the CAF using the dot product and frequency shifts has been
published to the package. This implementation is specific to this project in that it uses a sample size that is twice
that of the reference signal for the computation. A sample output slice will be shown in the Experiments section for the
CAF module in Fig. :ref:`caf-test-signal`.

.. figure:: caf_surface_example.png

   CAF Surface Example. :label:`caf-surface-example`

Hardware
========

The targeted hardware for this project is the Zynq processor on the PYNQ-Z1 board. However, this project is fully
synthesizable and should be able to be targeted for any other Xilinx board.


Python and PYNQ
---------------


The PYNQ development board designed by Xilinx provides a Zynq chip which has an ARM CPU running at 650 MHz and an FPGA
fabric that is programmable via an overlay :cite:`pynq`.
This performance allows for a RTOS to be run on the CPU.
The type of scheduler that is run on the OS is the main driver for whether it is an RTOS or not.
The RTOS in this case is Ubuntu, and hosts a Jupyter notebook to program and interface with the FPGA fabric using an
overlay.
This overlay contains mappings for ports and interfaces between the fabric and the CPU.
This functionality is very unique in that both an ARM core and a fabric are on the same board.
As shown by Fig. :ref:`pynq-overlay` the overlay sits between the processing system (CPU) and the programmable logic
(FPGA).
The overlay is represented by the yellow background with labels "Custom" and "Accelerator" and shows how the overlay is
a communication layer between the processing system and the programmable logic.

.. figure:: zynq_block_diagram.jpg

   The PYNQ processing overlay diagram. :cite:`pynq` :label:`pynq-overlay`

It also contains a bitfile that will properly configure the FPGA :cite:`bitfile`.
This bitfile is generated through the Vivado Design Suite that is provided by Xilinx.
If a different device is to be targeted, the new target device would also be configured using this design methodology of
creating a bitfile.

The Jupyter notebook itself is considered an interactive computing pool providing an interactive interface to do
computation and prototyping. In this implementation it is meant to be an easier way for a non-hardware oriented person
to be able to access a computational accelerator designed by a hardware engineer :cite:`pynq`.

A diagram of the processing and the programmable logic is shown in Fig. :ref:`pspl`. The processor system is the
Cortex-A9 processor that is running at 650MHz with 512MB of DDR3 RAM. The FPGA is a Zynq XC7020 part which has 13,300
logic slices, 53,200 6-input LUTs, 160,400 flip-flops, 630KB of block RAM, and 220 DSP slices. Later, a usage report is
provided with a description of how the logic was optimized to make use of these primitives. It is possible to access the
DRAM from the programmable logic (FPGA) through an AXI IP Core.

.. figure:: PSPL.png

   The PYNQ processing overlay diagram. :cite:`pynq` :label:`pspl`


Software
========

Xilinx Vivado WebPack 2018.2
----------------------------

The Vivado design tool provides a simulator along with the ability to synthesize and implement the design.
For this project, this built-in simulator was used exclusively.
Other simulators were not chosen because the other target devices that this project seeks to be implemented on are
likely to also be Xilinx products.
The tool is free to download for anyone to use, and allows the hardware engineer to develop and synthesize HDL designs
for Xilinx FPGA's.
There is also a Software Development Kit that allows an engineer to write in C code.
For this project, all modules are written in Verilog.
This was done because of the need to instantiate multiple submodules that provide functionality together.
When running the synthesis tool, the output was very useful in helping make incremental design changes to fully optimize
the board.
Although none were used in this project, Xilinx does offer many free IP Cores that can be used in designs. They are
black boxes that can be used in both simulation and the final implementation in HDL and block designs.

Python and Jupyter
------------------


This project made extensive use of the Python ecosystem through the use of pip, Jupyter, and many other packages. The
reader is encouraged to view the :code:`caf-verilog` source code and view the releases that have been made
:cite:`caf-verilog`.
All of the simulation source code and modules that are used in this project are generated through the Python package
that has been released.
While working on this project, many design changes occurred and test verification was done by loading the output files
and investigating them in the Jupyter notebook.
When designing modules, a first test of what a signal should look like when operated on was done using the interactive
plotting ability that is provided :cite:`jupyter`.
The generation of the modules was done using Jinja which provides both template parsing and rendering :cite:`jinja`.
Whenever a simulated signal was changed, instead of having to write out a file or test bench by hand, a template was
used to create the output and render it to the simulation directory.
The signals that are used to create the signal generator were first quantized by using the NumPy library and then
written to a file that gets used a memory buffer in the signal generator :cite:`numpydev`.
Most of the mathematical operations that are implemented were first verified using this library.
This project requires the use of orthogonal signals to ensure that the spectral density that is being tested is isolated
from the others. This was possible using the gps-helper module that implements the GPS gold codes that are orthogonal
PRN sequences :cite:`gps-helper`.


Experiments
===========

As the description of the CAF itself was discussed previously, the rest of this document will walk through the steps
taken to produce each module and the design decisions that were made and changed in order to produce a working
simulation. This project implemented the CAF as a bottom up approach, so the document will start with analyzing signals
and end with the implemented design and usage report.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.
