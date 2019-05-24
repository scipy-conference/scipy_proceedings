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
------------

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
----------
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
--------------
The main concepts necessary for the understanding of the CAF are topics that are covered in Modern Digital Signal
Processing, Communication Systems, and a digital design course. These concepts would be the Fast Fourier Transform
(FFT), integration in both infinite and discrete forms, frequency shifting, and digital design.
The main basis of this project then is to show a working implementation of digital design HDL modules implementing the
logic accurately with this given knowledge. Given the mathematical basis of this project, it is crucial to have a way to
test implementations against theory. This is the motivation for the discussion of using Python to help generate code
and test benches.

Project Overview
----------------
The goal of this project was to implement the CAF in an HDL such that the end product can be targeted to any device.
The execution of this goal was taken as a bottom up design approach, and as such the discussion starts from small
elements to larger ones. The steps taken were in the following order:

#. Obtain and generate a working CAF simulation
#. Break simulation into workable modules
#. Design modules
#. Verify and generate with test benches
#. Assemble larger modules
#. Synthesize and Implement using Vivado for the PYNQ-Z1 board

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


