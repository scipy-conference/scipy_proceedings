:author: Chiranth Siddappa
:email: csiddapp@uccs.edu
:institution: University of Colorado Colorado Springs
:bibliography: caf_summary

:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs

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

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


