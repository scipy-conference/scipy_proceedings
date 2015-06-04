:author: Sebastián Sepúlveda
:email: ssepulveda.sm@gmail.com
:institution: Escuela de Ingeniería Civil Biomédica, Facultad de Ingeniería, Universidad de Valparaiso

:author: Pablo Reyes
:email: pablo.reyes@uv.cl
:institution: Escuela de Ingeniería Civil Biomédica, Facultad de Ingeniería, Universidad de Valparaiso

:author: Alejandro Weinstein
:email: alejandro.weinstein@uv.cl
:institution: Escuela de Ingeniería Civil Biomédica, Facultad de Ingeniería, Universidad de Valparaiso

:video: http://bit.ly/1BHObxL
:video: http://bit.ly/1Ex0Ydy

------------------------------------------------
Visualizing physiological signals in real time
------------------------------------------------

.. class:: abstract

 This work presents a software, written in Python, to visualize and record in
 real time physiological signals, such as electrocardiography and
 electromyography. The software is also capable of doing real time processing,
 such as filtering and spectral estimation. The software is open source,
 extensible, multi-platform and has been tested on different Linux
 distributions, including conventional PCs and the RaspberryPi (ARM
 architecture). It leverages the use of several libraries, including PyQtGraph
 and the SciPy/NumPy stack.

.. class:: keywords

   real time processing, visualization, signal processing


.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.


Introduction
------------


A common task in biomedical research is to visualize and record physiological signals in real time. Although there are several options to do this, they are commonly based on  proprietary tools, associated to a particular signal acquisition device vendor. This work presents an open source software, written in Python, to visualize and record in real time physiological signals, such as electrocardiography and electromyography. The software is also capable of doing real time processing, such as filtering and spectral estimation. The software is open source [#]_  and extensible. It is easy to add new signal processing tasks and to use different signal sources (serial, Bluetooth, sockets, etc.). 

.. [#] A git repository is available at https://github.com/ssepulveda/RTGraph.

The main objective of the software is to display in real time multiple signals and to export them to a file. In the current implementation, The communication between the software and the acquisition device is through the serial port, and it is implemented using the PySerial library. Other communication protocols cab be easily added. The real time display of the signals  is implemented using the PyQtGraph library. The software has a multi-process architecture, based on the multiprocessing Python standard library. This allows having concurrent processes for receiving, processing, and displaying the data. Signal processing tasks, such as spectral estimation, are based on the SciPy stack. This architecture assures that no data is loosed and a fast response of the user interface. 

The software is multi-platform and runs in any machine and OS where Python and the corresponding dependencies can be installed. The software has been tested on different Linux distributions, including conventional PCs and the RaspberryPi (ARM architecture).

Software architecture
---------------------


The applications described is under the category of commonly "data loggers". Such applications are fairly simple, both in implementation and architecture. The activities of a data logger consists in acquire an stream of data, probably due some basic processing with it, and then add a timestamps to export it to a file, following simple and universal schemes such as CSV.
In the process of develop, evaluate or validate a new hardware of software, is important to control both the results of an algorithm, the correct acquisition of the data, and also, the performance reached from different transfer methods.

In Biomedical Engineering, the signal processing of biological signals needs to be reliable, whit specific control over the sampling frequency, first, and also, ensure that no data is lost in the process of acquisition and logging. Having to wait for the data to be stored and then viewed, or imported to other programs for analysis, could lead in a waste of time, and a tedious work that could make the final objective, from evaluating the results of a prototype of a medical device, to complications such as "is my CSV file comma separated values or dot-comma separated values?".

View what is being acquired is also important, to corroborate that the data being acquired is actually what is being expected. In that manner, a data logger capable of also plot the data, in real time is important tool. A simple, yet versatile application, capable of acquire data from sources like serial ports, Bluetooh or Zigbee or TCP/IP packets.

This application was developed to have a real time viewer and logger for an 9DOF IMU data acquisition, with the objective of reach the sample rates higher than 100 Hz. In consideration of that the data would be streamed as CSV, and a total of 9 values where transmitted, plus some algorithms to do sensor fusion to estimate orientation and other calculations, we where looking to plot and process 15 signals, in real time with logging, which, despite the simple of the application, it wasn't an easy task to accomplish.

A plotting library selection
============================
To start with, it was necessary to evaluate what plotting libraries where capable of...


Threading vs Multiprocessing
============================
There are know limitations of python regarding to threads...

The actual architecture
=======================

Figure :ref:`figSWarch` shows a diagram of the software architecture.

.. figure:: sw_architecture.pdf

   Diagram of the software architecture. :label:`figSWarch`

Programming details
-------------------

Relevant code snippets goes here. Perhaps this is unnecessary.

Results
-------

Figure xx shows a screenshot of the program showing an EMG signal.

Figure yy shows a photo of the device connected through the serial port.

See the following links for two examples where the software is used to acquire EMG signals from different devices: http://bit.ly/1BHObxL, http://bit.ly/1Ex0Ydy.


Conclusions
-----------

We are awesome.

Acknowledgments
---------------

This research was partially supported by the Advanced Center for Electrical and
Electronic Engineering, Basal Project FB0008, Conicyt.

References
----------
.. .. [Atr03] P. Atreides. *How to catch a sandworm*,
..           Transactions on Terraforming, 21(3):261-300, August 2003.


