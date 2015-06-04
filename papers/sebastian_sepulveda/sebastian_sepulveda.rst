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
 electromiography. The software is also capable of doing real time processing,
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


A common task in biomedical research is to visualize and record physiological signals in real time. Although there are several options to do this, they are commonly proprietary tools, associated to a particular signal acquisition device vendor. This work presents an open source software, written in Python, to visualize and record in real time physiological signals, such as electrocardiography and electromiography. The software is also capable of doing real time processing, such as filtering and spectral estimation. The software is open source (http://bit.ly/1CHJjgZ) and extensible. It is easy to add new signal processing tasks and to use different signal sources (serial, bluetooth, sockets, etc.). See the following links for two examples where the software is used to acquire EMG signals from different devices: http://bit.ly/1BHObxL, http://bit.ly/1Ex0Ydy.

The main objective of the software is to display in real time multiple signals and to export them to a file. The communication between the software and the acquisition device is through the serial port, and it is implemented using the PySerial library. Other communication protocols are also implemented. The real time display of the signals  is implemented using the PyQtGraph library. The software has a multi-process architecture, based on the multiprocessing standard library. This allows having concurrent processes for receiving and processing the data. Signal processing tasks, such as spectral estimation, are based on the SciPy stack. This architecture assures that no data is loosed and a fast response of the user interface. 

The software is multi-platform and runs in any machine and OS where Python and the corresponding dependencies can be installed. The software has been tested on different Linux distributions, including conventional PCs and the RaspberryPi (ARM architecture).

Software architecture
---------------------


The applications described is under the category of applications called "data loggers". Such applications are fairly simple, both in implementation and architecture. The activities of a data logger consists in adquire an stream of data, probably due some basic processing with it, and then add a timestamp to export it to a file, following simple and universal schemes as CSV.
In the process of develop, evaluate or validate a new hardware of software, is important to evaluate both the results of an algorithm, the correct adquisition of the data and also, the performance reached from different transfer methods.


Figure :ref:`figSWarch` shows a diagram of the software architecture.

.. figure:: sw_architecture.pdf

   Diagram of the software architecure. :label:`figSWarch`

Programming details
-------------------

Relevant code snippets goes here. Perhaps this is unnecesary.

Results
-------

Figure xx shows a screenshot of the program showing an EMG signal.
Figure yy shows a photo of the device connected through the serial port.


Conclusions
-----------

We are awsome.

Acknoledgements
---------------

This research was partially suported by the Advanced Center for Electrical and
Electronic Engineering, Basal Project FB0008, Conicyt.

References
----------
.. .. [Atr03] P. Atreides. *How to catch a sandworm*,
..           Transactions on Terraforming, 21(3):261-300, August 2003.


