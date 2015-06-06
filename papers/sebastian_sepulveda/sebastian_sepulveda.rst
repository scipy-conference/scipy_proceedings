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
 real time physiological signals, such as electrocardiography,
 electromyography and human movement. The software is also capable of doing real time processing,
 such as filtering and spectral estimation. The software is open source,
 extensible, multi-platform and has been tested on different Linux
 distributions, including conventional PC, Mac and the RaspberryPi (ARM
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


A common task in biomedical research is to record and visualize in real time physiological signals. Although there are several options to do this, they are commonly based on  proprietary tools, associated to a particular signal acquisition device vendor. This work presents an open source software, written in Python, to visualize and record in real time physiological signals, such as electrocardiography, electromyography and human movement. The software is also capable of doing real time processing, such as filtering and spectral estimation. The software is open source [#]_  and extensible. It is easy to add new signal processing tasks and to use different signal sources (serial, Bluetooth, Sockets, etc.), and customize the user interface for the applications needs.

.. [#] Available at https://github.com/ssepulveda/RTGraph.

The main objective of the software is to display in real time multiple signals and to export them to a file. In the current implementation, The communication between the software and the acquisition device is through the serial port, and it is implemented using the PySerial library. Other communication protocols can be easily added. The real time display of the signals  is implemented using the PyQtGraph library [#]_. The software has a multi-process architecture, based on the multiprocessing Python standard library. This allows having concurrent processes for receiving, processing, and displaying the data. Signal processing tasks, such as spectral estimation, are based on the SciPy stack. This architecture assures that no data is loosed and a fast response of the user interface. 

.. [#] Available at http://www.pyqtgraph.org.

The software is multi-platform and runs in any machine and OS where Python and the corresponding dependencies can be installed. The software has been tested on different Linux distributions, including conventional PC, Mac and the RaspberryPi (ARM architecture).

Software architecture
---------------------

The applications described in this work can be classified as a "data logger". A data logger needs to acquire a stream of data, add a time stamp to the data (if required), and export the time-stamped data to a file in a known file format, such as comma separated value (CSV) format. Optionally, the application can do some processing (filtering, spectral estimation, etc.) before saving the data. In addition, it is also useful to be able to visualize, in real time, the stream of data. 

When developing, evaluating, or validating a new hardware or software, it is important to have  control of the outcome of the algorithms and the fidelity and performance of the data acquisition process. In particular, in the field of Biomedical Engineering, the acquisition and signal processing of biological signals needs to be reliable and with a tight control over the sampling frequency. It is also fundamental to ensure that no data is lost during the acquisition and logging process. From a practical point of view, having to wait for the data to be stored before visualizing it (possibly in another program) is cumbersome and tedious, slowing down the development process. For these reasons, in this work we present a program able to: receive data from a variety of sources (serial port, Bluetooh, Zigbee, Sockets, etc.); process and visualize the data in real time; and to record the data in a file.


The first version of this program was developed for a research on Biomechanical engineering.  This research involves logging, processing, and real time displaying of the signals generated by a nine degrees of freedom inertial measurement unit (9DOF-IMU). This requires acquiring nine signals with a sampling rate of at least  100 Hz. Six additional signals are computed through a sensor fusion algorithm. A total of 15 signals are displayed and exported as a CSV file. We designed the architecture of the program with these requirements in mind.

Viewing what is being acquired is also important, to corroborate that the data being acquired is actually what is being expected. In that manner, a data logger capable of also plot the data, in real time is important tool. A simple, yet versatile application, capable of acquire data from sources a variety of sources and in different instances, over different platforms, and free, is a tool not easy to find.



A plotting library selection
============================
There are lots of tools and libraries capable of doing good plots. As Luke Campagnola [A]_ stated in the project's page, the main reasons for choosing PyQtGraph are Speed, portability and feature rich. First, we selected an UI framework to start working. We choose the Qt Framework, mainly because of the uniform look and feel that can be achieved cross platform, and the ability to import widgets.

The first attempts where done with the SciPy stack, using Matplotlib [B]_. This worked out of the box, with a not much complicated way to include the plot in an UI, and interact with other UI elements. For one data stream, the plot worked easily. But, after starting to use more data and signal processing, we ended noticing how Matplotlib is not intended to handle signals in real time (having an update of the plot of at least 30 Hz). PyQwt was the next to test, which give an easier integration with QT using QT Designer. Compared to Matplotlib design, PyQwt was left behind. From there, we started a research to find others libraries that where Python compatible and could used with Qt Framework in an easier way.

PyQtGraph, in difference to the other used libraries, was developed to do tasks like real time plotting of signals and also image processing. It's deeply integrated with Qt, giving easy and full compatibility with the Qt Framework, including GUI design interfaces like Qt Designer. From there, configuring and customizing the plots could be easily achieved with the live examples included in the library. There is also a integration with Numpy, giving those big performance improvement in the processing of the data, and the management of the Numpy data types. The graphical interface gives, as matter of fact, simple signal processing tools, like Fast Fourier Transform or Median calculation on signals in real time. This library, itself is a big part of this work and resolves almost all the plotting problems. But, the data management is not completely clear, and not an easy task for initial users to implement.

After using this library, and pushing his limits, we still couldn't achieve a reliable acquisition of data. Limitations of the threads (a good link to the 1 core thread of python here !), and the interaction between UI (main) threads and background threads, to acquire data, using Python's threads or QtThreads, wasn't giving the best results for heavy data transfers, both in speed and amount of data.


Threading vs Multiprocessing
============================
The global interpreter lock (GIL) [C]_ prevents threads to take advantage of multiprocessor systems. In short, it means that a mutex controls the access from the threads to the memory. There ways to workaround this, in fact, using Numpy itself, witch doesn't run under GIL, will improve the performance. But, in this specific application, there is a necessity to get most of the platform, to ensure the best processing, plotting and logging of the data without any loss.

The multiprocessing library workaround this problem by using subprocess instead of threads [D]_. This gives access to all the resources available on the platform, plus, letting the host OS to handle the subprocesses. With this library, the platform itself is the limit.

The remaining problematic, is to orchestrate the communication of the process, and more important, the communication between them. There are problems of synchronization of the data and also, access to their memory space, a subprocess shouldn't be able to access a memory space of other process. For this, there is a specific wait to communicate the threads, through queues or pipes.

The actual architecture
=======================

Figure :ref:`figSWarch` shows a diagram of the software architecture.

.. figure:: sw_architecture.pdf

   Diagram of the software architecture. There are two independent processes. The communication process reads the incoming data stream, parse it, add a timestamp (if necessary), and put the processed data into a queue. The main process reads the data from the queue, process the data, and then update the plot and log the data into a file. :label:`figSWarch` 


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

It is easy to modify by other users. Mention Lobos' application (is that the case?). 

Future work: do the signal processing in a different process, to take advantages of the multiple cores.

Acknowledgments
---------------

This research was partially supported by the Advanced Center for Electrical and
Electronic Engineering, Basal Project FB0008, Conicyt.

References
----------
.. [A] L. Campagnola. *PyQtGraph. Scientific Graphics and GUI Library for Python*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [B] J. D. Hunter. *Matplotlib: A 2D graphics environment*,
			Computing In Science \& Engineering, 9(3):90-95, IEEE COMPUTER SOC, 2007. http://dx.doi.org/10.5281/zenodo.15423

.. [C] http://en.wikipedia.org/wiki/Global_Interpreter_Lock

.. [D] https://docs.python.org/2/library/multiprocessing.html


