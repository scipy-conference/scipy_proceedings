:author: Marian-Leontin Pop
:email: popmarianleontin@gmail.com
:institution: Technical University of Cluj-Napoca

:author: Szilard Molnar
:email: molnarszilard10@gmail.com
:institution: Technical University of Cluj-Napoca

:author: Alexandru Pop
:email: Alexandru.Pop@aut.utcluj.ro
:institution: Technical University of Cluj-Napoca

:author: Benjamin Kelenyi
:email: Benjamin.Kelenyi@aut.utcluj.ro
:institution: Technical University of Cluj-Napoca

:author: Levente Tamas
:email: Levente.Tamas@aut.utcluj.ro
:institution: Technical University of Cluj-Napoca
:corresponding:

:author: Andrei Cozma
:email: Andrei.Cozma@analog.com
:institution: Analog Devices International


:bibliography: mybib



:video: https://www.youtube.com/watch?v=kANXhHwFrCo

------------------------------
CNN Based ToF Image Processing
------------------------------

.. class:: abstract

   ...followed by a real life application using artificial intelligence to estimate the human body pose 
   for applications such as gesture recognition, movement direction estimation or physical exercises monitoring. 
   The entire human pose estimation application implementation flow is described, from generating and training the pose estimation 
   AI model using Keras and TensorRT to deploying and running live on an Nvidia Xavier NX platform receiving data from an Analog Devices ToF camera.

.. class:: keywords

   transfer learning, ToF, python

Introduction
------------

Last years the evolution of CNN also affected the way in which the ToF specific images are processed...


ToF specific imaging
++++++++++++++++++++
.. TL part

Low level ToF image pre-processing (PCL based)
++++++++++++++++++++++++++++++++++++++++++++++
.. MSz part
In ToFNest we are approximating surface normals from depth images, recorded with Time-of-Flight cameras. The approximation is done using a neural network. The base of our neural network is the PyTorch library, since the whole process is done using Python 3.6 as our programming language.

The main pipeline of the data was the following: first we read the images with opencv, then we prepare them with numpy. From a numpy array it is easy to convert it to a torch tensor on the GPU, that then creates the predictions about the surface normals. An example of the prediction can be seen in the next image, where the direction of the normal vectors are decoded with RGB images. The results were accurate relative to other techniques, but the time was much less.

.. image:: ToFNest.png
  :width: 400
  :alt: Alternative text

Furthermore, in order to get a real-time visualization about the predictions, we used rospy to read the images from ROS topics, and also to publish the normal estimation values to another ROS topic, that we could visualize using Rviz. This can be seen in the demo video.

This whole pipeline and network, with some minor modifications can be also used to  smoothen the depth image, thus making the point cloud smoother as well.


PCL based pipeline for ToF.


CNN based solutions
-------------------
Jetson based solutions


Person detection from IR imaging
++++++++++++++++++++++++++++++++
.. BK part

Jetson detection with ToF node

Action recognition from IR images
+++++++++++++++++++++++++++++++++
.. PM part

DS demo with skeleton detection

Volumetric estimates for depth images
+++++++++++++++++++++++++++++++++++++
.. PA part

Volume estimation using enhanced planar/corner detections


Use-cases
---------
Short description of demos

:cite:`hume48`


Conclusion
----------
Summary 



Acknowledgement
---------------
The authors are thankful for the support of Analog Devices Romania, 
for the equipment list (cameras, embedded devices, GPUs) offered as support 
to this work. 
This work was financially supported by the Romanian National Authority 
for Scientific Research, CNCS-UEFISCDI, project number PN-III-P2-2.1-PTE-2019-0367.
The authors are thankful for the generous donation from NVIDIA corporation for supporting this research.