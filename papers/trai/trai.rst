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

   The current report focuses on ToF specific data processing followed by a real life application using artificial intelligence to estimate the human body pose 
   for applications such as gesture recognition, movement direction estimation or physical exercises monitoring. 
   The entire human pose estimation application implementation flow is described, from generating and training the pose estimation 
   AI model using Keras and TensorRT to deploying and running live on an Nvidia Xavier NX platform receiving data from an Analog Devices ToF camera.

.. class:: keywords

   transfer learning, ToF, python

Introduction
------------
   
Last years the evolution of deep neuronal networks also affected the way in which the Time of Flight (ToF) specific images are processed. The images from the ToF cameras are usually obtained as synchronized depth and infrared (IR) image pairs.
The customization of the existing deep nets to the IR and depth images allows us to reuse the existing models and techniques from this emerging domain. The applications targeted are ranging from persond detection, counting, activity analysis to volumetric measurements, mapping and navigation with mobile agents.
In the following parts the introduction to the specific ToF imaging, custom data processing and CNN based solutions are presented.
   
.. figure:: tof.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: ToF camera overview

  Exemplification of ToF camera :label:`tof`


ToF specific imaging
++++++++++++++++++++
.. TL part

The 2D image processing part
is a customized IR image module based on transfer learning for bounding box estimation, 
skeleton extraction and hardware specific model translation.
The latter is relevant in order to have a light-weight embedded solution running on limited floating-point precision hardware platforms such as Jetson Nvidia Family. 
As the existing CNN models are mainly with the focus on colour images, thus ones has to adopt transfer learning as a method to finetune the existing CNN models such as VGG, MobileNet for the infrared or depth images specific to ToF cameras. 
This solution seemed to be effective in terms of precision and runtime on embedded devices (e.g Jetson Nx or AGX). 
For the skeleton detection part we relied on the real-time tensorflow optimized module for the Jetson product family, however for the generic GPU enabled devices we had to tailor our models since these are custom solutions.


Custom pipeline for ToF data
++++++++++++++++++++++++++++
The main role of the depth image preprocessing part is the filtering and bounding box estimation for the 3D ROI. 
The filtering is essential for the embedded device in order to reduce the computational overload. 
For the filtering pipeline we considered three interconnected filters: voxel, pass-through and outlier filter as this is visible in Figure below. All these implementations are open source library  based variants.

   
.. figure:: filters.png
  :width: 800
  :height: 400
  :scale: 80%
  :align: center
  :alt: Alternative text

  Processing pipeline for ToF camera :label:`filters`

Low level ToF image pre-processing - Tofnest
++++++++++++++++++++++++++++++++++++++++++++
.. MSz part


In ToFNest we are approximating surface normals from depth images, recorded with Time-of-Flight cameras. The approximation is done using a neural network. The base of our neural network is the PyTorch library, since the whole process is done using Python 3.6 as our programming language. Using PyTorch we have created a Feature Pyramid Network type model (:cite:’FPN2017’).

The main pipeline of the data was the following: first we read the depth images with opencv (alongside the depth information we could also use the infrared information or the rgb information from the camera as well, thus adding more information to work with), then we prepare them with numpy. From a numpy array it is easy to convert it to a torch tensor on the GPU, which then creates the predictions about the surface normals. An example of the prediction can be seen in the next image, where the direction of the normal vectors are decoded with RGB images. 

.. figure:: ToFNest.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  Exemplification of ToF normal estimation :label:`tofnest`

The results were accurate relative to other techniques, but the time was much less. The time being less means that at least 100 times faster. This can be due to the fact, that this method works with images, instead of point clouds as other methods do. This makes it much faster.

Our method was evaluated by verifying only the angles between the lines, not the exact directions of the vectors (this was the case in the other methods as well), but we can train that, although the results are going to get worse.

Furthermore, in order to get a real-time visualization about the predictions, we used rospy to read the images from ROS topics, and also to publish the normal estimation values to another ROS topic, that we could visualize using Rviz. This can be seen in the demo video.

Low level ToF image pre-processing - Tofsmooth
++++++++++++++++++++++++++++++++++++++++++++++

This whole pipeline and network, with some minor modifications can be also used to  smoothen the depth image, thus making the point cloud smoother as well.

For the dataset we added gaussian noise of 5 and 10 cm to the original data, while we smoothed the original data with PointCloudDenoising (:cite:`pistilli2020learning`) method.

Our method got pretty close to the ground truth value, in most of the cases. Although, in the case of the original (originally fairly smooth) data resulted slightly worse results, then some other methods (for instance the PointCloud Library :cite:’Rusu_ICRA2011_PCL’), when we tested the smoothing for much more noisy data, our results barely changed, while other methods were highly compromised. A comparison between these cases can be seen in the next image 3 images:

.. figure:: noise00.jpg
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  The average error for the original data :label:`noise00`

.. figure:: noise05.jpg
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  The average error for data with 5 cm gaussian noise :label:`noise05`

.. figure:: noise10.jpg
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  The average error for data with 10 cm gaussian noise :label:`noise10`

Here we can see that our method kept very much the same throughout all the cases same as DeepDepthDenoising method (:cite:’sterzentsenko2019denoising’), which is the only other method that we have found, that works with depth images as well, making it about the same as ours, but a little bit more polished. Also this method performs at the same speed as ours.

The jump in the error at the end of the scale is due to some denormalization bias that we need to fine-tune. 

Furthermore, in order to get a real-time visualization about the predictions, we used rospy to read the images from ROS topics, and also to publish the normal estimation values to another ROS topic, that we could visualize using Rviz. This can be seen in the demo video.

This whole pipeline and network, with some minor modifications can be also used to  smoothen the depth image, thus making the point cloud smoother as well.


.. PCL based pipeline for ToF.


CNN based solutions
-------------------
Jetson based solutions


Person detection from IR imaging
++++++++++++++++++++++++++++++++


Detectnet is a detection algorithm based on the jetson-inference repository. 
This repository uses NVIDIA TensorRT for efficient implementation of neural networks on the Jetson platform, improving performance and energy efficiency through graphical optimizations, kernel fusion and FP16/INT8 accuracy.

 |

.. figure:: DetectNetIR.PNG
  :width: 400
  :height: 400
  :scale: 24%
  :align: center
  :alt: Alternative text

  Exemplification of skeleton detection on infrared images based detection:label:`detection`

The pre-trained model accepts 3 channel images – RGB, by modifying the existing model, we have managed to detect and track people on the infrared image – 1 channel. With the help of the OpenCV library and the 3.7 python programming language version, we have developed a script that modifies the contrast of the IR image; thus, we obtained a much better result than if we had not used this approach. This result can be seen in the figure below, where we can see that the people are detected on the IR image with high confidence.

To be able to run the algorithm in real-time we used the rospy client. With the help of this API, we have developed an efficient way to pass a ROS topic as input to our model. The algorithm was tested on a Jetson AGX, and the camera used was from Analog Devices (AD-96TOF1-EBZ). The result can be seen in the attached demo video.


Action recognition from IR images
+++++++++++++++++++++++++++++++++
.. PM part

This is a small tutorial for detecting the skeleton of a person
from an infrared image. In our setup we used one of the Analog Devices
Time-of-Flight cameras, which provided us the infrared image, and an
NVIDIA Jetson Xavier NX module.

As a baseline architecture model, we used the pretrained model from one
of the NVIDIA-AI-IOT's repositories: https://github.com/NVIDIA-AI-IOT/trt_pose .
We used the TensorRT SDK for achieving a better performance in our model inference
pipeline.

We also used, some of the Robot Operating System's tools for retrieving
the camera infrared images and by using the rospy client library API
we managed to transfer our infrared images to the network model. While this
would have been an easy step using the CvBridge, which provides an interface
between ROS and OpenCV, this time wasn't the case, as we had some issues with
this library. Because we are working on Jetson Xavier NX board, which comes with
the latest OpenCV version, and CvBridge uses at its core an older version of
OpenCV, we replaced the conversion from image message type to OpenCV image array
made by CvBridge with a very useful numpy functionality which allowed us to make 
make this conversion flawlessly. So, we replaced:

.. code-block:: python

   ir_image = bridge.imgmsg_to_cv2(image_msg,-1)


with:


.. code-block:: python

   ir_image = np.frombuffer(
   image_msg.data,
   dtype=np.uint8).reshape(
                           image_msg.height,
                           image_msg.width,
                           -1)



.. figure:: ir_skeleton_detection.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text
  
  Exemplification of skeleton detection on infrared images :label:`skeleton`

After making this conversion, we preprocessed the infrared image before 
feeding it to the neural network, using the OpenCv library. 
After this step we supply the model input with this preprocessed image, and
we obtained the results which can be seen in the Figure :ref:`skeleton`.


Further more, we managed to extend the infrared people detection application
by integrating it with NVIDIA's Deepstream SDK. While this SDK
makes further improvements with regards to the model inference performance,
one of the base application which the Deepstream SDk supports is the fact
that is able to provide communication with a server and transmit the output of 
the neural network model for further data processing. This can be very useful 
in applications where we want to gather some sort of statistics or when our application
has to make some decisions based on the output of our trained model, but we don't want 
to affect the Jetson's inference performance. In the Figure :ref:`deepstream`, can be 
seen the people detection made by using the Deepstream SDK, and below is the network'S
output received on our custom configured server when a person is detected:

.. code-block:: json

  {
  "object" : {
  "id" : "-1",
  "speed" : 0.0,
  "direction" : 0.0,
  "orientation" : 0.0,
  "person" : {
    "age" : 45,
    "gender" : "male",
    "hair" : "black",
    "cap" : "none",
    "apparel" : "formal",
    "confidence" : -0.10000000149011612
  },
  "bbox" : {
    "topleftx" : 147,
    "toplefty" : 16,
    "bottomrightx" : 305,
    "bottomrighty" : 343
  },
  "location" : {
    "lat" : 0.0,
    "lon" : 0.0,
    "alt" : 0.0
  },
  "coordinate" : {
    "x" : 0.0,
    "y" : 0.0,
    "z" : 0.0
  }
  }


.. figure:: deepstream_people_detection.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  Here can be seen the people detection algorithm which 
  runs with the Deepstream SDK on the Jetson Xavier NX board :label:`deepstream`


Volumetric estimates for depth images
+++++++++++++++++++++++++++++++++++++
.. PA part

The goal of this research is to estimate the volume of objects using only depth images recorded with Time-of-Flight cameras. As a simplifying feature, we consider only box shaped objects, with clearly definable perpendicular planes. Two methods have been determined.The first method uses RANSAC algorithm to detect planes while the other one uses the ideas from Sommer et all. 

The first algorithm iteratively finds the largest plane using RANSAC and uses euclidean extraction to remove it from the point cloud. Once the planes are determined and checked to see if they are perpendicular, the intersection lines of the planes are determined by projecting between them. The projections approximate a line and the points with the largest component difference determine the length of the line. This way iteratively the 3 intersecting line lengths can be determined once the planes are determined and checked for orthogonality.

.. figure:: RANSAC_volume.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text
  
  Planar detection :label:`plamar`

An important observation is that it can compute the volume using 2 planes instead of 3. This is due to the fact that if 2 planes are orthogonal, the common line between them will be determined by 2 points that are also corner points for the object. By selecting a corner point and the two perpendicular planes, a third plane can be determined that is perpendicular to the other two and it contains the chosen point. Once the virtual third plane has been computed, the algorithm resumes as in the case with 3 determined planes.

An advantage of this method is that it uses readily avaible and studied functions for processing pointclouds. For a simple case of a box and floor plane, the algorithm accuracy depends on the level of noise the pointcloud has.
The downside of this method is that it can compute the volume only for one box. Noise and other objects in the scene can totaly disrupt the volumetric estimate.

Due to these shortcomings, a new method for measuring the volume is studied, based on the work by Sommer et all. Their paper details an algorithm that uses pointclouds with normals computed in each point in order to determine collections of point pairs for which their normals satisfy the orthogonality constraint.  
The point pair collections will approximate the orthogonal planes. By determining the points contained by each orthogonal plane, projections can be made that approximate the intersecting lines of the orthogonal planes. By selecting the 3 lines that have the edge points closest to each other, volume of a box can be computed.
The advantage of this method is that it allow the computation of the volume for multiple box shaped objects and it 

.. figure:: ortho_volume.png
  :width: 400
  :height: 400
  :scale: 40%
  :align: center
  :alt: Alternative text

  Corner detection :label:`corner`

Volume estimation using enhanced planar/corner detections was done using the training from :cite:`sommer2020`.


Conclusion
----------
In this report we provided some guidlines for the ToF specific image processing using python libraries. The demos are randing from 
basic pointlcoud processing to people detection and enhanced volume estimation.



Acknowledgement
---------------
The authors are thankful for the support of Analog Devices Romania, 
for the equipment list (cameras, embedded devices, GPUs) offered as support 
to this work. 
This work was financially supported by the Romanian National Authority 
for Scientific Research, CNCS-UEFISCDI, project number PN-III-P2-2.1-PTE-2019-0367.
The authors are thankful for the generous donation from NVIDIA corporation for supporting this research.
