:author: Virginia Ng
:email: virginia@mapbox.com
:institution: Mapbox


:author: Daniel Hofmann
:email: hofmann@mapbox.com
:institution: Mapbox


--------------------------------------------------------------
Scalable Feature Extraction with Aerial and Satellite Imagery
--------------------------------------------------------------

.. class:: abstract

   Computer vision has been one of the most rapidly growing fields in deep learning.
   It powers a variety of emerging technologies—from facial recognition to
   augmented reality to self-driving cars. In this paper we present the steps to developing
   deep learning pipelines, which allow us to perform object detection and semantic segmentation on aerial and satellite
   imagery at large scale. Practical applications of these pipelines
   include detection and automatically mapping turn lane markings , parking lots, roads, water,
   and buildings etc.

   We first present the process of collecting, cleaning, pre-processing imagery-based data at scale by constructing data engineering pipelines.
   We also demonstrate how these training sets can be built with little or no human annotations
   needed making use of available data such as OpenStreetMap.

   We then discuss the implementation of state-of-the-art detection and semantic segmentation systems such as
   YOLOv2, U-Net, Residual Network, and Pyramid Scene Parsing Network (PSPNet), as well as
   specific adaptations for the aerial and satellite imagery domain in our pipelines.


.. class:: keywords

   computer vision, deep learning, neural networks, satellite imagery, aerial imagery



I. Background
-------------

Mapbox is the location data platform for mobile and web applications. We
provide building blocks to add location features like maps, search, and
navigation into any experience software developers create.

In particular, our navigation products are focused on providing smart
turn-by-turn routing based on real-time traffic. Valuable Assets For
Routing, Maps, and Geocoding include turn Restrictions, turn lane markings,
parking lots, buildings, grass, trees, parks, water, bridges. We identify and map these assets, 
which in turn powers our routing engines to help determine the most optimal routes. 

We can manually map them, or we can abstract these navigation assets from imagery.


II. Designed with Open-Source Tools
-------------------------------------

We designed our processing pipelines and tools with open-source
libraries like Scipy, Rasterio, Fiona, Osium, JSOM[*]_, Keras, PyTorch,
OpenCV etc, while our training data was compiled from
OpenStreetMap [osm]_ and Mapbox Maps
API [mapbox]_.

.. [*] JOSM is an extensible editor for OpenStreetMap (OSM) for Java 8+. It supports loading GPX tracks, background imagery and OpenStreetMap data from local sources as well as from online sources and allows to edit the OpenStreetMap data (nodes, ways, and relations) and their metadata tags. It is open source and licensed under GPL. 


III. Scalable Computer Vision Pipelines
-----------------------------------------

The general workflow for our computer vision pipeline can be found in
Figure 1. Two examples we will present in this paper are turn lane markings
and parking lot segmentation with aerial and satellite imagery.

.. figure:: fig1.png
   :height: 100 px
   :width:  200 px
   :scale: 39 %

   Computer Vision Pipeline. 

We design these pipelines with two things in mind. We not only ensured scalability
in the amount of data we process, but also how easily it is to expand to perform
computer vision tasks on new objects of interests or other navigation assets.


1. Data
--------

**Data Preparation.** Before we explain the data involved in creating
our training set, we would first like to
explain the difference between object detection and semantic
segmentation. These are two different tasks in computer vision.
Object detection involves locating and classifying a variable
number of objects in an image. Figure 2 demonstrates how we use object
detection models to classify and locate turn lane markings from satellite
imagery. We are not just distinguishing five left-turn only lane markings
from two right-turn only lane markings like we would do in a classification problem,
but we also want to know where in the image these markings are located. Other
practical applications of object detection include face detection,
counting, visual search engine.

.. figure:: fig2.png
   :height: 75 px
   :width:  150 px
   :scale: 22 %

   Turn lane markings detection.

Semantic segmentation on the other hand, attempts to partition the image
into semantically meaningful parts, and to classify each part into one of
the pre-determined classes. One can also achieve the same goal by
classifying and labeling each pixel. For example, in addition to
recognizing the road from the buildings, we have also delineated the
boundaries of each object shown in Figure 3.

.. figure:: fig3.png
   :height: 100 px
   :width: 200 px
   :scale: 22 %

   Semantic segmentation on roads, buildings and vegetation

To prepare training data for detecting turn lane markings, we first find
where the turn lane markings are. OpenStreetMap is a collaborative
project to create a free editable map of the world. Turn lane markings
on OpenStreetMap are recorded as “ways” [osm-lanes]_. We used a tool
called Overpass Turbo [overpass]_ to query
OpenStreetMap turn lane markings. We then extracted GeoJSONs in five cities
from OpenStreetMap that have one of the following attributes - “\turn:lane=*”,
“\turn:lane:forward=*”, “\turn:lane:backward=*” - and
created a custom layer over the `mapbox.satellite
layer` [mapbox]_.
We have humans manually draw and verify bounding boxes around the turn lane markings six
classes of turn lane markings using JOSM [josm]_, a process called annotation.
The six classes of turn lane markings are “\Left”, “\Right”, “\Through”,
“\ThroughLeft”, “\ThroughRight”, “\Other” in five cities, over 53,000 turn
lane markings. We included turn lane markings of
all shapes and sizes, as well as ones that are partially covered by cars
and/or shadows. We excluded turn lane markings that are erased or fully
covered by cars seen in Figure 4b.

.. figure:: fig4.png
   :height: 75 px
   :width: 150 px
   :scale: 22 %

   Annotating turn lane markings - Draw bounding box around the turn lane markings.
   Figure 4b: Data Cleaning - Excluding turn lane arrows that are fully covered by car.

To prepare training data for parking lot segmentation, we first generate
polygons from OpenStreetMap tags [osm-parking]_ excluding features that are not visible
in aerial imagery. Explicitly, these are OpenStreetMap features with the
attributes “\tag:amenity=parking=*” except underground, sheds, carports,
garage_boxes. To prepare training data for building segmentation, we
generate polygons from tags with attributes “\building=*” except
construction, houseboat, static_caravan, stadium, conservatory ,
digester, greenhouse, ruins. We then use a tool called
Osmium [osmium]_ to annotate
these parking lots.

**Data Engineering.** We built a data engineering pipeline within the
larger object detection pipeline to create our training datasets. 
This data engineering pipeline is capable of streaming
any set of prefixes off of Amazon Simple Storage Service (Amazon S3)[s3]_ into prepared training sets. 
For turn lane marking detection, we first stream these turn lane markings,
which are stored as OpenStreetMap features, out of the GeoJSON files on S3
and merge classes and bounding boxes into feature
attributes. Next, we convert these into JSON image annotations grouped by
tile. During this step, the annotated bounding boxes are converted to
image pixel coordinates. The annotations are then randomly assigned to
training and testing datasets, following the classic 80/20 split rule. They
are then written to disk and joined by
imagery fetched from the Satellite layer of Mapbox Maps API. This is where the abstract
tile in the pipeline is replaced by real imagery. Finally, the training and test
data are zipped and uploaded to Amazon S3. For parking lot segmentation, we convert the annotated parking lots,
which are also stored as GeoJSON polygons, into single channel numpy arrays.
We then stack each of these single channel numpy arrays with its respective aerial
image tile, a three channel numpy array - Red, Green, and Blue.

In either of these cases, we first developed Python command line tools and libraries for our data preparation steps.
Examples of these command line tools can be found on our segmentation GitHub repository [robosat]_. These
scripts are then ran at large scale in parallel (multiple cities at
once) on Amazon Elastic Container Service. Amazon Elastic Container Service is a
highly scalable, fast, container management service that makes it easy
to run, terminate, and manage Docker containers on a cluster (grouping of
container instances). This pipeline is shown in Figure 5.

.. figure:: fig5.png
   :height: 200 px
   :width: 400 px
   :scale: 47 %

   Data engineering pipeline combines OpenStreetMap GeoJSON features with imagery fetched from Mapbox Maps API.

Our data engineering pipelines are generalizable to any OpenStreetMap
feature. Examples of other features we have implemented include buildings. 
Users can generate training sets with any OpenStreetMap feature
simply by writing their own Osmium handler to turn OpenStreetMap geometries into
polygons.

2. Model
---------

**Fully Convolutional Neural Networks.** Fully convolutional are neural
networks composed of convolutional layers without any fully-connected
layers or MLP usually found at the end of the network. A convolutional
neural network (CNN, or ConvNet)  with fully connected layers is just
as end-to-end learnable as a fully
convolutional one. The main difference is that the fully convolutional
net is learning filters everywhere. Even the decision-making layers at
the end of the network are filters. Traditional Convolutional neural
networks containing fully connected layers cannot manage different input
sizes , whereas fully convolutional networks can have only convolutional
layers or layers which can manage different input sizes and are faster
at that task.

A fully convolutional net tries to learn representations and make
decisions based on local spatial input. Appending a fully connected
layer enables the network to learn something using global information
where the spatial arrangement of the input falls away and need not
apply.

**Object Detection Models.**\ We implemented YOLOv2 [yolov2]_, a real-time object
detection system and is the improved version of YOLO [yolo]_, which was
introduced in 2015. YOLOv2 outperforms all other state-of-the-art
methods like Faster R-CNN with ResNet [resnet]_ and Single Shot MultiBox Detector (SSD)
in both speed and detection
accuracy [cite1]_. Our YOLOv2 was first pre-trained on
ImageNet 224x224 resolution imagery and then the network was resized and finetuned
for classification on higher resolution 448x448 turn lane marking imagery. A major feature of
the YOLO family is the use of anchor boxes to run prediction. There are
two ways of predicting the bounding boxes- directly predicting the
bounding box of the object or using a set of predefined bounding boxes
(anchor box) to predict the actual bounding box of the object. YOLO
predicts the coordinates of bounding boxes directly using fully
connected layers on top of the convolutional feature extractor. But, it
makes a significant amount of localization error. It is easier to
predict the offset based on anchor boxes than to predict the coordinates
directly. Instead of using pre-defined anchor boxes, YOLOv2 authors
performed K-means clustering on bounding boxes from the training data
set. In addition to using clustering on bounding boxes, YOLOv2 was able to
converge and regularize well through the use of batch normalization,

 

**Segmentation Models.** We implemented U-Net [unet]_ for parking lot
segmentation. The U-Net architecture can be found in Figure 6. It consists
of a contracting path to capture context and a symmetric expanding path that enables precise
localization. This type of network can be trained end-to-end with very
few training images and yields more precise segmentations than prior
best method such as the sliding-window convolutional network. This first part is 
called down or one may think it as the encoder part
where one apples convolution blocks followed by a maxpool downsampling to
encode the input image into feature representations at multiple
different levels. The second part of the network consists of upsample
and concatenation followed by regular convolution operations. Upsampling
in convolutional neural networks may be a new concept to some but the idea is
fairly simple: we are expanding the feature dimensions to meet the same
size with the corresponding concatenation blocks from the left. While
upsampling and going deeper in the network, we are simultaneously concatenating the
higher resolution features from down part with the upsampled features in
order to better localize and learn representations with following
convolutions. For parking lot segmentation, we perform binary
segmentation distinguishing parking lots from the background.

.. figure:: fig6.png
   :height: 150 px
   :width: 300 px
   :scale: 37 %

   U-Net Architecture

We also experimented with Pyramid Scene Parsing Network (PSPNet) [pspnet]_. PSPNet
is effective to produce good quality results on scenes that are complex, contain
multi-class and on dataset with great
diversity. We found that it was redundant with our parking lot segmenation where the
the number of categories are only binary - parking lot versus background. PSPNet adds a
multi-scale pooling on top of the backend model to aggregate different
scale of global information. The upsample layer is implemented by
bilinear interpolation. After concatenation, PSP fuses different levels of
feature with a 3x3 convolution.

**Hard Negative Mining.** This is a technique we used to improve model
performance by reducing the negative samples. A hard negative is when we
take that falsely detected patch, and explicitly create a negative
example out of that patch, and add that negative to our training set.
When we retrain our models with this extra
knowledge, they usually perform better and not make as many false positives.

Figure 7 shows probability mask over what our models believe are pixels
belonging to parking lots


.. figure:: fig7.png
   :height: 150 px
   :width: 300 px
   :scale: 37 %

   Probability Mask


3. Post-Processing
------------------

Figure 8 shows an example of the raw segmentation mask derived
from our U-Net model. It cannot be used directly as input into
OpenStreetMap. We performed a series of post-processing to improve the
quality of the segmentation mask and to transform the mask into the
right data format for OpenStreetMap.


.. figure:: fig8.png
   :height: 200 px
   :width: 200 px
   :scale: 38 %

   Raw segmentation mask derived from our U-Net model


.. figure:: fig9.png
   :height: 200 px
   :width: 200 px
   :scale: 39 %

   Clean polygon in the form of GeoJSON


**Noise Removal.** We remove noise in the data by performing two
morphological operations: erosion followed by dilation. Erosion removes
white noises, but it also shrinks our object. So we dilate it.

**Fill in holes.** We fill holes in the mask by performing dilation
followed by erosion. It is especially useful in closing small holes
inside the foreground objects, or small black points on the object. We
use this operator to deal with polygons within polygons.

**Contouring.** Contours are curves joining all the continuous points
that have same color or intensity.

**Simplification.** Douglas-Peucker Simplification takes a curve
compared of line segments and finds a similar curve with fewer points.
We get simple polygons that can be ingested by OpenStreetMap as feature type “nodes” and “ways”

**Transform Data.** Convert detection or segmentation results from pixel
space back into GeoJSONs (world coordinate).

**Removing tile border artifacts.** Query and match neighboring image
tiles.

**Deduplication.** Deduplicate by matching GeoJSONs with data that already exist on OpenStreetMap.

After performing all these post-processing steps, we have a clean mask
that is also a polygon in the form of GeoJSON. An example of such a mask can be
found in Figure 9. This can now be added to
OpenStreetMap as a parking lot feature.


4. Output
----------

With this pipeline design, we are able to run batch prediction at large
scale (on the world). The output of these processing pipelines are turn
lane markings and parking lots in the form of GeoJSONs. We can then add
these GeoJSONs back into OpenStreetMap as turn lane and parking lot
features. Our routing engines then take these OpenStreetMap features
into account when calculating routes. Shown in Figure 10 is a front-end UI that
allows users to pan around for instant turn lane markings detection.


.. figure:: fig10.png
   :height: 200 px
   :width: 400 px
   :scale: 42 %

   Front-end UI for instant turn lane markings detection


IV. Future Work
---------------

We have made Robosat[*]_, our end-to-end semantic segmantion pipeline publicly available in June 2018. 

We are in the process of making several improvements to our models. We are currently working on
replacing the standard U-Net encoder with pre-trained ResNet50 encoder. In addition, we are replacing learned deconvolutions
with upsampling and uses nearest neaighbor upsampling followed by a convolution for refinement instead.

We believe that this approach gives us more accurate results, while speeding up training and prediction, lowering memory usage. The drawback to such an approach is that it only works for three-channel inputs (RGB) and not with arbitrary channels.

.. [*] Robosat is generic ecosystem for feature extraction from aerial and satellite imagery https://github.com/mapbox/robosat


References
----------
.. [osm] OpenStreetMap, https://www.openstreetmap.org
.. [mapbox] Mapbox, https://www.mapbox.com/api-documentation/#maps
.. [osm-lanes] OpenStreetMap tags, https://wiki.openstreetmap.org/wiki/Lanes
.. [overpass] Overpass, https://overpass-turbo.eu/
.. [josm] JOSM, https://josm.openstreetmap.de/
.. [osm-parking] OpenStreetMap tags, https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dparking
.. [osmium] Osmium, https://wiki.openstreetmap.org/wiki/Osmium
.. [robosat] Robosat, https://github.com/mapbox/robosat#rs-extract
.. [s3] Amazon Simple Storage Service, https://aws.amazon.com/s3/
.. [yolov2] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [cite1] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [yolo] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640 [cs.CV], June 2015
.. [unet] Olaf Ronneberger, Philipp Fischer, Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*, arXiv:1505.04597 [cs.CV], May 2015.
.. [resnet] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun arXiv:1512.03385 [cs.CV], Dec 2015.
.. [pspnet] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, *Pyramid Scene Parsing Network*, arXiv:1612.01105 [cs.CV], Dec 2016.



