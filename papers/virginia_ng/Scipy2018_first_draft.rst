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

   Deep learning techniques have greatly advanced the performance of the already rapidly developing field of computer vision, which powers a variety of emerging technologies—from facial recognition to augmented reality to self-driving cars. The remote sensing and mapping communities are particularly interested in extracting, understanding and mapping physical elements in the landscape. These mappable physical elements are called features, and can include both natural and synthetic objects of any scale, complexity and character. Points or polygons representing sidewalks, glaciers, playgrounds, entire cities, and bicycles are all examples of features. In this paper we present a method to develop deep learning tools and pipelines that generate features from aerial and satellite imagery at large scale. Practical applications include object detection, semantic segmentation and automatic mapping of general-interest features such as turn lane markings on roads, parking lots, roads, water, building footprints.

   We give an overview of our data preparation process, in which data from the Mapbox Satellite layer, a global imagery collection, is annotated with labels created from OpenStreetMap data using minimal manual effort. We then discuss the implementation of various state-of-the-art detection and semantic segmentation systems, such as YOLOv2, modified U-Net and Pyramid Scene Parsing Network (PSPNet), as well as specific adaptations for the aerial and satellite imagery domain. We conclude by discussing our ongoing efforts in improving our models and expanding their applicability across classes of features, geographical regions, and relatively novel data sources such as street-level and drone imagery.


.. class:: keywords

   computer vision, deep learning, neural networks, satellite imagery, aerial imagery



I. Background
-------------

Location data is built into the fabric of our daily experiences, and is more important than ever with the introduction of new location-based technologies such as self-driving cars. Mapping communities, open source or proprietary, work to find, understand and map elements of the physical landscape. However, mappable physical elements are continually appearing, changing, and disappearing. For example, more than 1.2 million residential units were built in the United States alone in 2017 [buildings]_. Therefore, a major challenge faced by mapping communities is maintaining recency while expanding worldwide coverage. To increase the speed and accuracy of mapping, allowing better pace-keeping with change in the mappable landscape, we propose integrating deep neural network models into the mapping workflow. In particular, we have developed tools and pipelines to detect various geospatial features from satellite and aerial imagery at scale. We collaborate with the OpenStreetMap [osm]_ (OSM) community to create reliable geospatial datasets, validated by trained and local mappers.

Here we present two usecases to demonstrate our workflow for extracting street navigation indicators such as turn restrictions signs, turn lane markings, and parking lots, in order to improve our routing engines. We designed our processing pipelines and tools with open source libraries including Scipy, Rasterio, Fiona, Osium, JOSM [#]_, Keras, PyTorch, and OpenCV, while our training data was compiled from OpenStreetMap and the Mapbox Maps API [mapbox_api]_. Our tools are designed to be generalizable across geospatial feature classes and across data sources.

.. [#] JOSM [josm]_ is an extensible OpenStreetMap editor for Java 8+. At its core, it is an interface for editing OSM, i.e., manipulating the nodes, ways, relations, and tags that compose the OSM database. Compared to other OSM editors, JOSM is notable for its range of features, such as allowing the user to load arbitrary GPX tracks, background imagery, and OpenStreetMap data from local and online sources. It is open source and licensed under GPL.


II. Scalable Computer Vision Pipelines
-----------------------------------------

The general design for our deep learning based computer vision pipelines can be found in Figure 1, and is applicable to both object detection and semantic segmantation tasks. We design such pipelines with two things in mind: we must allow scalability to very large data volumes, which requires processing efficiency; and we must allow repurposability towards computer vision tasks on other geospatial features, which requires a general-purpose design. We present turn lane markings as an example of an object detection pipeline, and parking lots as an example of a semantic segmantation pipeline.

.. figure:: fig1.png
   :height: 100 px
   :width:  200 px
   :scale: 37 %

   Computer Vision Pipeline. 


1. Data
--------

The data needed to create training sets depends on the type of task: object detection or semantic segmentation. We first present our data preparation process for object detection, which means locating and classifying a variable number of objects in an image. Figure 2 demonstrates how object detection models are used to classify and locate turn lane markings from satellite imagery. There are many other practical applications of object detection such as face detection, counting, and visual search engines. In our case, detected turn lane markings become valuable navigation assets to our routing engines when determining the most optimal routes.

.. figure:: fig2.png
   :height: 75 px
   :width:  150 px
   :scale: 21 %

   Turn lane markings detection.

**Data Preparation For Object Detection.** The training data for turn lane marking detection was created by collecting imagery of various types of turn lane markings and manually drawing a bounding box around each marking. We used Overpass Turbo [overpass]_ to query the OpenStreetMap database for streets containing turn lane markings, i.e., those tagged with one of the following attributes: “\turn:lane=*”, “\turn:lane:forward=*”, “\turn:lane:backward=*” in OpenStreetMap. The marked street segments, as shown in Figure 3, were stored as GeoJSON features clipped into the tiling scheme [tile]_ of the Mapbox Satellite basemap [mapbox]_.. Figure 4 shows how skilled mappers used this map layer as a cue to manually draw bounding boxes around each turn lane marking using JOSM, a process called annotation. Each of these bounding boxes was stored as a GeoJSON polygon on Amazon S3 [s3]_.

.. figure:: fig3.png
   :height: 200 px
   :width: 400 px
   :scale: 32 %

   A custom layer created by clipping the locations of roads with turn lane markings to Mapbox Satellite. Streets with turn lane markings are rendered in red.

.. figure:: fig4.png
   :height: 150 px
   :width: 150 px
   :scale: 37 %
   
   Annotating turn lane markings by drawing bounding boxes.


Mappers annotated six classes of turn lane markings - “\Left”, “\Right”, “\Through”, “\ThroughLeft”, “\ThroughRight”, and “\Other” in five cities, creating a training set consists of over 54,000 turn lane markings. Turn lane markings of all shapes and sizes, as well as ones that are partially covered by cars and/or shadows were included in this training set. To ensure a high-quality training set, we had a separate group of mappers verify each of the bounding boxes drawn. We excluded turn lane markings that are not visible, as seen in Figure 5.

.. figure:: fig5.png
   :height: 75 px
   :width: 150 px
   :scale: 21 %

   Defaced or obscured turn lane markings, such as those covered by cars, are excluded.

Semantic segmentation, on the other hand, is the computer vision task that partitions an image into semantically meaningful parts, and classifies each part into one of any pre-determined classes. This can be understood as assinging a class to each pixel in the image, or equivalently as drawing non-overlapping polygons with associated classes over the image. As an example of the polygonal approach, in addition to distinguishing roads from buildings, we also delineate the boundaries of each object in Figure 6.

.. figure:: fig6.png
   :height: 75 px
   :width: 150 px
   :scale: 21 %

   Semantic segmentation of roads, buildings and vegetation.


**Data Preparation for Semantic Segmentation.** The training data for parking lot segmentation was created by combining imagery collected from Mapbox Satellite with binary masks for parking lots. The binary masks for parking lots were generated by querying the OpenStreetMap database with Osmium for polygons with attributes “\tag:amenity=parking=*”. These binary masks were stored as two-dimensional numpy arrays, clipped and scaled to the Mapbox Satellite tiling scheme. Each mask array was paired with its corresponding real (photographic) image tile. Conceptually, this can be compared to concatenating a fourth channel, the mask, onto a standard red, green, and blue image. We annotated 55,710 masks for parking lot segmentation for our initial training set.

**Data Engineering.** A data engineering pipeline was built within the larger object detection pipeline to create and process training sets in large quantities. This data engineering pipeline is capable of streaming any set of prefixes off of Amazon S3 into prepared training sets. Several pre-processing steps were taken to convert turn lane marking annotations to the appropriate data storage format before combining them with real imagery. As mentioned earlier, turn lane marking annotations were initially stored as GeoJSON polygons grouped by class. Each of these polygons was streamed out of the GeoJSON files on S3, converted to image pixel coordinates, and stored as JSON image attributes to abstract tiles [tile]_. The pre-processed annotations were then randomly assigned to training and testing datasets with a ratio of 4:1. The abstract tiles then replaced by the corresponding real image tiles, fetched from the Satellite layer of the Mapbox Maps API. At this point, each sample consisted of a photographic image paired with a mask image. Finally, the training and test sets were zipped and uploaded to Amazon S3.

Before we scaled up processing, we first developed Python command line tools and libraries for our data preparation steps. All of command line tools we developed for the segmentation task can be found on our GitHub repository [robosat]_. These scripts were then run on multiple cities in parallel on the Amazon Elastic Container Service [ecs]_. This data engineering pipeline is shown in Figure 7.

.. figure:: fig7.png
   :height: 200 px
   :width: 400 px
   :scale: 47 %

   A data engineering pipeline that converts OpenStreetMap GeoJSON features to image pixel space, then combines each feature with imagery fetched from the Mapbox Maps API.

The design of our data engineering pipelines can be generalized to any OpenStreetMap feature. For example, we also experimented with buildings. One can generate training sets for any OpenStreetMap feature in this way by writing custom Osmium handlers to convert OpenStreetMap geometries into polygons.

2. Model
---------

**Fully Convolutional Neural Networks.** Fully convolutional networks (FCNs) are neural networks composed only of convolutional layers. They are contrasted with more conventional networks that typically have fully connected layers or other non-convolutional subarchitectures as “decision-makers” just before the output. For the purposes considered here, FCNs show several significant advantages. First, FCNs can handle images of any size, while most alternatives require input dimensions to be hard-coded. Second, FCNs compute a full prediction field in a single pass, making efficient use of large-scale information. By contrast, sliding window approaches [cite0]_ compute a prediction from scratch at each position. Third, FCNs are well suited to handling complex spatial information, because locality information flows well through them. Networks with fully connected layers, for comparison, are not well suited to representing local detail [cite1]_.

**Object Detection Models.** Many of our applications require low latency prediction from their object detection algorithms. We implemented YOLOv2 [yolov2]_, the improved version of the real-time object detection system You Look Only Once (YOLO), in our turn lane markings detection pipeline. YOLOv2 outperforms other state-of-the-art methods, like Faster R-CNN with ResNet [resnet]_ and Single Shot MultiBox Detector (SSD), in both speed and detection accuracy [cite0]_. It works by first dividing the input image into 13 × 13 grid cells (i.e., there are 169 total cells for any input image). Each grid cell is responsible for generating 5 bounding boxes. Each bounding box is composed of its coordinates within its grid cell, a confidence score for "objectness," and an array of class probabilities. The "objectness" confidence score estimates how likely it is that the box fits an object. Specifically, it predicts the intersection over union of the ground truth and the proposed box. The class probabilities predict the conditional probability of each class for the proposed object, if it is an object.

6 classes were defined for the turn lane markings detection project. With 4 coordinates defining each box's geometry, the "objectness" confidence, and 6 class probabilities, each box object is comprised of 11 numbers. Multiplying by boxes per grid cell and grid cells per image, this project's YOLOv2 network therefore always yields 11 × 5 × 13 × 13 = 9,295 outputs per image.

The base feature extractor of YOLOv2 is Darknet-19, an FCN composed of 19 convolutional layers and 5 maxpooling layers. Detection is done by replacing the last convolutional layer of Darknet-19 with three 3 × 3 convolutional layers, each outputting 1024 channels. A final 1 × 1 convolutional layer is then applied to convert the 13 × 13 × 1024 output into 13 × 13 × 55. We followed two suggestions proposed by the YOLO authors when designing our model. The first was incorporating batch normalization after every convolutional layer. Batch normalization stabilizes training, improves the model convergence, and regularizes the model [yolov2_batch]_. The authors saw a 2% improvement in mAP from YOLO on the VOC2007 dataset [yolov2]_. The second suggestion that we implemented was the use of anchor boxes and dimension clusters to predict the actual bounding box of the object. This step was acheieved by running k-means clustering on the turn lane marking training set bounding boxes. As seen in Figure 8, the ground truth bounding boxes for turn lane markings follow specific height-width ratios. Instead of directly predicting bounding box coordinates, our model predicts the width and height of the box as offsets from cluster centroids. The center coordinates of the box relative to the location of filter application is predicted by using a sigmoid function.

.. figure:: fig8.png
   :height: 150 px
   :width: 150 px
   :scale: 38 %

   Clustering of box dimensions in the turn lane marking training set. We ran k-means clustering on the dimensions of bounding boxes to get anchor boxes for our model. We used k = 5, as suggested by the YOLO authors, who found that this cluster count gives a good tradeoff for recall v. complexity of the model.

Our model was first pre-trained on ImageNet 224 × 224 resolution imagery. The network was then resized and fine-tuned for classification on 448 × 448 turn lane marking imagery, to ensure that the relatively small features of interest were still reliably detected.

**Segmentation Models.** For parking lot segmentation, we selected an approach of binary segmentation (distinguishing parking lots from the background), and found U-Net [unet]_ to be a suitable architecture. The U-Net architecture can be found in Figure 9. It consists of a contracting path, to capture context, and a symmetric expanding path, which allows precise localization. This type of network can be trained end-to-end with very few training images and can yield more precise segmentations than prior state-of-the-art methods such as sliding-window convolutional networks. The first part of the U-Net network downsamples, and is similar in design and purpose to the encoding part of an autoencoder. It repeatedly applies convolution blocks followed by maxpool downsamplings, encoding the input image into increasingly abstract representations at successively deeper levels. The second part of the network consists of upsampling and concatenation, followed by ordinary convolution operations. Concatenation combines relatively “raw” information with relatively “processed” information. This can be understood as allowing the network to assign a class to a pixel with sensitivity to small-scale, less-abstract information about the pixel and its immediate neighborhood (e.g., whether it is gray) and simultaneously with sensitivity to large-scale, more-abstract information about the pixel’s context (e.g., whether there are nearby cars aligned in the patterns typical of parking lots). We have recently replaced the standard U-Net encoder with pre-trained ResNet50 [resnet]_ encoder. We have also replaced learned deconvolutions with nearest neighbor upsampling followed by a convolution for refinement. We saw a 1% improvement in accuracy after making these changes.

.. figure:: fig9.png
   :height: 125 px
   :width: 200 px
   :scale: 38 %

   U-Net architecture.

We experimented with a Pyramid Scene Parsing Network (PSPNet) [pspnet]_ architecture for a 4-class segmentation task on buildings, roads, water, and vegetation. PSPNet adds a multi-scale pooling on top of the backend model to aggregate different scales of information. The upsampling layer is implemented by bilinear interpolation. After concatenation, PSPNet fuses different levels of feature with a 3x3 convolution. As seen in Figure 6, PSPNet produced good-quality segmentation masks in our tests on on scenes with complex features. For the 2-class parking lot task, however, we found PSPNet unecessarily complex.

**Hard Negative Mining.** This is a technique we have applied to improve model accuracy. During a training session, when a model produces a false positive detection for a sample in the testing set, we move that sample to the training set. The model thus “learns from its mistakes.” This method speeds learning by reducing false positives.

Figure 10 shows a model's output as a probability mask overlaid on Mapbox Satellite. Increasingly opaque red indicates an increasingly high probability estimate of the underlying pixel belonging to a parking lot. We use this type of visualization to find representative falsely detected patches for use as hard negatives in hard negative mining. The average over multiple IOU (AP) of our baseline U-Net model is 46.7 on a test set of 900 samples.


.. figure:: fig10.png
   :height: 150 px
   :width: 150 px
   :scale: 48 %

   A probability mask marking the pixels that our model believes belong to parking lots.


3. Post-Processing
------------------

Figure 11 shows an example of the raw segmentation mask derived from our U-Net model. It cannot be used directly as input for OpenStreetMap. We performed a series of post-processing steps to refine and transform the mask until it met quality and format requirements for OpenStreetMap consumption.


.. figure:: fig11.png
   :height: 150 px
   :width: 150 px
   :scale: 47 %

   An example of border artifacts and holes in raw segmentation masks produced by our U-Net model.


**Noise Removal.** Noise in the output mask is removed by two morphological operations: erosion followed by dilation. Erosion removes some positive speckle noise ("islands"), but it also shrinks objects. Dilation re-expands the objects.

**Fill in holes.** The converse of the previous step, removing "lakes" (small false or topologically inconvenient negatives) in the mask.

**Contouring.** During this step, continuous pixels, having same color or intensity, along the boundary of the mask are joined. The output is a binary mask with contours.

**Simplification.** We apply Douglas-Peucker simplification, which takes a curve composed of line segments and gives a similar curve with fewer vertexes. OpenStreetMap favors polygons with the least number of vertexes necessary to represent the ground truth accurately, so this step is important to increase the data's quality as percieved by its end users.

**Transform Data.** Polygons are converted from in-tile pixel coordinates to GeoJSONs in geographic coordinates (longitude and latitude).

**Merging multiple polygons.** The merge tool [merge]_ combines polygons that are nearly overlapping, such as those that represent a single feature broken by tile boundaries, into a single polygon. See Figure 12.

**Deduplication.** Cleaned GeoJSON polygons are compared against parking lot polygons that already exist in OpenStreetMap, so that only previously unmapped features are uploaded.


.. figure:: fig12.png
   :height: 400 px
   :width: 800 px
   :scale: 35 %

   Polygons crossing tile boundaries, and other adjacent polygons, are combined.



4. Output
----------

With these pipeline designs, we are able to run batch feature prediction on millions of image tiles. The outputs of the processing pipelines discussed are turn lane markings and parking lots in the form of GeoJSON features suitable for adding to OpenStreetMap. Mapbox routing engines then take these OpenStreetMap features into account when calculating optimal navigation routes. As we make various improvements to our baseline model (see below), we keep human control over the final decision to add a given feature to OpenStreetMap.


.. figure:: fig14.png
   :height: 200 px
   :width: 400 px
   :scale: 25 %

   Front-end UI for instant turn lane marking detection.


IV. Ongoing Work
----------------
Here we demonstrated the steps to building deep learning-based computer vision pipelines that can run object detection and segmentation tasks at scale. We open sourced an end-to-end semantic segmantion pipeline, Robosat [robosat]_, along with all its tools in June 2018 and ran parking lot segmentation over Atlanta, Baltimore, Sacramento, and Seattle. We designed our tools and pipelines with the intent that other practitioners would find it straightforward to adapt them to other landscapes, landscape features, and imagery data sources. We are now working on refining the post-processing steps as mentioned in the previous section. Currently our post-processing handler specifically designed for parking lot features is tuned with thresholds for zoom level 18 imagery [osm_zoom]_. We are working on generalizing these thresholds, base them on meters so we are able to expand to multiple resolutions.
We also need to reorder the simplication and merging steps. Currently, we perform simplication first and then run buffering, unioning, then unbuffering to merge polygons across tile boundaries. This leads to polygons that are no longer simplified. We also need a more sophisticated simplication besides Douglas-Peucker.


 We also plan to implement a feature pyramid-based deep convolutional network called Feature Pyramid Networks [FPN]. It sits on top of ResNet and is a practical and accurate solution to multi-scale object detection.


We will continue looking for ways to bring different sources and structures of data together to build better computer vision pipelines.


For future work we hope to use similar techniques for building footprint extraction. We experimented with building segmentation in unmanned aerial vehicle (UAV) imagery from the OpenAerialMap project in Tanzania [tanzania]_, to explore the challenges of very high-resolution imagery and a landscape unlike the ones we have trained on thus far.





References
----------
.. [buildings] United States Census Bureau. *New Residential Construction*, Jul 2018.
.. [osm] OpenStreetMap, https://www.openstreetmap.org
.. [mapbox] Mapbox, https://www.mapbox.com/about/
.. [mapbox_api] Mapbox Maps API, https://www.mapbox.com/api-documentation/#maps, https://www.openstreetmap.org/user/pratikyadav/diary/43954
.. [osm-lanes] OpenStreetMap tags, https://wiki.openstreetmap.org/wiki/Lanes
.. [overpass] Overpass, https://overpass-turbo.eu/
.. [josm] JOSM, https://josm.openstreetmap.de/
.. [osm-parking] OpenStreetMap tags, https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dparking
.. [osmium] Osmium, https://wiki.openstreetmap.org/wiki/Osmium
.. [tile] tile scheme, https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
.. [robosat] Robosat, https://github.com/mapbox/robosat#rs-extract
.. [s3] Amazon Simple Storage Service, https://aws.amazon.com/s3/
.. [ecs] Amazon Elastic Container Service, https://aws.amazon.com/ecs/
.. [yolo-drawbacks] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [yolov2] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [yolov2_batch] S. Ioffe and C. Szegedy. *Batch normalization: Accelerating deep network training by reducing internal covariate shift*, arXiv preprint arXiv:1502.03167, 2015. 2, 5
.. [cite0] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [cite1] Jonathan Long, Evan Shelhamer, Trevor Darrell *Fully Convolutional Networks for Semantic Segmentation*, https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf, 2015 
.. [yolo] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640 [cs.CV], June 2015
.. [unet] Olaf Ronneberger, Philipp Fischer, Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*, arXiv:1505.04597 [cs.CV], May 2015.
.. [resnet] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, arXiv:1512.03385 [cs.CV], Dec 2015.
.. [pspnet] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, *Pyramid Scene Parsing Network*, arXiv:1612.01105 [cs.CV], Dec 2016.
.. [merge] https://s3.amazonaws.com/robosat-public/3339d9df-e8bc-4c78-82bf-cb4a67ec0c8e/features/index.html#16.37/33.776449/-84.41297
.. [robosat] Mapbox 2018
.. [osm_zoom] https://wiki.openstreetmap.org/wiki/Zoom_levels
.. [FPN] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. *Feature Pyramid Networks for Object Detection*, arXiv:1612.03144 [cs.CV] Dec 2016



