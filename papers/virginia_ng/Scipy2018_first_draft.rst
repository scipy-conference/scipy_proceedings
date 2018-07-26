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

**Data Preparation For Object Detection.** The training data for turn lane marking detection was created by collecting imagery of various types of turn lane markings and manually drawing a bounding box around each marking. We used Overpass Turbo [#]_ to query the OpenStreetMap database for streets containing turn lane markings, i.e., those tagged with one of the following attributes: “\turn:lane=*”, “\turn:lane:forward=*”, “\turn:lane:backward=*” in OpenStreetMap. The marked street segments, as shown in Figure 3, were stored as GeoJSON features clipped into the tiling scheme [tile]_ of the Mapbox Satellite basemap [mapbox]_.. Figure 4 shows how skilled mappers used this map layer as a cue to manually draw bounding boxes around each turn lane marking using JOSM, a process called annotation. Each of these bounding boxes was stored as a GeoJSON polygon on Amazon S3 [s3]_.

.. [#] Overpass Turbo [overpass]_ is a web based data mining tool for OpenStreetMap. It runs any kind of Overpass API query and shows the results on an interactive map.


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

Semantic segmentation, on the other hand, is the computer vision task that partitions an image into semantically meaningful parts, and classifies each part into one of any pre-determined classes. This can be understood as assinging a class to each pixel in the image, or equivalently as drawing non-overlapping polygons with associated classes over the image. As an example of the polygonal approach, in addition to distinguishing roads from buildings and vegetation, we also delineate the boundaries of each object in Figure 6.

.. figure:: fig6.png
   :height: 75 px
   :width: 150 px
   :scale: 21 %

   Semantic segmentation of roads, buildings and vegetation.


**Data Preparation for Semantic Segmentation.** The training data for parking lot segmentation was created by combining imagery collected from Mapbox Satellite with binary masks for parking lots. The binary masks for parking lots were generated by querying the OpenStreetMap database with Osmium [osmium]_ for polygons with attributes “\tag:amenity=parking=*”. These binary masks were stored as two-dimensional numpy arrays, clipped and scaled to the Mapbox Satellite tiling scheme. Each mask array was paired with its corresponding real (photographic) image tile. Conceptually, this can be compared to concatenating a fourth channel, the mask, onto a standard red, green, and blue image. We annotated 55,710 masks for parking lot segmentation for our initial training set.

**Data Engineering.** A data engineering pipeline was built within the larger object detection pipeline to create and process training sets in large quantities. This data engineering pipeline is capable of streaming any set of prefixes off of Amazon S3 into prepared training sets. Several pre-processing steps were taken to convert annotations to the appropriate data storage format before combining them with real imagery. As mentioned earlier, turn lane marking annotations were initially stored as GeoJSON polygons grouped by class. Each of these polygons was streamed out of the GeoJSON files on S3, converted to image pixel coordinates, and stored as JSON image attributes to abstract tiles [tile]_. The pre-processed annotations were then randomly assigned to training and testing datasets with a ratio of 4:1. The abstract tiles were then replaced by the corresponding real image tiles, fetched from the Satellite layer of the Mapbox Maps API. At this point, each object detection sample consisted of a photographic image paired with its corresponding JSON image attribute for object detection, while each semantic segmentation sample consisted of a photographic image paired with a mask image. Finally, the training and test sets were zipped and uploaded to Amazon S3.

Before we scaled up processing, we first developed Python command line tools and libraries for our data preparation steps. All of command line tools we developed for the segmentation task can be found on our GitHub repository [robosat]_. These scripts were then run on multiple cities in parallel on the Amazon Elastic Container Service [#]_. This data engineering pipeline is shown in Figure 7.

.. figure:: fig7.png
   :height: 200 px
   :width: 400 px
   :scale: 47 %

   A data engineering pipeline that converts OpenStreetMap GeoJSON features to image pixel space, then combines each feature with imagery fetched from the Mapbox Maps API.

The design of our data engineering pipelines can be generalized to any OpenStreetMap feature and any data source. For example, we also experimented with building segmentation in unmanned aerial vehicle (UAV) imagery from the OpenAerialMap project in Tanzania [tanzania]_. One can generate training sets for any OpenStreetMap feature in this way by writing custom Osmium handlers to convert OpenStreetMap geometries into polygons.


.. [#] Osmium [osmium]_ is a fast and flexible C++ library for working with OpenStreetMap data.
.. [#] Amazon ECS [ecs]_ is a highly scalable, fast, container management service that makes it easy to run, stop, and manage Docker containers on specified type of instances




2. Model
---------

**Fully Convolutional Neural Networks.** Fully convolutional networks (FCNs) are neural networks composed only of convolutional layers. They are contrasted with more conventional networks that typically have fully connected layers or other non-convolutional subarchitectures as “decision-makers” just before the output. For the purposes considered here, FCNs show several significant advantages. First, FCNs can handle input images of different resolutions, while most alternatives require input dimensions to be of a certain size [FCN]_. For example, architectures like AlexNet can only work with input images sizes that are 224 x 224 x 3 [FCN]_. Second, FCNs are well suited to handling spatially dense prediction tasks like segmentation because one would no longer be constrained by the number of object categories or complexity of the scenes. Networks with fully connect layers, in contrast, generally lose spatial information in these layers because all output neurons are connected to all input neurons [FCN]_.

**Object Detection Models.** Many of our applications require low latency prediction from their object detection algorithms. We implemented YOLOv2 [yolov2]_, the improved version of the real-time object detection system You Look Only Once (YOLO) [yolo]_, in our turn lane markings detection pipeline. YOLOv2 outperforms other state-of-the-art methods, like Faster R-CNN with ResNet [resnet]_ and Single Shot MultiBox Detector (SSD) [ssd]_, in both speed and detection accuracy [yolov2]_. It works by first dividing the input image into 13 × 13 grid cells (i.e., there are 169 total cells for any input image). Each grid cell is responsible for generating 5 bounding boxes. Each bounding box is composed of its center coordinates relative to the location of its corresponding grid cell, its normalized width and height, a confidence score for "objectness," and an array of class probabilities. A logistic activation is used to constrain the network’s location prediction to fall between 0 and 1, so that the network is more stable. The objectness predicts the intersection over union (IOU) of the ground truth and the proposed box. The class probabilities predict the conditional probability of each class for the proposed object, given that there is an object in the box [yolov2]_.

6 classes were defined for the turn lane markings detection project. With 4 coordinates defining each box's geometry, the "objectness" confidence, and 6 class probabilities, each bounding box object is comprised of 11 numbers. Multiplying by boxes per grid cell and grid cells per image, this project's YOLOv2 network therefore always yields 13 x 13 x 5 x 11 = 9,295 outputs per image.

The base feature extractor of YOLOv2 is Darknet-19, a FCN composed of 19 convolutional layers and 5 maxpooling layers. Detection is done by replacing the last convolutional layer of Darknet-19 with three 3 × 3 convolutional layers, each outputting 1024 channels. A final 1 × 1 convolutional layer is then applied to convert the 13 × 13 × 1024 output into 13 × 13 × 55. We followed two suggestions proposed by the YOLO authors when designing our model. The first was incorporating batch normalization after every convolutional layer. Batch normalization stabilizes training, improves the model convergence, and regularizes the model [yolov2_batch]_. The authors saw a 2% improvement in mAP from YOLO on the VOC2007 dataset [yolov2]_. The second suggestion that we implemented was the use of anchor boxes and dimension clusters to predict the actual bounding box of the object. This step was acheieved by running k-means clustering on the turn lane marking training set bounding boxes. As seen in Figure 8, the ground truth bounding boxes for turn lane markings follow specific height-width ratios. Instead of directly predicting bounding box coordinates, our model predicts the width and height of the box as offsets from cluster centroids. The center coordinates of the box relative to the location of filter application is predicted by using a sigmoid function.

.. figure:: fig8.png
   :height: 150 px
   :width: 150 px
   :scale: 38 %

   Clustering of box dimensions in the turn lane marking training set. We ran k-means clustering on the dimensions of bounding boxes to get anchor boxes for our model. We used k = 5, as suggested by the YOLO authors, who found that this cluster count gives a good tradeoff for recall v. complexity of the model.

Our model was first pre-trained on ImageNet 224 × 224 resolution imagery. The network was then resized and fine-tuned for classification on 448 × 448 turn lane marking imagery, to ensure that the relatively small features of interest were still reliably detected.

**Segmentation Models.** For parking lot segmentation, we selected an approach of binary segmentation (distinguishing parking lots from the background), and found U-Net [unet]_ to be a suitable architecture. The U-Net architecture can be found in Figure 9. It consists of a contracting path, to capture context, and a symmetric expanding path, which allows precise localization. This type of network can be trained end-to-end with very few training images and can yield more precise segmentations than prior state-of-the-art methods such as sliding-window convolutional networks. The first part of the U-Net network downsamples, and is similar in design and purpose to the encoding part of an autoencoder. It repeatedly applies convolution blocks followed by maxpool downsamplings, encoding the input image into increasingly abstract representations at successively deeper levels. The second part of the network consists of upsampling and concatenation, followed by ordinary convolution operations. Concatenation combines relatively “raw” information with relatively “processed” information. This can be understood as allowing the network to assign a class to a pixel with sensitivity to small-scale, less-abstract information about the pixel and its immediate neighborhood (e.g., whether it is gray) and simultaneously with sensitivity to large-scale, more-abstract information about the pixel’s context (e.g., whether there are nearby cars aligned in the patterns typical of parking lots). We have recently replaced the standard U-Net encoder with pre-trained ResNet50 [resnet]_ encoder and also the learned deconvolutions with nearest neighbor upsampling followed by a convolution for refinement. We saw a modest 1% improvement in accuracy after making these changes.

.. figure:: fig9.png
   :height: 125 px
   :width: 200 px
   :scale: 38 %

   U-Net architecture.

We experimented with a Pyramid Scene Parsing Network (PSPNet) [pspnet]_ architecture for a 4-class segmentation task on buildings, roads, water, and vegetation. PSPNet is one of the few pixel-wise segmentation methods that focuses on global priors, while most methods fuse low-level, high resolution features with high-level, low resolution ones to develope comprehensive feature representations. Global priors can be especially useful for objects that have similar spatial features. For instance, runways and freeways have similar color and texture features, but they belong to different classes, which can be discriminated by adding car and building information. PSPNet uses pre-trained ResNet to generate a feature map that is 1/8 the size of the input image. The feature map is then fed through the pyramid parsing module, a hierarchical global prior that aggregates different scales of information. After upsampling and concatenation, the final feature representatation is fused with a 3 x 3 convolution to produce the final prediction map. As seen in Figure 6, PSPNet produced good-quality segmentation masks in our tests on scenes with complex features. For the 2-class parking lot task, however, we found PSPNet unnecessarily complex.

**Hard Negative Mining.** This is a technique we have applied to improve model accuracy [hnm]_ . We first train a model with an initial subset of negative examples, and collect negative examples that are incorrectly classified by this initial model to form a set of hard negatives. A new model is then trained with the hard negative examples and the process may be repeated a few times.

Figure 10 shows a model's output as a probability mask overlaid on Mapbox Satellite. Increasingly opaque red indicates an increasingly high probability estimate of the underlying pixel belonging to a parking lot. We use this type of visualization to find representative falsely detected patches for use as hard negatives in hard negative mining.

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

**Contouring.** During this step, continuous pixels having same color or intensity along the boundary of the mask are joined. The output is a binary mask with contours.

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

With these pipeline designs, we are able to run batch feature prediction on millions of image tiles. The outputs of the processing pipelines discussed are turn lane markings and parking lots in the form of GeoJSON features suitable for adding to OpenStreetMap. Mapbox routing engines then take these OpenStreetMap features into account when calculating optimal navigation routes. As we make various improvements to our baseline model and post-processing algorithms (see below), we keep human control over the final decision to add a given feature to OpenStreetMap.


.. figure:: fig14.png
   :height: 200 px
   :width: 400 px
   :scale: 25 %

   Front-end UI for instant turn lane marking detection.


IV. Ongoing Work
----------------
Here we demonstrated the steps to building deep learning-based computer vision pipelines that can run object detection and segmentation tasks at scale. We open sourced an end-to-end semantic segmantion pipeline, Robosat [robosat]_, along with all its tools in June 2018 and ran parking lot segmentation over Atlanta, Baltimore, Sacramento, and Seattle.

We are now working on making a few improvements to this Robosat pipeline so that it becomes more flexible in handling input image of different resolutions. For instance, our existing post-processing handler was designed for parking lot features and was specifically tuned with thresholds set for zoom level 18 imagery [osm_zoom]_. We are replacing these hard-coded thresholds with generalized ones that are calculated based on resolution in meters per pixel. We also plan to experiment with a feature pyramid-based deep convolutional network called Feature Pyramid Network (FPN) [FPN]_. It is a practical and accurate solution to multi-scale object detection. Similar to U-Net, the FPN has lateral connections between the bottom-up pyramid (left) and the top-down pyramid (right). The main difference is where U-net only copies features and appends them, FPN applies a 1x1 convolution layer before adding the features. We will most likely follow the authors' footsteps and use ResNet as the backbone of this network.

There two other modifications planned for the post-processing steps. First, we want to experiement with a more sophisticated polygon simplication algorithm besides Douglas-Peucker. Second, we are rethinking the ordering of doing simplication then merging. The current post-process workflow performs simplication on individual extracted polygons and then merges polygons that are across imagery tiles together. The resulting polygons may no longer be in the simplest shape.

We design our tools and pipelines with the intent that other practitioners would find it straightforward to adapt them to other landscapes, landscape features, and imagery data sources. For future work we will continue to look for ways to bring different sources and structures of data together to build better computer vision pipelines.



References
----------
.. [buildings] United States Census Bureau. *New Residential Construction* Jul 2018, https://www.census.gov/construction/nrc/index.html
.. [osm] OpenSteetMap. OpenStreetMap contributors. April 2018, https://www.openstreetmap.org
.. [mapbox] Mapbox. https://www.mapbox.com/about/
.. [mapbox_api] Mapbox. Maps API Documentation. May 2018, https://www.mapbox.com/api-documentation/#maps
.. [osm-lanes] OpenStreetMap tags, https://wiki.openstreetmap.org/wiki/Lanes
.. [overpass] Martin Raifer. Overpass Turbo. Jan 2017, https://overpass-turbo.eu/
.. [josm] Immanuel Scholz, Dirk Stöcker. Java OpenStreetMap Editor. May 2017, https://josm.openstreetmap.de/
.. [osm-parking] OpenStreetMap Wiki. Tags, https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dparking
.. [osmium] Jochen Topf. Osmium. April 2018, https://github.com/osmcode/libosmium
.. [tile] OpenStreetMap Wiki. Tile Scheme. 1 June 2018, https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
.. [tanzania] Daniel Hofmann. RoboSat loves Tanzania. July 2018, https://www.openstreetmap.org/user/daniel-j-h/diary/44321
.. [robosat] Mapbox. Robosat. June 2018, https://github.com/mapbox/robosat
.. [s3] Amazon. Amazon Simple Storage Service, https://aws.amazon.com/s3/
.. [ecs] Amazon. Amazon Elastic Container Service, https://aws.amazon.com/ecs/
.. [yolo] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640 [cs.CV], Jun 2015
.. [ssd] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg *SSD: Single Shot MultiBox Detector*, arXiv:1512.02325 [cs.CV], Dec 2015
.. [yolov2] Joseph Redmon, Ali Farhadi. *YOLO9000: Better, Faster, Stronger*, arXiv:1612.08242 [cs.CV], Dec 2016
.. [yolov2_batch] S. Ioffe and C. Szegedy. *Batch normalization: Accelerating deep network training by reducing internal covariate shift*, arXiv preprint arXiv:1502.03167, Feb 2015.
.. [FCN] Jonathan Long, Evan Shelhamer, Trevor Darrell *Fully Convolutional Networks for Semantic Segmentation*. 2015, https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
.. [unet] Olaf Ronneberger, Philipp Fischer, Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*, arXiv:1505.04597 [cs.CV], May 2015.
.. [resnet] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, arXiv:1512.03385 [cs.CV], Dec 2015.
.. [pspnet] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, *Pyramid Scene Parsing Network*, arXiv:1612.01105 [cs.CV], Dec 2016.
.. [hnm] N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in IEEE Conference on Computer Vision and Pattern Recognition, 2005.
.. [merge] https://s3.amazonaws.com/robosat-public/3339d9df-e8bc-4c78-82bf-cb4a67ec0c8e/features/index.html#16.37/33.776449/-84.41297
.. [osm_zoom] OpenStreetMap Wiki. Zoom Levels. 20 June 2018, https://wiki.openstreetmap.org/wiki/Zoom_levels
.. [FPN] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. *Feature Pyramid Networks for Object Detection*, arXiv:1612.03144 [cs.CV] Dec 2016



