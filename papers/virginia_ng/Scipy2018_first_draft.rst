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

   Deep learning techniques have greatly advanced the performance of the already rapidly developing field of computer vision, which powers a variety of emerging technologies—from facial recognition to augmented reality to self-driving cars. The remote sensing and mapping communities are particularly interested in extracting, understanding and mapping physical elements in the landscape. These mappable physical elements are called features, and can include both natural and synthetic objects of any scale, complexity and character. Points or polygons representing sidewalks, glaciers, playgrounds, entire cities, and bicycles are all examples of features. In this paper we present a method to develop deep learning tools and pipelines that generate features from aerial and satellite imagery at large scale. Practical applications include object detection, semantic segmentation and classification, and automatic mapping of general-interest features such as turn lane markings on roads, parking lots, roads, water, building footprints.

   We give an overview of our data preparation process, in which data from the Mapbox Satellite layer, a global imagery collection, is annotated with labels created from OpenStreetMap data using minimal manual effort. We then discuss the implementation of various state-of-the-art detection and semantic segmentation systems, such as YOLOv2, U-Net, Residual Network, and Pyramid Scene Parsing Network (PSPNet), as well as specific adaptations for the aerial and satellite imagery domain. We conclude by discussing our ongoing efforts in improving our models and expanding their applicability across classes of features, geographical regions, and relatively novel data sources such as street-level and drone imagery.


.. class:: keywords

   computer vision, deep learning, neural networks, satellite imagery, aerial imagery



I. Background
-------------

Location data is built into the fabric of our daily experiences, and is more important than ever with the introduction of new location-based technologies such as self-driving cars. Mapping communities, open source or proprietary, work to find, understand and map elements of the physical landscape. However, mappable physical elements are continually appearing, changing, and disappearing. For example, more than 1.2 million residential units were built in the United States alone in 2017 [buildings]_. Therefore, a major challenge faced by mapping communities is maintaining recency while expanding worldwide coverage. To increase the speed and accuracy of mapping, allowing better pace-keeping with change in the mappable landscape, we propose integrating deep neural network models into the mapping workflow. In particular, we have developed tools and pipelines to detect various geospatial features from satellite and aerial imagery at scale. We collaborate with the OpenStreetMap [osm]_ (OSM) community to create reliable geospatial datasets, validated by trained and local mappers.

Here we present two usecases to demonstrate our workflow for extracting street navigation indicators such as turn restrictions signs, turn lane markings, and parking lots, in order to improve our routing engines. We designed our processing pipelines and tools with open source libraries including Scipy, Rasterio, Fiona, Osium, JOSM [#]_, Keras, PyTorch, and OpenCV, while our training data was compiled from OpenStreetMap and the Mapbox Maps API [mapbox_api]_. Our tools are designed to be generalizable across geospatial feature classes and across data sources.

.. [#] JOSM [josm]_ is an extensible OpenStreetMap editor for Java 8+. At its core, it is an interface for editing OSM, i.e., manipulating the nodes, ways, relations, and tags that compose the OSM database. Compared to other OSM editors, JOSM is notable for its range of features, such as allowing the user to load arbitrary GPX tracks, background imagery, and OpenStreetMap data from local and online sources. It is open source and licensed under GPL.


II. Scalable Computer Vision Pipelines
-----------------------------------------

The general design for our deep learning based computer vision pipelines can be found in Figure 1, and is applicable to both object detection and semantic segmantation tasks. We design such pipelines with two things in mind: we must allow scalability to very large data volumes, which requires processing efficiency; and we must allow repurposability towards computer vision tasks on other geospatial features, which requires a general-purpose design. We present turn lane markings as an example of an object detection pipeline, and parking lot segmentation as an example of a semantic segmantation pipeline.

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

**Data Preparation For Object Detection.** The training data for turn lane marking detection was created by collecting imagery of various types of turn lane markings and manually drawing a bounding box around each marking. We used Overpass Turbo [overpass]_ to query the OpenStreetMap database for streets containing turn lane markings, i.e., those tagged with one of the following attributes - “\turn:lane=*”, “\turn:lane:forward=*”, “\turn:lane:backward=*” in OpenStreetMap. The marked street segments, as shown in Figure 3, were stored as GeoJSON features clipped into the tiling scheme of the Mapbox Satellite basemap [mapbox]_.. Figure 4 shows how skilled mappers used this map layer as a cue to manually draw bounding boxes around each turn lane marking using JOSM, a process called annotation. Each of these bounding boxes was stored as a GeoJSON polygon on Amazon S3 [s3]_.

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


Mappers annotated six classes of turn lane markings - “\Left”, “\Right”, “\Through”, “\ThroughLeft”, “\ThroughRight”, and “\Other” in five cities, creating a training set consists of over 54,000 turn lane markings. Turn lane markings of all shapes and sizes, as well as ones that are partially covered by cars and/or shadows were included in this training set. To ensure a high-quality training set, we had a separate group of mappers verify each of the bounding boxes drawn. We excluded turn lane markings that are erased or fully covered by cars, seen in Figure 5.

.. figure:: fig5.png
   :height: 75 px
   :width: 150 px
   :scale: 21 %

   Obscured turn lane markings, such as those covered by cars, are excluded.

Semantic segmentation, on the other hand, is the computer vision task that partitions an image into semantically meaningful parts, and classifies each part into one of any pre-determined classes. This can be understood as assinging a class to each pixel in the image, or equivalently as drawing non-overlapping polygons with associated classes over the image. For example, in addition to distinguishing the road from the buildings, we also delineate the boundaries of each object shown in Figure 6.

.. figure:: fig6.png
   :height: 75 px
   :width: 150 px
   :scale: 21 %

   Semantic segmentation on roads, buildings and vegetation


**Data Preparation For Semantic Segmentation.** The training data for parking lot segmentation was created by combining imagery collected from Mapbox Satellite with binary masks for parking lots. The binary masks for parking lots were generated from OpenStreetMap polygons with attributes “\tag:amenity=parking=*” using a tool called Osmium [osmium]_. These binary masks were stored as single channel numpy arrays. Each of these single channel numpy arrays were then stacked with its respective aerial image tile in raster format, which is a three channel numpy array - Red, Green, and Blue. We annotated 55,710 masks for parking lot segmentation for our initial training set.

**Data Engineering.** A data engineering pipeline was built within the larger object detection pipeline to create and process training sets in large quantities. This data engineering pipeline is capable of streaming any set of prefixes off of Amazon S3 into prepared training sets. Several pre-processing steps were taken to convert turn lane marking annotations to the appropriate data storage format before combining them with real imagery. As mentioned earlier, turn lane marking annotations were initially stored as GeoJSON polygons grouped by class. Each of these polygons had to be streamed out of the GeoJSON files on S3, converted to image pixel coordinates, and stored as JSON image attributes to actract tiles [tile]_. The pre-processed annotations were then randomly assigned to training and testing datasets, following the 80/20 split rule. Annotations were written to disk and joined by imagery fetched from the Satellite layer of Mapbox Maps API. During this step, the abstract tiles in the pipeline were replaced by real image tiles. Finally, the training and test sets were zipped and uploaded to Amazon S3. 

Before we scaled up processing, we first developed Python command line tools and libraries for our data preparation steps. All of command line tools we developed for the segmentation task can be found on our GitHub repository [robosat]_. These scripts were then ran at large scale, multiple cities in parallel on Amazon Elastic Container Service [ecs]_. This data engineering pipeline is shown in Figure 7.

.. figure:: fig7.png
   :height: 200 px
   :width: 400 px
   :scale: 47 %

   Data engineering pipeline converts OpenStreetMap GeoJSON features to image pixel space and combines each feature with imagery fetched from Mapbox Maps API.

The design of our data engineering pipelines can be generalized to any OpenStreetMap feature. Buildings is another example of an OpenStreetMap feature that we experimented with. One can generate training sets with any OpenStreetMap feature simply by writing customized Osmium handlers to convert OpenStreetMap geometries into polygons.

2. Model
---------

**Fully Convolutional Neural Networks.** Fully convolutional are neural networks composed of convolutional layers without any fully-connected
layers or multilayer perceptron (MLP) usually found at the end of the network. All learning layers in the fully convolutional network are convolutional, including the decision-making layers at the end. There are a few advantages of using fully convolutional neural networks. This type of network can handle variable input image sizes. Convolutional layers are capable of managing different input sizes and are faster at this task, while fully connected layer expects inputs of a certain size. Therefore, by leaving fully connected layers out of a network architecture, one can apply the network to images of virtually any size. Convolutions also enable computation of predictions at different positions in an image in an optimized way. Fully convolutional neural networks used for object detection tasks are therefore more computational efficient than sliding window approaches [cite0]_, in which predictions are computed separately at every potential position. A third advantage is that one would no longer be constrained by the number of object categories or complexity of the scenes when performing spatially dense prediction tasks like segmentation using fully convolutional networks. All output neurons are connected to all input neurons in fully connected layers and therefore generally cause loss of spatial information [cite1]_. 

**Object Detection Models.** Many of our applications require low latency prediction from their object detection algorithms. We implemented YOLOv2 [yolov2]_, the improved version of the real-time object detection system You Look Only Once (YOLO) in our turn lane markings detection pipeline. YOLOv2 outperforms all other state-of-the-art methods like Faster R-CNN with ResNet [resnet]_ and Single Shot MultiBox Detector (SSD) in both speed and detection accuracy [cite0]_. It works by first dividing the input image into an 13 × 13 grid cells. Each grid cell is responsible for generating 5 bounding boxes. Each bounding box contains x, y, w, h, a box confidence score for objectness, and a C class probabilities. x and y are location coordinates relative to the location of the corresponding grid cell. Logistic activation is used to constrain and stabilize the network’s predictions to fall between 0 and 1. Bounding box width w and height h are normalized by the image width and height. The confidence score, or the objectness reflects how likely the box contains an object and how accurate the bounding box is. More specifically, it predicts the IOU of the ground truth and the proposed box. The class probabilities predict the conditional probability of that class given that there is an object. 6 classes of turn lane markings were defined for the turn lane markings detection project. The output of our YOLOv2 network is therefore 13 x 13 x 55, i.e. 5 bounding boxes with 11 parameters: 55 parameters per 13 x 13 grid cell. The base feature extractor of YOLOv2 is called Darknet-19, a fully convolutional neural network composed of 19 convolutional layers and 5 max-pooling layers. Detection is done by replacing the last convolutional layer of Darknet-19 with three 3 × 3 convolutional layers, each outputting 1024 output channels. A final 1 × 1 convolutional layer is then applied to convert the 13 × 13 × 1024 output into 13 × 13 × 55. We followed two suggestions proposed by the YOLO authors when designing our model. The first was incorporating batch normalization after every convolutional layer. Batch normalization stabilizes training, improves the model convergence, and regularizes the model [yolov2_batch]_. The authors saw a 2% improvement in mAP from YOLO on the VOC2007 dataset [yolov2]_. The second suggestion that we implemented was the use of anchor boxes and dimension clusters to predict the actual bounding box of the object. This step was acheieved by running k-means clustering on the turn lane marking training set bounding boxes. As seen in Figure 8, the ground truth bounding boxes for turn lane markings follow specific height-width ratios. Instead of directly predicting bounding box coordinates, our model predicts the width and height of the box as offsets from cluster centroids. The center coordinates of the box relative to the location of filter application is predicted by using a sigmoid function.

.. figure:: fig8.png
   :height: 150 px
   :width: 150 px
   :scale: 38 %

   Clustering box dimensions on turn lane marking training set. We run k-means clustering on the dimensions of bounding boxes to get anchor boxes for our model. We used the suggested k = 5 as suggested by the YOLO authors, who found that k = 5 gives a good tradeoff for recall vs. complexity of the model.

Our model was first pre-trained on ImageNet 224x224 resolution imagery. The network was then resized and finetuned for classification on higher resolution 448x448 turn lane marking imagery to ensure that smaller objects like turn lane markings in a scene are detected.

**Segmentation Models.** For parking lot segmentation, we performed binary segmentation distinguishing parking lots from the background and found U-Net [unet]_ to be a suitable architecture for this task. The U-Net architecture can be found in Figure 9. It consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. This type of network can be trained end-to-end with very few training images and can yield more precise segmentations than prior state-of-the-art methods such as the sliding-window convolutional networks. This first part is called down or one may think it as the encoder part where one apples convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels. The second part of the network consists of upsample and concatenation followed by regular convolution operations. Upsampling in convolutional neural networks is equivalent to expanding the feature dimensions to meet the same size with the corresponding concatenation blocks from the left. While upsampling and going deeper in the network, the higher resolution features from down part is simultaneously concatenated with the upsampled features. This better localizes and learns representations from following convolutions.

.. figure:: fig9.png
   :height: 125 px
   :width: 200 px
   :scale: 38 %

   U-Net Architecture

We initially experimented with Pyramid Scene Parsing Network (PSPNet) [pspnet]_ when performing multi-class segmentation task on building, roads, water, vegetation. As seen in Figure 6, PSPNet was quite effective in producing good quality segmentation masks on multi-class segmentation tasks or scenes that are complex, i.e. scenes that contain visually diverse pixels or objects. We found that it was redundant with our binary usecase for parking lot segmenation where there are only two categories - parking lot versus background. PSPNet adds a multi-scale pooling on top of the backend model to aggregate different scale of global information. The upsample layer is implemented by bilinear interpolation. After concatenation, PSPNet fuses different levels of feature with a 3x3 convolution.

**Hard Negative Mining.** This is a technique we have been using in multiple deep learning-based computer vision pipelines to improve model performance. A hard negative is a negative sample that is explicitly created from a falsely detected patch and added back to our training set after the initial round of inference. When we retrain our models with this extra knowledge, they usually perform better and produce less false positives.

Figure 10 shows an example of a probability mask which is rendered as a custom layer overlaying on top of Mapbox Satellite. It helps us visualize what our model predicts are pixels belonging to parking lots. The average over multiple IOU (AP) of our baseline model U-Net is 46.7 on a test set of 900 samples. We use this probability mask to do visual debugging and find falsely detected patches to create hard negatives for hard negative mining.


.. figure:: fig10.png
   :height: 150 px
   :width: 150 px
   :scale: 48 %

   Probability mask specifying the pixels that our model believes belong to parking lots.


3. Post-Processing
------------------

Figure 11 shows an example of the raw segmentation mask derived from our U-Net model. It cannot be used directly as input into OpenStreetMap. We performed a series of post-processing to improve the quality of the segmentation mask and to transform the mask into the right data format for OpenStreetMap consumption.


.. figure:: fig11.png
   :height: 150 px
   :width: 150 px
   :scale: 47 %

   An example of border artifacts and holes observed in raw segmentation masks derived from our U-Net model


**Noise Removal.** Noise in the output mask is removed by performing two
morphological operations: erosion followed by dilation. Erosion removes
white noises, but it also shrinks an object. Therefore, dilation is used as a second step.

**Fill in holes.** Holes in the output mask are filled by performing dilation
followed by erosion. This morphological operation is especially useful in closing small holes
inside the foreground objects, or small black points on an object. Specifically for our parking lot polygons, this operator is used to deal with polygons within polygons.

**Contouring.** During this step, binary masks are drawn given its rasterized contours - a set of continuous points along the boundary, having same color or intensity.

**Simplification.** Douglas-Peucker Simplification takes a curve composed of line segments and finds a similar curve with fewer points.
Cleaner polygons with the minimum number vertices are created during this step so that their shapes are more consistent with that of OpenStreetMap features.

**Transform Data.** Detection and segmentation results are converted from pixel space back into GeoJSONs (world coordinate).


**Merging multiple polygons.** Shown in Figure 12, this tool merges GeoJSON features crossing tile boundaries as well as adjacent features
into a single polygon [merge]_.

**Deduplication.** Deduplicates by matching GeoJSONs with data that already exist on OpenStreetMap, so that only detections that have yet been mapped are uploaded.

After the series post-processing steps described above is performed, we have a clean GeoJSON polygon. An example of such a polygon can be found in Figure 13. This can now be added to OpenStreetMap as a parking lot feature.


.. figure:: fig12.png
   :height: 400 px
   :width: 800 px
   :scale: 35 %

   GeoJSON features crossing tile boundaries as well as adjacent features are merged into a single polygon



.. figure:: fig13.png
   :height: 250 px
   :width: 250 px
   :scale: 49 %

   Clean mask in the form of GeoJSON polygon




4. Output
----------

With these pipeline designs, we are able to run batch prediction at large scale on millions of image tiles. The output of these processing pipelines are turn lane markings and parking lots in the form of GeoJSON features. We can then add these GeoJSON features back into OpenStreetMap as turn lane and parking lot features. Mapbox routing engines then take these OpenStreetMap features into account when calculating the optimal routes. We are still in the process of making various improvements to our baseline model, therefore we include two manual steps performed by humans as a stopgap. First step is verification and inspection of our model results. Second step is to manually map only the true positive results in OpenStreetMap. Shown in Figure 14 is a front-end UI that allows users to pan around for instant turn lane markings detection.

.. figure:: fig14.png
   :height: 200 px
   :width: 400 px
   :scale: 25 %

   Front-end UI for instant turn lane markings detection


IV. Ongoing Work
----------------
We demonstrated the steps to building deep learning-based computer vision pipelines which enables us to run object detection and segmentation tasks at scale. We built our tools and pipelines so that users can easily expand to other physical elements in the landscape or to other geographical regions of interest. Going forward, we plan on experimenting with the new published and improved YOLOv3 [yolov3]_ for our object detection pipelines. For segmentation, we open sourced our end-to-end semantic segmantion pipeline called Robosat [#]_, along with all its tools in June 2018. We ran the first round of large-scale parking lot segmentation over Atlanta, Baltimore, Sacremanto, and Seattle. We are currently scaling up to run prediction over all of North America where we have great high resolution imagery coverage. We are also experimenting with building segmentation on drone imagery from the OpenAerialMap project in the area of Tanzania [tanzania]_. We are in the process of making several improvements to our models. We recently performed one round of hard negative mining and added 49,969 negative samples to our training set. We are also currently working on replacing the standard U-Net encoder with pre-trained ResNet50 encoder. In addition to these improvements, we are replacing learned deconvolutions with nearest neighbor upsampling followed by a convolution for refinement instead. We believe that this approach gives us more accurate results, while speeding up training and prediction, lowering memory usage. The drawback to such an approach is that it only works for three-channel inputs (RGB) and not with arbitrary channels.

.. [#] Robosat is an end-to-end pipeline for extracting physical elements in the landscape that can be mapped from aerial and satellite imagery https://github.com/mapbox/robosat


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
.. [resnet] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun arXiv:1512.03385 [cs.CV], Dec 2015.
.. [pspnet] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, *Pyramid Scene Parsing Network*, arXiv:1612.01105 [cs.CV], Dec 2016.
.. [merge] https://s3.amazonaws.com/robosat-public/3339d9df-e8bc-4c78-82bf-cb4a67ec0c8e/features/index.html#16.37/33.776449/-84.41297
.. [yolov3]    Joseph Redmon, Ali Farhadi. *YOLOv3: An Incremental Improvement*, arXiv:1804.02767 [cs.CV], Apr 2018
.. [tanzania] daniel-j-h, https://www.openstreetmap.org/user/daniel-j-h/diary/44321



