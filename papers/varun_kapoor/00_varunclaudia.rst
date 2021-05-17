:author: Varun Kapoor
:email: varun.kapoor@curie.fr
:institution: Institut Curie
:institution: Paris, France

:corresponding:

:author: Claudia Carabana Garcia
:email: claudia.carabana-garcia@curie.fr
:institution: Institut Curie
:institution: Paris, France

:bibliography: ``vollseg``

------------------------------------------------------------------------------------------------
Cell Tracking in 3D using deep learning segmentations
------------------------------------------------------------------------------------------------

.. class:: abstract


We present tools for doing segmentation, tracking and track analysis of biological cells. Cells coming from different organisms can be highly irregular in shape and are often imaged in 3D. They move across Z planes making it difficult to be detect and track them. Tracking requires detection of cells with accurate shape information, we developed an easy to use deep learning algorithm with jupyter notebook interface to segment cells of differnet shapes and intensity. Jupyter notebook interface comes with training, prediction and other required notebooks to obtain segmentations on images coming directly out of the microscope. For cell tracking we used the codebase of popular Fiji software Trackmate and customised it to do cell tracking inside the tissue with auto removal of un-real tracklets of dividing trajectories. We also created an interface in Napari, an Euler angle based viewer to view the tracks along a chosen view making is possible to follow a cell along the plane of motion. We make a customised Napari widget with front end jupyter notebooks to do track analysis.




.. class:: keywords

   segmentation, tracking, deep learning, irregular shaped cells

Introduction
------------
Studying the dynamics of cells is key to understanding biological questions. Such questions range from morphological changes of developing embryo to how cell decide their fate in developing embryos. Cells are often tagged with fluroscent proteins which express over time and such expression can be recorded as intensity changes of the cell. For some such and many other biological questions microscopy is a key technique to obtain the needed data. Data analysis requires use of image analysis techniques which have received a major boost in bio-image processing since the advent of deep learning :cite:`Rasse2020`. Image analysis almost always requires segmentation of cells/tissues, tracking developement of tissue or movements of cells over time or other metric to quantify biological aspects of cellular changes. Imaging conditions can be highly variable and have different sources of noise which degrades the quality of the image, this makes the task of segmentation tedious and of recent we can use deep learning based restoration techniques for denoising such images to make the downstream analysis easy :cite: `Florian2019`. Such restoration techniques enable easy segmentation of cells using conventional computer vision based tools. The restoration training often requires creating image pairs of high and low noise and it could be not possible for certain organisms or microscopy techniques (Claudia your example maybe?). It is also now possible to have image restoration when there are no training pairs available :cite:`n2v` after which segmentation task can be carried out using standard or with deep learning techniques. Deep learning techniques for restoration and segmentation are especially useful when doing live image of cells since during such experiments the cells can not be exposed to intense laser light as phototoxicity is harmful for such experiments and when imaged in low light the signal to noise ratio is low. There could also be non uniform illumination which makes some regions of the image darker than the other or the signal may become very weak over the cours eof the experiment that can last for hours to days. For such challenges deep learning based restorations and segmentations are more appropriate for having a parameter free segmentation that does not have to be tuned for each experiment as networks can be trained on a varriety of faint and bright cells alike. For doing semantic segmentation we use a U-net network that classifies pixels as foreground and background. For overlapping cells it was shown that stardist is superior compared to multi-class U-net networks for identifying overlapping cells as it learns the concept of a cell as a distance map and approximates the cell shape as a converx polygon. Such an approach works well for roundish cells but performs poorly for irregular shaped cells in 3D. In our algorithm we combine the strengths of semantic and instance segmentation networks to create a seed pool in 3D of all the cells present in the image and use the distance map of stardist to do a marker controlled watershed with U-net result serving as a mask for the watershed process. We show that this approach leads to proper reconstruction of overlapping irregular shaped cells. 

Reliable segmentation is neccessary for any tracking software to track cell/tissue motion. Cell tracking is also challenging in biology as cells have an erratic motion and can move in and out of frames creating gaps in tracks. Using simply the cetroid information to track the cells is not very accurate in such cases and requires customizable cost function matrix to link the cells in the first instance and then link the segments of the connected cells in the second instance. The first instance is always needed to have a high accuracy of cells being linked based on their size/shape and intensity attributes and the second instance of linking segments is needed to account for dividing and splitting trajectories. With such customized cost function the tracking software also needs a track editing interface to correct for the mistakes or override the autoamted tracks the software detects. Such a software exists as a Fiji plugin called Trackmate, we developed our tracking solution using its codebase. In our tracking solution called btrackmate we only track the cells that are inside a tissue and this can be input to the program as image files of csv file of the cell atrributes. We provide Jupyter notebooks to create such csv files that serve as an input to the tracker. Furthermore we also add some biological context in the tracking process of segment linking where after segment linking is done a track inspector removes tracklets that are shorter than a user defined time length. This avoids the tedious manual correction of removing such unphysical tracklets. For viewing of cells in 3D the default viewer of Fiji which is a hyperstack displayer may be a bit cumbersome hence we created a bridge between the Fiji and python world by providing a track xml file exporter to view the tracks in Napari, which is an Euler angle based viewer providing easy to navigate interface to visualize the cells and tracks along a user chosen view. We also provide track analysis tools to extract biological information from these tracks along with saving of the tracks along a chosen view. All the tools we created are available as pip packages and Fiji plugins that can be obtained from the Fiji update cite. 

.. _vollseg: https://github.com/kapoorlab/VollSeg
.. _bTrackmate: https://github.com/kapoorlab/BTrackMate
.. _napatrackmater: https://github.com/kapoorlab/NapaTrackMater





Segmentation
-----------------
Our segmentation task required segmentation of cells coming from developing mouse embryo in 3D. These cells are imaged in low light to avoid phototoxicity and are irregular in shape and intensity. Any bio image analysis task starts with segmentation of such cells coming out of a microscope. In order to avoid phototoxicity that leads to cell death the imaging conditions have to be modulated to not have too high laser intensity under which the cells are imaged in. This leads to a low signal to noise ratio image dataset. Segmentation of such cells could be tedious with the conventional computer vision based techniques alone which almost always will lead to over segmentation in such images :cite:`Rasse2020` . However given enough training data, deep learning networks can be trained to achieve the same task with high degree of accuracy. Segmentation tasks can broadly be divided into semantic or instance segmentation methods. In the semantic segmentation approach only background-foreground segmentation is performed where the pixels are classified either belonging to an object class or not, in the instance segmentation approach the object pixels are classified as belonging to object A or B. In our case we use U-net to perform semantic segmentation of the cells in 3D. U-net is independent of shape of the cell hence can do a good semantic segmentation task, if the cells do not overlap connected component analysis alone is enough to segment the cells. But often in timelapses the cells often overlap and this requires a network that can do instance segmentation. Stardist has proven to be network that performs well in such segmentation tasks compared to other available networks for biological cells. Stardist is an N + 1 channel U-net network where N output channels are distance from the center of the cell to the boundary over a range of angles and a single channel for foreground-background pixel probability map. Using this distance information a mathematically abstract representation of a cell can be learnt by the network. The limitation of this network is that it works reliably for star-convex shapes and does not perform well if the shape of the cells is irregular. Furthermore it is dependent on two parameters to avoid over/under-segmentation, the probability threshold and the non-maximal supression threshold.

We combine the strengths of both the networks in the following way: We perform the semantic segmentation using U-net, the foreground predicted pixels serve as the mask image we use later. Then we use noise to void to denoise the image prior to applying stardist prediction on them. The stardist prediction gives us convex polygon approximation to the cells and a distance map of the cell. Given our denoising step we are able to obtain a distance map that can then be used as base image of performin gthe watershed operation on. THe convex polygons are shrunk down to obtain seeds, then use do connected components on the U-net result to obtain a label image and for each label we search in the stardist seed pool for existence of a seed. If no such seed is found the U-net seed is accepted as a valid seed else it is rejected. Post this seed pooling we perform watershed on the distance map and the overlapping/non-overlapping cells are basins of the energy map. With such an approach we are able to segment fain tand bright cells alike in the same frame and obtain reliable shape as shown in Fig.

Network Training
---------------------

To train U-net and stardist networks for the segmentation task we created labelled training dataset of the cells in 3D. There are several network hyperparameters that have to be chosen to ensure that the model is not over or under fitting to the data. Such hyperparameters include the network depth, the starting number of convolutional filters that double with depth thereby increasing the number of optimization parameters of the network. For a network to generalize well on unseen data we need to fine tune these parameters. 
 
We trained several networks, compared their training and validation losses and also measured their performance on ground truth data the networks to asses their performance. In order to assess the performance of the segmentation we use object level metric used in :cite:`schmidt2018` :cite:`weigert2020`. We compute true positive (TP)  as intersection over union of the predicted and the ground truth being greater than a given threshold, :math:`$\tau \in [0,1]$` Unmatched objects are false positives (FP)  and unmatched ground truth objects are false negatives (FN). We then compute average precision :math:`$AP_\tau= \frac{TP_\tau}{TP_\tau+ FP_\tau + FN_\tau} $`

evaluated across several Z stacks. We also compute mean squared error between the ground truth and the predicted results. In Fig. we show the stardist, unet and results from our approach (vollseg). We also show the results as plots in Fig.:ref:metrics U-net has low performance when it comes to object level segmentation as two channel unet can not do instance segmentation and hence shows poor object level detection scores but good true positive rate. But at a semantic level U-net is better than stardist at resolving the shape of the objects, vollseg even has a better performance compared to both due to our pooling approach that obtains the instance level information from stardist and cell shape information from U-net. Fig.:ref:mse. 

.. _fig-metrics:

.. figure:: figs/Metrics.png

   Metric of comparision between 1) VollSeg, 2) Stardist, 3) Unet.
   
.. _fig-mse:
   
.. figure:: figs/MSE.png

   Mean Squared error comparision between VollSeg,  Stardist, Unet.
   
   
.. _fig-GTVoll:

.. figure:: figs/GTVoll.png

   Visual 3D segmentation comparision between 1) GT segmentation (top) and 2) VollSeg segmentation (bottom).
   
.. _fig-GTUnet:
   
.. figure:: figs/GTUnet

   Visual  3D segmentation comparision between 1) GT segmentation (top) and 2) Unet segmentation (bottom).     
   
   
.. _fig-GTStar:
   
.. figure:: figs/GTStar.png

   Visual 3D segmentation comparision between 1) GT segmentation (top) and 2) Stardist segmentation (bottom).  
   



The code for the seed criteria is shown below

.. code-block:: python

  def iou3D(boxA, centroid):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [Conditioncheck(centroid, boxA, p, ndim)
     for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside

  def Conditioncheck(centroid, boxA, p, ndim):
    
      condition = False
    
      if centroid[p] >= boxA[p] 
      and centroid[p] <= boxA[p + ndim]:
          
           condition = True
           
      return condition 
      
      
After obtaining the pool of seeds we can perform watershedding on either the distance map coming from stardist or the pixel probability map that is also an output of the stardist algorithm. We use U-net semantic segmentation as a mask in the watershedding process. The code for doing so is shown below     

.. code-block:: python     


  def WatershedwithMask3D(Image, Label,mask, grid): 
  
    properties = measure.regionprops(Label, Image) 
    binaryproperties = 
    measure.regionprops(label(mask), Image) 
    cord = 
    [prop.centroid for prop in properties] 
    bin_cord =
    [prop.centroid for prop in binaryproperties]
    Binarybbox = 
    [prop.bbox for prop in binaryproperties]
    cord = sorted(cord , 
    key=lambda k: [k[0], k[1], k[2]]) 
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = 
                [iou3D(box, star) for star in cord]
                
                if not any(inside) :
                         cord.append(bin_cord[i])    
                         
    
    cord.append((0,0,0))
    cord = np.asarray(cord)
    cord_int = np.round(cord).astype(int) 
    
    markers_raw = np.zeros_like(Image) 
    markers_raw[tuple(cord_int.T)] =
    1 + np.arange(len(cord)) 
    markers = 
    morphology.dilation(markers_raw,
    morphology.ball(2))

    watershedImage = 
    watershed(-Image, markers, mask) 
    
    return watershedImage, markers 
    
Here the Label comes from stardist prediction and mask comes from the U-net prediction. 
The result of this approach is a 3D instance segmentation which we obtain for the luminal cells as shown in Fig.{1}. In the software package we provide training and prediction notebooks for training the base U-net and stardist networks on your own dataset. The package comes with jupyter notebooks for training and prediction on local GPU servers and also on Google Colab.

Interactive codebase
-----------------------------

To train your networks using vollseg, install the code via pip install vollseg in your tensorflow enviornment with python > 3.7 and < 3.9. The first notebook needed is the one with takes the training dataset as input and creates npz file for U-net training, specify the path to your data that contains the folder of Raw and Segmentation images with the same name of images to create training pairs. Also to be specified is the name of the generated npz file along with the model directory to store the h5 files of the trained model and the model name.

.. code-block:: python

  Data_dir = '/data/'
  NPZ_filename = 'VolumeSeg'
  Model_dir = '/data/'
  Model_Name = 'VolumeSeg'
  
  
In the next cell specify the model parameters, these parameters are the patch size chosen for training in XYZ for making overlapping patches for training, the number of patches to make the training data are also to be specified. The network depth is an important hyperparameter, more the network depth more are the number of parameters in the network and the image patch size has to be big enough so that when downsampling happens with increasing depth the size of the image in the inner most layer is still greater than 1. Start number of convolutional filters is another crucial hyperparameter controlling the network learning capacity. The number of filters are double at each layer of the network and depending on the size of the training dataset and of the GPU memory capacity this parameter can be tuned when doing hyperparameter optimization to obtain the best model for the given dataset. In this cell as a first step we generate the npz file for U-net training by setting the boolean GenerateNPZ to be true. Then in the next cell we can either train U-net and stardist network sequentially by setting TrainUNET and TrainSTAR booleans to be true or the users can split the training task between two GPUs by making a copy of the notebook and training one network per notebook. The other parameters to be chosen are the number of epochs for training, kernel size of the convolutional filter, the number of rays for stardist network to create a distance map along these directions. Additionally some of the OpenCL computations can be perfromed on a GPU using gputools library and if that is installed in the enviornment you can set use_gpu_opencl to be true. 
  
.. code-block:: python

  #Network training parameters
  NetworkDepth = 5
  Epochs = 100
  LearningRate = 1.0E-4 
  batch_size = 1
  PatchX = 256
  PatchY = 256
  PatchZ = 64 
  Kernel = 3
  n_patches_per_image = 16
  Rays = 128 
  startfilter = 48
  use_gpu_opencl = True
  GenerateNPZ = True
  TrainUNET = False
  TrainSTAR = False  
  
After the network has been trained it will save the config files of the training configuration for both the networks along with the weight vector file as h5 files that will be used by the prediction notebook.  
  
For running the network prediction on XYZ shape images use the prediction notebook either locally or on Colab. In this notebook you only have to specify the path to the image and the model directory. The only two parameters to be set here are the number of tiles (for creating image patches to fit in the GPU memory) and min_size in pixel units to discard segmented objects below that size. Since we perform watershed on either the probability map or the distance map coming out of stardist the users can choose the former by setting UseProbability variable to true or by default we use the distance map.  The code below operates on a directory of XYZ shape images.

.. code-block:: python
 
     ImageDir = 'data/tiffiles/'
     Model_Dir = 'data/' 
     SaveDir = ImageDir + 'Results/'
     UNETModelName = 'UNETVolumeSeg'
     StarModelName = 'VolumeSeg'

     UnetModel = CARE(config = None, 
     name = UNETModelName, 
     basedir = Model_Dir)
     StarModel = StarDist3D(config = None, 
     name = StarModelName, 
     basedir = Model_Dir)
     Raw_path = 
     os.path.join(ImageDir, '*.tif')
     filesRaw =
     glob.glob(Raw_path)
     filesRaw.sort
     min_size = 5 
     n_tiles = (1,1,1)
     for fname in filesRaw:
     
          SmartSeedPrediction3D(ImageDir,
          SaveDir, fname, 
          UnetModel, StarModel, 
          min_size = min_size, 
          n_tiles = n_tiles, 
          UseProbability = False)


Tracking
------------

After we obtain the segmentation using our approach we create a csv file fo the cell attributes that include their location, size and volume of the segmented cells. We use this csv file of the cell attributes as input to the tracker along with the Raw image. The Raw image is used to measure the intensity signal of the segmented cels while the segmentation is used to do the localization of the cells which we want to track. We do the tracking in Fiji, which is a popular software among the biologists. We developed our code over the existing tracking solution called Trackmate :cite:`Tinevez2017`. Trackmate uses linear assingment  problem (LAP) algorithm to do linking of the cells and uses Jaqman linker for linking the segments for dividing and merging trajectories. We introduced a new parameter of minimum tracklet length to aid in the track editing tools also provided in the software. Hence by introducing a biological context of not having very short trajectories we reduce the track editing effort to correct for the linking mistakes made by the program. For testing our tracking program we used a freely available dataset from the cell tracking challenge of a developing C.elegans embryo. Using our software we can remove cells from tracking which do not fit certain criteria such as being too small (hence most likely a segmentation mistake) or being low in intensity or outside the region of interest such as when we want to track cells only inside a tissue. For this dataset we kept 12,000 cells and after filtering short tracks kept about 50 tracks with and without division events. The track information is saved as an XML file and can be re-opened to perform track editing from the last saved checkpoint. This is particularly useful when editing tracks coming from a huge dataset.

For this dataset the track scheme along with overlayed tracks in shown in Fig. The trackscheme is interactive as selecting a node in the trackscheme highlights the cell in Green and by selecting a cell in the image highlights its location in the trackscheme. Extensive manual for using the track editing is available on Fiji wiki.


.. _fig-trackscheme:

.. figure:: figs/trackscheme.png

   Trackscheme display for the C-elegans dataset.
   
   

Track Analysis
------------------------

After obtaining the tracks from bTrackmate we save them as Trackmate XML file, this file contains the information about all the cells in a track. Since the cells can be highly erratic in their motions and move in not just the XY plane but also in Z we needed an Euler angle based viewer to view such tracks from different camera positions, recently a new and easy to use viewer based on python called Napari came into existence. Using this viewer we can easily navigate along multi dimensions, zoom and pan the view, toggle the visibility of image layers etc. We made a python package to bridge the gap between the Fiji and the Napari world by providing a track exporter that can read in the track XML files coming from the Fiji world and convert them into the tracks layer coming form the Fiji world. We use this viewer not just to view the tracks but also to analyze and extract the track information. As a first step we separate the dividing trajectories from the non-dividing trajectories, then in one notebook we compute the distance of the cells in the track from the tissue boundary and record the starting and the end distance of the root tracks and the succeeding tracklets of the daughter cells post division for dividing trajectories and only the root track for the non-dividing trajectory. This information is used to determine how cell chooses its fate, does it start from inside the tissue and remain inside during the duration of the experiment or does it move closer to the tissue boundary. This information is crucial when studying the organism in the early stage of development where the cells are highly dynamic and their fate is not known a priori.

Also another quantity of interest that can be obtained from the tools is quantification of intensity oscillations over time. In certain conditions there could be an intensity oscillation in the cells due to certain protein expression that leads to such oscillations, the biological question of interest is if such oscillations are stable and if so what is the period of the oscillation :cite:`Lahmann2019`. Using our tool intensity of individual tracklet can be obtained which is then Fourier transformed to show the oscillation frequency if any. With this information we can see the contribution of each tracklet in the intensity oscillation and precisely associate the time when this oscillation began and ended.

.. _fig-distancediv:

.. figure:: figs/DistanceDividing1.png

   Parent cell before division.
   
.. _fig-distancediv2:

.. figure:: figs/DistanceDividing2.png

   Parent cell after division, one daughter cells moves inside while other stays close to the boundary.   
   
      
    


        

References
--------------------
..  [Stardist] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers,Cell detection with star-convex polygons, in Proceedings of MICCAI'18, 2018, pp. 265-273.
..  [Unet] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, in Proceedings of MICCAI'15, 2015, pp. 234-241.
..  [Ines] Lahmann I, Brohl D, Zyrianova T, et al. Oscillations of MyoD and Hes1 proteins regulate the maintenance of activated muscle stem cells. Genes & Development. 2019 May;33(9-10):524-535. DOI: 10.1101/gad.322818.118.
..  [TM] Tinevez JY, Perry N, Schindelin J, Hoopes GM, Reynolds GD, Laplantine E, Bednarek SY, Shorte SL, Eliceiri KW. TrackMate: An open and extensible platform for single-particle tracking. Methods. 2017 Feb 15;115:80-90. doi: 10.1016 j.ymeth.2016.09.016. Epub 2016 Oct 3. PMID: 27713081.




