:author: Varun Kapoor
:email: varun.kapoor@curie.fr
:institution: Institut Curie
:institution: Paris, France
:orcid: 
:corresponding:

:author: Claudia Carabana Garcia
:email: claudia.carabana-garcia@curie.fr
:institution: Institut Curie
:institution: Paris, France
:orcid: 



:video: 

------------------------------------------------
Cell Tracking in 3D using deep learning segmentations
------------------------------------------------

.. class:: abstract

Biological cells can be highly irregular in shape and move across planes making it difficult to be detect and track them in 3D with high level of accuracy. In order to solve the detection problem of such cells we created SmartSeeds algorithm to do instance segmentation of oddly shaped cells in 3D and provide jupyter notebooks for training and applying model prediction with our use case.
We also present an open source tool that can use used for tracking such cells using customised cost function to solve linear assignment problem and Jaqman linker for linking the tracks of dividing and merging cells. The tool is available as a Fiji plugin and post analysis of tracks is a python and Napari based tool for plotting relevant information coming out of the tracks.



.. class:: keywords

   segmentation, tracking, deep learning, irregular shaped cells

Introduction
------------
Studying the dynamics of biological cells is key to understanding key biological questions. Such questions involve imaging the cells under different conditions, tracking their development over time and extracting relevant dynamical parameters such as cell intensity variation, cell size change, cell velocity, cell to tissue boundary distance change over time etc. Imaging conditions can be highly variable and have different sources of noise which degrades the quality of the image and with the increasing size of the data acquired using these microscopes it is imperative to have automated algorithms to enable their quantification. Such analysis requires reliable segmentation of cells followed by tracking software to track their motion and finally a track analysis software to extract the relevant information. In out work we develop a technique to segment cells if irregular shape in 3D including the cells that are very faint in their intensity signal. We use a combination of deep learning and computer vision techniques and by using a combination of these two we obtain a parameter free segmentation dependent only on the quality of the training data. 




Condition to check if the seed based on U-net has already been found by stardist algorithm, if so no new seed is introduced coming for U-net and only stardist seed is accepted as valid.


.. code-block:: python

  def iou3D(boxA, centroid):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [Conditioncheck(centroid, boxA, p, ndim) for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside

  def Conditioncheck(centroid, boxA, p, ndim):
    
      condition = False
    
      if centroid[p] >= boxA[p] and centroid[p] <= boxA[p + ndim]:
          
           condition = True
           
      return condition    
      
    

U-net model prediction:

.. code-block:: python

  def UNETPrediction3D(image, model, n_tiles, axis):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0
    #Postprocessing steps
    Filled = binary_fill_holes(Binary)
    Finalimage = label(Filled)
    Finalimage = fill_label_holes(Finalimage)
    Finalimage = relabel_sequential(Finalimage)[0]
    
          
    return Finalimage

Stardist model prediction:

.. code-block:: python

  def STARPrediction3D(image, model, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True):
    
      copymodel = model
      image = normalize(image, 1, 99.8, axis = (0,1,2))
      shape = [image.shape[1], image.shape[2]]
      image = zero_pad_time(image, 64, 64)
      grid = copymodel.config.grid
      MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
      SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)
      StarImage = MidImage[:image.shape[0],:shape[0],:shape[1]]
      SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
      Probability = np.zeros([SmallProbability.shape[0] * grid[0],SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2] ])
      Distance = np.zeros([SmallDistance.shape[0] * grid[0], SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2] ])
    	 #We only allow for the grid parameter to be 1 along the Z axis
    	for i in range(0, SmallProbability.shape[0]):
             Probability[i,:] = cv2.resize(SmallProbability[i,:], dsize=(SmallProbability.shape[2] * grid[2] , SmallProbability.shape[1] * grid[1] ))
             Distance[i,:] = cv2.resize(SmallDistance[i,:], dsize=(SmallDistance.shape[2] * grid[2] , SmallDistance.shape[1] * grid[1] ))
    
        if UseProbability:
        
        			MaxProjectDistance = Probability[:image.shape[0],:shape[0],:shape[1]]

        else:
        
        			MaxProjectDistance = Distance[:image.shape[0],:shape[0],:shape[1]]

    	if MaskImage is not None:
        
       		if smartcorrection is None: 
          
         		 Watershed, Markers = WatershedwithMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid )
         		 Watershed = fill_label_holes(Watershed.astype('uint16'))
    
       		if smartcorrection is not None:
           
          		Watershed, Markers = WatershedSmartCorrection3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid, smartcorrection = smartcorrection )
          		Watershed = fill_label_holes(Watershed.astype('uint16'))

    	if MaskImage is None:

       		 Watershed, Markers = WatershedNOMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), grid)
       

      return Watershed, Markers, StarImage  
        
Watershedding is done on either the probability map or the distance map coming from stardist using the seeds coming from a combination of U-net and stardist predictions.        


.. code-block:: python     


  def WatershedwithMask3D(Image, Label,mask, grid): 
  
    properties = measure.regionprops(Label, Image) 
    binaryproperties = measure.regionprops(label(mask), Image) 
    
    
    Coordinates = [prop.centroid for prop in properties] 
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    
    Binarybbox = [prop.bbox for prop in binaryproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1], k[2]]) 
    
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iou3D(box, star) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])    
                         
    
    Coordinates.append((0,0,0))


    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int) 
    
    markers_raw = np.zeros_like(Image) 
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates)) 
    markers = morphology.dilation(markers_raw.astype('uint16'), morphology.ball(2))


    watershedImage = watershed(-Image, markers, mask = mask.copy()) 
    
    return watershedImage, markers         

References
----------
..  [Stardist] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers,Cell detection with star-convex polygons, in Proceedings of MICCAI'18, 2018, pp. 265-273.
..  [Unet] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, in Proceedings of MICCAI'15, 2015, pp. 234-241.


