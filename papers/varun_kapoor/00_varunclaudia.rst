:author: Varun Kapoor
:email: varun.kapoor@curie.fr
:institution: Institut Curie
:institution: Paris, France
:orcid: 
:corresponding:

:author: Claudia Carabana Garcia
:email: claudia.carabana@curie.fr
:institution: Institut Curie
:institution: Paris, France
:orcid: 



:video: 

------------------------------------------------
Cell Tracking in 3D using deep learning segmentations
------------------------------------------------

.. class:: abstract

Biological cells can be highly irregular in shape and move across planes making it difficult to be detect and track them in 3D with high level of accuracy. In order to solve the detection problem of such cells we use deep learning based segmentations which can reliably segment oddly and differently shaped cells in the same image in 3D. We created SmartSeeds algorithm todo instance segmentation of oddly shaped cells in 3D and provide jupyter notebooks for training and applying model prediction with different use cases.We also present an open source tool that can use used for tracking such cells using customised cost function to solve linear assignment problem and Jaqman linker for linking the tracks of dividing and merging cells. The tool is available as a Fiji plugin and post analysis of tracks is a python and Napari based plugin for plotting relevant information coming out of the tracks.



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



Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }
   
Inline code looks like this: :code:`chunk of code`.

Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption.:code:`chunk of code` inside of it. :label:`egfig` 

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).
   This caption also has :code:`chunk of code`.

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it.  :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


