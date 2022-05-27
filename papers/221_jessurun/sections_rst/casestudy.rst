.. _seccasestudy:

Case Study
==========
Both the inspiration and developing efforts for S3A were initially driven by optical printed circuit board (PCB) assurance needs. In this domain, high-resolution images can contain thousands of complex objects in a scene, as seen in Figure :ref:`figpcb`. Moreover, numerous components are not representable by cardinal shapes such as rectangles, circles, etc. Hence, high-count polygonal regions dominated a significant portion of the annotated regions. The computational overhead from displaying large images and substantial numbers of complex regions either crashed most annotation platforms or prevented real-time interaction. In response, S3A was designed to fill the gap in open-source annotation platforms that addressed each issue while requiring minimal setup and allowing easy prototyping of arbitrary image processing tasks. The subsections below describe how the S3A labeling platform was utilized to collect a large database of PCB annotations along with their associated metadata [#]_.

.. raw:: latex

    \makePcbFig

Large Images with Many Annotations
----------------------------------
In optical PCB assurance, one method of identifying component defects is to localize and characterize all objects in the image. Each component can then be cross-referenced against genuine properties such as length/width, associated text, allowed orientations, etc. However, PCB surfaces can contain hundreds to thousands of components at several magnitudes of size, necessitating high-resolution images for in-line scanning. To handle this problem more generally, S3A separates the editing and viewing experiences. In other words, annotation time is orders of magnitude faster since only edits in one region at a time and on a small subset of the full image are considered during assisted segmentation. All other annotations are read-only until selected for alteration. For instance, Figure :ref:`figregionEdit` depicts user inputs on a small ROI out of a much larger image. The resulting component shape is proposed within seconds and can either be accepted or modified further by the user. While PCB annotations initially inspired this approach, it is worth noting that the architectural approach applies to arbitrary domains of image segmentation.

Another key performance improvement comes from resizing the processed region to a user-defined maximum size. For instance, if an ROI is specified across a large portion of the image but the maximum processing size is 500x500 pixels, the processed area will be downsampled to a maximum dimension length of 500 before intensive algorithms are run. The final output will be upsampled back to the initial region size. In this manner, optionally sacrificing a small amount of output accuracy can drastically accelerate runtime performance for larger annotated objects.

.. raw:: latex

    \makeRegionEditFig

Complex Vertices/Semantic Segmentation
--------------------------------------
Multiple types of PCB components possess complex shapes which might contain holes or noncontiguous regions. Subsequently, it is beneficial for software like S3A to represent these features inherently with a ``ComplexXYVertices`` object: that is, a collection of polygons which either describe foreground regions or holes. Subsequently, any possible shape or shape grouping can be represented as needed. Example components difficult to accomodate with single-polygon annotation formats are illustrated in Figure :ref:`figcomplexRegion`.

.. raw:: latex

    \makeComplexRegionFig

At the same time, S3A also supports high-count polygons with no performance losses. Since region edits are performed by image processing algorithms, there is no need for each vertex to be manually placed or altered by human input. Thus, such non-interactive shapes can simply be rendered as a filled path without a large number of event listeners present. This is the key performance improvement when thousands of regions (each with thousands of points) are in the same field of view. When low polygon counts *are* required, S3A also supports RDP polygon simplification down to a user-specified epsilon parameter :cite:`ramer_iterative_1972`.


..
    % PCB example: In PCB assurance, it can be desirable to semantically annotate PCB components $\rightarrow$ e.g. component footprint analysis. Some compoenents have complex shapes and even holes $\rightarrow$ Complex Vertices case. For many other annotation tools, complex annotations either cannot be done (annotator only supports bounding boxes), must be done completely manually (point-by-point), doesn't support holes, and/or can result in lagging/slowdown. On the other hand, S3A can deal with the complex vertices case because X, Y, and Z. 


..
    % show off screenshot of complex annotation(s) such as a 64-pin IC, large jack, or large connector - bonus if there's an annotation with holes in it bc idk many other annotation apps that allow annotations to have holes -- TODO: insert fig of transistor with holes in heat sink 


..
    % Extension Example: Though this aspect of S3A was inspired by PCB component footprint analysis, there are certainly other applications that deal with complex vertices such as in biomedical, Y, and Z. -- Might not need to be justified since it's a case study, but I'm definitely open to alternative suggestions


Complex Metadata
----------------
Most annotation software support robust implementation of image region, class, and various text tags ("metadata"). However, this paradigm makes collecting type-checked or input-sanitized metadata more difficult. This includes label categories such as object rotation, multiclass specifications, dropdown selections, and more. In contrast, S3A treats each metadata field the same way as object vertices, where they can be algorithm-assisted, directly input by the user, or part of a machine learning prediction framework. Note that simple properties such as text strings or numbers can be directly input in the table cells with minimal need for annotation assistance [#]_. In conrast, custom fields can provide plugin specifications which allow more advanced user interaction. Finally, auto-populated fields like annotation timestamp or author can easily be constructed by providing a factory function instead of default value in the parameter specification.

This capability is particularly relevant in the field of optical PCB assurance. White markings on the PCB surface, known as silkscreen, indicate important aspects of nearby components. Thus, understanding the silkscreen's orientation, alphanumeric characters, associated component, logos present, and more provide several methods by which to characterize / identify features of their respective devices. Both default and customized input validators were applied to each field using parameter specifications, custom plugins, or simple factories as described above. A summary of the metadata collected for one component is shown in Figure :ref:`figmetadata`.

.. raw:: latex

    \makeMetadataFig


..
    % PCB example: In PCB assurance, it can be desirable to associate metadata with elements on a PCB $\rightarrow$ e.g. design rule checking. For example, annotating reference designators, it can be useful to note the text values, text orientation, and any components the reference designator is referrring to $\rightarrow$ complex metadata case. For many other annotation tools, complex metadata either cannot be captured (annotator only supports classes), has a limited number of types of metadata that can be captured (e.g. only string data), and/or doesn't support between-annotation assocaition. On the other hand, S3A can deal with the complex metadata case because X, Y, and Z. 


..
    % show off screenshot of PCB OCR metadata and/or yaml file/table of different types of metadata you can collect. especially highlight the comp locator for complex reference designators (e.g. designators that refer to multiple comps such as R1-R5) -- TODO: insrt fig of yaml file for OCR and link to https://pyqtgraph.readthedocs.io/en/latest/parametertree/parametertypes.html for extensions 


..
    % Extension Example: Though this aspect of S3A was inspired by PCB design rule checking, there are certainly other applications that deal with complex vertices such as in human annotation, Y, and Z. 




.. [#] For those curious, the dataset and associated paper are accessible at `https://www.trust-hub.org/\#/data/pcb-images <https://www.trust-hub.org/\#/data/pcb-images>`_.
.. [#] For a list of input validators and supported primitive types, refer to PyQtGraph's `Parameter <https://pyqtgraph.readthedocs.io/en/latest/parametertree/parametertypes.html>`_ documentation.