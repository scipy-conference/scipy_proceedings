.. _secappFeatures:

Application Overview
====================
Design decisions throughout S3A's architecture have been driven by the following objectives: metadata should have significance rather than be treated as an afterthought; high-resolution images should have minimal impact on the annotation workflow; ROI density and complexity should not limit annotation workflow; and prototyping should not be hindered by application complexity.
These motives were selected upon noticing the general lack of solutions for related problems in previous literature and tooling. Moreover, applications that *do* address multiple aspects of complex region annotation often require an enterprise service and cannot be accessed under open-source policies.

While the first three points are highlighted in the case study, the subsections below outline pieces of S3A's architecture that prove useful for iterative algorithm prototyping and dataset generation as depicted in Figure :ref:`figfeedbackLoop`. Note that beyond the facets illustrated here, S3A possesses multiple additional characteristics as outlined in its documentation (`https://gitlab.com/s3a/s3a/-/wikis/docs/User's-Guide <https://gitlab.com/s3a/s3a/-/wikis/docs/User's-Guide>`_).

.. raw:: latex

    \makeFeedbackLoopFig

.. _secprocFramework:

Processing Framework
--------------------
At the root of S3A's functionality and configurability lies its adaptive processing framework. Functions exposed within S3A are thinly wrapped using a ``Process`` structure responsible for parsing signature information to provide documentation, parameter information, and more to the UI. Hence, all graphical depictions are abstracted beyond the concern of the user while remaining trivial to specify (but can be modified or customized if desired). As a result, incorporating additional/customized application functionality can require as little as one line of code. Processes interface with PyQtGraph parameters to gain access to data-customized widget types and more (`https://github.com/pyqtgraph/pyqtgraph <https://github.com/pyqtgraph/pyqtgraph>`_).


..
    % .. raw:: latex


    \makeAtomicProcFig

These processes can also be arbitrarily nested and chained, which is critical for developing hierarchical image processing models, an example of which is shown in Figure :ref:`figregionAnalytics`. This framework is used for all image and region processing within S3A. Note that for image processes, each portion of the hierarchy yields intermediate outputs to determine which stage of the process flow is responsible for various changes. This, in turn, reduces the effort required to determine which parameters must be adjusted to achieve optimal performance. 

.. raw:: latex

    \makeRegionAnalyticsFig

.. _secplugins:

Plugins for User Extensions
---------------------------
The previous section briefly described how custom user functions are easily be wrapped within a process, exposing its parameters within S3A in a GUI format. A rich plugin interface is built on top of this capability in which custom functions, table field predictors, default action hooks, and more can be directly integrated into S3A. In all cases, only a few lines of code are required to achieve most integrations between user code and plugin interface specifications. The core plugin infrastructure consists of a function/property registration mechanism and an interaction window that shows them in the UI. As such, arbitrary user functions can be `registered` in one line of code to a plugin, where it will be effectively exposed to the user within S3A.


..
    % .. raw:: latex


    \makeCustomMiscFuncFig

Plugin features are heavily oriented toward easing the process of automation both for general annotation needs and niche datasets. In either case, incorporating existing library functions is converted into a trivial task directly resulting in lower annotation and higher labeling accuracy.

Adaptable I/O
-------------
An extendable I/O framework allows annotations to be used in a myriad of ways. Out-of-the-box, S3A easily supports instance-level segmentation outputs, facilitating deep learning model training. As an example, Figure :ref:`figcropExports` illustrates how each instance in the image becomes its own pair of image and mask data. When several instances overlap, each is uniquely distinguishable depending on the characteristic of their label field. Particularly helpful for models with fixed input sizes, these exports can optionally be forced to have a uniform shape (e.g. 512x512 pixels) while maintaining their aspect ratio. This is accomplished by incorporating additional scene pixels around each object until the appropriate size is obtained. Models trained on these exports can be directly plugged back into S3A's processing framework, allowing them to generate new annotations or refine preliminary user efforts. The described I/O framework is also heavily modularized such that custom dataset specifications can easily be incorporated. In this manner, future versions of S3A will facilitate interoperability with popular formats such as COCO and Pascal VOC.

.. raw:: latex

    \makeCropExportsFig

Deep, Portable Customizability
------------------------------
Beyond the features previously outlined, S3A provides numerous avenues to configure shortcuts, color schemes, and algorithm workflows. Several examples of each can be seen in the `user guide <https://gitlab.com/s3a/s3a/-/wikis/docs/user's-guide>`_. Most customizable components prototyped within S3A can also be easily ported to external workflows after development. Hierarchical processes have states saved in YAML files describing all parameters, which can be reloaded to create user profiles. Alternatively, these same files can describe ideal parameter combinations for functions outside S3A in the event they are utilized in a different framework.

