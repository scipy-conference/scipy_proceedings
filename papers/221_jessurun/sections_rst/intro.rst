Introduction
============
Labeled image data is essential for tuning and evaluating the performance of machine learning applications.
Such labels are typically defined with approximate enclosing shapes (i.e., simple polygons or parametric shapes), which tend to misrepresent more complex components.
When high accuracy is required, labels must be specified at or close to the pixel-level - a process known as semantic labeling or semantic segmentation.
A detailed description of this process is given in :cite:`chengSurveyAnalysisAutomatic2018`.
Examples can readily be found in several popular datasets such as COCO, depicted in Figure :ref:`figsampleSegData`.

.. raw:: latex

    \makeSampleSegFig

Semantic segmentation is important in numerous domains including printed circuit board assembly (PCBA) inspection (discussed later in the case study) :cite:`paradis2020color,azhaganReviewAutomaticBill2019`, quality control during manufacturing :cite:`fergusonDetectionSegmentationManufacturing2018,anagnostopoulosComputerVisionApproach2001,anagnostopoulosHighPerformanceComputing2002`, manuscript restoration / digitization :cite:`gatosSegmentationfreeRecognitionTechnique2004,kesimanNewSchemeText2016,jainTextSegmentationUsing1992,taxtSegmentationDocumentImages1989,fujisawaSegmentationMethodsCharacter1992`, and effective patient diagnosis :cite:`seifertSemanticAnnotationMedical2010,rajchlDeepCutObjectSegmentation2017,yushkevichUserguided3DActive2006,iakovidisRatsnakeVersatileImage2014`.
In all these cases, imprecise annotations severely limit the development of automated solutions and can decrease the accuracy of standard trained segmentation models.

Quality semantic segmentation is difficult due to a reliance on large, high-quality datasets, which are often created by manually labeling each image.
Manual annotation is error-prone, costly, and greatly hinders scalability.
As such, several tools have been proposed to alleviate the burden of collecting these ground-truth labels :cite:`BestImageAnnotation`.
Unfortunately, existing tools are heavily biased toward lower-resolution images with few regions of interest (ROI), similar to Figure :ref:`figsampleSegData`.
While this may not be an issue for some datasets, such assumptions are *crippling* for high-fidelity images with hundreds of annotated ROIs :cite:`Ladicky_whatWhereCombiningCRFs,Wang_multiLabelImageAnnotation`.

With improving hardware capabilities and increasing need for high-resolution ground truth segmentation, there are a continually growing number of applications that *require* high-resolution imaging with the previously described characteristics :cite:`Mohajerani_cloudRemoteSensing,Demochkina_improvingOneShotXray`.
In these cases, the existing annotation tooling greatly impacts productivity due to the previously referenced assumptions and lack of support :cite:`SpaceNet2020-lb`.

In response to these bottlenecks, *we present the Semi-Supervised Semantic Annotation (S3A) annotation and prototyping platform -- an application which eases the process of pixel-level labeling in large, complex scenes.* [#]_
Its graphical user interface is shown in Figure :ref:`figappOverview`.
The software includes live app-level property customization, real-time algorithm modification and feedback, region prediction assistance, constrained component table editing based on allowed data types, various data export formats, and a highly adaptable set of plugin interfaces for domain-specific extensions to S3A.
Beyond software improvements, these features play significant roles in bridging the gap between human annotation efforts and scalable, automated segmentation methods :cite:`Branson_humansInLoop`.

.. raw:: latex

    \makeAppOverviewFig


.. [#] A preliminary version was introduced in an earlier publication :cite:`jessurunComponentDetectionEvaluation2020`, but significant changes to the framework and tool capabilities have been employed since then.