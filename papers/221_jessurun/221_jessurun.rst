:author: Nathan Jessurun
:email: njessurun@ufl.edu
:institution: University of Florida

:author: Dan E. Capecci
:email: dcapecci@ufl.edu
:institution: University of Florida

:author: Olivia P. Dizon-Paradis
:email: paradiso@ufl.edu
:institution: University of Florida

:author: Damon L. Woodard
:email: dwoodard@ece.ufl.edu
:institution: University of Florida

:author: Navid Asadizanjani
:email: nasadi@ece.ufl.edu
:institution: University of Florida

:bibliography: references

:video: http://www.youtube.com/watch?v=dhRUe-gz690

----------------------------------------------------------------------------
Semi-Supervised Semantic Annotator (S3A): Toward Efficient Semantic Labeling
----------------------------------------------------------------------------

.. class:: abstract

   Most semantic image annotation platforms suffer severe bottlenecks when handling large images, complex regions of interest, or numerous distinct foreground regions in a single image. We have developed the Semi-Supervised Semantic Annotator (S3A) to address each of these issues and facilitate rapid collection of ground truth pixel-level labeled data. Such a feat is accomplished through a robust and easy-to-extend integration of arbitrary python image processing functions into the semantic labeling process. Importantly, the framework devised for this application allows easy visualization and machine learning prediction of arbitrary formats and amounts of per-component metadata. To our knowledge, the ease and flexibility offered are unique to S3A among all open-source alternatives.

.. class:: keywords

   Semantic annotation, Image labeling, Semi-supervised, Region of interest
   
.. raw:: latex

   \input{figures/makefigs}
   \input{sections/intro}
   \input{sections/methods}
   \input{sections/case_study}
   \input{sections/conclusion}