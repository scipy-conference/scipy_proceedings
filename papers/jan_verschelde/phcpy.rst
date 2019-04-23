:author: Jasmine Otto
:email: jtotto@ucsc.edu
:institution: University of California, Santa Cruz

:author: Angus Forbes
:email: angus@ucsc.edu
:institution: University of California, Santa Cruz

:author: Jan Verschelde
:email: jan@mah.uic.edu
:institution: University of Illinois at Chicago

-------------------------------------
Solving Polynomial Systems with phcpy
-------------------------------------

.. class:: abstract

   The solutions of a system of polynomials in several variables are often 
   needed, e.g.: in the design of mechanical systems, and 
   in phase-space analyses of nonlinear biological dynamics. 
   Reliable, accurate, and comprehensive numerical solutions are available 
   through PHCpack, a FOSS package for solving polynomial systems with 
   homotopy continuation.

   This paper explores the development of phcpy, a scripting interface for 
   PHCpack, over the past five years. One result is the availability of phcpy
   through a JupyterHub featuring Python2, Python3, and SageMath kernels.

Introduction
------------

Although phcpy is only a couple of years out,
three instances in the research literature mention its application
in the computation of the following:

* The number of embeddings of minimally rigid graphs [BELT18]_.

* Roots of Alexander polynomials [CD18]_.

* Critical points of equilibrium problems [SWM16]_ .

The cited publications above appear respectively in symbolic computation,
geometry and topology, chemical engineering.

Acknowledgments
---------------

This material is based upon work supported by the National Science
Foundation under Grant No. 1440534.

References
----------

.. [BELT18] E. Bartzos, I. Z. Emiris, J. Legersky, and E. Tsigaridas.
            *On the maximal number of real embeddings of spatial minimally
            rigid graphs*.
            In the Proceedings of the 2018 International Symposium on Symbolic 
            and Algebraic Computation (ISSAC 2018), pages 55-62, ACM 2018. 
            DOI 10.1145/3208976.3208994

.. [CD18] M. Culler and N. M. Dunfield.
          *Orderability and Dehn filling.*
          Geometry and Topology, 22:1405--1457, 2018.
          DOI 10.2140/gt.2018.22.1405

.. [SWM16] H. Sidky, J. K. Whitmer, and D. Mehta.
           *Reliable mixture critical point computation using 
           polynomial homotopy continuation.*
           AIChE Journal. Thermodynamics and Molecular-Scale Phenomena,
           62(12):4497--4507, 2016.  DOI 10.1002/aic.15319.

.. [Ver99] J. Verschelde.
           *Algorithm 795: PHCpack: A general-purpose solver for polynomial
           systems by homotopy continuation*,
           ACM Trans. Math. Softw., 25(2):251-276, 1999.
           DOI 10.1145/317275.317286

.. [Ver14] J. Verschelde.
           *Modernizing PHCpack through phcpy.*
           In P. de Buyl and N. Varoquaux, editors, 
           Proceedings of the 6th
           European Conference on Python in Science (EuroSciPy 2013),
           pages 71--76, 2014.
