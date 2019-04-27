:author: Jasmine Otto
:email: jtotto@ucsc.edu
:institution: University of California, Santa Cruz

:author: Angus Forbes
:email: angus@ucsc.edu
:institution: University of California, Santa Cruz

:author: Jan Verschelde
:email: jan@math.uic.edu
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

The python package phcpy [Ver14]_ provides an alternative to the
command line executable of PHCpack [Ver99]_ to solve polynomial systems
by homotopy continuation methods.  
Scripts replace command line options and text menus.
Data remains persistent in a session, decreasing the dependency on files.

The meaning of *solving* evolved from computing approximations to
all isolated solution into the numerical irreducible decomposition
of the solution set.  The numerical irreducible decomposition includes
not only the isolated solutions, but also representations for all
positive dimensional solution sets.  Such representations consist
of sets of *generic points*, partitioned along the irreducible factors.
Parallel versions of the software described in [SVW03]_
were recently developed [Ver18]_ and added to phcpy.

Although phcpy is only a couple of years out,
three instances in the research literature mention its application
in the computation of the following:

* The number of embeddings of minimally rigid graphs [BELT18]_.

* Roots of Alexander polynomials [CD18]_.

* Critical points of equilibrium problems [SWM16]_ .

The cited publications above appear respectively in symbolic computation,
geometry and topology, chemical engineering.

Code Snippets
-------------

Code snippets suggest typical applications and guide novice user.

The figure below shows the code snippet
with an example of use of the blackbox solver.

.. figure:: ./bbsolvesnippet2.png
   :align: center
   :figclass: h

   The code snipped for the blackbox solver.

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

.. [SVW03] A. J. Sommese, J. Verschelde, and C. W. Wampler.
           *Numerical irreducible decomposition using PHCpack.*
           In Algebra, Geometry and Software Systems,
           edited by M. Joswig and N. Takayama, pages 109-130, 
           Springer-Verlag 2003.
           DOI 10.1007/978-3-662-05148-1_6.

.. [SVW05] A. J. Sommese, J. Verschelde, and C. W. Wampler.
           *Introduction to numerical algebraic geometry.*
           In Solving Polynomial Equations, 
           Foundations, Algorithms, and Applications,
           edited by A. Dickenstein and I. Z. Emiris, pages 301-337, 
           Springer-Verlag 2005.
           DOI 10.1007/3-540-27357-3_8.

.. [Ver99] J. Verschelde.
           *Algorithm 795: PHCpack: A general-purpose solver for polynomial
           systems by homotopy continuation*,
           ACM Trans. Math. Softw., 25(2):251-276, 1999.
           DOI 10.1145/317275.317286.

.. [Ver14] J. Verschelde.
           *Modernizing PHCpack through phcpy.*
           Proceedings of the 6th
           European Conference on Python in Science (EuroSciPy 2013),
           edited by P. de Buyl and N. Varoquaux, pages 71--76, 2014.

.. [Ver18] J. Verschelde.
           *A Blackbox Polynomial System Solver for Shared Memory Parallel
           Computers.*
           In Computer Algebra in Scientific Computing,
           20th International Workshop, CASC 2018, Lille, France, 
           edited by
           V. P. Gerdt, W. Koepf, W. M. Seiler, and E. V. Vorozhtsov,
           volume 11077 of Lecture Notes in Computer Science, pages 361--375.
           Springer-Verlag, 2018.
           DOI 10.1007/978-3-319-99639-4_25.
