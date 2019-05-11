:author: Jasmine Otto
:email: jtotto@ucsc.edu
:institution: University of California, Santa Cruz

:author: Angus Forbes
:email: angus@ucsc.edu
:institution: University of California, Santa Cruz

:author: Jan Verschelde
:email: jan@math.uic.edu
:institution: University of Illinois at Chicago

.. |eacute| unicode:: U+00E9 .. eacute
   :trim:

.. |iacute| unicode:: U+00ED .. iacute
   :trim:

.. |Ccaron| unicode:: U+010C .. Ccaron
   :trim:

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
============

The Python package phcpy [Ver14]_ provides an alternative to the
command line executable ``phc`` of PHCpack [Ver99]_ to solve polynomial 
systems by homotopy continuation methods.
In the phcpy interface, Python scripts replace command line options and text menus, and data persists in a session without temporary files. This also makes PHCpack accessible from Jupyter notebooks, including a JupyterHub available online [Pascal]_.

The meaning of *solving* evolved from computing approximations to
all isolated solutions into the numerical irreducible decomposition
of the solution set.  The numerical irreducible decomposition includes
not only the isolated solutions, but also representations for all
positive dimensional solution sets. Such representations consist
of sets of *generic points*, partitioned along the irreducible factors.

The focus of this paper is on the application of new technology
to solve polynomial systems, in particular, cloud computing [BSVY15]_
and multicore shared memory parallelism
accelerated with graphics processing units [VY15]_.
Our web interface offers phcpy in a SageMath [Sage]_, [SJ05]_ kernel
or in a Python kernel of a Jupyter notebook [Klu16]_.

Although phcpy has been released for only five years,
three instances in the research literature of symbolic computation, geometry and topology, and chemical engineering (respectively) mention its application to their computations.

* The number of embeddings of minimally rigid graphs [BELT18]_.
* Roots of Alexander polynomials [CD18]_.
* Critical points of equilibrium problems [SWM16]_.

phcpy is in ongoing development. At the time of writing, this paper is based on version 0.9.4 of phcpy, whereas version 0.1.5 was current at the time of [Ver14]_. An example of these changes is that the software described in [SVW03]_ was recently parallelized for phcy [Ver18]_.

Mission
-------

The mission of phcpy is to bring polynomial homotopy continuation,
which is at the origin of many of the algorithms in numerical algebraic geometry [SVW05]_,
into Python's computational ecosystem.

phcpy wraps the compiled code provided as shared object files by PHCpack, an approach which benefits accessibility of the methods without sacrificing their efficiency.
First, the wrapping transfers the implementation of the many available homotopy algorithms in a direct way into Python modules.
Second, we do not sacrifice the efficiency of the compiled code. Scripts replace the input/output movements and interactions with the user, but not the computationally intensive algorithms.

User Interactions
=================

CGI Scripting
-------------

We previously developed a collection of Python scripts (mediated through HTML forms), following common programming patterns [Chu06]_, as a web interface to ``phc``. MySLQdb does the management of user data, including a) names and encrypted passwords, b) generic, random folder names to store data files, and c) file names with polynomial systems solved. With the module smtplib, we defined email exchanges for an automatic 2-step registration process and password recovery protocol.

Jupyter and JupyterHub
----------------------

The Jupyter notebook supports language agnostic computations,
supporting execution environments in several dozen languages.
With JupyterHub, we can run the code in a Python Terminal session,
in a Jupyter notebook running Python, or in a SageMath session.

For the user administration, we refreshed our first web interface. Interfacing the existing MySQL database required a custom AuthManager, and the e-mail prompts were hooked to a new Tornado page. The setup requires some system administration expertise. [FIXME]

With JupyterHub, we provide user accounts on our server.

* At login time, a new process is spawned.
* Users have generic, random login names.
* Actions of users must be isolated from each other.


Code Snippets
-------------

We use the extension [...] to provide code snippets suggesting typical applications to guide the novice user.

The screen shot in Fig. :ref:`figsnippet` shows the code snippet
with an example of using the blackbox solver.

.. figure:: ./bbsolvesnippet2.png
   :align: center
   :height: 400 px
   :figclass: h

   The code snippet for the blackbox solver.  :label:`figsnippet`


Intensive Methods
=================

For its fast mixed volume computation, the software incorporates MixedVol [GLW05]_ and DEMiCs [MT08]_. High-precision double double and quad double arithmetic is performed by the algorithms in QDlib [HLB01]_.

Speedup and Quality Up
----------------------

An obvious benefit of running on many cores is the speedup.
The *quality up* question asks the following:
if we can afford to spend the same time,
by how much can we improve the solution using *p* processors?

The function defined below returns the elapsed performance
of the blackbox solver on the cyclic 7-roots benchmark problem,
for a number of tasks and a precision equal to double, double double,
or quad double arithmetic.

.. code-block:: python

    def qualityup(nbtasks=0, precflag='d'):
        """
        Runs the blackbox solver on a system.
        The default uses no tasks and no multiprecision.
        The elapsed performance is returned.
        """
        from phcpy.families import cyclic
        from phcpy.solver import solve
        from time import perf_counter
        c7 = cyclic(7)
        tstart = perf_counter()
        s = solve(c7, verbose=False, tasks=nbtasks, \
                  precision=precflag, checkin=False)
        return perf_counter() - tstart

If the quality of the solutions is defined as the working precision,
then the quality up question ask for the number of processors needed
to compensate for the overhead of the multiprecision arithmetic.

Applications
============

Mechanism Design
----------------

Fig. :ref:`fig4barcoupler` illustration a reproduction
of a result in the mechanism design literature [MW90]_.
Given five points, the problem is to determine the length of two bars
so their coupler curve passes through the five given points.

.. figure:: ./fbarcoupler.png
   :align: center
   :figclass: h

   The design of a 4-bar mechanism.  :label:`fig4barcoupler`

This example is part of the tutorial of phcpy and the scripts 
to reproduce the results are in its source code distribution.
The equations are generated with sympy [SymPy]_
and the plots are made with matplotlib [Hun07]_.

Biodynamics
-----------



Tangent Circles
---------------

Another example from the phcpy tutorial, that runs in real-time.


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
            DOI 10.1145/3208976.3208994.

.. [BSVY15] N. Bliss, J. Sommars, J. Verschelde, X. Yu.
            *Solving polynomial systems in the cloud with polynomial
            homotopy continuation.*
            In the Proceedings of the 17th International Workshop on Computer 
            Algebra in Scientific Computing (CASC 2015),
            edited by V. P. Gerdt, W. Koepf, W. M. Seiler,
            and E. V. Vorozhtsov, volume 9301 of Lecture Notes in 
            Computer Science, pages 87-100, Springer-Verlag, 2015. 
            DOI 10.1007/978-3-319-24021-3_7.

.. [Chu06] W. J. Chun. *Core Python Programming.*
           Prentice Hall, 2nd Edition, 2006.

.. [CD18] M. Culler and N. M. Dunfield.
          *Orderability and Dehn filling.*
          Geometry and Topology, 22: 1405-1457, 2018.
          DOI 10.2140/gt.2018.22.1405.

.. [HLB01] Y. Hida, X. S. Li, and D. H. Bailey.
           *Algorithms for quad-double precision floating point arithmetic.*
           In the Proceedings  of the 15th IEEE Symposium on Computer 
           Arithmetic (Arith-15 2001), pages 155--162. IEEE Computer Society,
           2001.  DOI 10.1109/ARITH.2001.930115.

.. [Hun07] J. D. Hunter.
           *Matplotlib: A 2D Graphics Environment.*
           Computing in Science and Engineering 9(3): 90-95, 2007.
           DOI 10.1109/MCSE.2007.55.

.. [GLW05] T. Gao, T.Y. Li, and M. Wu.
           *Algorithm 846: MixedVol: a software package for mixed-volume
           computation.*
           ACM Trans. Math. Softw., 31(4):555-560, 2005.
	       DOI 10.1145/1114268.1114274.

.. [SymPy] D. Joyner, O. :math:`~\!` |Ccaron| ert |iacute| k, 
           A. Meurer, and B. E. Granger.
           *Open source computer algebra systems: SymPy.*
           ACM Communications in Computer Algebra 45(4): 225-234 , 2011.
           DOI 10.1145/2110170.2110185.

.. [Klu16] T. Kluyver, B. Ragan-Kelley, F. P |eacute| rez, B. Granger,
           M. Bussonnier, J. Frederic, K. Kelley, J. Hamrick, J. Grout,
           S. Corlay, P. Ivanov, D. Avila, S. Abdalla, C. Willing,
           and Jupyter Development Team.
           *Jupyter Notebooks -- a publishing format for reproducible
           computational workflows*.
           In Positioning and Power in Academic Publishing: Players, Agents, 
           and Agendas, edited by F. Loizides and B. Schmidt, 
           pages 87-90. IOS Press, 2016.
           DOI 10.3233/978-1-61499-649-1-87.

.. [MT08] T. Mizutani and A. Takeda.
          *DEMiCs: A software package for computing the mixed volume via
          dynamic enumeration of all mixed cells.*
          In Software for Algebraic Geometry, edited by M. E. Stillman,
          N. Takayama, and J. Verschelde,
          volume 148 of The IMA Volumes in Mathematics and its Applications,
          pages 59-79. Springer-Verlag, 2008.
          DOI 10.1007/978-0-387-78133-4.

.. [MW90] A. P. Morgan and C. W. Wampler.
          *Solving a Planar Four-Bar Design Using Continuation.*
          Journal of Mechanical Design, 112(4): 544-550, 1990.
          DOI 10.1115/1.2912644.

.. [Sage] The Sage Developers.
          *SageMath, the Sage Mathematics Software System, Version 7.6*.
          https://www.sagemath.org, 2016.
          DOI 10.5281/zenodo.820864.

.. [SJ05] W. Stein and D. Joyner.
          *Sage: System for algebra and geometry experimentation.*
          ACM SIGSAM Bulletin 39(2): 61-64, 2005.
          DOI 10.1145/1101884.1101889.

.. [SWM16] H. Sidky, J. K. Whitmer, and D. Mehta.
           *Reliable mixture critical point computation using 
           polynomial homotopy continuation.*
           AIChE Journal. Thermodynamics and Molecular-Scale Phenomena,
           62(12): 4497-4507, 2016.  DOI 10.1002/aic.15319.

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
           edited by P. de Buyl and N. Varoquaux, pages 71-76, 2014.

.. [Ver18] J. Verschelde.
           *A Blackbox Polynomial System Solver for Shared Memory Parallel
           Computers.*
           In Computer Algebra in Scientific Computing,
           20th International Workshop, CASC 2018, Lille, France, 
           edited by
           V. P. Gerdt, W. Koepf, W. M. Seiler, and E. V. Vorozhtsov,
           volume 11077 of Lecture Notes in Computer Science, pages 361-375.
           Springer-Verlag, 2018.
           DOI 10.1007/978-3-319-99639-4_25.

.. [VY15] J. Verschelde and X. Yu
          *Polynomial Homotopy Continuation on GPUs.*
          ACM Communications in Computer Algebra, volume 49, issue 4, 
          pages 130-133, 2015. 
          DOI 10.1145/2893803.2893810.

.. [Pascal] 
          *JupyterHub deployment of phcpy.*
          https://pascal.math.uic.edu, 20
