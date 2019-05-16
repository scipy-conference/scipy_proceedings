:author: Jasmine Otto
:email: jtotto@ucsc.edu
:institution: University of California, Santa Cruz

:author: Angus Forbes
:email: angus@ucsc.edu
:institution: University of California, Santa Cruz

:author: Jan Verschelde
:email: jan@math.uic.edu
:institution: University of Illinois at Chicago

.. |cacute| unicode:: U+0107 .. cacute
   :trim:

.. |eacute| unicode:: U+00E9 .. eacute
   :trim:

.. |iacute| unicode:: U+00ED .. iacute
   :trim:

.. |Ccaron| unicode:: U+010C .. Ccaron
   :trim:

.. |oumlaut| unicode:: U+00F6 .. oumlaut
   :trim:

-------------------------------------
Solving Polynomial Systems with phcpy
-------------------------------------

.. class:: abstract

   The solutions of a system of polynomials in several variables are often    needed, e.g.: in the design of mechanical systems, and    in phase-space analyses of nonlinear biological dynamics.    Reliable, accurate, and comprehensive numerical solutions are available    through PHCpack, a FOSS package for solving polynomial systems with    homotopy continuation.

   This paper explores the development of phcpy, a scripting interface for    PHCpack, over the past five years. One result is the availability of phcpy   through a JupyterHub featuring Python2, Python3, and SageMath kernels.

Introduction
============

The Python package phcpy [Ver14]_ provides an alternative to the
command line executable ``phc`` of PHCpack [Ver99]_ to solve polynomial 
systems by homotopy continuation methods. In the phcpy interface, Python scripts replace command line options and text menus, and data persists in a session without temporary files. This also makes PHCpack accessible from Jupyter notebooks, including a JupyterHub available online [Pascal]_.

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
---------

The mission of phcpy is to bring polynomial homotopy continuation, which is at the origin of many of the algorithms in numerical algebraic geometry [SVW05]_, into Python's computational ecosystem.

phcpy wraps the shared object files of a compiled PHCpack, which makes the methods more accessible without sacrificing their efficiency.
First, the wrapping transfers the implementation of the many available homotopy algorithms in a direct way into Python modules.
Second, we do not sacrifice the efficiency of the compiled code. Scripts replace the input/output movements and interactions with the user, but not the computationally intensive algorithms.

Related Software
----------------
Limiting to free and open source software, currently under development,
with a presence on github, we can list three related software packages:
Bertini 2.0 [Bertini2.0]_, HomotopyContinuation.jl [HCJL]_,
and NAG4M2 [NAG4M2]_.

NAG4M2 as a Macaulay2 [M2]_ package, described in [Ley11]_,
provided the starting point for the development of PHCpack.m2 [GPV13]_.
The Julia package [HCJL]_ was presented at ICMS 2018 [BT18]_.
An earlier version to [Bertini2.0]_ is explained in [BHSW13]_.

User Interaction
================

CGI Scripting
-------------

In our first design of a web interface to ``phc``, we developed a collection of Python scripts (mediated through HTML forms), following common programming patterns [Chu06]_.  MySLQdb does the management of user data, including a) names and encrypted passwords, b) generic, random folder names to store data files, and c) file names with polynomial systems solved. With the module smtplib, we defined email exchanges for an automatic 2-step registration process and password recovery protocol.

As of the middle of May 2019, our web server has 145 user accounts.

JupyterHub
----------

With JupyterHub, we provide user accounts on our server,
which has both phcpy and SageMath pre-installed.

The hub's notebook environment supports language-agnostic computations,
supporting execution environments in several dozen languages.
We can also run the code in a Python Terminal session.

For the user administration, we refreshed our first web interface. A custom JupyterHub Authenticator connects to the existing MySQL database, and triggers a SystemdSpawner that isolates the actions of users to separate processes and logins in generic home folders.

The account management prompts by e-mail were hooked to a new Tornado Handler.

Code Snippets
-------------

In our JupyterHub deployment, we use the snippets menu provided by nbextensions [JUP15]_ to suggest typical applications to guide the novice user.

The screen shot in Fig. :ref:`figsnippet` shows the code snippet with an example of use of the blackbox solver.

.. figure:: ./bbsolvesnippet2.png
   :align: center
   :height: 400 px
   :figclass: h

   The code snippet for the blackbox solver.  :label:`figsnippet`

Direct Manipulation
-------------------

[Discuss Javascript and d3.js support in Jupyter Notebook. Relevance to computational geometry.]


Intensive Methods
=================

For its fast mixed volume computation, the software incorporates MixedVol [GLW05]_ and DEMiCs [MT08]_. High-precision double double and quad double arithmetic is performed by the algorithms in QDlib [HLB01]_.

Speedup and Quality Up
----------------------

The solution paths defined by polynomial homotopies can be tracked
independently, providing obvious opportunities for parallel execution.
This section reports on computations on our server, a 44-core computer.

An obvious benefit of running on many cores is the speedup. The *quality up* question asks the following: if we can afford to spend the same time, by how much can we improve the solution using *p* processors?

The function defined below returns the elapsed performance of the blackbox solver on the cyclic 7-roots benchmark problem, for a number of tasks and a precision equal to double, double double, or quad double arithmetic.

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


The function above is applied in an interactive Python script,
prompting the user for the number of tasks and precision,
This scripts runs in a Terminal window and prints the elapsed performance
returned by the function.
If the quality of the solutions is defined as the working precision,
then the quality up question ask for the number of processors needed
to compensate for the overhead of the multiprecision arithmetic.

Although cyclic 7-roots is a small system for modern computers,
the cost of tracking all solution paths in double double and 
quad double arithmetic causes significant overhead.
The script above was executed on a 2.2 GHz Intel Xeon E5-2699 processor
in a CentOS Linux workstation with 256 GB RAM
and the elapsed performance is in Table :ref:`perfcyc7overhead`.

.. table:: Elapsed performance of the blackbox solver in double,
           double double, and quad double precision. :label:`perfcyc7overhead`

   +------------------+------+-------+--------+
   | precision        |  d   |   dd  |   qd   |
   +==================+======+=======+========+
   | elapsed perform. | 5.45 | 42.41 | 604.91 |
   +------------------+------+-------+--------+
   | overhead factor  | 1.00 |  7.41 | 110.99 |
   +------------------+------+-------+--------+

Table :ref:`perfcyc7parallel` demonstrates the reduction of the
overhead caused by the multiprecision arithmetic by multitasking.

.. table:: Elapsed performance of the blackbox solver 
           with 8, 16, and 32 path tracking tasks, in double double
           and quad double precision.  :label:`perfcyc7parallel`

   +-------+-------+-------+-------+
   | tasks |   8   |   16  |   32  |
   +=======+=======+=======+=======+
   |  dd   | 42.41 |  5.07 |  3.88 |
   +-------+-------+-------+-------+
   |  qd   | 96.08 | 65.82 | 44.35 |
   +-------+-------+-------+-------+

Notice that the 5.07 in Table :ref:`perfcyc7parallel`
is less than the 5.45 of Table :ref:`perfcyc7overhead`:
with 16 tasks we doubled the precision and finished the computations
in about the same time.
The 42.41 and 44.35 in Table :ref:`perfcyc7parallel` are similar enough
to state that with 32 instead of 8 tasks we doubled the precision from 
double double to quad double precision in about the same time.

Precision is a crude measure of quality.
Another motivation for quality up by parallelism is to compensate
for the cost overhead caused by arithmetic with power series.
Power series are hybrid symbolic-numeric representations
for algebraic curves.

Positive Dimensional Solution Sets
----------------------------------

As solving evolved from approximating all isolated solutions
of a polynomial system into computing a numerical irreducible decomposition,
the meaning of a solution expanded as well.
To illustrate this expansion, 
we consider again the family of cyclic *n*-roots problems, 
now for :math:`n = 8`, [BF94]_.
While for :math:`n = 7` all roots are isolated points,
there is a one dimensional solution curve of cyclic 8-roots of degree 144.
This curve decomposes in 16 irreducible factors,
eight factors of degree 16 and eight quadratic factors,
adding up to :math:`8 \times 16 + 8 \times 2 = 144`.

Consider the following code snippet.

.. code-block:: python

    from phcpy.phcpy2c3 import py2c_set_seed
    from phcpy.factor import solve
    from phcpy.families import cyclic
    py2c_set_seed(201905091)
    c8 = cyclic(8)
    sols = solve(8, 1, c8, verbose=False)
    witpols, witsols, factors = sols[1]
        deg = len(witsols)
    print('degree of solution set at dimension 1 :', deg)
    print('number of factors : ', len(factors))
    _, isosols = sols[0]
    print('number of isolated solutions :', len(isosols))

The output of the script is

::

    degree of solution set at dimension 1 : 144
    number of factors :  16
    number of isolated solutions : 1152

Survey of Applications
======================

We consider some examples from various literatures which apply polynomial 
constraint solving, two of which are tutorialized for phcpy.

[DRAFT NOTE: None of these run on the public phcpy deployment, except possibly Apollonius circles. However, they do all seem to use the Python bindings.]

Real-Time Interaction
---------------------

Another example from the phcpy tutorial is the circle problem of Apollonius...

[cite scipy poster]

Rigid Graph Theory
------------------

[BELT18]_

Also, a simpler example of mechanism design:

Fig. :ref:`fig4barcoupler` illustration a reproduction
of a result in the mechanism design literature [MW90]_.
Given five points, the problem is to determine the length of two bars
so their coupler curve passes through the five given points.

.. figure:: ./fbarcoupler.png
   :align: center
   :figclass: h
   :height: 300 px

   The design of a 4-bar mechanism.  :label:`fig4barcoupler`

This example is part of the tutorial of phcpy and the scripts 
to reproduce the results are in its source code distribution.
The equations are generated with sympy [SymPy]_
and the plots are made with matplotlib [Hun07]_.

Critical Point Computation
--------------------------

[SWM16]_

(Consider also methods not implemented with phcpack that could be. multiobjective optimization? 
http://www-leland.stanford.edu/group/SOL/reports/SOL-2010-1.pdf
)

Conclusion
==========


Acknowledgments
---------------

This material is based upon work supported by the National Science Foundation under Grant No. 1440534.

References
----------

.. [BHSW13] D. J. Bates, J. D. Hauenstein, A. J. Sommese, and C. W. Wampler.
            *Numerically solving polynomial systems with Bertini*, 
            volume 25 of Software, Environments, and Tools, SIAM, 2013.

.. [BELT18] E. Bartzos, I. Z. Emiris, J. Legersky, and E. Tsigaridas.
            *On the maximal number of real embeddings of spatial minimally
            rigid graphs*.
            In the Proceedings of the 2018 International Symposium on Symbolic 
            and Algebraic Computation (ISSAC 2018), pages 55-62, ACM 2018. 
            DOI 10.1145/3208976.3208994.

.. [Bertini2.0] Bertini 2.0: The redevelopment of Bertini in C++.
                https://github.com/bertiniteam/b2

.. [BF91] J. Backelin and R. Fr |oumlaut| berg.
          *How we proved that there are exactly 924 cyclic 7-roots.*
          In the Proceedings of the 1991 International Symposium on
          Symbolic and Algebraic Computation (ISSAC'91), pages 103-111,
          ACM, 1991.  DOI 10.1145/120694.120708.

.. [BF94] G. Bj |oumlaut| rck and R. Fr |oumlaut| berg.
          *Methods to ``divide out'' certain solutions from systems of 
          algebraic equations, applied to find all cyclic 8-roots.*
          In Analysis, Algebra and Computers in Mathematical Research,
          Proceedings of the twenty-first Nordic congress of
          mathematicians, edited by M. Gyllenberg and L. E. Persson, 
          volume 564 of Lecture Notes in Pure and Applied Mathematics,
          pages 57-70.  Dekker, 1994.

.. [BSVY15] N. Bliss, J. Sommars, J. Verschelde, X. Yu.
            *Solving polynomial systems in the cloud with polynomial
            homotopy continuation.*
            In the Proceedings of the 17th International Workshop on Computer 
            Algebra in Scientific Computing (CASC 2015),
            edited by V. P. Gerdt, W. Koepf, W. M. Seiler,
            and E. V. Vorozhtsov, volume 9301 of Lecture Notes in 
            Computer Science, pages 87-100, Springer-Verlag, 2015. 
            DOI 10.1007/978-3-319-24021-3_7.

.. [BT18] P. Breiding and S. Timme.
          *HomotopyContinuation.jl: A package for homotopy continuation in
          Julia.*
          In the proceedings of ICMS 2018, the 6th International Conference
          on Mathematical Software, South Bend, IN, USA, July 24-27, 2018,
          edited by J. H. Davenport, M. Kauers, G. Labahn, and J. Urban,
          volume 10931 of Lecture Notes in Computer Science, pages 458-465.
          Springer-Verlag, 2018.  DOI 10.1007/978-3-319-96418-8.

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

.. [HCJL] A Julia package for solving systems of 
          polynomials via homotopy continuation.
          https://github.com/JuliaHomotopyContinuation

.. [Hun07] J. D. Hunter.
           *Matplotlib: A 2D Graphics Environment.*
           Computing in Science and Engineering 9(3): 90-95, 2007.
           DOI 10.1109/MCSE.2007.55.

.. [GLW05] T. Gao, T.Y. Li, and M. Wu.
           *Algorithm 846: MixedVol: a software package for mixed-volume computation.*
           ACM Trans. Math. Softw., 31(4):555-560, 2005.
           DOI 10.1145/1114268.1114274.

.. [GPV13] E. Gross, S. Petrovi |cacute|, and J. Verschelde.
           *Interfacing with PHCpack.*
           The Journal of Software for Algebra and Geometry: Macaulay2,
           5:20-25, 2013.  DOI 10.2140/jsag.2013.5.20.

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

.. [Ley11] A. Leykin.  *Numerical algebraic geometry.*
           The Journal of Software for Algebra and Geometry: Macaulay2,
           3:5-10, 2011.  DOI 10.2140/jsag.2011.3.5.

.. [M2] D. R. Grayson and M. E. Stillman.
        Macaulay2, a software system for research in algebraic geometry.
        http://www.math.uiuc.edu/Macaulay2

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

.. [NAG4M2] Branch NAG of M2 repository.
            https://github.com/antonleykin/M2/tree/NAG

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

.. [BNN16] D. J. Bates, A. J. Newell, & M. Niemerg
  *BertiniLab: A MATLAB interface for solving systems of polynomial equations.*
  Numerical Algorithms, 71, pages 229–244, 2016.
  DOI 10.1007/s11075-015-0014-6.

.. [BNN17] D. J. Bates, A. J. Newell, & M. E. Niemerg
  *Decoupling highly structured polynomial systems.*
  Journal of Symbolic Computation, 79, pages 508–515, 2017.
  DOI 10.1016/j.jsc.2016.07.016.

.. [BM16] E. Bogart & C. R. Myers
  *Multiscale Metabolic Modeling of C4 Plants: Connecting Nonlinear Genome-Scale Models to Leaf-Scale Metabolism in Developing Maize Leaves.*
  PLOS ONE, 11, e0151722, 2016.
  DOI 10.1371/journal.pone.0151722.

.. [D3] M. Bostock, V. Ogievetsky, & J. Heer
  *D3 Data-Driven Documents.*
  IEEE Transactions on Visualization and Computer Graphics, 17, pages 2301–2309, 2011.
  DOI 10.1109/TVCG.2011.185.

.. [DSG18] S. Dura-Bernal, B. A. Suter, P. Gleeson, M. Cantarelli, A. Quintana, F. Rodriguez, D. J. Kedziora, G. L. Chadderdon, C. C. Kerr, S. A. Neymotin, R. McDougal, M. Hines, G. M. G. Shepherd, & W. W. Lytton
  *NetPyNE: a tool for data-driven multiscale modeling of brain circuits.*
  bioRxiv, 461137, 2018.
  DOI 10.1101/461137.

.. [FSC13] T. Fischbacher & F. Synatschke-Czerwonka
  *FlowPy—A numerical solver for functional renormalization group equations.*
  Computer Physics Communications, 184, pages 1931–1945, 2013.
  DOI 10.1016/j.cpc.2013.03.002.

.. [GWW09] J. E. Guyer, D. Wheeler, & J. A. Warren
  *FiPy: Partial Differential Equations with Python.*
  Computing in Science Engineering, 11, pages 6–15, 2009.
  DOI 10.1109/MCSE.2009.52.

.. [KMC18] C. Knoll, D. Mehta, T. Chen, & F. Pernkopf
  *Fixed Points of Belief Propagation—An Analysis via Polynomial Homotopy Continuation.*
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 40, pages 2124–2136, 2018.
  DOI 10.1109/TPAMI.2017.2749575.

.. [LBC10] J. Liepe, C. Barnes, E. Cule, K. Erguler, P. Kirk, T. Toni, & M. P. H. Stumpf
  *ABC-SysBio—approximate Bayesian computation in Python with GPU support.*
  Bioinformatics, 26, pages 1797–1799, 2010.
  DOI 10.1093/bioinformatics/btq278.

.. [SBS18] D. G. A. Smith, L. A. Burns, D. A. Sirianni, D. R. Nascimento, A. Kumar, A. M. James, J. B. Schriber, T. Zhang, B. Zhang, A. S. Abbott, E. J. Berquist, M. H. Lechner, L. A. Cunha, A. G. Heide, J. M. Waldrop, T. Y. Takeshita, A. Alenaizan, D. Neuhauser, R. A. King, A. C. Simmonett, J. M. Turney, H. F. Schaefer, F. A. Evangelista, A. E. DePrince, T. D. Crawford, K. Patkowski, & C. D. Sherrill
  *Psi4NumPy: An Interactive Quantum Chemistry Programming Environment for Reference Implementations and Rapid Development.*
  Journal of Chemical Theory and Computation, 14, pages 3504–3511, 2018.
  DOI 10.1021/acs.jctc.8b00286.

.. [AD18] A. Dickenstein
    *Algebraic geometry in the interface of pure and applied mathematics.*
    Rio Intelligencer, ICM, 2018.
    http://mate.dm.uba.ar/~alidick/DickensteinIntelligencerWithoutFigures.

.. [DB15] D. Brake
    *Advances in Software in Numerical Algebraic Geometry.*
    Slides presented at Advances @ SIAM AG15, U Notre Dame, 2015.
    https://danielleamethyst.org/resources/presentations/talks/siam_AG2015_numerical_AG_overview.pdf.

.. [Pascal] *JupyterHub deployment of phcpy.*
    Website, accessed May 2019, 2017.
    https://pascal.math.uic.edu.

.. [JUP15] *Jupyter notebook snippets menu.*
     jupyter contrib nbextensions 0.5.0 documentation, 2015.
     https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/snippets_menu/readme.html.
