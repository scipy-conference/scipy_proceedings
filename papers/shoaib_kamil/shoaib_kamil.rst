:author: Shoaib Kamil
:email: skamil@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

:author: Derrick Coetzee
:email: dcoetzee@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

:author: Armando Fox
:email: fox@cs.berkeley.edu
:institution: Department of Computer Science, UC Berkeley

------------------------------------------------
Bringing Parallel Performance to Python 
------------------------------------------------

.. class:: abstract

    Due to physical limits, processor clock scaling is no longer the path
    to better performance.  Instead, hardware designers are using Moore's law
    scaling to increase the available hardware parallelism on modern processors.
    At the same time, domain scientists are increasingly using modern scripting
    languages such as Python, augmented with C libraries, for productive,
    exploratory science. However, due to Python's limited support for parallelism, these programmers
    have not been able to take advantage of increasingly powerful hardware; in
    addition, many domain scientists do not have the expertise to directly write
    parallel codes for many different kinds of hardware, each with specific
    idiosyncrasies.
    Instead, we propose SEJITS, a methodology that uses high-level abstractions and the
    capabilities of powerful scripting languages to bridge this
    performance-productivity gap.  SEJITS, or Selective Embedded Just-In-Time Specialization,
    takes code written to use domain-specific abstractions and selectively generates efficient, parallel,
    low-level C++ code, compiles it and runs it, all invisibly to the user.  Efficiency programmers, who 
    know how to obtain the highest performance from a parallel machine, encapsulate their knowledge into 
    domain-specific "specializers", which translate abstractions into parallel code.

.. class:: keywords

   parallel programming, specialization

Introduction
------------
Intro.

Vision: SEJITS and Asp
----------------------
1 page.

Approach/Mechanics of Asp
-------------------------
2 pages including the next 2 sections.  Need to make sure we differentiate between the host language and the transformation language.


Walkthru Example
----------------
Here is where the walkthru of the stencil example goes.


Results
-------
Results.


Other Specializers
------------------
1 page.  Talk about GMM and Akx.


Status and Future Plans
------------------------
0.5 page.  AspDB, platform detection.


Related Work
------------
0.5 page.  Auto-tuning, Pochoir, Python stuff.


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


