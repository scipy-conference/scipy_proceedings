:author: Rollin Thomas
:email: rcthomas@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0002-2834-4257
:corresponding:

:author: Laurie Stephey
:email: lastephey@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0003-3868-6178

:author: Annette Greiner
:email: amgreiner@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0001-6465-7456

:author: Brandon Cook
:email: bgcook@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720

:video: http://www.youtube.com/watch?v=dhRUe-gz690

=====================================================
Monitoring Scientific Python Usage on a Supercomputer
=====================================================

.. class:: abstract

   This is the abstract.

.. class:: keywords

   keywords, procrastination

Introduction
============

..
   Why is the work important?

* What is workload characterization all about

  * Treat the system and human behavior on it as a phenomenon that can be measured
  * Asking users what they do isn't always accurate or precise
  * What is MODS and why do we have it

* Reasons to monitor Python at this level

  * General statements about why we care about Python
  * Ability to engage vendors and developers with actual user data
  * Identify potential bottlenecks in user performance and/or productivity
  * Why aren't they scaling bigger?
  * Do they need help with single-node performance?
  * Do they need help porting to GPUs?
  * Do they need help with containers?
  * What else?

* Characterize the workload

  * For procurement
  * For stakeholders (HQ, users, and facilities)
  * For developers to understand how their stuff is used
  * For NERSC staff to make adjustments, improve docs, etc
  * Optimize deployment

    * Containers don't have to contain dependencies nobody uses

* Why do we do it this way?

  * Test dog food
  * Able to interact with the data using Python which allows more sophisticated analysis
  * Lends itself to a very appealing prototype-to-production flow

    * We make something that works
    * Show it to stakeholder, get feedback,
    * Iterate on the actual notebook in a job
    * Productionize that notebook without rewriting to scripts etc

* Outline of the paper

Related Work
============

..
   What is the context for the work?

* Colin's original CUG paper, see CUG site [Mac17]_
* Link-time injectors like altd, xalt, etc.
* Lmod monitoring and such, see website
* LDMS, ask Taylor/Eric for ref and refs
* PyTokio, ask Glenn for ref and refs
* OMNI (is there a paper?)
* Other stories we can find? [Ordering of above probably not right]
* TACC paper Laurie

Methods
=======

..
   How was the work done?

* Short background on OMNI, MODS
* Strategy of exit hook; potential shortcomings, implications and mitigations
  * Injection, but we do it from /opt (local to the node) and users can deactivate
  * Libraries monitored is a subset of the whole
  * Slurm may kill the job before it fires the exit hook
  * Mpi4py also: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
  * What if monitoring downstream fails (canary jobs)
* Explain the code
* Path we take from exit hook execution through syslog/kafka(?), elastic
* Talk about the analysis flow: Papermill, Dask, Jupyter, Voila
* Talk about how/why we choose these various pieces
* Compare GPU-based vs CPU-based analysis?

Results
=======

..
   What were the results of the work?  What did we learn, discover, etc?

* GPU vs CPU analysis
* Other alternatives analyses
* How hard was it to set up, experiment with, maintain
* May need to follow up with users
* Most jobs are one node
* Plotting/viz libraries rank higher than expected
* Even on our GPU system, there are lots of CPU imports (unclear how high GPU utilization really is)
* For Dask, users may be/sometimes unaware they are actually using it
* Multiprocessing use is really heavy
* Quantitative statements like
  * Top 10 libraries
  * Mean job size
  * Job size as a function of library
  * Correlated libraries and dependency patterns

Discussion
==========

..
   What do the results mean?  What are the implications and directions for future work?

* "Typical" Python user on our systems does what?
* Qualitative statements about our process and its refinement
* How did we proceed and are there things others could learn from it?
* Revisit limitations, implications, and mitigations

Conclusion
==========

..
   Summarize what was done, learned, and where to go next.

* Invite developers to suggest packages for monitoring
* But we may try monitoring all imports and dropping stdlib anyway
* Abstraction helped with the design
* Future work includes watching users transition to new GPU-based system
  * Do these users run the same kind of workflow?
  * Do they change in response to the system change?
* More sophisticated, AI-based analysis and responses for further insights
  * Anomaly/problem detection and alert to us/user?

Acknowledgments
===============

References
==========
.. [Mac17] C. MacLean. *Python Usage Metrics on Blue Waters*
           Cray User Group, Redmond, WA, 2017.
