.. -*- mode: rst; mode: visual-line; fill-column: 9999; coding: utf-8 -*-

:author: Shujie Fan
:email: sfan19@asu.edu
:institution: Arizona State University
:equal-contributor:	      

:author: Max Linke
:email: max.linke88@gmail.com
:institution: Max Planck Institute of Biophysics
:equal-contributor:
	      
:author: Ioannis Paraskevakos
:email: i.paraskev@rutgers.edu
:institution: Rutgers University

:author: Richard J. Gowers
:email: richardjgowers@gmail.com
:institution: University of New Hampshire

:author: Michael Gecht
:email: michael.gecht@biophys.mpg.de
:institution: Max Planck Institute of Biophysics

:author: Oliver Beckstein
:email: obeckste@asu.edu 
:institution: Arizona State University 
:corresponding:

:bibliography: pmda


.. STYLE GUIDE
.. ===========
.. .
.. Writing
..  - use past tense to report results
..  - use present tense for intro/general conclusions
.. .
.. Formatting
..  - restructured text
..  - hard line breaks after complete sentences (after period)
..  - paragraphs: empty line (two hard line breaks)
.. .
.. Workflow
..  - use PRs (keep them small and manageable)
..  - build the paper locally from the top level
..       rm -r output/shujie_fan      # sometimes needed to recover from errors
..       make_paper.sh papers/shujie_fan/
..       open  output/shujie_fan/paper.pdf
..   
   
.. definitions (like \newcommand)

.. |Calpha| replace:: :math:`\mathrm{C}_\alpha`
.. |tN| replace:: :math:`t_N`
.. |tcomp| replace:: :math:`t_\text{comp}`
.. |tIO| replace:: :math:`t_\text{I/O}`
.. |tcomptIO| replace:: :math:`t_\text{comp}+t_\text{I/O}`
.. |avg_tcomp| replace:: :math:`\langle t_\text{compute} \rangle`
.. |avg_tIO| replace:: :math:`\langle t_\text{I/O} \rangle`
.. |Ncores| replace:: :math:`N`

---------------------------------------------
 PMDA - Parallel Molecular Dynamics Analysis
---------------------------------------------

.. class:: abstract

   MDAnalysis_ is an object-oriented Python library to analyze trajectories from molecular dynamics (MD) simulations in many popular formats.
   With the development of highly optimized molecular dynamics software (MD) packages on HPC resources, the size of simulation trajectories is growing to terabyte size.
   Thus efficient analysis of MD simulations becomes a challenge for MDAnalysis, which does not yet provide a standard interface for parallel analysis.
   To address the challenge, we developed PMDA_, a Python library that provides parallel analysis algorithms based on MDAnalysis.
   PMDA parallelizes common analysis algorithms in MDAnalysis through a task-based approach with the Dask_ library.
   We implement a simple split-apply-combine scheme for parallel trajectory analysis.
   The trajectory is split into blocks and analysis is performed separately and in parallel on each block ("apply").
   The results from each block are gathered and combined.
   PMDA allows one to perform parallel trajectory analysis with pre-defined analysis tasks.
   In addition, it provides a common interface that makes it easy to create user-defined parallel analysis modules.
   PMDA supports all schedulers in Dask, and one can run analysis in a distributed fashion on HPC or ad-hoc clusters or on a single machine.
   We tested the performance of PMDA on single node and multiple nodes on local supercomputing resources and workstations.
   The results show that parallelization improves the performance of trajectory analysis.
   Although still in alpha stage, it is already used on resources ranging from multi-core laptops to XSEDE supercomputers to speed up analysis of molecular dynamics trajectories.
   PMDA is available under the GNU General Public License, version 2.

.. class:: Keywords

   MDAnalysis, High Performance Computing, Dask


Introduction
============

MDAnalysis  :cite:`Michaud-Agrawal:2011fu,Gowers:2016aa`

Dask :cite:`Rocklin:2015aa`


split-apply-combine for trajectory analysis :cite:`Khoshlessan:2017ab,Paraskevakos:2018aa`

Methods
=======




Results and Discussion
======================





Conclusions
===========




Acknowledgments
===============

SF and IP were supported by grant ACI-1443054 from the National Science Foundation.
OB was supported in part by grant ACI-1443054 from the National Science Foundation.
Computational resources were in provided the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1053575 (allocation MCB130177 to OB.


References
==========

.. We use a bibtex file ``pmda.bib`` and use
.. :cite:`Michaud-Agrawal:2011fu` for citations; do not use manual
.. citations


.. _PMDA: https://www.mdanalysis.org/pmda/
.. _MDAnalysis: https://www.mdanalysis.org
.. _Dask: https://dask.org
