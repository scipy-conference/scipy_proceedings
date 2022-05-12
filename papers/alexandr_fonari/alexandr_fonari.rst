:author: Alexandr Fonari
:email: sasha.fonari@schrodinger.com
:institution: Schrödinger Inc., 1540 Broadway, 24th Floor. New York, NY 10036
:corresponding:

:author: Farshad Fallah
:email: farshad.fallah@schrodinger.com
:institution: Schrödinger Inc., 1540 Broadway, 24th Floor. New York, NY 10036


--------------------------------------------------------------------------------------------------------------------------------------
Utilizing SciPy and other open source packages to provide a powerful API for materials manipulation in the Schrödinger Materials Suite
--------------------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

The use of several open source scientific packages in The Schrödinger Materials Suite will be discussed. A common workflow for materials discovery will be described, discussing how open source packages have been incorporated at every stage. Some recent implementations of machine learning for materials discovery will be discussed, and how open source packages were leveraged to achieve results faster and more efficiently.



.. class:: keywords

   materials, active learning, OLED, deposition, evaporation

Introduction
------------

Tools for computational materials discovery can be facilitated by utilizing existing libraries that cover the fundamental mathematics used in the calculations in an optimized fashion. This allows developers to devote more time to developing truly new features instead of re-inventing the wheel, improves the performance of the software, and reduces maintenance. The Schrödinger Materials Suite [Schr]_ uses a wide variety of scientific packages both in the calculation stage and visualization of results.

A common materials discovery practice is to start with reading an experimental structure of a material, compute its properties of interest (e.g. elastic constants, electrical conductivity), tune the material by modifying its structure (e.g. doping) or adding and removing atoms (deposition, evaporation), and then recompute the properties of the modified material. This cycle can be run in a high-throughput manner, enumerating different structure modifications in a systematic fashion, such as doping ratio in a semiconductor or depositing different adsorbates. Every step of this workflow can benefit from open source code.


Materials import and generation
-------------------------------

For reading and writing of material structures, several open source packages (e.g. OpenBabel [Obabel]_, RDKit [Rdkit]_) have implemented functionality for working with several extensively used formats (e.g. CIF, PDB, mol, xyz). Experimental periodic structures of materials, mainly coming from single crystal X-ray diffraction, are distributed in CIF (Crystallographic Information File), PDB (Protein Data Bank) and lately mmCIF formats [Formats]_. Correctly reading experimental structures is of significant importance, since the rest of the materials discovery workflow depends on it. In addition to  atom coordinates and periodic cell information, structural data also contains symmetry operations (listed explicitly or by the means of providing a space group) that can be used to decrease the number of computations required for a particular system taking symmetry into account. This can be important for scaling high-throughput calculations.  Both formats allow description of the positional disorder (one example being a solvent molecule not having a stable position within the cell can be described by two or more sets of coordinates). Another complication is  that experimental data spans an interval of almost a century, one of the oldest crystal structures deposited in the Cambridge Structural Database (CSD) [CSD]_, dates to 1924 [Grph]_. All these nuances present non-trivial technical challenges. Thus, it has been a continuous effort by Schrödinger (at least 39 commits went into this project) and others to correctly read and convert periodic structures in OpenBabel. By version 3.1.1 (the most recent at writing time), this effort resulted in very few (if any) known cases where OpenBabel reads incorrect structure from the experimental data. OpenBabel contains a versatile Python API that exposes most functionality of the underlying C code in a clean and ergonomic way. Non-periodic molecular formats are simpler since they only contain atom coordinates and no cell or symmetry information.

An important application of structure generation is modeling of substitutional disorder in solid alloys and materials with point defects (intermetallics, semiconductors, oxides and their crystalline surfaces). In this case the unit cell and atomic sites of the crystal or surface slab are well defined while the chemical species occupying the site may vary. In order to simulate substitutional disorder one must generate the ensemble of structures that includes all statistically significant atomic distribution in a given unit cell.  This can be achieved by a brute force enumeration of all symmetrically unique atomic structures with a given number of vacancies, impurities or solute atoms. The enumlib library implements algorithms for such a systematic enumeration of periodic structures [Enumlib]_. This allows to generate a big set of symmetrically nonequivalent materials with different compositions (e.g. doping or defect concentration). Recently, we applied this approach in simultaneous study of the activity and stability of Pt based core-shell type catalysts for the oxygen reduction reaction. We generated a set of stable doped Pt/transition metal/nitrogen surfaces using periodic enumeration. Using Quantum ESPRESSO (QE) [QE]_ to perform periodic density functional theory (DFT) calculations, we assessed surface phase diagrams for Pt alloys and  and identified the avenues for stabilizing the cost effective  core-shell systems by a judicious choice of the catalyst core material. Such catalysts may prove critical in electrocatalysis for the fuel cell applications. [TM]_.

Workflow capabilities
---------------------

In order to be able to run a massively parallel screening of materials a highly scalable and stable queuing system (job scheduler) is required. We have implemented a job queuing system on top of the most used queuing systems (SLURM, torque, etc.) and exposed a Python API to submit and monitor jobs. In order to accommodate job dependencies in the workflows, for each job a parent job (or multiple parent jobs) can be defined forming a directed graph of jobs. If a job fails (and can not be restarted), all its children (if any) will not start, thus saving queuing and computational time (some examples of workflows are provided below). This allowed us and our customers to perform massive screenings of materials and their properties. We reported a massive screening of 250,000 charge-conducting organic materials, totaling approximately 3,619,000 DFT SCF (self-consistent field) single-molecule calculations using quantum mechanics (QM) Jaguar code [Jaguar]_ that took 457,265 CPU hours (~52 years) [CScreen]_. Another case study is high-throughput molecular dynamics simulations (MD) of thermophysical properties of polymers for various applications [MDS]_. There, using Desmond code [Desmond]_ we computed the glass transition temperature (Tg) of 315 polymers and compared the results with experimental measurements [Bicerano]_. This study took advantage of GPU (graphics processing unit) support as implemented in Desmond, as well as the job scheduler API described above.

For soft materials (polymers, organic small molecules and substrates composed of soft molecules), convex hull and related mathematical methods are important for finding possible accessible solvent voids (during submerging or sorption) and adsorbate sites (during molecular deposition). These methods are conveniently implemented in SciPy [Scipy]. We implemented molecular deposition and evaporation workflows using the Desmond MD engine as the backend. This workflow enables simulation of the deposition and evaporation of the small molecules on a substrate. As an example, organic light-emitting diodes (OLEDs) are fabricated using a stepwise process, where layers are deposited on top of previous layers. Both vacuum and solution deposition processes have been used to prepare these films, primarily as "amorphous" thin film active layers lacking long-range order. Each of these deposition techniques introduces changes to the film structure and consequently, different charge-transfer and luminescent properties [Deposition]_.



References
----------
.. [Schr] Schrödinger Release (2021). Schrödinger Release 2021-2: Materials Science Suite. New York, NY: Schrödinger, LLC. http://www.schrodinger.com/materials/

.. [Obabel] N. M. O'Boyle, et al. *Open Babel: An open chemical toolbox*, Journal of cheminformatics 3.1 (2011): 1-14. https://openbabel.org/

.. [Rdkit] G. Landrum. *RDKit: A software suite for cheminformatics, computational chemistry, and predictive modeling*, (2013). http://www.rdkit.org/

.. [Formats] J. D. Westbrook, and P. MD Fitzgerald. *The PDB format, mmCIF formats, and other data formats*, Structural bioinformatics 2 (2003): 271-291.

.. [CSD] C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward. *The Cambridge Structural Database*, Acta Cryst. B72: (2016): 171-179. DOI: 10.1107/S2052520616003954

.. [Grph] O Hassel, H Mark. *The Crystal Structure of Graphite*, Zeitschrift für Physik (Journal of Physics), 25, pages 317–337, (1924): 10.1007/BF01327534

.. [Enumlib] G. LW Hart, and R. W. Forcade. *Algorithm for generating derivative structures*, Physical Review B 77.22 (2008): 224115. https://github.com/msg-byu/enumlib/

.. [QE] P. Giannozzi, et al. *Advanced capabilities for materials modelling with Quantum ESPRESSO*, Journal of physics: Condensed matter 29.46 (2017): 465901. https://www.quantum-espresso.org/

.. [TM] T. Mustard, et al. *Surface reactivity and stability of core-shell solid catalysts from ab initio combinatorial calculations*, ABSTRACTS OF PAPERS OF THE AMERICAN CHEMICAL SOCIETY. 258. (2019).

.. [Jaguar] A. D. Bochevarov, et al. *Jaguar: A high‐performance quantum chemistry software program with strengths in life and materials sciences*, International Journal of Quantum Chemistry 113.18 (2013): 2110-2142.

.. [CScreen] N. N. Matsuzawa, et al. *Massive theoretical screen of hole conducting organic materials in the heteroacene family by using a cloud-computing environment*, The Journal of Physical Chemistry A 124.10 (2020): 1981-1992.

.. [MDS] M. Atif F. Afzal, et al. *High-throughput molecular dynamics simulations and validation of thermophysical properties of polymers for various applications*, ACS Applied Polymer Materials 3.2 (2020): 620-630.

.. [Desmond] D. E. Shaw, et al. *Anton 2: Raising the Bar for Performance and Programmability in a Special-Purpose Molecular Dynamics Supercomputer*, SC14: International Conference for High Performance Computing, Networking, Storage and Analysis: 41 (2014). DOI: 10.1109/SC.2014.9

.. [Bicerano] J Bicerano. *Prediction of polymer properties.* cRc Press, 2002.

.. [Scipy] P. Virtanen, et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*, Nature Methods, 17(3), 261-272. https://scipy.org/

.. [Deposition] P. Winget, et al. *Organic Thin Films for OLED Applications: Influence of Molecular Structure, Deposition Method, and Deposition Conditions*, International Conference on the Science and Technology of Synthetic Metals (2022).
