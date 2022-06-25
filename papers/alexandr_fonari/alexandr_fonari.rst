:author: Alexandr Fonari
:email: sasha.fonari@schrodinger.com
:institution: Schrödinger Inc., 1540 Broadway, 24th Floor. New York, NY 10036
:corresponding:

:author: Farshad Fallah
:email: farshad.fallah@schrodinger.com
:institution: Schrödinger Inc., 1540 Broadway, 24th Floor. New York, NY 10036

:author: Michael Rauch
:email: michael.rauch@schrodinger.com
:institution: Schrödinger Inc., 1540 Broadway, 24th Floor. New York, NY 10036


--------------------------------------------------------------------------------------------------------------------------------------
Utilizing SciPy and other open source packages to provide a powerful API for materials manipulation in the Schrödinger Materials Suite
--------------------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

The use of several open source scientific packages in the Schrödinger Materials Science Suite will be discussed.
A typical workflow for materials discovery will be described, discussing how open source packages have been incorporated at every stage.
Some recent implementations of machine learning for materials discovery will be discussed, as well as how open source packages were leveraged to achieve results faster and more efficiently.



.. class:: keywords

   materials, active learning, OLED, deposition, evaporation


Introduction
------------

A common materials discovery practice or workflow is to start with reading an experimental structure of a material or generating a structure in silico, computing its properties of interest (e.g. elastic constants, electrical conductivity), tuning the material by modifying its structure (e.g. doping) or adding and removing atoms (deposition, evaporation), and then recomputing the properties of the modified material (Figure :ref:`fig1`).
Computational materials discovery leverages such workflows to empower researchers to explore vast design spaces and uncover root causes without (or in conjunction with) laboratory experimentation.

.. figure:: fig_scheme.png
   :align: center
   :figclass: w
   :scale: 60%

   Example of a workflow for computational materials discovery. :label:`fig1`

Software tools for computational materials discovery can be facilitated by utilizing existing libraries that cover the fundamental mathematics used in the calculations in an optimized fashion.
This use of existing libraries allows developers to devote more time to developing new features instead of re-inventing established methods.
As a result, such a complementary approach improves the performance of computational materials software and reduces overall maintenance.

The Schrödinger Materials Science Suite [Schr]_ is a proprietary computational chemistry/physics platform that streamlines materials discovery workflows into a single graphical user interface (Materials Science Maestro).
The interface is a single portal for structure building and enumeration, physics-based modeling and machine learning, visualization and analysis.
Tying together the various modules are a wide variety of scientific packages, some of which are proprietary to Schrödinger, Inc., some of which are open-source and many of which blend the two to optimize capabilities and efficiency.
For example, the main simulation engine for molecular quantum mechanics is Jaguar [Jaguar]_ proprietary code.
Proprietary classical molecular dynamics code Desmond (distributed by Schrödinger, Inc.) [Desmond]_ is used to obtain physical properties of soft materials, surfaces and polymers.
For periodic quantum mechanics, the main simulation engine is Quantum ESPRESSO (QE) [QE]_ open source code.
One of the co-authors of this proceedings (A. Fonari) contributes to the QE code in order to make integration with the Materials suite more seamless and less error-prone.
Also there is a push to use portable ``XML`` format as the input/output format for QE, this has been implemented in the Python open source qeschema code [qeschema]_.

Figure :ref:`fig2` gives an overview of some of the various products that compose the Schrödinger Materials Science Suite.
The various workflows are implemented mainly in Python (some of them described below), calling on proprietary or open-source code where appropriate, again, to improve the performance of the software and reduce overall maintenance.

The materials discovery cycle can be run in a high-throughput manner, enumerating different structure modifications in a systematic fashion, such as doping ratio in a semiconductor or depositing different adsorbates.
As we will detail herein, there are several open source packages that allow the user to generate a large number of structures, run calculations in high throughput manner and analyze the results.
For example, pymatgen open source package [pymatgen]_ facilitates generation and analysis of periodic structures.
It can generate inputs and read outputs of several packages, such as QE and also commercial VASP and Gaussian codes, etc.
To run and manage workflow jobs in a high-throughput manner, open source packages such as Custodian [pymatgen]_ and AiiDA [AiiDA]_ can be used.

.. figure:: fig_product.png
   :align: center
   :figclass: w
   :scale: 60%

   Some example products that compose the Schrödinger Materials Science Suite. :label:`fig2`


Materials import and generation
-------------------------------

For reading and writing of material structures, several open source packages (*e.g.* OpenBabel [Obabel]_, RDKit [RDKit]_) have implemented functionality for working with several commonly used formats (e.g. CIF, PDB, mol, xyz).
Experimental periodic structures of materials, mainly determined by single crystal X-ray diffraction, are distributed in CIF (Crystallographic Information File), PDB (Protein Data Bank) and lately, mmCIF formats [Formats]_.
Correctly reading experimental structures is of significant importance, since the rest of the materials discovery workflow depends on it.
In addition to atom coordinates and periodic cell information, structural data also contains symmetry operations (listed explicitly or by the means of providing a space group) that can be used to decrease the number of computations required for a particular system by accounting for symmetry.
This can be important, especially when scaling high-throughput calculations.
From file, structure is read in a structure object through which atomic coordinates (as a NumPy array) and chemical information of the material can be accessed and updated.
Structure object is similar to the one implemented in open source packages such as pymatgen [pymatgen]_ and ASE [ASE]_.
All the structure manipulations during the workflows are done using structure object interface.
Example of Structure object definition in pymatgen:

.. code-block:: python

   class Structure:

      def __init__(self, lattice, species, coords, ...):
          """
          Create a periodic structure.
          ...
          """

One consideration of note is that PBD, CIF and mmCIF structure formats allow description of the positional disorder (for example, a solvent molecule without a stable position within the cell which can be described by multiple sets of coordinates).
Another complication is that experimental data spans an interval of almost a century, one of the oldest crystal structures deposited in the Cambridge Structural Database (CSD) [CSD]_, dates to 1924 [Grph]_.
These nuances  and others present nontrivial technical challenges for developers.
Thus, it has been a continuous effort by Schrödinger, Inc. (at least 39 commits and several weeks of work went into this project) and others to correctly read and convert periodic structures in OpenBabel.
By version 3.1.1 (the most recent at writing time), there are no known structures read incorrectly by OpenBabel that the authors are aware of.
In general, non-periodic molecular formats are simpler to handle because they only contain atom coordinates and no cell or symmetry information.
OpenBabel has Python bindings but due to the GPL license limitation, it is called as a subprocess from the Schrödinger suite.

Another important consideration in structure generation is modeling of substitutional disorder in solid alloys and materials with point defects (intermetallics, semiconductors, oxides and their crystalline surfaces).
In such cases, the unit cell and atomic sites of the crystal or surface slab are well defined while the chemical species occupying the site may vary.
In order to simulate substitutional disorder, one must generate the ensemble of structures that includes all statistically significant atomic distributions in a given unit cell.
This can be achieved by a brute force enumeration of all symmetrically unique atomic structures with a given number of vacancies, impurities or solute atoms.
Open source enumlib library [Enumlib]_ implements algorithms for such a systematic enumeration of periodic structures.
enumlib consists of several Fortran binaries and Python scripts that can be run as a subprocess (no Python bindings).
This allows the user to generate a large set of symmetrically nonequivalent materials with different compositions (e.g. doping or defect concentration).

Recently, we applied this approach in simultaneous study of the activity and stability of Pt based core-shell type catalysts for the oxygen reduction reaction [TM]_.
We generated a set of stable doped Pt/transition metal/nitrogen surfaces using periodic enumeration.
Using QE to perform periodic density functional theory (DFT) calculations, we assessed surface phase diagrams for Pt alloys and identified the avenues for stabilizing the cost effective core-shell systems by a judicious choice of the catalyst core material.
Such catalysts may prove critical in electrocatalysis for fuel cell applications.

Workflow capabilities
---------------------

In the last section, we briefly described a complete workflow from structure generation and enumeration to periodic DFT calculations to analysis.
In order to be able to run a massively parallel screening of materials, a highly scalable and stable queuing system (job scheduler) is required.
We have implemented a job queuing system on top of the most used queuing systems (LSF, PBS, SGE, SLURM, TORQUE, UGE) and exposed a Python API to submit and monitor jobs.
In line with technological advancements, cloud is also supported by means of a virtual cluster configured with SLURM.
This allows the user to submit a large number of jobs, limited only by SLURM scheduling capabilities and cloud resources.
In order to accommodate job dependencies in workflows, for each job, a parent job (or multiple parent jobs) can be defined forming a directed graph of jobs (Figure :ref:`fig3`).

.. figure:: fig_job_scheme.png
   :align: center
   :figclass: w
   :scale: 60%

   Example of the job submission process. :label:`fig3`

There could be several reasons for a job to fail and there are several restart and recovery mechanisms in place.
The lowest level is the restart mechanism (in SLURM it is called *requeue*) which is performed by the queuing system itself.
This is triggered when a node goes down.
On the cloud, preemptible instances (nodes) can go offline at any moment.
In addition, workflows implemented in the proprietary Schrödinger Materials Science Suite have built-in methods for handling various types of failure.
For example, in case when the simulation is not converging to a requested energy accuracy, it is wasteful to blindly restart the calculation without changing some input parameters.
However, in the case of full disk space failure, it is reasonable to try restart with hopes to get a node with more empty disk space.
If a job fails (and can not be restarted), all its children (if any) will not start, thus saving queuing and computational time.

Having developed robust systems for running calculations, job queuing and troubleshooting (autonomously, when applicable), the developed workflows have allowed us and our customers to perform massive screenings of materials and their properties.
For example, we reported a massive screening of 250,000 charge-conducting organic materials, totaling approximately 3,619,000 DFT SCF (self-consistent field) single-molecule calculations using Jaguar that took 457,265 CPU hours (~52 years) [CScreen]_.
Another similar case study is the high-throughput molecular dynamics simulations (MD) of thermophysical properties of polymers for various applications [MDS]_.
There, using Desmond we computed the glass transition temperature (:math:`T_g`) of 315 polymers and compared the results with experimental measurements [Bicerano]_.
This study took advantage of GPU (graphics processing unit) support as implemented in Desmond, as well as the job scheduler API described above.

Other workflows implemented in the Schrödinger Materials Science Suite utilize open source packages as well.
For soft materials (polymers, organic small molecules and substrates composed of soft molecules), convex hull and related mathematical methods are important for finding possible accessible solvent voids (during submerging or sorption) and adsorbate sites (during molecular deposition).
These methods are conveniently implemented in the open source SciPy package [Scipy]_.
Thus, we implemented molecular deposition and evaporation workflows by using the Desmond MD engine as the backend in tandem with the convex hull functionality.
This workflow enables simulation of the deposition and evaporation of the small molecules on a substrate.
We utilized the aforementioned deposition workflow in the study of organic light-emitting diodes (OLEDs), which are fabricated using a stepwise process, where new layers are deposited on top of previous layers.
Both vacuum and solution deposition processes have been used to prepare these films, primarily as amorphous thin film active layers lacking long-range order.
Each of these deposition techniques introduces changes to the film structure and consequently, different charge-transfer and luminescent properties [Deposition]_.

As can be seen from above a workflow is usually some sort of structure modification through the structure object with a subsequent call to a backend code and analysis of its output (input for the next iteration depends on the output of the previous iteration is some workflows) after it successful (or not) exit.
Due to the large chemical and manipulation space of the materials, sometimes it very tricky to keep code for all workflows follow the same code logic.
For every workflow and/or functionality, some sort of peer reviewed material (publication, conference presentation) is created where implemented algorithms are described to facilitate reproducibility.

Data fitting algorithms and use cases
-------------------------------------

Materials simulation engines for QM, periodic DFT, and classical MD (referred to herein as backends) are frequently written in compiled languages with enabled parallelization for CPU or GPU hardware.
These backends are called from Python workflows using the job queuing systems described above.
Meanwhile, packages such as SciPy and NumPy provide sophisticated numerical function optimization and fitting capabilities.
Here, we describe examples of how the Schrödinger suite can be used to combine materials simulations with popular optimization routines in the SciPy ecosystem.

Recently we implemented convex analysis of the stress strain curve (as described here [Patrone]_).
``scipy.optimize.minimize`` [ScipyOptimize]_ is used for a constrained minimization with boundary conditions of a function related to the stress strain curve.
The stress strain curve is obtained from a series of MD simulations on deformed cells (cell deformations are defined by strain type and deformation step).
The pressure tensor of a deformed cell is related to stress.
This analysis allowed prediction of elongation at yield for high density polyethylene polymer.
Figure :ref:`fig4` shows obtained calculated yield of 10% *vs.* experimental value within 9-18% range [Convex]_.

.. figure:: fig_stress_strain.png
   :align: center
   :scale: 60%

   Left: The uniaxial stress/strain curve of a polymer calculated using Desmond through the stress strain workflow, grey band indicates inflection point (yield)). Right: Constant strain simulation with convex analysis indicates elongation at yield. Red curve - simulated stress versus strain. Blue curve - convex analysis. :label:`fig4`

The ``scipy.optimize`` package is used for a least-squares fit of the bulk energies at different cell volumes (compressed and expanded) in order to obtain the bulk modulus and equation of state (EOS) of a material.
In the Schrödinger suite this was implemented as a part of an EOS workflow, in which fitting is performed on the results obtained from a series of QE calculations performed on the original as well as compressed and expanded (deformed) cells.
An example of deformation applied to a structure in pymatgen:

.. code-block:: python

   deform = Deformation([
      [1.0, 0.02, 0.02],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]])

   latt = Lattice([
      [3.84, 0.00, 0.00],
      [1.92, 3.326, 0.00],
      [0.00, -2.22, 3.14],
   ])
   st = Structure(
      latt,
      ["Si", "Si"],
      [[0, 0, 0], [0.75, 0.5, 0.75]])

   strained_st = deform.apply_to_structure(st)

This is also an example of loosely coupled (embarrassingly parallel) jobs.
In particular, calculations of the deformed cells only depend on the bulk calculation and do not depend on each other.
Thus, all the deformation jobs can be submitted in parallel, greatly facilitating high-throughput runs.

Experimental structure refinement from powder diffraction is another example where more complex optimization is used.
Powder diffraction is a widely used method in drug discovery to assess purity of the material and discover known or unknown crystal polymorphs [Powder]_.
In particular, there is interest in fitting of the experimental powder diffraction intensity peaks to the indexed peaks (Pawley refinement) [Jansen]_.
Here we employed the open source ``lmfit`` package [Lmfit]_ to perform a minimization of the multivariable Voigt-like function that represents the entire diffraction spectrum.
This allows the user to refine (optimize) unit cell parameters coming from the indexing data as a result goodness of fit (:math:`R`-factor) between experimental and simulated spectrum is reported.

Machine learning techniques
---------------------------

Of late, there is great interest in machine learning assisted materials discovery.
There are several components required to perform machine learning assisted materials discovery.
In order to train a model, benchmark data from simulation and/or experimental data is required.
Besides benchmark data, computation of the relevant descriptors is required (see below).
Finally, a model based on benchmark data and descriptors is generated that allows prediction of properties for novel materials.
There are several techniques to generate the model, spawning from linear or non-linear fitting to neural networks, open source DeepChem [DeepChem]_ and AutoQSAR [AutoQSAR]_ from the Schrödinger suite.
Depending on the type of materials, benchmark data can be obtained using different codes available in the Schrödinger suite:
- small molecules and finite systems -  Jaguar
- periodic systems - Quantum ESPRESSO
- larger polymeric and similar systems - Desmond

Different materials systems require different descriptors for featurization.
For example, for crystalline periodic systems, we have implemented several sets of tailored descriptors.
Generation of these descriptors again uses a mix of open source and Schrödinger proprietary tools.
Specifically:

- elemental features such as atomic weight, number of valence electrons in *s*, *p* and *d*-shells, electronegativity
- structural features such as density, volume per atom, and packing fraction descriptors implemented in the open source matminer package [Matminer]_
- intercalation descriptors such as cation and anion counts, crystal packing fraction, average neighbor ionicity [Sendek]_ implemented in the Schrödinger suite
- three-dimensional smooth overlap of atomic positions (SOAP) descriptors implemented in the open source DScribe package [DScribe]_.

We are currently training models that use these descriptors to predict properties, such as bulk modulus, of a set of Li-containing battery related compounds [Chandrasekaran]_.
Several methods to generate model will be compared, such as kernel regression methods (as implemented in the open source scikit-learn code [SkLearn]_) and AutoQSAR.

For isolated small molecules and extended non-periodic systems, RDKit can be used to generate a large number of atomic and molecular descriptors.
A lot of effort has been devoted to ensure that RDKit can be used on a wide variety of materials that are supported by the Schrödinger suite.
At the time of writing, the 4th most active contributor to RDKit is Ricardo Rodriguez-Schmidt from Schrödinger [RDKitC]_.

Recently, active learning (AL) combined with DFT has received much attention to address the challenge of leveraging exhaustive libraries in materials informatics [Vasudevan]_, [Schleder]_.
On our side, we have implemented a workflow that employs active learning (AL) for intelligent and iterative identification of promising materials candidates within a large dataset.
In the framework of AL, the predicted value with associated uncertainty is considered to decide what materials to be added in each iteration, aiming to improve the model performance in the next iteration (Figure :ref:`figal`).

.. figure:: fig_al.png
   :align: center
   :figclass: w

   Active learning workflow for the design and discovery of novel optoelectronics molecules. :label:`figal`

Since it could be important to consider multiple properties simultaneously in material discovery, multiple property optimization (MPO) has also been implemented as a part of the AL workflow [Kwak]_.
MPO allows scaling and combining multiple properties into a single score.
We employed the AL workflow to determine the top candidates for hole (positively charged carrier) transport layer (HTL) by evaluating 550 molecules in 10 iterations using DFT calculations for a dataset of ~9,000 molecules [Abroshan]_.
Resulting model was validated by randomly picking a molecule from the dataset, computing properties with DFT and comparing those to the predicted values.
According to the semiclassical Marcus equation [Marcus]_, high rates of hole transfer are inversely proportional to hole reorganization energies.
Thus, MPO scores were computed based on minimizing hole reorganization energy and targeting oxidation potential to an appropriate level to ensure a low energy barrier for hole injection from the anode into the emissive layer.
In this workflow, we used RDKit to compute descriptors for the chemical structures.
These descriptors generated on the initial subset of structures are given as vectors to an algorithm based on Random Forest Regressor as implemented in scikit-learn [SKRFR]_.
Bayesian optimization is employed to tune the hyperparameters of the model.
In each iteration, a trained model is applied for making predictions on the remaining materials in the dataset.
Figure :ref:`figalplot` (A) displays MPO scores for the HTL dataset estimated by AL as a function of hole reorganization energies that are separately calculated for all the materials.
This figure indicates that there are many materials in the dataset with desired low hole reorganization energies but are not suitable for HTL due to their improper oxidation potentials, suggesting that MPO is important to evaluate the optoelectronic performance of the materials.
Figure :ref:`figalplot` (B) presents MPO scores of the materials used in the training dataset of AL, demonstrating that the feedback loop in the AL workflow efficiently guides the data collection as the size of the training set increases.

.. figure:: fig_al_plot.png
   :align: center
   :figclass: w

   A: MPO score of all materials in the HTL dataset. B: Those used in the training set as a function of the hole reorganization energy - :math:`\lambda_h`. :label:`figalplot`

To appreciate the computational efficiency of such an approach, it is worth noting that performing DFT calculations for all of the 9,000 molecules in the dataset would increase the computational cost by a factor of 15 versus the AL workflow.
It seems that AL approach can be useful in the cases where problem space is broad (like chemical space), but there are many clusters of similar items (similar molecules).
In this case, benchmark data is only needed for few representatives of each cluster.
We are currently working on applying this approach to train models for predicting physical properties of soft materials (polymers).

Conclusions
-----------

We present several examples of how Schrödinger Materials Suite integrates open source software packages.
There is a wide range of applications in materials science that can benefit from already existing open source code.
Where possible, we report issues to the package authors and submit improvements and bug fixes in the form of the pull requests.
We are thankful to all who have contributed to open source libraries, and have made it possible for us to develop a platform for accelerating innovation in materials and drug discovery.
We will continue contributing to these projects and we hope to further give back to the scientific community by facilitating research in both academia and industry.
We hope that this report will inspire other scientific companies to give back to the open source community in order to improve the computational materials field and make science more reproducible.


References
----------
.. [Schr] Schrödinger Release (2021). Schrödinger Release 2021-2: Materials Science Suite. New York, NY: Schrödinger, LLC. http://www.schrodinger.com/materials/
.. [pymatgen] S. P. Ong, et al. *Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis*, Computational Materials Science, 68: 314–319 (2013). https://pymatgen.org/
.. [AiiDA] S. P. Huber et al., *AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance*, Scientific Data 7: 300 (2020). https://www.aiida.net/
.. [Obabel] N. M. O'Boyle, et al. *Open Babel: An open chemical toolbox*, Journal of cheminformatics 3.1 (2011): 1-14. https://openbabel.org/
.. [RDKit] G. Landrum. *RDKit: A software suite for cheminformatics, computational chemistry, and predictive modeling*, (2013). http://www.rdkit.org/
.. [Formats] J. D. Westbrook, and P. MD Fitzgerald. *The PDB format, mmCIF formats, and other data formats*, Structural bioinformatics 2: 271-291 (2003).
.. [ASE] A. H. Larsen et al. *The atomic simulation environment—a Python library for working with atoms.* J. Phys. Cond. Matt. 29 (27): 273002 (2017). https://wiki.fysik.dtu.dk/ase/
.. [CSD] C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward. *The Cambridge Structural Database*, Acta Cryst. B72: 171-179 (2016).
.. [Grph] O Hassel, H Mark. *The Crystal Structure of Graphite*, Zeitschrift für Physik (Journal of Physics), 25: 317–337 (1924).
.. [Enumlib] G. LW Hart, and R. W. Forcade. *Algorithm for generating derivative structures*, Physical Review B 77 (22): 224115 (2008). https://github.com/msg-byu/enumlib/
.. [QE] P. Giannozzi, et al. *Advanced capabilities for materials modelling with Quantum ESPRESSO*, Journal of physics: Condensed matter 29 (46): 465901 (2017). https://www.quantum-espresso.org/
.. [qeschema] D. Brunato, et al. *qeschema*. https://github.com/QEF/qeschema
.. [TM] T. Mustard, et al. *Surface reactivity and stability of core-shell solid catalysts from ab initio combinatorial calculations*, ABSTRACTS OF PAPERS OF THE AMERICAN CHEMICAL SOCIETY. 258. (2019).
.. [Jaguar] A. D. Bochevarov, et al. *Jaguar: A high‐performance quantum chemistry software program with strengths in life and materials sciences*, International Journal of Quantum Chemistry 113 (18): 2110-2142 (2013).
.. [CScreen] N. N. Matsuzawa, et al. *Massive theoretical screen of hole conducting organic materials in the heteroacene family by using a cloud-computing environment*, The Journal of Physical Chemistry A 124 (10): 1981-1992 (2020).
.. [MDS] M. Atif F. Afzal, et al. *High-throughput molecular dynamics simulations and validation of thermophysical properties of polymers for various applications*, ACS Applied Polymer Materials 3 (2): 620-630 (2020).
.. [Desmond] D. E. Shaw, et al. *Anton 2: Raising the Bar for Performance and Programmability in a Special-Purpose Molecular Dynamics Supercomputer*, SC14: International Conference for High Performance Computing, Networking, Storage and Analysis: 41 (2014).
.. [Bicerano] J Bicerano. *Prediction of polymer properties.* cRc Press, 2002.
.. [Scipy] P. Virtanen, et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*, Nature Methods, 17(3): 261-272 (2020). https://scipy.org/
.. [Deposition] P. Winget, et al. *Organic Thin Films for OLED Applications: Influence of Molecular Structure, Deposition Method, and Deposition Conditions*, International Conference on the Science and Technology of Synthetic Metals (2022).
.. [Patrone] P. Patrone, A. Kearsley, A. Dienstfrey. *The role of data analysis in uncertainty quantification: Case studies for materials modeling*, 2018 AIAA Non-Deterministic Approaches Conference. 2018.
.. [ScipyOptimize] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. [Convex] A. R. Browning, M. A. F. Afzal, J. Sanders, A. Goldberg, A. Chandrasekaran, H. S. Kwak, M. D. Halls. *Polyolefin Molecular Simulation for Critical Physical Characteristics*, International Polyolefins Conference. 2020.
.. [Jansen] J. Jansen, R. T. Peschar, H. Schenk. *On the determination of accurate intensities from powder diffraction data. I. Whole-pattern fitting with a least-squares procedure*, Journal of applied crystallography 25(2): 231-236 (1992).
.. [Lmfit] M. Newville, et al. *LMFIT: Non-linear least-square minimization and curve-fitting for Python*, Astrophysics Source Code Library (2016): ascl-1606. https://lmfit.github.io/lmfit-py/
.. [Powder] J. A. Kaduk, et al., *Powder diffraction*, Nature Reviews Methods Primers 1: 77 (2021).
.. [DeepChem] B. Ramsundar, et al., *Deep Learning for the Life Sciences.* O'Reilly Media, 2019.
.. [AutoQSAR] S. L. Dixon, et al. *AutoQSAR: an automated machine learning tool for best-practice quantitative structure–activity relationship modeling*, Future medicinal chemistry 8 (15): 1825-1839 (2016).
.. [Matminer] L. Ward, et al., *Matminer: An open source toolkit for materials data mining*, Computational Materials Science 152: 60-69 (2018). https://hackingmaterials.lbl.gov/matminer/
.. [Sendek] A. D. Sendek, et al., *Holistic computational structure screening of more than 12000 candidates for solid lithium-ion conductor materials.* Energy & Environmental Science 10 (1): 306-320: (2017).
.. [DScribe] L. Himanen, et al. *DScribe: Library of descriptors for machine learning in materials science*, Computer Physics Communications 247: 106949 (2020). https://singroup.github.io/dscribe/latest/
.. [SkLearn] F. Pedregosa, et al., *Scikit-learn: Machine learning in Python.*, Journal of Machine Learning Research 12: 2825-2830 (2011). https://scikit-learn.org/
.. [Chandrasekaran] A. Chandrasekaran *Active Learning Accelerated Design of Ionic Material*, in progress.
.. [RDKitC] https://github.com/rdkit/rdkit/graphs/contributors
.. [Vasudevan] R. Vasudevan, et al., *Machine learning for materials design and discovery.*, Journal of Applied Physics 129(7): 070401 (2021).
.. [Schleder] G. R. Schleder, et al., *From DFT to machine learning: recent approaches to materials science–a review*, Journal of Physics: Materials 2(3): 032001 (2019).
.. [Marcus] R. A. Marcus, *Electron Transfer Reactions in Chemistry. Theory and experiment.*, Rev. Mod. Phys. 65: 599–610 (1993).
.. [SKRFR] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
.. [Abroshan] H. Abroshan, et al., *Active Learning Accelerates Design and Optimization of Hole-Transporting Materials for Organic Electronics* Frontiers in Chemistry 9 (2021).
.. [Kwak] H. S. Kwak, et al., *Design of organic electronic materials with a goal-directed generative model powered by deep neural networks and high-throughput molecular simulations.*, Frontiers in Chemistry 9: 800370 (2022).
