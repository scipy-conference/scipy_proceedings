:author: Ralph Kube
:email: rkube@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:author: R Michael Churchill
:email: rchurchi@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:author: Jong Youl Choi
:email: choij@ornl.gov
:institution: Oak Ridge National Laboratory

:author: Ruonan Wang
:email: wangr1@ornl.gov
:institution: Oak Ridge National Laboratory

:author: Scott Klasky
:email: klasky@ornl.gov
:institution: Oak Ridge National Laboratory

:author: CS Chang
:email: cschang@pppl.gov
:institution: Princeton Plasma Physics Laboratory

:bibliography: mybib


:video: http://www.youtube.com/watch?v=PSfrMksr0GY

----------------------------------------------------------------------
Leading magnetic fusion energy science into the big-and-fast data lane
----------------------------------------------------------------------

.. class:: abstract

We present the DELTA framework, a Python framework for efficient wide-area network 
transfer of high-velocity high-dimensional data streams from remote scientific experiments, sent to 
HPC resources for parallelized processing of typical scientific analysis workflows. Targeting 
magnetic fusion research, we accelerate the analysis of a plasma imaging diagnostic that produces
a high-dimensional data stream upwards of 500MB/sec. Using the DELTA framework we adapt the existing
Python code-base that runs in batch mode using single-core methods to a code processing large data 
streams with modern, distributed HPC architectures. We facilitate all data transfers using the ADIOS2
I/O middleware and implement the data analysis tasks using the PoolExecutor model. 

By reducing such data streams
into analysis results, available to scientists in near real-time, the process of scientific discovery
can be accelerated. 

.. class:: keywords

   streaming analysis, mpi4py, queue, adios, HPC


Introduction
------------

If you could harvest the energy from controlled nuclear fusion reactions you would have 
a potentially unlimited, environmentally friendly energy source. Nuclear fusion reactions
are the opposite to nuclear fusion reactions, which are used in todays nuclear power plants.
In a fusion reaction two light atomic nuclei merge into a heavier one, while converting a 
small fraction of the binding energy of the reactants into kinetic energy of the products.
As a nuclear reaction, the amount of energy released is many orders of magnitude larger
than for a chemical reaction, such as oxidization of carbon when burning coal.
At the same time nuclear fusion reactions are inherently safe. To bring positively charged
atomic nuclei close enough together so that they fuse requires temperatures upwards of
100 million degrees. Such a requirement unfortunately excludes any material container to
confine a fusion fuel. The most promising approach to confine a fusion fuel is in the 
state of a plasma - a hot gas where the atoms are stripped of their electrons. Such a 
plasma can be confined in a donut-shaped by strong magnetic fields. Since the energy yield 
of a fusion reaction is so large, only a small amount of fusion plasma needs to be confined
to power a fusion reactor. To produce 1 GW of fusion power, enough to power about 700,000 homes, 
only 2 kg of fusion plasma would need to be burned per day [Ent18]_. Thus, a catastrophic event
such as loss of plasma confinement would only lead to local damage to the plasma vessel. with
Fusion, no uncontrolled chain reactions are possible. Under operation, plasma facing components of
the vessel will be activated. Due to the characteristic energies of fusion reaction, the weakly 
active materials will be safe to handle after about 10-20 years. Fuels for fusion reactions are readily
extracted from sea water, which is available in near-infinite quantities. 

The plasma confinement devices with the best performance have a toroidal geometry, such as a 
tokamak. Medium-sized tokamaks, such as DIII-D [D3D]_, NSTX-U [NSTX]_, ASDEX Upgrade [AUG]_,
MAST [MAST]_, TCV [TCV]_ or KSTAR [KSTAR], have a major radius R≈1-1.5m and a minor radius a≈0.2-0.7m. 
During a typical experimental campaign, about 10-30 ``shots`` are performed on a given day. In each shot
the machine confines a plasma for a duration of a few seconds to up to several minutes. Numerous
measurements of the plasma are taken in each shot in order to better understand the physics of fusion 
plasmas and to optimize the design of a future fusion power plant. A common type of measurement is the
samping of the emission intensity by free electrons in the plasma, which allows to infer 
the temperature of the electrons as a function of radius [Cos74]_. This so-called Electron Cyclotron
Emission Imaging (ECEI) diagnostic produces a high-dimensional stream of high-velocity data. The
ECEI diagnostic installed in KSTAR [Yun14]_ produces data streams of up to 500 MB/sec.

Given the operational size and scale of fusion research, hundreds to thousands of people per 
experimental size, and hundreds of research groups of all sizes scattered around the globe,
measurement data is of interest to numerous researchers. 


Inter-shot optimization used in combination with ML f.ex. TAE optometrist [Bal17]_


This presentation will guide the audience through the adaptation process and will demonstrate how 
HPC python packages including mpi4py, threading, and queue are used and optimized for performance. 
To establish a baseline, we start by describing the implementation and limitations of the original 
code base. 

Proceeding we connect the data generation site, the KSTAR fusion facility in Korea, with the 
compute facility, the Cori Cray XC-40 supercomputer in California, USA. Cori is operated by the 
U.S. National Energy Research Scientific Computing Center and ranks 13 on the Top500 list. For this 
task, we use the ADIOS2 I/O middleware for the wide-area network (WAN) data transfer, which is part 
of the exascale computing project. We highlight the newly developed WAN-capabilities of ADIOS2 for 
low-latency streaming I/O. We show how to efficiently implement asynchronous data processing, using 
threading and queue to process the data streams into analysis workers. 

We continue by showing how the original code-base is adapted to the distributed computing 
architecture of modern HPC facilities by factoring the diagnostic routines into computational 
kernels and interfacing them to a pool executor. Using the executor models implemented by 
the mpi4py package we demonstrate the scaling of the new analysis routines on Cori. We further 
explore how embarrassingly parallel diagnostic kernels are accelerated using task-based 
parallelization and vectorization and some pitfalls to avoid. 


Analysis of measurements taken in experiments on magnetic fusion energy are
typically performed batch-wise after the experiment has concluded. 


.. table:: This is the caption for the materials table.

    +---------------+------------------+--------------------+
    |    Task       | Time-scale       | code-name          |
    +===============+==================+====================+
    | real-time     | ms               | [Bel18]_           |
    | control       |                  |                    |
    +---------------+------------------+--------------------+
    | inter-shot    | seconds,         | delta              |
    | analysis      | minutes          |                    |
    +---------------+------------------+--------------------+
    | post-shot     | hours,days,weeks | fluctana           |
    | batch analysis|                  |                    |
    +---------------+------------------+--------------------+



Designing the streaming framework
---------------------------------



Designing abstractions for the diagnostic data
----------------------------------------------

How do program?


Refactoring the analysis code
-----------------------------

What new?



Performance analysis
--------------------

Is new fast?


Acknowledgements
----------------
The authors would like to acknowledge support from engineers and developers at the National Energy 
Research Scientific Computing Center. This work used resources of the National Energy Research 
Scientific Computing Center (NERSC), a U.S. DOE Office of Science User Facility operated under
Contract No. DE-AC02-05CH11231.

References
----------

.. [Ent18] S. Entler, J. Horacek, T. Dlouhy and V. Dostal *Approximation of the economy of fusion energy*
           Energy 152 p. 489 (2018)

.. [D3D] DIII-D http://www.ga.com/diii-d

.. [NSTX] NSTX https://www.pppl.gov/nstx

.. [KSTAR] KSTAR Tokamak https://www.nfri.re.kr/kor/index

.. [AUG] ASDEX Upgrade https://www.ipp.mpg.de/16195/asdex

.. [MAST] Mega Amp Spherical Tokamak https://ccfe.ukaea.uk/research/mast-upgrade/

.. [TCV] https://www.epfl.ch/research/domains/swiss-plasma-center/research/tcv/research_tcv_tokamak/

.. [Cos74] A.E Costley, R.J. Hastie, J.W.M. Paul, and J. Chamberlain *Electron Cyclotron Emission from a Tokamak Plasma: Experiment and Theory*
           Phys. Rev. Lett. 33 p. 758 (1974).

.. [Yun14] G.S. Yun, W. Lee, M.J. Choi et al. *Quasi 3D ECE imaging system for study of MHD instabilities in KSTAR*
           Rev. Sci. Instr. 85 11D820 (2014)
           http://dx.doi.org/10.1063/1.4890401

.. [Bal17] E.A. Baltz, E. Trask, M. Binderbauer et al. *chievement of Sustained Net Plasma Heating in a Fusion Experiment with the Optometrist Algorithm*
           Sci. Rep 7m 6425 (@017)
           https://doi.org/10.1038/s41598-017-06645-7

.. [Bel18] V. A. Belyakov and A. A. *Kavin Fundamentals of Magnetic Thermonuclear Reactor Design*
           Chapter 8 Woodhead Publishing Series in Energy

.. [nerscdtn] https://docs.nersc.gov/systems/dtn/