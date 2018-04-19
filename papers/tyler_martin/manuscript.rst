:author: Tyler B. Martin
:email: tyler.martin@nist.gov
:institution: National Institute of Standards and Technology
:bibliography: mybib

:author: Thomas E. Gartner III
:email: tgartner@udel.edu
:institution: Chemical and Biomolecular Engineering, University of Delaware

:author: Ronald L. Jones
:email: ronald.jones@nist.gov
:institution: National Institute of Standards and Technology

:author: Chad R. Snyder
:email: chad.snyder@nist.gov
:institution: National Institute of Standards and Technology

:author: Arthi Jayaraman
:email: arthij@udel.edu
:institution: Chemical and Biomolecular Engineering, University of Delaware
:institution: Materials Science and Engineering, University of Delaware


----------------------------------------------------------------------------------------------
pyPRISM: A Computational Tool for Liquid-State Theory Calculations of Macromolecular Materials
----------------------------------------------------------------------------------------------

.. class:: abstract

	Polymer Reference Interaction Site Model (PRISM) theory describes the
	equilibrium spatial-correlations, thermodynamics, and structure of liquid-like
	polymer systems and macromolecular materials. Here, we present a Python-based,
	open-source framework, pyPRISM, for conducting PRISM theory calculations.
	pyPRISM provides data structures, functions, and classes that streamline
	predictive PRISM calculations and can be extended for use in other tasks such
	as the coarse-graining of atomistic simulation force-fields or the modeling of
	experimental scattering data. The goal of providing this framework is to reduce
	the barrier to correctly and appropriately using PRISM theory and to provide a
	platform for rapid calculations of the structure and thermodynamics of
	polymeric fluids and nanocomposites.  

.. class:: keywords

  	polymer, nanocomposite, liquid state, theory, materials science, modeling

Introduction
------------

Free and open-source (FOSS) scientific software lowers the barriers to applying
theoretical knowledge by codifying complex techniques into usable tools that can
be leveraged by non-experts. Here, we present pyPRISM, a Python tool which
implements Polymer Reference Interaction Site Model (PRISM) theory.
:cite:`pyPRISM,disclaimer` PRISM theory is an integral equation formalism that
describes the structure and thermodynamics of polymer liquids. This category of
materials includes polymer melts, blends, solutions, and nanocomposites with
varying homopolymer and copolymer chemistry and chain topology.
:cite:`PRISMreview,PRISMorig` Despite the success of PRISM in past studies on
these complex systems, the use of PRISM in the soft-matter community has been
low compared to other theory and simulation methods which have available
open-source tools such as Self-Consistent Field Theory (SCFT),
:cite:`pscf1,pscf2` Molecular Dynamics (MD),
:cite:`hoomd1,hoomd2,hoomd3,lammps1,lammps2` or Monte Carlo (MC),
:cite:`simpatico1,cassandra1,cassandra2`. A primary factor contributing to this
reduced usage is the complexities associated with implementing the theory and
the lack of an available open-source codebase.  In the following sections, we
will discuss the basics of PRISM theory, our implementation of the theory in
pyPRISM, our approach to teaching the scientific community pyPRISM, and finally
our view for the future of the tool.

.. figure:: figure1.pdf

    Schematic of PRISM theory numerical solution process. :label:`numerical`


.. figure:: figure2.pdf
    :figclass: w
    :align: center
    :scale: 40%

    Overview of codebase and class organization. A full description of the
    codebase classes and methods can be found in the online documentation.
    :cite:`pyPRISMdocs`. :label:`code`

PRISM Theory
------------

PRISM theory describes the spatial correlations in a liquid-like polymer system
made up of spherical interacting "sites". Like an MD or MC simulation, sites can
represent atoms or coarse-grained beads, but, in contrast to these methods,
PRISM treats all of the sites of a given type as indistinguishable and does not
track the individual positions of each site in space. Instead, the structure of
the system is described through pre-averaged correlation functions. The
fundamental PRISM equation for multi-component systems is represented in
Fourier-space as a matrix equation of these correlations functions.

.. math::
    :label: PRISMeq

    \hat{H}(k)  = \hat{\Omega}(k) \hat{C}(k) 
                  \left[ \hat{\Omega}(k) + \hat{H}(k) \right]

In this expression, :math:`\hat{H}(k)` is the inter-molecular total correlation
function matrix, :math:`\hat{C}(k)` is the inter-molecular direct correlation
function matrix, and :math:`\hat{\Omega}(k)` is the intra-molecular correlation
function matrix. Each of these matrices is function of wavenumber :math:`k`
which returns an :math:`n \times n` matrix, with :math:`n` being the number of
site-types in the calculation. Each element of a correlation function matrix
(e.g.  :math:`\hat{H}_{\alpha,\beta}(k)`) represents the value of that
correlation function between site types :math:`\alpha` and :math:`\beta` at a
given wavenumber :math:`k`. 

While the PRISM equation can be solved analytically in select cases, we focus on
a more general numerical approach in pyPRISM. Figure :ref:`numerical` shows a
schematic of our approach. For all site-types or site-type pairs, the user
provides input values for :math:`\hat{\Omega}_{\alpha,\beta}(k)`, site-site pair
potentials :math:`U_{\alpha,\beta}(r)`, and site-type densities
:math:`\rho_{\alpha}`. The user also must supply an initial guess for the
optimization variable :math:`\Gamma(r)` which is defined in real-space as

.. math::
    :label: gamma

    \Gamma_{\alpha,\beta}(r) = H_{\alpha,\beta}(r) - C_{\alpha,\beta}(r)

Equation :ref:`PRISMeq`, as written, has one unspecified degree of freedom for
each site-type pair therefore mathematical relationships must be supplied in
order to solve it. These relationships are called closures and are derived in
various ways from fundamental liquid-state theory. Closures are also how the
chemistry of a system is specified *via* pairwise interaction potentials
:math:`U_{\alpha,\beta}(r)`. An example closure is the Percus-Yevick closure
shown below

.. math:: 
    :label: percusyevick

    C_{\alpha,\beta}(r) = \left(e^{-U_{\alpha,\beta}(r)} - 1.0 \right) 
                          \left(1.0 + \Gamma_{\alpha,\beta}(r) \right)

After the user supplies all necessary parameters and input correlation
functions, we apply a numerical optimization routine, such as a Newton-Krylov
method, :cite:`newton-krylov` to minimize a self-consistent cost function. After
the cost function is minimized, the PRISM equation is considered "solved" and
the resultant correlation functions can be used for calculations.

Knowledge of :math:`\hat{H}(k)`, :math:`\hat{C}(k)`, and :math:`\hat{\Omega}(k)`
for a given system allows one to calculate a range of important structural and
thermodynamic parameters, e.g., structure factors, radial distribution
functions, second virial coefficients, Flory-Huggins :math:`\chi` parameters,
bulk isothermal compressibilities, and spinodal decomposition temperatures. A
full description of PRISM theory and the nature of these correlation functions
can be found in our recent work. :cite:`pyPRISM`

Implementation Details
----------------------

pyPRISM is a Python library that has been tested on Linux, OS X, and Windows
with the CPython 2.7, 3.5 and 3.6 interpreters and only strictly depends on
Numpy :cite:`numpy1,numpy2` and Scipy :cite:`scipy1,scipy2` for core
functionality.  Optionally, pyPRISM provides a unit conversion utility if the
Pint :cite:`pint` library is available and a simulation trajectory calculation
tool if pyPRISM is compiled with Cython :cite:`cython`. pyPRISM is available on
GitHub, :cite:`pyPRISMgithub`,  conda-forge :cite:`pyPRISMconda` and the Python
Package Index (PyPI) :cite:`pyPRISMpypi` for download. It can be installed from
the command line *via*

.. code:: sh

    $ conda install -c conda-forge pyPRISM


or alternatively

.. code:: sh

    $ pip install pyPRISM

Full installation instructions can be found in the documentation.
:cite:`pyPRISMdocs`

Codebase Description 
--------------------

Figure :ref:`code`, shows an overview of the available classes and functions in
pyPRISM and how they relate categorically. To begin, we consider the core data
structures listed in the left column of the figure.  Parameters and data in
PRISM theory fall into two categories: those that define the properties of a
single site-type (e.g., density, diameter) and those which others define
properties for a site-type pair (e.g., closure, potential, omega). pyPRISM
defines two base container classes based on this concept which inherit from a
parent :code:`pyPRISM.Table` class: :code:`pyPRISM.ValueTable` and
:code:`pyPRISM.PairTable`.  These classes store numerical and non-numerical
data, support complex iteration, and provide a :code:`.check()` method that is
used to ensure that all parameters are fully specified. Both
:code:`pyPRISM.Table` subclasses also support setting multiple pair-data at
once, thereby making scripts easier to maintain *via* reduced visual noise and
repetition. 

.. code:: python
    :linenos:

    '''
    Example of pyPRISM.ValueTable usage
    '''
    import pyPRISM

    PT = pyPRISM.PairTable(types=['A','B','C'],
                           name='potential')

    # Set the A-A pair
    PT['A','A'] = 'Lennard-Jones'

    # Set the B-A, A-B, B-B, B-C, and C-B pairs
    PT['B',['A','B','C'] ] = 'Weeks-Chandler-Andersen'

    try:
        # Raises ValueError b/c not all pairs are set
        PT.check() 
    except ValueError:
        print('Not all pairs are set in ValueTable!')

    # Set the C-A, A-C, C-C pairs
    PT['C',['A','C']]  = 'Exponential'

    # No-op as all pairs are set
    PT.check() 

    for i,t,v in PT.iterpairs():
        print('{} {}-{} is {}'.format(i,t[0],t[1],v))

    # The above loop prints the following:
    #   (0, 0) A-A is Lennard-Jones
    #   (0, 1) A-B is Weeks-Chandler-Andersen
    #   (0, 2) A-C is Exponential
    #   (1, 1) B-B is Weeks-Chandler-Andersen
    #   (1, 2) B-C is Weeks-Chandler-Andersen
    #   (2, 2) C-C is Exponential

    for i,t,v in PT.iterpairs(full=True):
        print('{} {}-{} is {}'.format(i,t[0],t[1],v))

    # The above loop prints the following:
    #   (0, 0) A-A is Lennard-Jones
    #   (0, 1) A-B is Weeks-Chandler-Andersen
    #   (0, 2) A-C is Exponential
    #   (1, 0) B-A is Weeks-Chandler-Andersen
    #   (1, 1) B-B is Weeks-Chandler-Andersen
    #   (1, 2) B-C is Weeks-Chandler-Andersen
    #   (2, 0) C-A is Exponential
    #   (2, 1) C-B is Weeks-Chandler-Andersen
    #   (2, 2) C-C is Exponential

In some cases where additional logic or error checking is needed, we have
created more specialized container classes. For example, both the site volumes
and the site-site contact distances are functions of the individual site
diameters. The :code:`pyPRISM.Diameter` class contains multiple
:code:`pyPRISM.Table` objects which are dynamically updated as the user defines
site-type diameters. The :code:`pyPRISM.Density` class was created for analogous
reasons so that the pair-density matrix, 

.. math::

    \rho^{pair}_{\alpha,\beta} = \rho_{\alpha} \rho_{\beta}

and the site-density matrix, 

.. math::

    \rho^{site}_{\alpha,\beta} = 
        \begin{cases}
            \rho_{\alpha}                & \text{if } i = j \\
            \rho_{\alpha} + \rho_{\beta} & \text{if } i \neq j
        \end{cases}

and the total site density,

.. math::
        
    \rho^{total} = \sum_{\alpha} \rho^{site}_{\alpha,\alpha}

can all be calculated dynamically as the user specifies or modifies the
individual site-type densities :math:`\rho_{\alpha}`.

An additional specialized container is :code:`pyPRISM.Domain`. This class
specifies the discretized real- and Fourier-space grids over which the PRISM
equation is solved and is instantiated by specifying the length (i.e. number of
gridpoints) and grid spacing in real- or Fourier space (i.e. :math:`dr` or
:math:`dk`). A detail of the PRISM cost function which is not discussed above is
that correlation functions need to be transformed to and from Fourier space
during the cost function evaluation. :code:`pyPRISM.Domain` also contains the
Fast Fourier Transform (FFT) methods needed efficiently carry out these
transforms. The mathematics behind these FFTs, which are implemented as Type II
and III Discrete Sine Transforms (DST-II and DST-III), are discussed in previous
work.  :cite:`pyPRISM`

The :code:`pyPRISM.System` class contains multiple :code:`pyPRISM.ValueTable`
and :code:`pyPRISM.PairTable` objects in addition to the specialized container
classes described above. The goal of the :code:`pyPRISM.System` class is to be a
super-container which can validate that a system is fully and correctly
specified before allowing the user to attempt to solve the PRISM equations.

While :code:`pyPRISM.System` primarily houses input property tables,
:code:`pyPRISM.PRISM` represents a fully specified PRISM calculation and
contains the cost function to be numerically minimized. The correlation
functions shown in Equation :ref:`PRISMeq` are stored in the
:code:`pyPRISM.PRISM` object as :code:`pyPRISM.MatrixArray` objects which are
similar to :code:`pyPRISM.ValueTable` objects, but with a focus on mathematics
rather than storage. :code:`pyPRISM.MatrixArray` objects can only contain
numerical data, are space-aware, and provide many operators and methods which
simplify PRISM theory mathematics. The core data structure underlying the
:code:`pyPRISM.MatrixArray` is a three-dimensional Numpy ndarray as :math:`m`
:math:`n \times n` matrices, where :math:`m` is the length of the
:code:`pyPRISM.Domain`.

.. code:: python
    :linenos:

    '''
    Example of MatrixArray usage.
    '''
    ## Setup ##
    length = 1024      # number of gridpoints 
    dr = 0.1           # real-space grid spacing
    rank = 2           # number of site-types
    types = ['A', 'B'] # name of site-types

    domain = pyPRISM.Domain(length,dr)
    rho = pyPRISM.Density(types)      

    # Total and intra-molecular correlation functions
    # dataH and dataW are length x rank x rank 
    # numpy ndarrays that are assumed to be in memory
    kwargs = dict(length=length,rank=rank,types=types)
    H = pyPRISM.MatrixArray(data=dataH,**kwargs)
    W = pyPRISM.MatrixArray(data=dataW,**kwargs)

    ## Example Calculation of Structure Factor ##
    S = (W + H)/rho.site
    S_AB = S['A','B'] # extract S_AB from MatrixArray

    ## MatrixArray by Scalar Operations ##
    # All matrices in W are modified by the scalar x
    x = 1 # arbitrary scalar 
    W+x; W-x; W*x; W/x; # elementwise ops
    
    ## MatrixArray by Matrix Operations ##
    # All matrices in W are modified by the matrix rho
    W+rho; W-rho; W*rho; W/rho;   # elementwise ops
    W.dot(rho)                    # matrix mult.
    
    ## MatrixArray by MatrixArray Operations ##
    # Operations are matrix to corresponding matrix
    W+H; W-H; W*H; W/H;   # elementwise ops
    W.dot(H)              # matrix mult.
    
    ##  Fourier Transformations ##
    # Transform a single array versus all functions
    # in a MatrixArray
    W_AA = domain.to_real(W['A','A']) # one function
    domain.MatrixArray_to_fourier(H)  # all functions
    
    ## Other Operations ##
    W.invert()     # invert each matrix in W
    W['A','B']     # set or get function for pair A-B
    W.getMatrix(i) # get matrix i in MatrixArray
    W.iterpairs()  # iterate over all 1-D functions
    

The :code:`pyPRISM.PRISM` object is solved by calling the :code:`.solve()`
method which invokes a numerical algorithm to minimize the output of the
:code:`.cost()` method by varying the input :math:`\Gamma_{\alpha,\beta}(r)`.
Once a :code:`pyPRISM.PRISM` object is numerically solved, it can be passed to a
calculator which processes the optimized correlation functions and returns
various structural and thermodynamic data. The current list of available
calculators is shown in the rightmost column of Figure :ref:`code` and fully
described in the documentation. :cite:`pyPRISMdocs`

Beyond the core data structures, pyPRISM defines classes which are meant to
represent various theoretical equations or ideas. Classes which inherit from
:code:`pyPRISM.Potential`, :code:`pyPRISM.Closure`, or :code:`pyPRISM.Omega`
represent interaction potentials, theoretical closures, or *intra*-molecular
correlation functions :math:`\hat{\Omega}_{\alpha,\beta}(k)` respectively. These
properties must be specified for all site-type pairs before a
:code:`pyPRISM.PRISM` object can be created. In order to ensure that new-users
can easily add new potentials, closures, and
:math:`\hat{\Omega}_{\alpha,\beta}(k)` to the codebase, we have kept the
required contract of these classes as simple as possible. Users only must ensure
that the subclass inherits from the proper parent class and that the class
implements a :code:`.calculate()` method which takes a vector representing the
real- or Fourier-space solution grid as input and returns a vector of calculated
values. 

The classes and methods in pyPRISM define a scripting API (application
programming interface) that allows users to construct calculations and
numerically solve the PRISM equation (Equation :ref:`PRISMeq`) for a range of
polymer liquid-like systems. Providing a scripting API rather than an
"input-file" based scheme gives users the ability to use the full power of
Python for complex PRISM-based calculations. For example, one could use
parallelized loops to fill a database with PRISM results using Python's built-in
support for thread or process pools. Alternatively, pyPRISM could easily be
coupled to a simulation engine by calling the engine *via* subprocess,
processing the engine output, and then feeding that output to to a pyPRISM
calculation.

pyPRISM Example 
----------------

.. figure:: figure3.pdf
    :scale: 60%
    
    A schematic representation of the components of a coarse grained
    nanocomposite made up of bead-spring polymer chains and large spherical
    nanoparticles. This system is the focus of reference :cite:`composite`. In
    this example, there are two site-types: a monomer site-type in green and a
    nanoparticle site-type in yellow. :label:`nanocomposite`

.. figure:: figure4.pdf
    :scale: 75%

    All pair-correlation functions from the pyPRISM example for the
    nanocomposite system depicted in Figure :ref:`nanocomposite`.
    :label:`results`

.. code:: python
    :linenos:
    
    '''
    pyPRISM script calculating the pair correlation 
    function and structure factor of a polymer 
    nanocomposite.
    '''
    import pyPRISM
    
    sys = pyPRISM.System(['particle','polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.01,length=4096)
        
    sys.density['polymer']  = 0.75
    sys.density['particle'] = 6e-6
    
    sys.diameter['polymer']  = 1.0
    sys.diameter['particle'] = 5.0
    
    sys.omega['polymer','polymer']   = \
    pyPRISM.omega.FreelyJointedChain(length=100,l=4/3)
    sys.omega['polymer','particle']  = \
    pyPRISM.omega.InterMolecular()
    sys.omega['particle','particle'] = \
    pyPRISM.omega.SingleSite()
    
    sys.potential['polymer','polymer']   = \
    pyPRISM.potential.HardSphere()
    sys.potential['polymer','particle']  = \
    pyPRISM.potential.Exponential(alpha=0.5,epsilon=1.0)
    sys.potential['particle','particle'] = \
    pyPRISM.potential.HardSphere()
    
    sys.closure['polymer',['polymer','particle']]   = \
    pyPRISM.closure.PercusYevick()
    sys.closure['particle','particle'] = \
    pyPRISM.closure.HyperNettedChain()
    
    PRISM = sys.solve()

    pcf = pyPRISM.calculate.pair_correlation(PRISM)
    pcf_11 = pcf['particle','particle']

    chi = pyPRISM.calculate.chi(PRISM)
    chi_12 = pcf['particle','polymer']

The code above shows how to use pyPRISM to calculate the properties of a
nanocomposite made of linear polymer chains and spherical nanoparticles, as
depicted in Figure :ref:`nanocomposite`. Users first create a
:code:`pyPRISM.System` object by defining the names of the site-types for the
calculation and the thermal energy level, :math:`k_{B}T`. In this case, we have
two site-types which we (arbitrarily) call *polymer* and *particle*. 

Next, various container objects in the :code:`pyPRISM.System` object are
populated to define the molecular structure and interactions of the system. This
includes all of the inputs described in Figure :ref:`numerical`. Note how the
subclasses of :code:`pyPRISM.Potential`, :code:`pyPRISM.Closure` and
:code:`pyPRISM.Omega` are used to easily specify these properties.  When all
properties are defined, the user calls the :code:`pyPRISM.System.solve()` method
which first conducts a number of sanity checks, and issues any relevant
exceptions or warnings if issues are found. If no issues are found, a PRISM
object is created and minimization is attempted. 

Once the minimization completes, a :code:`pyPRISM.PRISM` object is returned
which contains the final solutions for :math:`H(r)` and :math:`C(r)`. The
:code:`pyPRISM.PRISM` object is then passed through the
:code:`pyPRISM.calculate.pair_correlation` and :code:`pyPRISM.calculate.chi`
calculators. Both of these methods return :code:`pyPRISM.ValueTables` which can
be subscripted to access the individual data. In the example, we extract the
particle-particle pair correlation function and the particle-polymer
:math:`\chi` parameter. 

While it would be feasible to study this nanocomposite system *via* simulation
methods such as MD or MC, the use of PRISM theory offers some distinct
advantages. PRISM theory does not suffer from finite-size or equilibration
effects, both of which limit simulation methods. Furthermore, a simulation of
sufficient size to study the large nanoparticles and relatively long polymer
chains in this example would be computationally expensive. In comparison, the
PRISM equations can be solved in seconds, even on modest (e.g., laptop)
hardware. Finally, once the PRISM equation is solved, a variety of properties
can quickly be screened without having to process large simulation trajectories.
While PRISM theory does have limitations, as described in Section IV.D of
:cite:`pyPRISM`, it provides a powerful alternative or complement to traditional
simulation approaches. 

Pedagogy
--------

.. figure:: figure5.pdf

    Depiction of the tutorial tracks we provide for users of different
    backgrounds and trainings. See the Tutorial page :cite:`pyPRISMtut` for more
    information. :label:`tutorial`
    
It is our stated goal to not only create a platform for polymer liquid state
theorists to innovate on, but to also lower the barriers to using PRISM theory
for the greater polymer science community. In this effort, we have identified
two primary challenges:

1) The process of understanding and numerically solving PRISM theory is complex
   and filled with pitfalls and opportunities for error.

2) Many of those who would benefit most from PRISM theory do not have a strong
   programming background 

The primary solution to both of these challenges is a strong focus on providing
pedagogical resources to users. To start, we have put significant effort into
our documentation. Every page of the API documentation :cite:`pyPRISMdocs`
contains a written description of the theory being implemented, all necessary
mathematics, descriptions of all input and output parameters, links to any
relevant journal articles, and a detailed and relevant example. While
including these features in our documentation is not a new idea, we are focusing
on providing these resource immediately upon release. 

Moving beyond API documentation, we also have created knowledgebase materials
which provide more nuanced information about using and numerically solving PRISM
theory. This includes everything from concise lists of systems and properties
that can be studied with pyPRISM to tips and tricks for reaching convergence of
the numerical solver.  In reference to Challenge 2 above, we also recognize that
a significant barrier to using these tools is the installation process. Our
installation documentation :cite:`pyPRISMdocs` attempts to be holistic and
provide detailed instructions for the several different ways that users can
install pyPRISM. 

While a user could learn PRISM theory and pyPRISM from the resources described
above, we have also created a self-guided tutorial in the form of Jupyter
notebooks. :cite:`pyPRISMtut,jupyter1` The tutorial notebooks cover everything
from a basic introduction to Python to how to add new features to pyPRISM. The
tutorial also has several case-study focused notebooks which walk users through the
process of reproducing PRISM results from the literature.  Figure
:ref:`tutorial` shows our recommendations for how users of different backgrounds
and skill levels might move through the tutorial. In order to ensure the widest
audience possible can take advantage of this tutorial, we have also set up a
binder instance :cite:`pyPRISMbinder`, which allows users to try out pyPRISM and
run the tutorial instantly without installing any software. This feature is also
a great benefit to users who fall under Challenge 2 above. 

Future Directions
-----------------

While pyPRISM is a step forward in providing a central platform for polymer
liquid-state theory calculations, we intend to significantly extend the tool
beyond its release state. The most obvious avenue for extension will be to add
new potentials, closures, and :math:`\hat{\Omega}(k)` to the codebase. As
described above, we hope that a significant portion of these classes will be
contributed back by users. Another source of :math:`\hat{\Omega}(k)`, is to
calculate them from simulation trajectories. While we do provide a
Cython-enhanced tool to do the calculation, we also plan to add features to more
easily couple pyPRISM to common MD and MC simulation packages.
:cite:`hoomd1,lammps1,simpatico1,cassandra1` These linkages would also make it
easier for users to carry out the Self-Consistent PRISM (SCPRISM) method, as
described previously. :cite:`pyPRISM`

PRISM theory also has advanced applications which are not possible in the
currently pyPRISM workflow. One example of this, is that PRISM theory can be
used to translate a detailed atomistic simulation model to a less detailed, less
computationally expensive coarse-grained model in a methodology called Integral
Equation Coarse Graining (IECG). :cite:`iecg1,iecg2,iecg3,iecg4` We hope to
provide utilities in the pyPRISM codebase that aid in carrying out this method.
PRISM theory can also be used to model or fit neutron and X-ray scattering data.
In particular, PRISM theory can be used to take existing scattering models for
single particles or polymer chains and introduce intermolecular interactions.
This approach would greatly extend the applicability of existing scattering
models which, on their own, would only be valid in the infinite dilute
concentration limit but could be combined with pyPRISM to model high
concentration.

Summary
-------

pyPRISM is an open-source tool with the goal of greatly increasing the usage of
PRISM theory, a polymer liquid-state theory.  Compared to more highly used
simulation methods such as MD and MC, PRISM theory is significantly more
computationally efficient, does not need to be equilibrated, and does not suffer
from finite size effects.  pyPRISM lowers the barriers to PRISM by providing a
simple scripting interface for setting up and numerically solving the theory. In
order to ensure users correctly and appropriately use pyPRISM, we have created
extensive pedagogical materials in the form of API documentation, knowledgebase
materials, and Jupyter-notebook powered tutorials. 


Acknowledgements
----------------

TBM is supported by the NIST/NRC fellowship program and, in addition, this work
has been supported by the members of the NIST nSoft consortium
(nist.gov/nsoft). TEG and AJ thank NSF DMR-CMMT grant number 1609543 for
financial support. This research was supported in part through the use of
Information Technologies (IT) resources at the University of Delaware,
specifically the high-performance computing resources of the Farber
supercomputing cluster. This work used the Extreme Science and Engineering
Discovery Environment (XSEDE) Stampede cluster at the University of Texas
through allocation MCB100140 (AJ), which is supported by National Science
Foundation grant number ACI-1548562. All authors thank Dr. Boualem Hammouda,
Dr. Debra Audus, Dr. Ivan Lyubimov, Dr. Nils Persson, Jannat Nayem, and Dr.
Ahmad Ghobadi for help with reviewing and developing the pyPRISM manuscript,
documentation, and tutorial. Finally, all authors thank Prof. Kenneth Schweizer
for his critical reading and valuable suggestions for improvement of the
pyPRISM manuscript.


