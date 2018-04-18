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
theoretical knowledge by codifying complex techniques into usable tools that
can be leveraged by non-experts. Here, we present pyPRISM, a Python tool which
implements Polymer Reference Interaction Site Model (PRISM) theory.
:cite:`pyPRISM, disclaimer` PRISM theory is an integral equation formalize
that describes the structure and thermodynamics of polymer liquids. This
category of materials includes polymer melts, blends, solutions, and
nanocomposites with varying homopolymer and copolymer chemistry and chain
topology. :cite:`PRISMreview` Despite the success of PRISM in past studies on these
complex systems, the use of PRISM in the soft-matter community has be low
compared to other theory and simulation methods such as Self-Consistent Field
Theory (SCFT), Molecular Dynamics (MD), or Monte Carlo (MC). A primary factor
contributing to this reduced usage is the complexities associated with
implementing the theory and the lack of an available open-source codebase. In
the following sections, we will discuss the basics of PRISM theory, our
implementation of the theory in pyPRISM, our approach to educating the
scientific community on PRISM theory and pyPRISM, and finally our view for the
future of the tool.

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
the system is described through pre-averaged correlation functions.  The
fundamental PRISM equation for multi-component systems is represented in
Fourier-space as a matrix equation of these correlations functions.

.. math::
    :label: PRISMeq

    \hat{H}(k)  = \hat{\Omega}(k) \hat{C}(k) \left[ \hat{\Omega}(k) + \hat{H}(k) \right]

In this expression, :math:`\hat{H}(k)` is the inter-molecular total correlation
function matrix, :math:`\hat{C}(k)` is the inter-molecular direct correlation
function matrix, and :math:`\hat{\Omega}(k)` is the intra-molecular correlation
function matrix. Each of these matrices is an :math:`n \times n` matrix function
of wavenumber :math:`k`: with :math:`n` being the number of site-types in the
calculation. Each element of a correlation function matrix (e.g.
:math:`\hat{H}_{\alpha,\beta}(k)`) represents the value of that correlation
function between site types :math:`\alpha` and :math:`\beta` at a given
wavenumber :math:`k`. Knowledge of these three correlation functions allows one
to calculate a range of important structural and thermodynamic parameters, e.g.,
structure factors, radial distribution functions, second virial coefficients,
Flory-Huggins :math:`\chi` parameters, bulk isothermal compressibilities, and
spinodal decomposition temperatures. A full description of PRISM theory and the
nature of these correlation functions is beyond the scope of this document and
we refer readers to our recent work for more details on PRISM theory.
:cite:`pyPRISM`

While the PRISM equation can be solved analytically in select cases, we focus on a
more general numerical approach in pyPRISM. Figure :ref:`numerical` shows a
schematic of our approach. In short, after the user supplies a number of
parameters and input correlation functions, we apply a numerical optimization
routine, such as a Newton-Krylov method, :cite:`newton-krylov` to minimize a cost
function. After the cost function is minimized, the PRISM equation is considered
‘solved’ and the resultant H(r) and C(r) can be used for calculations.

Python Implementation
---------------------

pyPRISM is a Python library that has been tested on the CPython 2.7.x, 3.5.x,
and 3.6.x and only strictly depends on Numpy :cite:`numpy` and Scipy
:cite:`scipy` for core functionality. Optionally, pyPRISM provides a unit
conversion utility if the Pint :cite:`pint` library is available and a
simulation trajectory calculation tool if pyPRISM is compiled with Cython
:cite:`cython`. Figure :ref:`code`, shows an overview of the available classes
and functions in pyPRISM and how they relate categorically. In this section, we
will briefly overview and describe the concepts behind these classes. 

Parameters and data in PRISM theory fall into two categories: those that define
the properties of a single site-type (e.g., density, diameter) which others
define properties for a site-type pair (e.g., closure, potential, omega).
pyPRISM defines two base container classes based on this concept:
:code:`pyPRISM.ValueTable` and :code:`pyPRISM.PairTable`. These classes store
numerical and non-numerical data, support simple and complex iteration, and
provide :code:`check()` methods that are used to ensure that all parameters are
fully specified. Both :code:`pyPRISM.Table` classes also support setting multiple
pair-data at once making scripts easier to maintain *via* reduced visual noise.

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

    # Set the C-A, A-C, C-C pairs
    PT['C',['A','C']]  = 'Exponential'

In some cases where additional logic or error checking is needed, we have
created more specialized container classes. For example, both the site volumes
and the site-site contact distances are functions of the individual site
diameters. The :code:`pyPRISM.Diameter` class contains multiple
:code:`pyPRISM.Table` objects which are dynamically updated as the user defines
site-type diameters. 

Another specialized container is the :code:`pyPRISM.Domain` class which
specifies the discretized grid in both real- and Fourier- space over which the
PRISM equation is solved. :code:`pyPRISM.Domain` also contains the fast-Fourier
Transform (FFT) methods needed to transform correlation functions between the
two spaces. 

The :code:`pyPRISM.System` class contains multiple :code:`pyPRISM.ValueTable`
and :code:`pyPRISM.PairTable` objects in addition to the specialized container
classes described above. The goal of the :code:`pyPRISM.System` class is to be a
super-container which can validate that a system is fully and correctly
specified before allowing the user to attempt to solve the PRISM equations.

While :code:`pyPRISM.System` is primarily a super-container which houses
input property tables, the :code:`pyPRISM.PRISM` represents a fully specified
PRISM calculation and contains the cost function which must be numerically
minimized to solve. The correlation functions shown in Equation :ref:`PRISMeq`
are stored as :code:`pyPRISM.MatrixArray` objects which are similar to
:code:`pyPRISM.ValueTable` objects, but with a focus on mathematics rather than
storage. :code:`pyPRISM.MatrixArray` objects can only contain numerical data,
are space-aware, and provide many operators and methods to simplify
implementing PRISM theory mathematics. 

Once a :code:`pyPRISM.PRISM` object is numerically solved, it can be passed to a
calculator which processes the optimized correlation functions and returns various
structural and thermodynamic data. The current list of available calculators is
shown in the rightmost column of Figure :ref:`code` and fully described in the
documentation. :cite:`pyPRISMdocs`

Beyond the core data structures, pyPRISM defines classes which are meant to
represent various theoretical equations or ideas. Classes which inherit from
:code:`pyPRISM.Potential`, :code:`pyPRISM.Closure`, or code:`pyPRISM.Omega`
represent interaction potentials, theoretical closures, or *intra*-molecular
correlation functions :math:`\hat{\Omega}_{\alpha,\beta}(k)` respectively. These
properties must be specified for all site-type pairs before a
:code:`pyPRISM.PRISM` object can be created. In order to ensure that new-users
can easily add new potentials, closures, and
:math:`\hat{\Omega}_{\alpha,\beta}(k)` to the codebase, we have kept the
required contract of these classes as simple as possible. Users only must ensure
that the subclass inherits from the proper parent class and that the class
implements a :code:`calculate()` method which takes a vector representing the
real- or Fourier-space solution grid and returns a vector of calculated values. 

pyPRISM Example 
----------------

.. figure:: figure3.pdf
    :scale: 50%
    
    A schematic representation of the components of a coarse grained
    nanocomposite made up of a bead-spring polymer chain and large spherical
    nanoparticles. 

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
    
    sys.closure['polymer','polymer']   = \
    pyPRISM.closure.PercusYevick()
    sys.closure['polymer','particle']  = \
    pyPRISM.closure.PercusYevick()
    sys.closure['particle','particle'] = \
    pyPRISM.closure.HyperNettedChain()
    
    PRISM = sys.solve()

    pcf = pyPRISM.calculate.pair_correlation(PRISM)
    pcf_11 = pcf['particle','particle']

    chi = pyPRISM.calculate.chi(PRISM)
    chi_12 = pcf['particle','polymer']

The classes and methods in pyPRISM define a scripting API that allows users to
construct calculations and numerically solve the PRISM equation (Equation
:ref:`PRISMeq`) for a range of liquid-like systems. The code above shows how to
use pyPRISM to calculate the structure of a nanocomposite made of linear polymer
chains and spherical nanoparticles.  Users first create a :code:`pyPRISM.System`
object by defining the names of the site-types for the calculation. In this
case, we have two site-types which we call 'polymer' and 'particle'. Next,
various containers in the :code:`pyPRISM.System` object are populated to define
the molecular structure and interactions of the system.  When all properties are
defined, the user calls the :code:`pyPRISM.System.solve()` method which first
conducts a number of sanity checks  and then attempts to numerically solve the
PRISM equations. If successful, a :code:`pyPRISM.PRISM` object is created which
contains the final solutions for :math:`H(r)` and :math:`C(r)`. The
:code:`pyPRISM.PRISM` object can then be passed through any of the calculators
in pyPRISM to calculate various thermodynamic and structural parameters.

While it would be feasible to study this nanocomposite system *via* simulation
methods such as MD or MC, the use of PRISM theory offers some distinct
advantages. PRISM theory does not suffer from finite-size or equilibration
effects, both of which limit simulation methods. Furthermore, a simulation of
sufficient size to study the large nanoparticles and relatively long polymer
chains in this example would be computationally expensive, while the PRISM
equations can be solved in seconds on modest (e.g., laptop) hardware. Finally,
once the PRISM equation is solved, a variety of properties can quickly be
screened without having to process large simulation trajectories. While PRISM
theory does have limitations (as described in Section IV.D of :cite:`pyPRISM`),
it provides a powerful alternative or complement to traditional simulation
approaches. 


Pedagogy
--------

It is our stated goal to not only create a platform for polymer liquid state
theorists to innovate on, but to also lower the barriers to using PRISM theory
for the greater polymer science community. In this effort, we have identified
two primary challenges:

1) The process of understanding and numerically solving PRISM theory is complex
   and has many places for error

2) Many of those who would benefit most from PRISM theory do not have a strong
   programming background

In order to


Future Directions
-----------------

- Greater breadth of potentials, closures, and omega

- Coupling with popular simulatin pacakges
- Utilities for coare-graining

