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
can be leveraged by non-experts. Here, we present pyPRISM, a tool which
implements Polymer Reference Interaction Site Model (PRISM)
theory. :cite:`pyPRISM, disclaimer` PRISM theory describes the structure and
thermodynamics of polymer liquids. This category of materials includes polymer
melts, blends, solutions, and nanocomposites with varying homopolymer and
copolymer chemistry and chain topology. :cite:`dummy` Despite the success of PRISM in
past studies on these complex systems, the use of PRISM in the soft-matter
community has be low compared to other theory and simulation methods such as
Self-Consistent Field Theory (SCFT), Molecular Dynamics (MD), or Monte Carlo
(MC). A primary factor contributing to this reduced usage is the complexities
associated with implementing the theory and the lack of an available
open-source codebase. In the following sections, we will discuss the basics of
PRISM theory, our implementation of the theory in pyPRISM, our approach to
educating the scientific community on PRISM theory and pyPRISM, and finally our
view for the future of the tool.

PRISM Theory
------------

The fundamental PRISM equation for multi-component systems is represented in
Fourier-space as a matrix equation of correlations functions.

.. math::
    :label: PRISMeq

    \hat{H}(k)  = \hat{\Omega}(k) \hat{C}(k) \left[ \hat{\Omega}(k) + \hat{H}(k) \right]

In this expression, :math:`\hat{H}(k)` is the inter-molecular total correlation
function matrix, :math:`\hat{C}(k)` is the inter-molecular direct correlation
function matrix, and :math:`\hat{\Omega}(k)` is the intra-molecular correlation
function matrix. Each of these matrices is an n×n function of wavenumber k with
n being the number of site-types (i.e., atoms or coarse-grained bead types) in
the calculation. Each element of a correlation function matrix (e.g.
:math:`\hat{H}_{\alpha,\beta}(k)`) represents the value of that correlation
function between site types :math:`\alpha` and :math:`\beta` at a given
wavenumber :math:`k`. Knowledge of these three correlation functions allows one
to calculate a range of important structural and thermodynamic parameters,
e.g., structure factors, radial distribution functions, second virial
coefficients, Flory-Huggins :math:`\chi` parameters, bulk isothermal
compressibilities, and spinodal decomposition temperatures. A full description
of PRISM theory and the nature of these correlation functions is beyond the
scope of this document and we refer readers to our recent work for more details
on PRISM theory. :cite:`pyPRISM`

.. figure:: figure1.pdf

    Schematic of PRISM theory numerical solution process. :label:`numerical`

While the PRISM equation can be solved analytically in select cases, we focus on a
more general numerical approach in pyPRISM. Figure :ref:`numerical` shows a
schematic of our approach. In short, after the user supplies a number of
parameter and input correlation functions, we apply a numerical optimization
routine, such as Picard iteration or Newton-Krylov methods, to minimize a cost
function. After the cost function is minimized, the PRISM equation is considered
‘solved’ and the resultant H(r) and C(r) can be used for calculations

pyPRISM Example 
----------------

.. code:: python
    :linenos:
    
    '''
    pyPRISM script calculating the pair correlation 
    function and structure factor of a polymer blend.
    '''
    import pyPRISM

    sys = pyPRISM.System(['poly1','poly2'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.05,length=8192)
    sys.density['poly1']  = 0.7
    sys.density['poly2']  = 0.2

    sys.diameter[sys.types] = 1.0
    sys.closure[sys.types,sys.types] = 
        pyPRISM.closure.PercusYevick()
    sys.potential[sys.types,sys.types] = 
        pyPRISM.potential.HardSphere()

    sys.omega['poly1','poly1']  = 
        pyPRISM.omega.Gaussian(sigma=1,length=20000)
    sys.omega['poly2','poly2']  = 
        pyPRISM.omega.Gaussian(sigma=1,length=10000)
    sys.omega['poly1','poly2']  = 
        pyPRISM.omega.InterMolecular()

    PRISM = sys.solve()

    rdf = pyPRISM.calculate.pair_correlation(PRISM)
    rdf_11 = rdf['poly1','poly1']
    rdf_12 = rdf['poly1','poly2']

    sk  = pyPRISM.calculate.structure_factor(PRISM)
    sk_11 = sk['poly1','poly1']
    sk_12 = sk['poly1','poly2']

pyPRISM defines a scripting API that allows users to construct calculations and
numerically solve the PRISM equation (Equation :ref:`PRISMeq`) for a range of
liquid-like systems. The code above shows how to use pyPRISM to calculate the
structure of a simple bidisperse polymer blend.  Users first create a
:code:`pyPRISM.System` object by defining the names of the site-types for the
calculation. In this case, we have two site-types which we call 'poly1' and
'poly2'. Next, various containers in the :code:`pyPRISM.System` object are
populated to define the molecular structure and interactions of the system.
Some of these parameters define the properties of a single site-type (e.g.,
density, diameter) which others define properties for a site-type pair (.e.g,
closure, potential, omega). When all properties are defined, the user calls the
:code:`pyPRISM.System.solve()` method which first conducts a number of sanity
checks  and then attempts to numerically solve the PRISM equations. If
successful, a :code:`pyPRISM.PRISM` object is created which contains the final
solutions for :math:`H(r)` and :math:`C(r)`. The :code:`pyPRISM.PRISM` object
can then be passed through any of the calculators in pyPRISM to calcuate
various thermodynamic and structural parameters.

In comparison 

This calculation takes seconds to finish, representing.  Compare to MD/MC


Implementation Details
----------------------

.. figure:: figure2.pdf

    Overview of codebase and class organization. :label:`code`

pyPRISM is a Python library that has been tested on the CPython 2.7.x, 3.5.x,
and 3.6.x series and only strictly depends on Numpy :cite:`numpy` and Scipy
:cite:`scipy` for core functionality. The current set of classes and functions
are showing in Figure :ref:`code`.




MatrixArray Discussion
Other Containers?


Pedagogy
--------

It is our stated goal to not only create a platform for polymer liquid state
theorists to innovate on, but to also lower the barriers to using PRISM theory
for the greater polymer science community. 

In this effort, we have recognized many of those who would benefit most from
pyPRISM theory will struggle with the details of both the theory and
programming. 

We have strove to ensure that the
scripting API and namespaces are descriptive and clear with as limited jargon as
possible. 
pyPRISM
We have also worked to make the interface intuitive 

1. API should be descriptive, clear, and forgiving

2. Easy to add features

3. Detailed API documentation, knowledgebase, tutorial


To this end, pyPRISM has been designed to be accessible to users with varying
levels of training in theory or programming. This means that 


Future Directions
-----------------

