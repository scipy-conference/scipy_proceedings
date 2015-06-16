:author: Kathryn Huff
:email: katyhuff@gmail.com
:institution: University of California, Berkeley

-----------------------------------------------------
PyRK: A Python Package For Nuclear Reactor Kinetics
-----------------------------------------------------

.. class:: abstract

   In this work, a new python package, PyRK (Python for Reactor Kinetics), is
   introduced and demonstrated.  PyRK has been designed to simulate, in zero
   dimensions, the transient, coupled, thermal-hydraulics and neutronics of
   time-dependent behavior in nuclear reactors. PyRK is intended for analysis
   of many commonly studied transient scenarios including normal reactor
   startup and shutdown as well as abnormal scenarios including Beyond Design
   Basis Events (BDBEs) such as Accident Transients Without Scram (ATWS). For
   robustness, this package employs various tools within the scientific python
   ecosystem. For additional ease of use, it employes an object-oriented data
   model, allowing nuclear engineers to rapidly prototype nuclear reactor
   control and safety systems in the context of their novel nuclear reactor
   designs. An application of PyRK, capturing the evolution of an accident
   scenario in a novel reactor design, is demonstrated.


.. class:: keywords

   engineering, nuclear reactor, package

Introduction
------------

Transient analysis is fundamental to nuclear reactor design. That is,
time-dependent fluctuations in neutron population, fluid flow, and heat transfer are
essential to understanding the performance and safety of a reactor. Such
_transients_ include normal reactor startup and shutdown as well as abnormal scenarios
including Beyond Design Basis Events (BDBEs) such as Accident Transients
Without Scram (ATWS). However, no open source tool currently exists for
reactor transient analysis. To fill this gap, PyRK (Python for Reactor
Kinetics)[Huff2015]_, a new python package for nuclear reactor kinetics was
created.  No other open source tool exists for this application, so PyRK is the
first to offer an open source tool designed to conduct:

- time-dependent,
- lumped parameter thermal-hydraulics,
- coupled with neutron kinetics,
- in 0-dimensions,
- for nuclear reactor analysis,
- in an object-oriented context.


For background, this paper will introduce necessary concepts for understanding
the PyRK model and will describe the differential equations representing the
coupled physics at hand. Next, the implementation of the data model, simulation
framework, and numerical solution will be described. This discussion will
include the use, in PyRK, of many parts of the scientific python software
ecosystem: NumPy for array manipulation, SciPy for ODE and PDE solvers, nose
for testing, distutils for packaging, Pint for unit-checking, and Sphinx for
documentation. Finally, this paper conclude with a demonstration of accident
simulations conducted in the context of the Pebble-Bed, Fluoride-Salt-Cooled,
High-Temperature Reactor \cite{andreades_technical_2014}.


Background
----------

Fundamentally, nuclear reactor transient analyses must characterize the
relationship between neutron population and temperature. These two parameters are coupled
together by reactivity, :math:`\rho` , which characterizes the departure of the
nuclear reactor from criticality:

.. math::
   :type: align
   :label: reactivity

   \rho &= \frac{k-1}{k}
   \intertext{where}
   \rho &= \mbox{reactivity}\\
   k &= \mbox{multiplication factor}



In all power reactors, the scalar flux of neutrons determines the reactor's power. The reactor power, in
turn, affects the temperature. Reactivity feedback then results, due the
temperature dependence of geometry, material densities, the neutron spectrum,
and microscopic cross sections [Bell1970]_.

One common method for approaching these transient simulations is a
zero-dimensional approximation which results in differential equations called
the Point Reactor Kinetics Equations (PRKE). PyRK provides a simulation
interface that drives the solution of these equations in a modular, reactor
design agnostic manner. In particular, PyRK provides an object oriented data
model for generically representing a nuclear reactor system and provides the
capability to exchange solution methods from one simulation to another.


The Point Reactor Kinetics Equations can only be understood in the context of
neutronics, thermal-hydraulics, reactivity, delayed neutrons, and reactor
control.

Neutronics
************

The heat produced in a nuclear reactor is due to nuclear fission reactions. In
a fission reaction, a neutron collides inellastically with a 'fissionable'
isotope, which subsequently splits. This reaction emits both heat and neutrons.
When the emitted neutrons go on to collide with another isotope, this is called
a nuclear chain reaction and is the basis of power production in a nuclear
reactor. The study of the population, speed, direction, and energy spectrum of
neutrons in a reactor as well as the related rate of fission at a particular
moment is called neutronics or neutron transport. Neutronics simulations
characterize the production and destruction of neutrons in a reactor and
dependend on many reactor material properties and design choices (e.g.,
atomic densities and material configurations).

Thermal-Hydraulics
********************

Reactor thermal hydraulics describes the mechanics of flow and heat in fluids
present in the reactor core. As fluids are heated or cooled in a reactor core
(e.g. due to changes in fission power) pressure, density, flow, and other
parameters of the system respond accordingly.  The fluid of interest in a
nuclear reactor is typically the coolant.  The hydraulic properties of this
fluid depend primarily on its intrinsic properties and the characteristics of
the cooling system. Thermal hydraulics is also concerned with the heat transfer
between thevarious components of the reactor (e.g., heat generation in the
reactor fuel heat removal by the coolant). Heat transfer behavior depends on
everything from the moderator density and temperature to the neutron-driven
power production in the fuel.


Reactivity
****************
The two physics are coupled by the notion of reactivity, which is related to
the probility of fission due to material properties. The temperature and
density of materials can increase or decrease this probability, which directly
impacts the neutron production and destruction rates and therefore, the reactor
power. TODO, this bit is badly writ.


The PRKE
*********
The Point Reactor Kinetics Equations (PRKE) are the set of equations that
capture neutronics and thermal hydraulics when geometry is neglected. The two
physics are coupled primarily by reactivity, but have very different
characteristic time scales, so the equations are quite stiff.

.. math::
   :type: equation
   :label: fullprke

   \frac{d}{dt}\left[
    \begin{array}{c}
      p\\
      \zeta_1\\
      .\\
      .\\
      .\\
      \zeta_j\\
      .\\
      .\\
      .\\
      \zeta_J\\
      \omega_1\\
      .\\
      .\\
      .\\
      \omega_k\\
      .\\
      .\\
      .\\
      \omega_K\\
      T_{fuel}\\
      T_{cool}\\
      T_{refl}\\
      T_{matr}\\
      T_{grph}\\
      .\\
      .\\
      .\\
    \end{array}
    \right]
    =
    \left[
      \begin{array}{ c }
        \frac{\rho(t,T^{fuel},T_{cool},\cdots)-\beta}{\Lambda}p +
        \displaystyle\sum^{j=J}_{j=1}\lambda_j\zeta_j\\
        \frac{\beta_1}{\Lambda} p - \lambda_1\zeta_1\\
        .\\
        .\\
        .\\
        \frac{\beta_j}{\Lambda}p-\lambda_j\zeta_j\\
        .\\
        .\\
        .\\
        \frac{\beta_J}{\Lambda}p-\lambda_J\zeta_J\\
        \kappa_1p - \lambda_1\omega_1\\
        .\\
        .\\
        .\\
        \kappa_kp - \lambda_k\omega_k\\
        .\\
        .\\
        .\\
        \kappa_{k p} - \lambda_k\omega_{k}\\
        f_{fuel}(p, C_p^{fuel}, T_{fuel}, T_{cool},\cdots)\\
        f_{cool}(C_p^{cool}, T_{fuel}, T_{cool},\cdots)\\
        f_{refl}(C_p^{refl}, T_{fuel}, T_{refl},\cdots)\\
        f_{matr}(C_p^{matr}, T_{fuel}, T_{matr},\cdots)\\
        f_{grph}(C_p^{grph}, T_{fuel}, T_{grph},\cdots)\\
        .\\
        .\\
        .\\
      \end{array}
      \right]


In the above matrix equation, the following variable definitions are used:

.. math::
   :type: align
   :label: n_data

    \rho(t,&T_{fuel},T_{cool},T_{mod}, T_{refl}) = \mbox{ reactivity, [pcm]}\\
    \beta &= \mbox{ fraction of neutrons that are delayed}\\
    \beta_j &= \mbox{ fraction of delayed neutrons from precursor group j}\\
    \zeta_j &= \mbox{ concentration of precursors of group j}\\
    \lambda^{d}_j &= \mbox{ decay constant of precursor group j}\\
    \Lambda &= \mbox{ mean generation time }\\
    \omega_k &= \mbox{ decay heat from FP group k}\\
    \kappa_k &= \mbox{ heat per fission for decay FP group k}\\
    \lambda^{FP}_k &= \mbox{ decay constant for decay FP group k}

The PRKE in equation :ref:`fullprke` can be solved in numerous ways, using
either loose or tight coupling.  Operator splitting, loosely coupled in time,
is a stable technique that neglects higher order nonlinear terms in exchange
for solution stability.  Under this approach, the system can be split clearly
into a neutronics sub-block and a thermal-hydraulics sub-block which can be
solved independently at each time step, combined, and solved again for the next
time step.

.. math::
   :type: align
   :label: os

   U^n &= \left[
          \begin{array}{ c }
            N^n\\
            T^n\\
          \end{array}
          \right]\\
   N^{n+1} &= N^n + kf(U^n)\\
   \nonumber\\
   U^* &= \left[
          \begin{array}{ c }
            N^{n+1}\\
            T^n\\
          \end{array}
          \right]\\
   T^{n+1} &= T^n + kf(U^*)


PyRK Implementation
--------------------

Now that the premise of the problem is clear, the implementation of the package
can be discussed. Fundamentally,  PyRK is object oriented and modular. The
important object classes in PyRK are:

- SimInfo: Reads the input file, manages the solution matrix, Timer, and
  communication between neutronics and thermal hydraulics.
- Neutronics : calculates dP=dt, d!j=dt, based on dTi=dt and the external
  reactivity insertion.
- THSystem : manages various THComponents, facilitates their communication.
- THComponent : Conducts lumped parameter calculation. Other thermal models can
  inherit from it and replace it in the simulation.
- Material : A class for defining the intensive properties of a material
  (:math:`c_p`, :math:`\rho`, :math:`k_{th}`). Currently, subclasses include
  Flibe, Graphite, and Kernel.

Each of these classes will be discussed in detail in this section.

SimInfo
********

PyRK has implemented a casual context manager pattern by encapsulating
simulation information in a SimInfo object. This class keeps track of the neutronics
system and its data, the thermal hydraulics system (THSystem) and its
components (THComponents), as well as timing and other simulation-wide
parameters.

In particular, the SimInfo object is responsible for capturing the information
conveyed in the input file.  The input file is a python file holding parameters
specific to the reactor design and transient scenario. However, a more robust
solution is anticipated for future versions of the code, relying on a json
input file rather than python, for more robust validation options.

The current output is a plain text log of the input, runtime messages, and the
solution matrix. The driver automatically generates a number of plots.  However,
a more robust solution is anticipated for v0.2, relying on an output database
backend in hdf5, via the pytables package.


Neutronics Class
******************
The neutronics class holds the first 1+j+k equations in the right hand side of
the matrix equation in :ref:`fullprke`.

Additionally, the accident scenario can be driven by an insertion of reactvity
(e.g. due to the removal of a control rod). In PyRK, this reactivity insertion
capability is captured in the ReactivityInsertion class, from which reactivity
insertions can be selected and customized as in figure :ref:`figri`.

.. figure:: ri.png

   The reactivity insertion that can drives the PyRK simulator can be selected
   and customized from three models. :label:`figri`


Nuclear data encapsulating the fractions of delayed neutron precursors and
their precursor group halflives are stored in the PrecursorData class.

THSystem
**********

Each neutronic object needs a temperature. To determine that temperature,
The neutronics class holds the first 1+j+k equations in the right hand side of
the matrix equation in :ref:`fullprke`.

THComponent
***********

The THSystem class is made up of THComponent objects, linked together at
runtime by interfaces defined in the input class.

Object-Oriented Simulation Model
---------------------------------

The world is made of objects, so an object-oriented data model provides the
most intuitive user experience in a simulation environment [citationneeded]_.

Quality Assurance
-----------------

For robustness, a number of tools were used to improve robustness and
reproducibility in this package. These include:

- [github]_ : for versio control
- [matplotlib]_ : for plotting
- [nose]_ : for unit testing
- [numpy]_ : for holding and manipulating arrays of floats
- [pint]_ : for dimensional analysis and unit conversions
- [scipy]_ : for ode solvers
- [sphinx]_ : for automated documentation
- [travis-ci]_ : for continuous integration

Together, these tools create a functional framework for distribution and reuse.

Unit Validation
*****************

Of particular note, the Pint package (pint.readthedocs.org/en/0.6/) is
used keeping track of units, converting between them, and throwing
errors when unit conversions are not sane. For example, in the code below,
the user is able to initialize the material object with :math:`k_{th}` and
:math:`c_p` in any valid unit for those quantities. Upon initialization of
those member variables, the input values are converted to SI using Pint.

.. code-block:: python

    class Material(object):
        """This class represents a material. Its attributes
        are material properties and behaviors."""

        def __init__(self, name=None,
                     k=0*units.watt/units.meter/units.kelvin,
                     cp=0*units.joule/units.kg/units.kelvin,
                     dm=DensityModel()):
            """Initalizes a material

            :param name: The name of the component...
            :type name: str.
            :param k: thermal conductivity, :math:`k_{th}`
            :type k: float, pint.unit.Quantity
            :param cp: specific heat capacity, :math:`c_p`
            :type cp: float, pint.unit.Quantity
            :param dm: The density of the material
            :type dm: DensityModel object
            """
            self.name = name
            self.k = k.to('watt/meter/kelvin')
            validation.validate_ge("k", k,
                0*units.watt/units.meter/units.kelvin)
            self.cp = cp.to('joule/kg/kelvin')
            validation.validate_ge("cp", cp,
                0*units.joule/units.kg/units.kelvin)
            self.dm = dm

The above code employs a validation utility written for PyRK and used
throughout the code to confirm (at runtime) types, units, and valid ranges for
parameters of questionable validity.  Those validators are simple, but
versatile, and in combination with the Pint package, provide a robust
environment for users to experiment with parameters in the safe confines of
dimensional accuracy.


Conclusions and Future Work
----------------------------

The PyRK library provides a modular simulation environment for a common and
essential calculation in nuclear engineering. PyRK is the first freely
distributed tool for neutron kinetics. By supplying a library of ANSI standard
precursor data, a modular material definition framework, and coupled lumped
parameter thermal hydraulics with zero-dimensional neutron kinetics in an
object-oriented modeling paradigm, PyRK provides design-agnostic toolkit for
accident analysis potentially useful to all nuclear reactor designers and analysts.

References
----------

.. [Andreades2014] Andreades, etc.

.. [Huff2015] Huff

.. [Bell1970] Bell and Glasstone

.. [matplotlib]

.. [nose]

.. [numpy]

.. [pint]

.. [scipy]

.. [travis-ci]

.. [github]

.. [sphinx]

.. [citationneeded]
