:author: Brandon Butler
:email: butlerbr@umich.edu
:institution: University of Michigan, Department of Chemical Engineering

:author: Joshua Anderson
:institution: University of Michigan, Department of Chemical Engineering

:author: Sharon C. Glotzer
:institution: University of Michigan, Department of Chemical Engineering
:institution: University of Michigan, Department of Material Science and Engineering
:institution: University of Michigan, Department of Physics
:institution: University of Michigan, Biointerfaces Institute

:bibliography: references

------------------------------------------------------------
HOOMD v3 A Modern, Extensible, Flexible, Object-Oriented API
------------------------------------------------------------

.. class:: abstract

    .. TODO Abstract goes here

.. class:: keywords

    molecular dynamics, particle simulations, Monte Carlo simulations, object-oriented

Switch to Object Oriented API
-----------------------------

HOOMD-blue was first created in 2008 by Joshua Anderson. Soon after, he added a Python interface for
writing simulations scripts. The first Python API was inspired by other simulation software such as
LAMMPS :cite:`TODO` which use(s/d) an imperative style scripting interface. This larger remained the
same as HOOMD-blue released its version 2.0. However, as the package transitions into a 3.0 release,
the API has been rethought from the ground up to present a thoroughly object oriented and Pythonic
interface for users. Other Python packages like SciPy :cite:`TODO`, NumPy :cite:`TODO`, scikit-learn
:cite:`TODO`, matplotlib :cite:`TODO`, and others have inspired us in this pursuit. In this
endeavour we have founds ways to make HOOMD-blue more flexible, extensible, and integrable with the
SciPy community as well.

Each simulation in HOOMD-blue now is controlled through 3 main objects which are joined by the
:code:`hoomd.Simulation` class: the :code:`hoomd.Device`, :code:`hoomd.State`,
:code:`hoomd.Operations` classes. Each :code:`Simulation` object holds the requisite information to
run a full molecular dynamics (MD) or Monte Carlo (MC) simulation. The :code:`Device` class denotes
whether a simulation should be run on CPU or GPU and the number of cores/GPUS it should run on. In
addition, the device manages thinks like custom memory tracebacks and MPI communicator. The system
data (e.g. particle positions, orientations, velocities) is stored in the :code:`State` while
the :code:`Operations` object stores objects which act on this data. The design with brief
descriptions can be seen in Figure (:ref:`base-classes`).

Over the next few sections, we will use examples of HOOMD-blue's version 3 API to highlight
changes in the package's extensibility, flexibility, or Pythonic interface.

Triggers
--------

Previously in HOOMD-blue, everything that was not run every timestep had a period associated with
it and phase associated with it. The timesteps the operation was run on could then be determined by
the following expression.

.. code-block:: python

    timestep % period - phase == 0

In our refactoring and development, we recognized that this concept could be made much more general
and consequently more flexible. The generalization is that objects did not have to be run on a
periodic timescale; they just needed some indication of when to run. In other words, the operations
needed to be triggered. The :code:`Trigger` class encapsulates such a concept with some other
functionality like minor caching of results. Each operation that requires triggering is now
associated with a corresponding :code:`Trigger` instance. Some examples of the new flexibility this
approach provides can be seen in the currently implemented subclasses of :code:`Trigger` such as :code:`And`, :code:`Or`, and :code:`Not` whose function can be understood by recognizing that a :code:`Trigger` is essentially a functor that returns a Boolean value.

In addition, to the flexibility the :code:`Trigger` class provides, abstracting out the concept of
triggering an operation, we can provide through pybind11 a way to subclass :code:`Trigger` in
Python. This allows users to create their own triggers in pure Python. An example of such
subclassing reimplementing the functionality of HOOMD-blue version 2.x can be seen in the below --
this functionality already exists in the :code:`Periodic` class.

.. code-block:: python

    from hoomd.trigger import Trigger

    class CustomTrigger(Trigger):
        def __init__(self, period, phase=0):
            self.period = period
            self.phase = phase

        def __call__(self, timestep):
            return timestep % self.period - self.phase == 0

While this example is quite simple, user created subclasses of :code:`Trigger` need not be. They
could implement arbitrarily complex Python code for more caching, examining the simulation state,
etc.

Variants
--------

Similar to :code:`Trigger`, we generalized our ability to linear interpolate values
(:code:`hoomd.variant.liner_interp` in HOOMD v2) across timesteps to a base class :code:`Variant`
which generalizes the concept of functions in the semi-infinite domain of timesteps
:math:`t \in [0,\infty), t \in \mathbb{Z}`. This allows sinusoidal cycling, non-uniform ramps, and
various other behaviors -- as many as there are functions in the non-negative integer domain and
real range. Like :code:`Trigger`, :code:`Variant` is able to be directly subclassed from the C++
class. An example of a sinusoidal cycled variant is shown below.

.. code-block:: python

    from math import sin
    from hoomd.variant import Variant

    class SinVariant(Variant):
        def __init__(self, period, amplitude,
                     phase=0, center=0):
            self.period = period
            self.amplitude = amplitude
            self.phase = phase
            self.center = center

        def __call__(self, timestep):
            tmp = sin(self.period + self.phase)
            return self.amplitude * tmp + self.center

        def _min(self):
            return -self.amplitude

        def _max(self):
            return self.amplitude

ParticleFilters
---------------

We will round off the examples of directly inheritable classes in HOOMD-blue v3 with
:code:`ParticleFitler`. Unlike :code:`Trigger` or :code:`Variant`, :code:`ParticleFitler` is not a
generalization of an existing concept but the splitting of one class into two. However, we will find
that this affords us a similar flexibility. In HOOMD v2, the :code:`ParticleGroup` class and
subclasses served to provide a subset of particles within a simulation for file output, application
of thermodynamic integrators, and other purposes. The class hosted both the logic for storing the
subset of particles and filtering them out from all the system. After the refactoring,
:code:`ParticleGroup` still exists but just for the logic to store and preform some basic operations
on particle tags (a means of individuating particles). The new class :code:`ParticleFilter`
implements the selection logic. This choice makes :code:`ParticleFilter` objects much more
lightweight and provide a :code:`State` specific cache of :code:`ParticleFilter` objects. The latter
ensures that we do not create multiple of the same :code:`ParticleGroup` which can occupy large
amounts of memory. The separation also allows the creation of large numbers of the same
:code:`ParticleFitler` object without needing to worry about memory constraints. Finally, this
separation makes, :code:`ParticleFilter` a suitable class to subclass since its scope is limited and
does not have to deal with many of the internal details that the :code:`ParticleGroup` class does.
For this reason, :code:`ParticleGroup` instances are private in HOOMD v3.
