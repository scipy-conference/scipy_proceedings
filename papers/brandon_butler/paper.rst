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

    HOOMD-blue is a Python library for running molecular dynamics and hard particle Monte Carlo
    simulations. HOOMD-blue is written in C++ with a Python interface through pybind11. The packages
    is designed to scale from a single CPU core to multiple GPU's and anything in between. In
    developing HOOMD-blue v3, a large focus has been to improve the application protocol interface
    (API) by making it more flexible, extensible, and Pythonic. We have also striven to provide
    performant ways to interface Python code with the C++ core. In this paper, we show how those
    goals have been acheived and explain design decisions through examples of the new API.

.. class:: keywords

    molecular dynamics, particle simulations, Monte Carlo simulations, object-oriented

Introduction
------------

Two methods of determining equilibrium properties of particulate matter, Monte Carlo (MC) and
molecular dynamics (MD), have existed in some form since the 1950's :cite:`TODO`. Today many
packages exist for this purpose: LAMMPS :cite:`TODO`, Gromacs :cite:`TODO`, OpenMM :cite:`TODO`, and
HOOMD-blue :cite:`TODO` to name a few. Algorithms for improving performance using GPU's :cite:`TODO`
or other means :cite:`TODO` and greater accessibility to greater computational power has improved
tremendously the length and time scales of simulations. However, having well designed application
protocol interfaces (API) is also important for simulation engines. They must provide scientists
with little computer science exercise a facile way to interface with the code while also providing
those experienced with software the power to customize and tailor the code to their needs.
HOOMD-blue always has aimed to provide such an interface, and changes in its API for version 3 have
continued to improve upon that aim.

HOOMD-blue was first created in 2008 by Joshua Anderson. Soon after, he added a Python interface for
writing simulations scripts. The first Python API was inspired by other simulation software such as
LAMMPS. This largely remained the same as HOOMD-blue released its version 2.0. However, as the
package transitions into a 3.0 release, the API has been rethought from the ground up to present a
thoroughly object oriented and Pythonic interface for users. In addition, where possible we have
sought to provide performant ways to add custom Python objects into HOOMD-blue's run loop. Other
Python packages like SciPy :cite:`TODO`, NumPy :cite:`TODO`, scikit-learn :cite:`TODO`, matplotlib
:cite:`TODO`, and others have inspired us in this pursuit. In this endeavour we have founds ways to
make HOOMD-blue more flexible, extensible, and integrable with the SciPy community as well.  Over
the next few sections, we will use examples of HOOMD-blue's version 3 API to highlight changes in
the package's extensibility, flexibility, or Pythonic interface.

General API Design
------------------

Simulation, Device, State, Operations
+++++++++++++++++++++++++++++++++++++

Each simulation in HOOMD-blue now is controlled through 3 main objects which are joined together by
the :code:`Simulation` class: the :code:`Device`, :code:`State`, :code:`Operations` classes. Each
:code:`Simulation` object holds the requisite information to run a full molecular dynamics (MD) or
Monte Carlo (MC) simulation.  The :code:`Device` class denotes whether a simulation should be run on
CPU or GPU and the number of cores/GPUS it should run on. In addition, the device manages thinks
like custom memory tracebacks and MPI communicator.

.. figure:: figures/object-diagram.pdf
    :align: center

    Diagram of core objects with some attributes and methods.

The :code:`State` class stores the system data (e.g. particle positions, orientations, velocities,
the system box). The :code:`State` class also exposes this data and allows setting it in two
fundamental ways. Through the snapshot API, users interface with a single object exposing many NumPy
arrays of system data. To construct a snapshot all system data stored across MPI ranks must be
gathered and combined. Changing the :code:`State` through the snapshot object requires setting the
snapshot property to an entirely new snapshot. The advantages to this approach come from its ease of
use which can be seen in the snippet below.

.. code-block:: python

    snap = sim.state.snapshot
    # set all z positions to 0
    if snap.exists:
        snap.particles.position[:, 2] = 0
        sim.state.snapshot = snap

The other API for accessing :code:`State` data is through a zero copy MPI rank local access. The
data buffers are exposed as NumPy arrays, and support quick read and write access. To ensure data
integrity we use a context-manager to remove access to the data buffers. This approach is faster,
but requires the user to properly deal with MPI ranks. The following code snippet shows this
approach.

.. code-block:: python

    with sim.state.local_snapshot as data:
        data.particles.position[:, 2] = 0

Both approaches allow the complete use of the SciPy ecosystem as they use NumPy arrays. In addition
to these two methods though, we plan on exposing the data through the
:code:`__cuda_array_interface__` as well which would allow interoperability with cupy :cite:`TODO`,
numba's :cite:`TODO` GPU capabilities, and other packages which support the interface.

The final of the three :code:`Operations` holds the different "operations" that will act on the
simulation state. Broadly these consist of 3 categories: updaters which modify simulation state,
analyzers which observe system state, and tuners which tune other operation's hyperparameters for
performance.

The Internal Base Classes
+++++++++++++++++++++++++

To facilitate adding more features to HOOMD-blue, simplify the internal class logic, and provide a
more uniform interface, we wrote the :code:`_Operation` class. This base class is inherited by most
other user facing classes. Through it we provide object dependency handling, deferred C++
initialization (explained below), and our default way of exposing and syncing attributes between
Python and C++.

Likewise, to provide a Pythonic interface for interacting with object parameters, robust validation
on setting, and syncing with C++ when "attached" to a :code:`Simulation`, we created two solutions
: one for parameters that are type dependent and those that were not.  Through the
:code:`ParameterDict` class, we ensure syncing between C++ objects variable and Python variables
while exposing the dictionaries keys as attributes. For type dependent attributes, we use
:code:`TypeParameter` and :code:`TypeParameterDict` to provide syncing with C++. These type dependent
quantities are exposed a dictionary-like attributes for the containing class.

Both classes support validation of each of keys, and the :code:`TypeParameterDict` can be used to
define validation of arbitrarily nested structures of dictionaries, lists, and tuples. In addition,
both classes support a similar level of default specification to their validation. An example
object specification and initialization can be seen below.

.. code-block:: python

    TypeParameterDict(
        num=float,
        list_of_str=[str],
        nesting={len_three_vec=(float, float, float)},
        len_keys=2
        )


An example of the interface for both can be seen in the following code example of the hard particle
MC :code:`Sphere` integrator.


.. code-block:: python

    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere(seed=42)
    # example using ParameterDict
    sphere.nselect = 2
    # examples using TypeParameter and TypeParameterDict
    sphere.shape['A'] = {'diameter': 1.}
    sphere.shape[['B', 'C', 'D']] = {'diameter': 0.5}

In a similar vain to store lists that must be synced to C++, the :code:`SyncedList` class
allow syncing between the C++ and Python lists.

.. code-block:: python

    from hoomd import Operations
    from hoomd.dump import GSD

    ops = Operations()
    gsd = GSD('example.gsd')
    # use of SyncedList
    ops.analyzers.append(gsd)

Error Handling
**************

Another improvement to user experience is our improved error messaging and handling. An example
error message for accidentally trying to set :code:`sigma` for particle type 'A' in the
Lennard-Jones pair potential to a string would provide the error message, TODO.

Deferred C++ Initialization
+++++++++++++++++++++++++++

Many objects in C++ in HOOMD-blue require either a :code:`System` or a :code:`SystemState` object
(both C++ classes) in order to be correctly instantiated. The requirement is foremost due to the
interconnected nature of many things in a simulation. However, this requires a very strict order in
which objects can be created. Having to create a full simulation state to create a
:code:`PairPotential` object limits the utility and ease of Python plugins to HOOMD-blue. For
instance, a package that wanted to automatically generate a particular force-field in response to
some user inputs would have to have access to the :code:`State` it was to operate on. Our decision
in HOOMD-blue v3 was to defer the initialization of C++ objects until they are "attached" to a
:code:`Simulation`. This has the benefit that most Python only plugins to HOOMD would not need to
worry about initializing context or state. Another motivating force for the decision, deferred
initialization provides more leniency to users creating scripts.

This deferring also has an additional benefit in the ability to more easily store the object's
state. We take advantage of this by making an object's state a loggable quantity for the
:code:`Logger` object, and providing a :code:`from_state` factory method for all operations in HOOMD
which can reconstruct the object from the state.

Logging and Accessing Data
--------------------------

Another area that made a switch from an imperative to object oriented style can be seen in the new
HOOMD-blue v3 logging system. Through extensive use of properties, we now directly expose object
data such as the per-particle potential energy in all our pair potentials rather than require it to
be logged first. When logging data is desired, we have created a Python :code:`Logger` class that
creates an intermediate representation of the logged information when called. By using an
intermediate representation, we get the added flexibility of supporting multiple official and
user-created "back-ends" for the logger.

Traditional simulation output such as standard out are fairly easy to implement in Python while
other back-ends like MongoDB, Pandas, and Python pickles are possible. In addition to this improved
flexibility in storage, for HOOMD-blue v3 we have extensively added properties to different objects
to directly expose their data rather than require it to be logged first -- in fact, this is how the
:code:`Logger` class acquires the data. An example of this is how pairwise potentials expose the
total energy of their potential in the system at a given time. Furthermore, to improve integration
with Python packages, we have exposed data from C++ that previously was not available in Python. To
continue with the pairwise potential example, the force on individual particles for a given pairwise
potential is now accessible in Python through properties.

Logger
++++++

The :code:`Logger` class aims to provide a simple interface for logging most HOOMD-Blue object and
custom user quantities. Through the :code:`Loggable` metaclass, all subclasses that inherit from
:code:`_Operation` expose their loggable quantities. Adding an object to a logger for logging is as
simply as :code:`logger += obj`. The utility of this class, however, lies in its intermediate
representation of the data. Using the HOOMD-blue namespace as the basis for separating logged
quantities, we map logged quantities into a nested dictionary. For example, logging the
Lennard-Jones pair potential's total energy would be produce this dictionary by a :code:`Logger`
object :code:`{'md': {'pair': {'LJ': {'energy': (-1.4, 'scalar')}}}}` where the :code:`'scalar'` is
a flag to make processing logged values more easily. In real use cases, the dictionary would likely
be filled with many other quantities. This intermediate form allows developers and users to more
easily create different back ends that a :code:`Logger` object can plug into for outputting data.

User Customization
------------------

Triggers
++++++++

In HOOMD-blue v2, everything that was not run every timestep had a period associated with it and
phase associated with it. The timesteps the operation was run on could then be determined by the
expression, :code:`timestep % period - phase == 0`.  In our refactoring and development, we
recognized that this concept could be made much more general and consequently more flexible, objects
do not have to be run on a periodic timescale; they just need some indication of when to run. In
other words, the operations needed to be "triggered". The :code:`Trigger` class encapsulates such a
concept with some other functionality like minor caching of results, providing a uniform way of
specifying when an object should run without limiting options. Each operation that requires
triggering is now associated with a corresponding :code:`Trigger` instance. Some examples of the new
possibilities this approach provides can be seen in the currently implemented subclasses of
:code:`Trigger` such as :code:`And`, :code:`Or`, and :code:`Not` whose function can be understood by
recognizing that a :code:`Trigger` is essentially a functor that returns a Boolean value.

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

While this example is quite simple, user created subclasses of :code:`Trigger` need not be as seen
in an example in a further section. They could implement arbitrarily complex Python code for more
caching, examining the simulation state, etc.

Variants
++++++++

Similar to :code:`Trigger`, we generalized our ability to linear interpolate values
(:code:`hoomd.variant.liner_interp` in HOOMD v2) across timesteps to a base class :code:`Variant`
which generalizes the concept of functions in the semi-infinite domain of timesteps :math:`t \in
[0,\infty), t \in \mathbb{Z}`. This allows sinusoidal cycling, non-uniform ramps, and various other
behaviors -- as many as there are functions in the non-negative integer domain and real range. Like
:code:`Trigger`, :code:`Variant` is able to be directly subclassed from the C++ class.
:code:`Variant` objects are used in HOOMD-blue to specify temperature, pressure, and box size for
varying objects. An example of a sinusoidal cycled variant is shown below.

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
+++++++++++++++

Unlike :code:`Trigger` or :code:`Variant`, :code:`ParticleFitler` is not a generalization of an
existing concept but the splitting of one class into two. However, this affords us a similar
flexibility. In HOOMD v2, the :code:`ParticleGroup` class and subclasses served to provide a subset
of particles within a simulation for file output, application of thermodynamic integrators, and
other purposes. The class hosted both the logic for storing the subset of particles and filtering
them out from all the system. After the refactoring, :code:`ParticleGroup` still exists but just for
the logic to store and preform some basic operations on particle tags (a means of individuating
particles). The new class :code:`ParticleFilter` implements the selection logic. This choice makes
:code:`ParticleFilter` objects much more lightweight and provide a :code:`State` specific cache of
:code:`ParticleFilter` objects. The latter ensures that we do not create multiple of the same
:code:`ParticleGroup` which can occupy large amounts of memory. The separation also allows the
creation of large numbers of the same :code:`ParticleFitler` object without needing to worry about
memory constraints. Finally, this separation makes, :code:`CustomParticleFilter` which is a subclass
of :code:`ParticleFilter` with some added functionality a suitable class to subclass since its scope
is limited and does not have to deal with many of the internal details that the
:code:`ParticleGroup` class does.  For this reason, :code:`ParticleGroup` instances are private in
HOOMD v3. An example of a :code:`CustomParticleFilter` that selects only particle with positive
charge is given below.

.. code-block:: python

    class PositiveCharge(CustomParticleFilter):
        def __init__(self, state):
            super().__init__(state)

        def __hash__(self):
            return hash(self.__class__.__name__)

        def __eq__(self, other):
            return type(self) == type(other)

        def find_tags(self, state):
            with state.local_snapshot as data:
                mask = data.particles.charge > 0
                return data.particles.tag[mask]

Custom Operations
+++++++++++++++++

Through composition, HOOMD-blue v3 offers the ability to create custom actions in Python that run in
the standard :code:`Simulation` run loop. The feature makes user created actions behave
indistinguishable from native C++ actions. Through custom actions, users can modify state, tune
hyperparameters for performance, or just observe parts of the simulation. With the zero copy access
to the data on the CPU and zero copy access to the data on the GPU expected, custom actions also
allow for users to achieve higher performance using standard Python libraries like NumPy, SciPy,
numba, cupy and others. Furthermore, this performance comes without users having to worry about code
compilation, ABI, or other concerns in compiled languages.

Fuller Examples
---------------

In this section we will provide more substantial applications of features new to HOOMD-blue v3.

Trigger that determines nucleation (freud)
++++++++++++++++++++++++++++++++++++++++++

The first example is a :code:`Trigger` that only returns true when a threshold :math:`Q_6`
Steinhardt order parameter is reached. Such a :code:`Trigger` could be used for nucleation detection
which depending on the type of simulation could trigger a decrease in cooling rate, the more
frequent output of simulation trajectories, or any of numerous other possibilities. Also, in this
example we showcase the use of the local MPI rank data access uses ghost particles as well (ghost
particles are particles that an MPI ranks knows about, but is not directly responsible for
updating). Another approach to implement this class could use the snapshot approach and would be
simpler and shorter, but this approach should be significantly faster on large simulations.

.. code-block:: python

    import numpy as np
    import freud
    from mpi4py import MPI
    from hoomd.trigger import Trigger

    class Q6Trigger(Trigger):
        def __init__(self, simulation, threshold,
                     mpi_comm=None):
            super().__init__()
            self.threshold = threshold
            self.state = simulation.state
            nr = simulation.device.num_ranks
            if nr > 1 and mpi_comm is None:
                raise RuntimeError()
            elif nr > 1:
                self.comm = mpi_comm
            else:
                self.comm = None
            self.q6 = freud.order.Steinhardt(l=6)

        def __call__(self, timestep):
            with self.state.local_snapshot as data:
                part_data = data.particles
                box = data.box
                aabb_box = freud.locality.AABBQuery(
                    box,
                    part_data.positions_with_ghosts)
                nlist = aabb_box.query(
                    part_data.position,
                    {'num_neighbors': 12,
                     'exclude_ii': True})
                Q6 = np.mean(
                    self.q6.compute(
                        (box, part_data.position),
                        nlist).particle_order)
                if self.comm:
                    return self.comm.allreduce(
                        Q6 >= self.threshold,
                        op=MPI.LOR)
                else:
                    return Q6 >= self.threshold


Pandas Logger Back-end
++++++++++++++++++++++

Here we highlight the ability to creatively use the :code:`Logger` class to create novel back-ends
for simulation data. For this example, we will create a Pandas back-end. We will store the scalar
and string quantities in a single :code:`pandas.DataFrame` object while array-like objects will be
stored in a separate :code:`DataFrame` objects. All :code:`DataFrame` objects will be stored in an
dictionary.

.. code-block:: python

    import pandas as pd
    from hoomd import CustomAction
    from hoomd.util import (
        dict_flatten, dict_filter, dict_map)

    def is_flag(flags):
        def func(v):
            return v[1] in flags
        return func

    def not_none(v):
        return v[0] is not None

    def hnd_2D_arrays(v):
        if v[1] in ['scalar', 'string', 'state']:
            return v
        elif len(v[0].shape) == 2:
            return {
                str(i): col
                for i, col in enumerate(v[0].T)}


    class DataFrameBackEnd(CustomAction):
        def __init__(self, logger):
            self.logger = logger

        def act(self, timestep):
            log_dict = self.logger.log()
            is_scalar = is_flag(['scalar', 'string'])
            sc = dict_flatten(dict_map(dict_filter(
                log_dict,
                lambda x: not_none(x) and is_scalar(x)),
                lambda x: x[0]))
            rem = dict_flatten(dict_map(dict_filter(
                log_dict,
                lambda x: not_none(x) and not is_scalar(x)),
                hnd_2D_arrays))

            if not hasattr(self, 'data'):
                self.data = {
                    'scalar': pd.DataFrame(
                        columns=['.'.join(k) for k in sc]),
                    'array': {'.'.join(k): pd.DataFrame()
                            for k in rem}}

            sdf = pd.DataFrame(
                {'.'.join(k): v for k, v in sc.items()},
                index=[timestep])
            rdf = {'.'.join(k): pd.DataFrame(
                        v, columns=[timestep]).T
                for k,v in rem.items()}
            data = self.data
            data['scalar'] = data['scalar'].append(sdf)
            data['array'] = {
                k: v.append(rdf[k])
                for k, v in data['array'].items()}

Comparison between LAMMPS, OpenMM, and HOOMD v3
-----------------------------------------------

Initialing the State
++++++++++++++++++++

Setting the Forces
++++++++++++++++++

Writing Output
++++++++++++++

Logging
+++++++

Running the System
++++++++++++++++++
