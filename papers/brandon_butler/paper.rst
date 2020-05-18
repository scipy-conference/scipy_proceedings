:author: Brandon L. Butler
:email: butlerbr@umich.edu
:institution: University of Michigan, Department of Chemical Engineering

:author: Vyas Ramasubramani
:institution: University of Michigan, Department of Chemical Engineering

:author: Joshua A. Anderson
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
    simulations. HOOMD-blue is written in C++ with support for GPUs using (CUDA/HIP) with a Python
    interface through pybind11. The package is designed to scale from a single CPU core to multiple
    GPU's and anything in between. In developing HOOMD-blue version 3.0, a large focus has been to
    improve the application protocol interface (API) by making it more flexible, extensible, and
    Pythonic. We have also striven to provide performant ways to interface Python code with the C++
    core and internal data structures.  In this paper, we show how those goals have been acheived
    and explain design decisions through examples of the new developing API.

.. class:: keywords

    molecular dynamics, molecular simulations, Monte Carlo simulations, object-oriented

Introduction
------------

Two methods of determining equilibrium properties of molecular systems, Monte Carlo and
molecular dynamics simulations, have existed in some form since the 1950's
:cite:`metropolis.etal1953, alder.wainwright1959`. Molecular dynamics (MD) is the application of
Newton's laws of motion to molecular system, while Monte Carlo simulations (MC) use Monte Carlo
integration over a system's degrees of freedom to find equilibrium quantities. Since their inception
these tools have been used to study numerous systems, examples include colloids
:cite:`damasceno.etal2012`, metallic glasses :cite:`fan.etal2014`, and proteins
:cite:`dignon.etal2018a`.

Today many packages exist for this purpose: LAMMPS :cite:`plimpton1993`, GROMACS
:cite:`berendsen.etal1995, abraham.etal2015`, OpenMM :cite:`eastman.etal2017`, and HOOMD-blue
:cite:`anderson.etal2008, glaser.etal2015, anderson.etal2020` to name a few. Algorithms improving
performance using GPU's :cite:`spellings.etal2017` or other means :cite:`niethammer.etal2014` as
well as greater accessibility of computational power has improved tremendously the length
:cite:`byna.etal2012` and time :cite:`shaw.etal2009` scales of simulations from those conducted in
the mid 1900's.  However, having a well designed application protocol interface (API) is also
important for particle simulation software. To achieve this the various simulation engines provide
many different solutions: text file scripting, command line interfaces, and Python interfaces to
name some. Ultimately, the API must provide scientists a facile way to interface with the code while
also enabling tailoring and customizing the code to their needs.  HOOMD-blue always has aimed to
provide such an interface, and our developments for version 3.0 have continued to improve upon that
aim.

HOOMD-blue is a Python package written in C++ for MD and hard particle MC simulations. HOOMD-blue
was first released in 2008 by Joshua Anderson :cite:`anderson.etal2020`. Joshua developed HOOMD-blue
in C++/CUDA , for a molecular simulation engine that could utilize the still nascent area of GPU
scientific computing. In 2014, HOOMD-blue began supporting parallelization using domain
decomposition (separating a simulation box into local boxes, one for each rank) through MPI. Recent
development on HOOMD-blue enables support for both NVIDIA and AMD GPU's. At the time of writing,
HOOMD-blue's branch for version 3.0 development has 12,510 commits, 1,154 files and 187,382 lines of
code excluding blank lines and comments.

Soon after HOOMD-blue's first release, Joshua added an imperative Python interface for writing
simulations scripts. The first Python API was inspired by other simulation software such as LAMMPS.
This largely remained the same as HOOMD-blue released its version 2.0. However, as the package
transitions into a 3.0 release, the API has been rethought from the ground up to present a
thoroughly object oriented and Pythonic interface for users. In addition, where possible we have
sought to provide performant ways to use Python to interface with the HOOMD-blue C++ back-end.
Other Python packages like SciPy :cite:`virtanen.etal2020`, NumPy :cite:`vanderwalt.etal2011`,
scikit-learn :cite:`pedregosa.etal2011`, matplotlib :cite:`hunter2007`, and others have inspired us
in this pursuit. In this endeavour, we have found ways to make HOOMD-blue more flexible, extensible,
and integrable with the SciPy community as well.  Over the next few sections, we will use examples
of HOOMD-blue's version 3.0 API (which is still in development at the time of writing) to highlight
changes in the package's extensibility, flexibility, and Pythonic interface.

Example Script
--------------

Here we show a script that simulations a Lennard-Jones fluid using the current implementation of the
version 3.0 API. We also show a rendering of the particle configuration in Figure (:ref:`sim`).

.. code-block:: python

    import hoomd
    import hoomd.md as md
    import numpy as np

    device = hoomd.device.Auto()
    sim = hoomd.Simulation(device)

    # Create particles in simple cubic lattice
    N_per_side = 14
    N = N_per_side ** 3
    L = 20
    xs = np.linspace(0, 0.9, N_per_side)
    x, y, z = np.meshgrid(xs, xs, xs)
    coords = np.array(
        (x.ravel(), y.ravel(), z.ravel())).T

    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.configuration.box = hoomd.Box.cube(L)
    snap.particles.position[:] = (coords - 0.5) * L
    snap.particles.types = ['A']

    # Create state
    sim.create_state_from_snapshot(snap)

    # Create integrator and forces
    integrator = md.Integrator(dt=0.005)
    langevin = md.methods.Langevin(
        hoomd.filter.All(), kT=1., seed=42)

    nlist = md.nlist.Cell()
    lj = md.pair.LJ(nlist, r_cut=2.5)
    lj.params[('A', 'A')] = dict(
        sigma=1., epsilon=1.)

    integrator.methods.append(langevin)
    integrator.forces.append(lj)

    # Setup output
    gsd = hoomd.dump.GSD('dump.gsd', trigger=100)
    log = hoomd.Logger()
    log += lj
    gsd.log = log

    sim.operations.integrator = integrator
    sim.operations.analyzers.append(gsd)
    sim.run(100000)

.. figure:: figures/sim-output.png
    :align: center

    A rendering of the Lennard-Jones fluid simulation script output. Particles are colored by the
    Lennard-Jones potential energy that is logged using the HOOMD-blue :code:`Logger` and
    :code:`GSD` class objects. :label:`sim`

General API Design
------------------

Simulation, Device, State, Operations
+++++++++++++++++++++++++++++++++++++

Each simulation in HOOMD-blue now is controlled through 3 main objects which are joined together by
the :code:`Simulation` class: the :code:`Device`, :code:`State`, and :code:`Operations` classes. A
simple figure of this relationship with some core attributes/methods for each class is given in
Figure (:ref:`core-objects`). Each :code:`Simulation` object holds the requisite information to run
a full molecular dynamics (MD) or Monte Carlo (MC) simulation.  The :code:`Device` class denotes
whether a simulation should be run on CPU or GPU and the number of cores/GPUS it should run on. In
addition, the device manages custom memory tracebacks and the MPI communicator among other things.

.. figure:: figures/object-diagram.pdf
    :align: center

    Diagram of core objects with some attributes and methods. Figure made using Graphviz
    :cite:`ellson.etal2003, gansner.etal1993`. :label:`core-objects`

The :code:`State` class stores the system data (e.g. particle positions, orientations, velocities,
the system box). The :code:`State` class also exposes this data and allows setting it in two
fundamental ways. Through the snapshot API, users interface with a single object exposing many NumPy
arrays of system data. To construct a snapshot all system data stored across MPI ranks must be
gathered and combined to the root rank. To set the state using the snapshot API requires setting the
snapshot property to an entirely new snapshot. The advantages to this approach come from its ease of
use and the object holding the complete aggregation of state data. The following snippet showcases
this approach setting all particles z-axis position to zero.

.. code-block:: python

    snap = sim.state.snapshot
    # set all z positions to 0
    if snap.exists:
        snap.particles.position[:, 2] = 0
    sim.state.snapshot = snap

The other API for accessing :code:`State` data is via a zero copy MPI rank local access to the
state's data on either the GPU or CPU. On the CPU, we expose the buffers as :code:`numpy.ndarray`
like objects through provided hooks such as :code:`__array_ufunc__` and standards, i.e.
:code:`__array_interface__`. Similarly, on the GPU, we mock much of the functionality of CuPy's
:cite:`zotero-593` :code:`ndarray` class if it is installed; however, without the same degree of
hooks the integration is not as tight. Whether or not CuPy is installed though we use the
:code:`__cuda_array_interace__` protocol for GPU access. This provides support for libraries such as
numba's :cite:`lam.etal2015` GPU JIT and PyTorch :cite:`paszke.etal2019`. We chose to mock the
interfaces of both NumPy and CuPy rather than just expose :code:`ndarray` objects directly out of
consideration for memory safety. For both, to ensure data integrity we restrict the data to only be
accessible within a context-manager. Using zero copy and MPI rank local access, this approach is
much faster than using the snapshot API, but requires the user to deal with MPI and domain
decomposition directly.  The example below does the same thing as the previous but using the zero
copy access.

.. code-block:: python

    # CPU access
    with sim.state.local_snapshot as data:
        data.particles.position[:, 2] = 0

    # GPU access
    with sim.state.gpu_snapshot as data:
        data.particles.position[:, 2] = 0

The final of the three classes, :code:`Operations`, holds the different "operations" that will act
on the simulation state. Broadly these consist of 3 categories: updaters which modify simulation
state, analyzers which observe system state, and tuners which tune other operation's hyperparameters
for performance.

Deferred C++ Initialization
+++++++++++++++++++++++++++

Many objects in C++ in HOOMD-blue require either a :code:`System` or a :code:`SystemDefinition`
object (both C++ classes) in order to be correctly instantiated. The requirement is foremost due to
the interconnected nature of many things in a simulation. However, this requires a very strict order
in which objects can be created. Having to create a full simulation state to create a
:code:`PairPotential` object limits the utility and ease of Python plug-ins to HOOMD-blue. For
instance, a package that wanted to automatically generate a particular force-field in response to
some user inputs would have to have access to the :code:`State` it was to operate on. Our decision
in HOOMD-blue version 3.0 was to defer the initialization of C++ objects until they are "attached"
to a :code:`Simulation`. This has the benefit that most plug-ins to HOOMD would not need to worry
about initializing :code:`Device` or :code:`State` objects. Another motivating force for the
decision, deferred initialization provides more leniency to users creating scripts.

This deferring also has an additional benefits of more easily accesses the object's
state as well as allowing duck-typing of parameters more easily. We take advantage of the
accessibility of state by making an object's state a loggable quantity for the :code:`Logger`
object, and providing a :code:`from_state` factory method for all operations in HOOMD-blue which can
reconstruct the object from the state. This greatly increases the restartability of simulations
since the state of each object can be logged at the end of a given run, and read from at the start
of the next.

.. code-block:: python

    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere.from_state('example.gsd', frame=-1)

This code block would create a :code:`Sphere` object from the last frame of the gsd file
"example.gsd".


The Internal Base Classes
+++++++++++++++++++++++++

To facilitate adding more features to HOOMD-blue, simplify the internal class logic, and provide a
more uniform interface, we wrote the :code:`_Operation` class. This base class is inherited by most
other user facing classes. Through it we provide object dependency handling, deferred C++
initialization, and our default way of exposing and synchronizing attributes between Python and C++.

Likewise, to provide a Pythonic interface for interacting with object parameters, robust validation
on setting, and maintaining state between Python and C++ when "attached", we created two solutions:
one for parameters that are type dependent and one for those that were not.  Through the
:code:`ParameterDict` class, we ensure constancy between C++ object members and Python values while
exposing the dictionaries keys as attributes. For type dependent attributes, we use the
:code:`TypeParameter` and :code:`TypeParameterDict` classes. These type dependent quantities are
exposed through dictionary-like attributes with types as keys.

Each class support validation of their keys, and the :code:`TypeParameterDict` can be used to
define the structure and validation of arbitrarily nested structures of dictionaries, lists, and
tuples. In addition, both classes support a similar level of default specification to their
level of validation. An example object specification and initialization can be seen below.

.. code-block:: python

    TypeParameterDict(
        num=float,
        list_of_str=[str],
        nesting={len_three_vec=(float, float, float)},
        len_keys=2
        )

.. code-block:: python

    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere(seed=42)
    # example using ParameterDict
    sphere.nselect = 2
    # examples using TypeParameter and TypeParameterDict
    sphere.shape['A'] = {'diameter': 1.}
    # sets for 'B', 'C', and 'D'
    sphere.shape[['B', 'C', 'D']] = {'diameter': 0.5}

In a similar vain to store lists that must be synced to C++, the :code:`SyncedList` class allow
syncing between the C++ vectors and Python lists.

.. code-block:: python

    from hoomd import Operations
    from hoomd.dump import GSD

    ops = Operations()
    gsd = GSD('example.gsd')
    # use of SyncedList
    ops.analyzers.append(gsd)

Another improvement to user experience is the error messaging and handling for these objects. An
example error message for accidentally trying to set :code:`sigma` for 'A'-'A' interactions in the
Lennard-Jones pair potential to a string (i.e. :code:`lj.params[('A', 'A')] = {'sigma': 'foo',
'epsilon': 1.}` would provide the error message, "TypeConversionError: For types [('A', 'A')], error
In key sigma: Value foo of type <class 'str'> cannot be converted using OnlyType(float).  Raised
error: value foo not convertible into type <class 'float'>.".

Logging and Accessing Data
--------------------------

Another area that made a switch from an imperative to object oriented style can be seen in the new
HOOMD-blue version 3.0 logging system. Through extensive use of properties, we directly expose
object data such as the total potential energy in all our pair potentials, thereby encouraging users
to use object data directly rather than through a logging interface.  When logging data is desired,
we have created a Python :code:`Logger` class that creates an intermediate representation of the
logged information when called. By using an intermediate representation, we get the added
flexibility of supporting multiple official and user-created "back-ends" for logging. Furthermore,
the logging capabilities of HOOMD-blue version 3.0 are quite general, as they allow logging
scalars, strings, arrays, and even general Python objects.

Traditional simulation output such as standard out are fairly easy to implement in Python while
other back-ends like MongoDB, Pandas :cite:`mckinney2010`, and Python pickles are possible.
Consistent with this move towards providing numerous output options and thinking of HOOMD as a
Python simulation library first, HOOMD-blue version 3.0 chooses to make simulation output an opt-in
feature even for common simulation output like performance and thermodynamic quantities (e.g
temperature and pressure). In addition to this improved flexibility in storage possibilities, for
HOOMD-blue version 3.0 we have added new properties to objects to directly expose more of their data
than had previously been available. An example of this is how pairwise potentials expose the per
particle potential energy potential in the system at a given time (which is used to color Figure
(:ref:`sim`)).

Logger
++++++

The :code:`Logger` class aims to provide a simple interface for logging most HOOMD-blue objects and
custom user quantities. Through the :code:`Loggable` metaclass, all subclasses that inherit from
:code:`_Operation` expose their loggable quantities. Adding an HOOMD-blue object to a logger for
logging is as simple as :code:`logger += obj`. The utility of this class, however, lies in its
intermediate representation of the data. Using the HOOMD-blue namespace as the basis for
distinguishing between quantities, we map logged quantities into a nested dictionary. For example,
logging the Lennard-Jones pair potentials total energy would be produce this dictionary by a
:code:`Logger` object :code:`{'md': {'pair': {'LJ': {'energy': (-1.4, 'scalar')}}}}` where 
:code:`'scalar'` is a flag to make processing the logged output easier. In real use cases, the
dictionary would likely be filled with many other quantities. This intermediate form allows
developers and users to more easily create different back ends that a :code:`Logger` object can plug
into for outputting data.

User Customization
------------------

In HOOMD-blue version 3.0, we provide multiple means of "injecting" Python code into HOOMD-blue's
C++ core. We achieve this through two general means, inheriting from C++ classes through pybind11
:cite:`jakob.etal2017` and through wrapping user classes and functions in C++ classes. To guide the
choice between inheritance and composition, we looked at multiple factors: is the class simple (only
requires a few methods) and would inheritance expose internals, to name two. Regardless of the
method to add functionality to HOOMD-blue, we have prioritized adding and improving methods for
extending the package as the examples below highlight.

Triggers
++++++++

In HOOMD-blue version 2.x, everything that was not run every timestep had a period and phase
associated with it. The timesteps the operation was run on could then be determined by the
expression, :code:`timestep % period - phase == 0`.  In our refactoring and development, we
recognized that this concept could be made much more general and consequently more flexible. Objects
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
subclassing that reimplements the functionality of HOOMD-blue version 2.x can be seen in the below
-- this functionality already exists in the :code:`Periodic` class.

.. code-block:: python

    from hoomd.trigger import Trigger

    class CustomTrigger(Trigger):
        def __init__(self, period, phase=0):
            super().__init__()
            self.period = period
            self.phase = phase

        def __call__(self, timestep):
            v = timestep % self.period - self.phase == 0
            return v

While this example is quite simple, user created subclasses of :code:`Trigger` need not be as seen
in an example in a further section. They can implement arbitrarily complex Python code for more
caching, examining the simulation state, etc.

Variants
++++++++

Similar to :code:`Trigger`, we generalized our ability to linearly interpolate values
(:code:`hoomd.variant.liner_interp` in HOOMD-blue version 2.x) across timesteps to a base class
:code:`Variant` which generalizes the concept of functions in the semi-infinite domain of timesteps
:math:`t \in [0,\infty), t \in \mathbb{Z}`. This allows sinusoidal cycling, non-uniform ramps, and
various other behaviors -- as many as there are functions in the non-negative integer domain and
real range. Like :code:`Trigger`, :code:`Variant` is able to be directly subclassed from the C++
class.  :code:`Variant` objects are used in HOOMD-blue to specify quantities like temperature,
pressure, and box size for varying objects. An example of a sinusoidal cycled variant is shown
below.

.. code-block:: python

    from math import sin
    from hoomd.variant import Variant

    class SinVariant(Variant):
        def __init__(self, frequency, amplitude,
                    phase=0, center=0):
            super().__init__()
            self.frequency = frequency
            self.amplitude = amplitude
            self.phase = phase
            self.center = center

        def __call__(self, timestep):
            tmp = self.frequency * timestep
            tmp = sin(tmp + self.phase)
            return self.amplitude * tmp + self.center

        def _min(self):
            return -self.amplitude + self.center

        def _max(self):
            return self.amplitude + self.center

ParticleFilters
+++++++++++++++

Unlike :code:`Trigger` or :code:`Variant`, :code:`ParticleFitler` is not a generalization of an
existing concept but the splitting of one class into two. However, this affords us a similar
flexibility. In HOOMD-blue version 2.x, the :code:`ParticleGroup` class and subclasses served to
provide a subset of particles within a simulation for file output, application of thermodynamic
integrators, and other purposes. The class hosted both the logic for storing the subset of particles
and filtering them out from the system. After the refactoring, :code:`ParticleGroup` still
exists but just for the logic to store and preform some basic operations on particle tags (a means
of individuating particles). The new class :code:`ParticleFilter` implements the selection logic.
This choice makes :code:`ParticleFilter` objects much more lightweight and provides a means of
implementing a :code:`State` instance specific cache of :code:`ParticleFilter` objects. The latter
ensures that we do not create multiple of the same :code:`ParticleGroup` which can occupy large
amounts of memory. The caching also allows the creation of large numbers of the same
:code:`ParticleFitler` object without needing to worry about memory constraints.

.. TODO Update this section with whatever paradigm we decide to use for user customization.

Finally, this separation makes, :code:`CustomParticleFilter` which is a subclass of
:code:`ParticleFilter` with some added functionality a suitable class to subclass since its scope is
limited and does not have to deal with many of the internal details that the :code:`ParticleGroup`
class does.  For this reason, :code:`ParticleGroup` instances are private in HOOMD-blue version 3.0.
An example of a :code:`CustomParticleFilter` that selects only particle with positive charge is
given below.

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

Through composition, HOOMD-blue version 3.0 offers the ability to create custom actions (the object
within HOOMD-blue operations that performs some act with the :code:`Simulation`) in Python that run
in the standard :code:`Simulation` run loop. The feature makes user created actions behave
indistinguishably from native C++ actions. Through custom actions, users can modify state, tune
hyperparameters for performance, or just observe parts of the simulation. In addition, we are adding
a signal for Actions to send that would stop a :code:`Simulation.run` call. This would allow
actions to run until they are "done" rather than running for a large number of steps to ensure
completion or running for multiple short spurts and checking in between. With respect to
performance, with zero copy access to the data on the CPU or GPU, custom actions can also achieve
high performance using standard Python libraries like NumPy, SciPy, numba, CuPy and others.
Furthermore, this performance comes without users having to worry about manual code compilation,
ABI, or other concerns of compiled languages.

.. TODO need to add example

Larger Examples
---------------

In this section we will provide more substantial applications of features new to HOOMD-blue version 3.0.

Trigger that detects nucleation
+++++++++++++++++++++++++++++++

The first example is a :code:`Trigger` that only returns true when a threshold :math:`Q_6`
Steinhardt order parameter :cite:`steinhardt.etal1983` (as calculated by freud
:cite:`ramasubramani.etal2020`) is reached. Such a :code:`Trigger` could be used for BCC nucleation
detection which depending on the type of simulation could trigger a decrease in cooling rate, the
more frequent output of simulation trajectories, or any of numerous other possibilities. Also, in
this example we showcase the use of the zero copy local MPI rank data access . In this example, we
use ghost particles as well; ghost particles are particles that an MPI rank knows about, but is not
directly responsible for updating. They are used for force calculations and other things that
require looping over neighbors.

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

Most of the complexity in the logic comes from ensuring that we use as much data as possible and
strive for optimal performance. By using the ghost particles, more particles local to a rank will
have at least 12 neighbors. If we did not care about this, we would not need to construct
:code:`nlist` at all, and could just pass in :code:`(box, data.particle.position)` to the
:code:`compute` method. Another simplification to the :code:`Q6Trigger` class while still using all
the system data, would be to use the standard snapshot, though this would be slower (many times
slower if freud  if restricted to a single core like many packages such as NumPy are).

Pandas Logger Back-end
++++++++++++++++++++++

Here we highlight the ability to use the :code:`Logger` class to create novel back-ends
for simulation data. For this example, we will create a Pandas back-end. We will store the scalar
and string quantities in a single :code:`pandas.DataFrame` object while array-like objects will each
be stored in separate :code:`DataFrame` objects. All :code:`DataFrame` objects will be stored in a
single dictionary.

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
                lambda x: not_none(x) \
                    and not is_scalar(x)),
                hnd_2D_arrays))

            if not hasattr(self, 'data'):
                self.data = {
                    'scalar': pd.DataFrame(
                        columns=[
                            '.'.join(k) for k in sc]),
                    'array': {
                        '.'.join(k): pd.DataFrame()
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

If handling non-scalar values was not needed, then the code would become much simpler; likewise, if
using a file format such as HDF5 through h5py that has support for scalar and multi-dimensional
array quantities directly, the code is simplified further.

Conclusion
----------

HOOMD-blue version 3.0 presents a Pythonic API that encourages experimentation and customization.
Through subclassing C++ classes, providing wrappers for custom actions, and exposing data in
zero-copy arrays/buffers, we allow HOOMD-blue to utilize the full potential of Python and the
scientific Python community. Our examples have shown that often this customization is easy to
implement, and is only more verbose or difficult when the desired performance or the algorithm
itself requires.
