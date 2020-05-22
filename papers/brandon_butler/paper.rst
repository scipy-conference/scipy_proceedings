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

-----------------------------------------------------------------------------------------------------
HOOMD-blue version 3.0  A Modern, Extensible, Flexible, Object-Oriented API for Molecular Simulations
-----------------------------------------------------------------------------------------------------

.. class:: abstract

    HOOMD-blue is a library for running molecular dynamics and hard particle Monte Carlo simulations
    that uses pybind11 to provide a Python interface to fast C++ internals.  The package is designed
    to scale from a single CPU core to thousands of NVIDIA or AMD GPUs. In developing HOOMD-blue
    version 3.0, we significantly improve the application protocol interface (API) by making it more
    flexible, extensible, and Pythonic.  We have also striven to provide simpier and more performant
    entry points to the internal C++ classes and data structures. With these updates, we show how
    HOOMD-blue users will be able to write completely custom Python classes which integrate directly
    into the simulation run loop directly in Python and analyze previously inaccessible data.
    Throughout this paper, we focus on how these goals have been achieved and explain design
    decisions through examples of the newly developed API.
.. class:: keywords

    molecular dynamics, molecular simulations, Monte Carlo simulations, object-oriented

Introduction
------------

Molecular simulation has been an important technique for studying the equilibrium properties of
molecular systems since the 1950s. The two most common methods for this purpose are molecular
dynamics and Monte Carlo simulations :cite:`metropolis.etal1953, alder.wainwright1959`. Molecular
dynamics (MD) is the application of Newton's laws of motion to molecular system, while Monte Carlo
simulations explore a system's degrees of freedom through a Markov chain, to find equilibrium
quantities. Since their inception these tools have been used to study numerous systems, examples
include colloids :cite:`damasceno.etal2012`, metallic glasses :cite:`fan.etal2014`, and proteins
:cite:`dignon.etal2018a`, among others.

Today many software packages exist for this purpose: LAMMPS :cite:`plimpton1993`, GROMACS
:cite:`berendsen.etal1995, abraham.etal2015`, OpenMM :cite:`eastman.etal2017`, and Amber
:cite:`salomon-ferrer.etal2013, case.etal2005` to name a few. Implementations on high performance
GPUs :cite:`spellings.etal2017`, parallel architectures :cite:`niethammer.etal2014`, as well as
greater accessibility of computational power has improved tremendously the length
:cite:`byna.etal2012` and time :cite:`shaw.etal2009` scales of simulations from those conducted in
the mid 1900's. Due to the flexibility and generality of such tools, their usage has dramatically
increased the usage of molecular simulations, further increasing the demand for highly flexible and
customizable software packages that can be tailored to very specific simulation requirements.
Different tools have taken different approaches to enabling this, such as the text-file scripting in
LAMMPS, the command line interface provided by GROMACS, and the Python, C++, C, and Fortran bindings
of OpenMM. The desire to allow for the unparalleled flexibility of working within a fully-featured
programming language environment, HOOMD-blue :cite:`anderson.etal2008, glaser.etal2015,
anderson.etal2020` offers a full-featured Python interface. HOOMD-blue version 3.0 aims to further
improve this interface by providing simpler, more Pythonic ways to write simulations and providing
seamless interfaces for other tools in the Python ecosystem, providing a more usable and flexible
API than ever before.

HOOMD-blue is a Python package with a C++ for MD and MC simulations. It was first released in 2008
by with full support for NVIDIA GPUs using CUDA.  In 2014, we added support for MPI parallelization
using domain decomposition (separating a simulation box into local boxes, one for each rank). Recent
development on HOOMD-blue enables support for both NVIDIA and AMD GPUs. At the time of writing,
HOOMD-blue's branch for version 3.0 development has 12,510 commits, 1,154 files and 187,382 lines of
code excluding blank lines and comments.

The second release of HOOMD-blue added an imperative Python interface for writing simulations
scripts. The structure and available commands in the original Python API are largely
inspired by and reminiscent of the structure of other simulation software such as LAMMPS.  This
largely remained the same in HOOMD-blue version 2.0. However, as the package transitions into a 3.0
release, we have redesigned the API from the ground up to present a thoroughly object oriented and
Pythonic interface for users. Where possible we have sought to provide performant ways
to use Python to interface with the HOOMD-blue C++ back end.  Other Python packages like SciPy
:cite:`virtanen.etal2020`, NumPy :cite:`vanderwalt.etal2011`, scikit-learn
:cite:`pedregosa.etal2011`, matplotlib :cite:`hunter2007`, and others have inspired us through their
intuitive API's and motivated us increasing the ability to integrate such packages with HOOMD-blue.
In this endeavour, we have found ways to make HOOMD-blue more flexible, extensible, and integrable
with the scientific Python community as well.  Over the next few sections, we will use examples of
HOOMD-blue's version 3.0 API (which is still in development at the time of writing) to highlight
changes in the package's extensibility, flexibility, and Pythonic interface.

Example Script
--------------

Here we show a script that simulations a Lennard-Jones fluid using the current implementation of the
version 3.0 API. We also show a rendering of the particle configuration in Figure (:ref:`sim`).

.. code-block:: python

    import hoomd
    import hoomd.md
    import numpy as np

    device = hoomd.device.Auto()
    sim = hoomd.Simulation(device)

    # Place particles on simple cubic lattice
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
    integrator = hoomd.md.Integrator(dt=0.005)
    langevin = hoomd.md.methods.Langevin(
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
    :code:`GSD` class objects. Figure is rendered in OVITO :cite:`stukowski2009a` using the Tachyon
    :cite:`stone1998` render. :label:`sim`

General API Design
------------------

Simulation, Device, State, Operations
+++++++++++++++++++++++++++++++++++++

Each simulation in HOOMD-blue now is controlled through 3 main objects which are joined together by
the :code:`Simulation` class: the :code:`Device`, :code:`State`, and :code:`Operations` classes. A
simple figure of this relationship with some core attributes/methods for each class is given in
Figure (:ref:`core-objects`). Each :code:`Simulation` object holds the requisite information to run
a full molecular dynamics or Monte Carlo simulation.  The :code:`Device` class denotes
whether a simulation should be run on CPU or GPU and the number of cores/GPUS it should run on. In
addition, the device manages custom memory tracebacks, profiler configurations, and the MPI
communicator among other things.

.. figure:: figures/object-diagram.pdf
    :align: center

    Diagram of core objects with some attributes and methods. Classes are in bold and orange;
    attributes and methods are blue. Figure is made using Graphviz :cite:`ellson.etal2003,
    gansner.etal1993`. :label:`core-objects`

The :code:`State` class stores the system data (e.g. particle positions, orientations, velocities,
the system box). The :code:`State` class also exposes this data and allows setting it in two ways.
Through the snapshot API, users interface with a single object exposing NumPy arrays that store a
copy of the system data. To construct a snapshot all system data distributed across MPI ranks must
be gathered and combined to the root rank. To set the state using the snapshot API requires setting
the snapshot property to an new snapshot (i.e. all system data is reset upon setting). The
advantages to this approach come from its ease of use of working with a single object containing the
complete description of the state. The following snippet showcases hows this approach can be used
to set the z position of all particles to zero.

.. code-block:: python

    snap = sim.state.snapshot
    # snapshot only stores data on rank 0
    if snap.exists:
        # set all z positions to 0
        snap.particles.position[:, 2] = 0
    sim.state.snapshot = snap

The other API for accessing :code:`State` data is via a zero-copy, rank-local access to the
state's data on either the GPU or CPU. On the CPU, we expose the buffers as
:code:`numpy.ndarray`-like objects through provided hooks such as :code:`__array_ufunc__` and
:code:`__array_interface__`. Similarly, on the GPU, we mock much of the CuPy's
:cite:`zotero-593` :code:`ndarray` class if it is installed; however, at present the CuPy
package provides fewer hooks, so our integration is more limited.  Whether or not CuPy is installed
though we use the :code:`__cuda_array_interace__` protocol for GPU access. This provides support for
libraries such as numba's :cite:`lam.etal2015` GPU JIT and PyTorch :cite:`paszke.etal2019`. We chose
to mock the interfaces of both NumPy and CuPy rather than just expose :code:`ndarray` objects
directly out of consideration for memory safety. To ensure data integrity, we restrict the data to
only be accessible within a specific context manager. Using HOOMD-blue's data buffers directly, this
approach is much faster than using the snapshot API, but requires the user to deal the domain
decomposition directly. The example below modifies the previous example to instead use the zero-copy
API.

.. code-block:: python

    with sim.state.cpu_local_snapshot as data:
        data.particles.position[:, 2] = 0

    # assumes CuPy is installed
    with sim.state.gpu_local_snapshot as data:
        data.particles.position[:, 2] = 0

The final of the three classes, :code:`Operations`, holds the different *operations* that will act
on the simulation state. Broadly these consist of 3 categories: updaters which modify simulation
state, analyzers which observe system state, and tuners which tune other operation's hyperparameters
for performance.

For the rest of this section we will discuss our decision to make HOOMD-blue version 3.0
*operations* defer the initialization of their C++ objects. This will provide context to the
following section describing some base classes behind many of HOOMD-blue's classes and behavior.

Deferred C++ Initialization
+++++++++++++++++++++++++++

Most objects in C++ in HOOMD-blue require a :code:`Device` and the C++ implementation of a
:code:`State` object on creation. However, this establishes a strict order in which objects can be
created. Having to create a full simulation state just to create, for instance, a pair potential,
limits the composability of the Python interface and makes it harder to write modular simulation
protocols. For instance, if a package that wanted to generate a particular force-field
in response to some user inputs, it would require a previously instantiated :code:`Device` access to
the :code:`State` it was to operate on. This means that this functionality could only be invoked
after the user had already instantiated a specific simulation state. Moreover, this requirement
makes it more difficult for users to write simulation scripts, because it requires them be aware of
the order in which objects must be created. To circumvent these difficulties, the new API has moved
to a deferred initialization model in which C++ objects are not created until the corresponding
Python objects are *attached* to a :code:`Simulation`.

In addition to ameliorating the difficulties mentioned above, deferred initialization allows us to
more easily export an object's state (not to be confused with the simulation state) since it is
store in pure Python as well as enabling duck-typing of parameters. We make an object's state a
loggable quantity in HOOMD-blue's logging system, and provide a :code:`from_state` factory method
for all operations in HOOMD-blue which can reconstruct the object from the state.  This greatly
increases the restartability of simulations since the state of each object can be saved at the end
of a given run, and read at the start of the next.

.. code-block:: python

    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere.from_state('example.gsd', frame=-1)

This code block would create a :code:`Sphere` object with the parameters stored from the last frame
of the gsd file :code:`example.gsd`.


The Internal Base Classes
+++++++++++++++++++++++++

The :code:`_Operation` class facilitates adding more features to HOOMD-blue, simplifies the internal
class logic, and provides a more uniform interface. This base class is inherited by most other user
facing classes. Through it (and its metaclass :code:`Loggable`) we provide deferred C++
initialization, expose entry points for logging (explained in more detail in the Logging and
Accessing Data section), enable manage dependencies between objects in cases where they may be added
or removed from a :code:`Operations` object, and our default way of exposing and synchronizing
attributes between Python and C++.

We implemented two solution that provide a Pythonic interface for interacting with object
parameters, robust validation on setting, and maintaining state between Python and C++ when
attached: one for parameters that are type dependent and one for those that were not.  Through the
:code:`ParameterDict` class, we synchronize C++ and Python for standard attributes. For type
dependent attributes, we created the :code:`TypeParameter` and :code:`TypeParameterDict` classes.
These type dependent quantities are exposed through dictionary-like attributes with types as keys.

Each class supports validation of their keys, and they can be used to define the structure and
validation of arbitrarily nested structures of dictionaries, lists, and tuples. In addition,
the :code:`TypeParameterDict` class supports default specification, :code:`ParameterDict` has
defaults but these are equivalent to object attribute defaults. An example object specification and
initialization can be seen below.

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

To store lists that must be synchronized to C++, the analogous :code:`SyncedList` class
transparently handles synchronization of Python lists and C++ vectors.

.. code-block:: python

    from hoomd import Operations
    from hoomd.dump import GSD

    ops = Operations()
    gsd = GSD('example.gsd')
    # use of SyncedList
    ops.analyzers.append(gsd)

We also improve the user experience by improving the error messaging and handling through these
objects. An example error message for trying to set :code:`sigma` for *A-A* interactions in the
Lennard-Jones pair potential to a string (i.e. :code:`lj.params[('A', 'A')] = {'sigma': 'foo',
'epsilon': 1.}` would provide the error message,

.. code-block:: python

    TypeConversionError: For types [("A", "A")], error
    In key sigma: Value foo of type <class 'str'> cannot
    be converted using OnlyType(float).  Raised error:
    value foo not convertible into type <class 'float'>.
    
Previously, the equivalent error would be :code:`TypeError: must be real number, not str`, the error
would not be raised until running the simulation, and the line setting sigma would not be in the
stack trace given.

Logging and Accessing Data
--------------------------

Logging simulation data is critical for molecular simulation software packages.  Such data is
required post processing and analysis.  We use our Pythonic object oriented design to provide a
flexible and extensible logging system.  Through extensive use of properties, we directly expose
object data such as the total potential energy in all our pair potentials, the trial move acceptance
rate in MC integrators, and thermodynamic variables like temperature that users can use directly or
store through a logging interface.  The logging is quite general, and supports logging scalars,
strings, arrays, and even general Python objects.  By separating the data collection from the
writing to files, and by providing such a flexible intermediate representation, HOOMD can now
support a range of back ends for logging; moreover, it offers users the flexibility to define their
own.  For instance, logging data to text files or standard out is supported out of the box, but
other back ends like MongoDB, Pandas :cite:`mckinney2010`, and Python pickles would also be feasible
to implement.  Consistent with the new approach to logging, HOOMD-blue version 3.0 makes simulation
output an opt-in feature even for common simulation output like performance and thermodynamic
quantities (e.g temperature and pressure).  In addition to this improved flexibility in storage
possibilities, for HOOMD-blue version 3.0 we have added new properties to objects to directly expose
more of their data than had previously been available. For example, pair potentials now expose
*per-particle* potential energies at any given time (this data is used to color Figure
(:ref:`sim`)).

Logger
++++++

The :code:`Logger` class provides an interface for logging most HOOMD-blue objects and
custom user quantities. Through the :code:`Loggable` metaclass, all subclasses that inherit from
:code:`_Operation` expose their loggable quantities. Adding all loggable quantities of an object to
a logger for logging is as simple as :code:`logger += obj`. The utility of this class lies in its
intermediate representation of the data. Using the HOOMD-blue namespace as the basis for
distinguishing between quantities, the :code:`Logger` maps logged quantities into a nested
dictionary. For example, logging the Lennard-Jones pair potentials total energy would be produce
this dictionary by a :code:`Logger` object :code:`{'md': {'pair': {'LJ': {'energy': (-1.4,
'scalar')}}}}` where :code:`'scalar'` is a flag to make processing the logged output easier. In real
use cases, the dictionary would likely be filled with many other quantities. This intermediate form
allows developers and users to create different back ends for outputting data.

User Customization
------------------

We have added multiple means of injecting Python code into HOOMD-blue's C++ core simulation loop.
We achieve this through two general means, inheriting from C++ classes through pybind11
:cite:`jakob.etal2017` and through wrapping user classes and functions in C++ classes. To guide the
choice between inheritance and composition, we looked at multiple factors such as is the class
simple (only requires a few methods) and would inheritance expose internal data structures subject
to change.  We have prioritized adding and improving methods for extending the package
as the examples below highlight.

Triggers
++++++++

In HOOMD-blue version 2.x, everything that was not run every timestep had a period and phase
associated with it. The timesteps the operation was run on could then be determined by the
expression, :code:`timestep % period - phase == 0`.  In our refactoring and development, we
recognized that this concept could be made much more general and consequently more flexible. Objects
do not have to be run on a periodic timescale; they just need some indication of when to run. In
other words, the operations needed to be *triggered*. The :code:`Trigger` class encapsulates this
concept  providing a uniform way of specifying when an object should run without limiting options.
Each operation that requires triggering is now associated with a corresponding :code:`Trigger`
instance. The previous behavior is encapsulated in a single :code:`Periodic` class. However, this
approach enables much more triggering logic through composition of multiple triggers such as
:code:`Before` and :code:`After` which return :code:`True` before or after a given timestep with the
:code:`And`, :code:`Or`, and :code:`Not` subclasses whose function can be understood by recognizing
that a :code:`Trigger` is essentially a functor that returns a Boolean value.

In addition, to the flexibility the :code:`Trigger` class provides by abstracting out the concept of
triggering an operation, we can provide through pybind11 a way to subclass :code:`Trigger` in
Python. This allows users to create their own triggers in pure Python. An example of such
subclassing that reimplements the functionality of HOOMD-blue version 2.x can be seen in the below.

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

User created subclasses of :code:`Trigger` are not restricted to simple algorithms; they can
implement arbitrarily complex Python code as demonstrated in the Large Examples first code snippet
section. 

Variants
++++++++

:code:`Variant` objects are used in HOOMD-blue to specify quantities like temperature, pressure, and
box size for varying objects. Similar to :code:`Trigger`, we generalized our ability to linearly
interpolate values (:code:`hoomd.variant.liner_interp` in HOOMD-blue version 2.x)
across timesteps to a base class :code:`Variant` which generalizes the concept of functions in the
semi-infinite domain of timesteps :math:`t \in [0,\infty), t \in \mathbb{Z}`. This allows sinusoidal
cycling, non-uniform ramps, and other behaviors. Like :code:`Trigger`, :code:`Variant` is able to be
directly subclassed from the C++ class.   An example of a sinusoidal cycled variant is shown below.

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
existing concept but the splitting of one class into two. However, this split is also targeted at
increasing flexibility and extensibility. In HOOMD-blue version 2.x, the :code:`ParticleGroup` class
and subclasses served to provide a subset of particles within a simulation for file output,
application of thermodynamic integrators, and other purposes. The class hosted both the logic for
storing the subset of particles and filtering them out from the system. After the refactoring,
:code:`ParticleGroup` is only responsible for the logic to store and preform some basic operations
on particle tags (a means of identifying individual particles), while new class :code:`ParticleFilter`
implements the selection logic.  This choice makes :code:`ParticleFilter` objects lightweight and
provides a means of implementing a :code:`State` instance specific cache of :code:`ParticleFilter`
objects. The latter ensures that we do not create multiple of the same :code:`ParticleGroup` which
can occupy large amounts of memory.  The caching also allows the creation of many of the same
:code:`ParticleFitler` object without needing to worry about memory constraints.

.. TODO Update this section with whatever paradigm we decide to use for user customization.

:code:`ParticleFitler` can be subclassed (like :code:`Trigger` and :code:`Variant`), but only
through the :code:`CustomParticleFilter` class. This is necessary to prevent some internal details
from leaking to the user.  An example of a :code:`CustomParticleFilter` that selects only particle
with positive charge is given below.

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

Custom Actions
+++++++++++++++++

In HOOMD-blue, we distinguish between the object that performs an action on the simulation state
called *Actions* and their containing objects that deal with setting state and the user interface
*Operations*.  Through composition, HOOMD-blue offers the ability to create custom actions in Python
and wrap them in our :code:`_CustomOperation` subclasses (divide on the type of action performed)
allowing the execution of the action in the :code:`Simulation` run loop.  The feature makes user
created actions behave indistinguishably from native C++ actions. Through custom actions, users can
modify state, tune hyperparameters for performance, or observe parts of the simulation. In addition,
we are adding a signal for Actions to send that would stop a :code:`Simulation.run` call. This would
allow actions to stop the simulation when they complete.  With respect to performance, with zero
copy access to the data on the CPU or GPU, custom actions can also achieve high performance using
standard Python libraries like NumPy, SciPy, numba, CuPy and others.

.. TODO need to add example

Larger Examples
---------------

In this section we will provide more substantial applications of features new to HOOMD-blue.

Trigger that detects nucleation
+++++++++++++++++++++++++++++++

This example demonstrates a :code:`Trigger` that returns true when a threshold :math:`Q_6`
Steinhardt order parameter :cite:`steinhardt.etal1983` (as calculated by freud
:cite:`ramasubramani.etal2020`) is reached. Such a :code:`Trigger` could be used for BCC nucleation
detection which could trigger a decrease in cooling rate, the more frequent output of simulation
trajectories, or any other desired action. Also, in this example we showcase the use of the
zero-copy rank-local data access . This example also requires the use of ghost particles, which are
a subset of particles bordering a MPI rank's local box. Ghost particles are known by a rank, but the
rank is not responsible for updating them. In this case, those particles are required for computing
the :math:`Q_6` value for particles near the edges of the current rank's local simulation box.


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

Here we highlight the ability to use the :code:`Logger` class to create a Pandas back end
for simulation data. It will store the scalar and string quantities in a single
:code:`pandas.DataFrame` object while array-like objects are stored each in a separate
:code:`DataFrame` object. All :code:`DataFrame` objects are stored in a single dictionary.

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

Conclusion
----------

HOOMD-blue version 3.0 presents a Pythonic API that encourages experimentation and customization.
Through subclassing C++ classes, providing wrappers for custom actions, and exposing data in
zero-copy arrays/buffers, we allow HOOMD-blue users to utilize the full potential of Python and the
scientific Python community.
