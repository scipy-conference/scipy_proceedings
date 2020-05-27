:author: Jim Pivarski
:email: pivarski@princeton.edu
:institution: Princeton University

:author: Ianna Osborne
:email: Ianna.Osborne@cern.ch
:institution: Princeton University

:author: Pratyush Das
:email: reikdas@gmail.com
:institution: Institute of Engineering and Management

:author: Anish Biswas
:email: anishbiswas271@gmail.com
:institution: Manipal Institute of Technology

:author: Peter Elmer
:email: Peter.Elmer@cern.ch
:institution: Princeton University

------------------------------------------------
Awkward Array: JSON-like data, NumPy-like idioms
------------------------------------------------

.. class:: abstract

    NumPy simplifies and accelerates mathematical calculations in Python, but only for rectilinear arrays of numbers. Awkward Array provides a similar interface for JSON-like data: slicing, masking, broadcasting, and performing vectorized math on the attributes of objects, unequal-length nested lists (i.e. ragged/jagged arrays), and heterogeneous data types.

    Awkward Arrays are columnar data structures, like (and convertible to/from) Apache Arrow, with a focus on manipulation, rather than serialization/transport. These arrays can be passed between C++ and Python, and they can be used in functions that are JIT-compiled by Numba.

    Development of a GPU backend is in progress, which would allow data analyses written in array-programming style to run on GPUs without modification.

.. class:: keywords

   NumPy, Numba, Pandas, C++, Apache Arrow, Columnar data, AOS-to-SOA, Ragged array, Jagged array, JSON

Introduction
------------

NumPy [1]_ is a powerful tool for data processing, at the center of a large ecosystem of scientific software. Its built-in functions are general enough for many scientific domains, particularly those that analyze time series, images, or voxel grids. However, it is difficult to apply NumPy to tasks that require data structures beyond N-dimensional arrays of numbers.

More general data structures can be expressed as JSON and processed in pure Python, but at an expense of performance and often conciseness. NumPy is faster and more memory efficient than pure Python because its routines are precompiled and its arrays of numbers are packed contiguously in memory. Some expressions can be more concise in NumPy's "vectorized" notation, which describe actions to perform on whole arrays, rather than scalar values.

In this paper, we will describe Awkward Array [2]_ [3]_, a generalization of NumPy's core functions to the nested records, variable-length lists, missing values, and heterogeneity of JSON-like data. In doing so, we'll focus on the internal representation of data structures as columnar arrays, very similar to (and compatible with) Apache Arrow [4]_. The key feature of the Awkward Array library, however, is that calculations, including those that restructure data, are performed on the columnar arrays themselves, rather than instantiating objects. Since the logical data structure is implicit in the arrangement of columnar arrays and most operations only rearrange integer indexes, these functions can be precompiled for specialized data types, like NumPy.

Motivation from particle physics
--------------------------------

Awkward Array was created to make it easier to analyze particle physics data using scientific Python tools. Particle physics datasets are big and intrinsically structured: trillions of particle collisions result in thousands of particles each, grouped by type and clustered as jets. The operations to be performed depend on this structure.

The most basic operation reconstructs particles that decayed before they could be detected, using kinematic constraints on their visible decay products. For example, if a neutral kaon (:math:`K_S`) decays into two charged pions (:math:`\pi^+`, :math:`\pi^-`), the energy (:math:`E`) and momenta (:math:`\vec{p}`) of the observed pions can be combined to reconstruct the unobserved kaon's mass (:math:`m`):

.. math::

   m_{K_S} = \sqrt{(E_{\pi^+} + E_{\pi^-})^2 - \left|\vec{p}_{\pi^+} + \vec{p}_{\pi^-}\right|^2}

Since kaons have a well-defined mass, the :math:`m_{K_S}` values that correspond to real kaons form a peak in a histogram: see Figure :ref:`physics-example`. Not knowing which pions are due to kaon decays, every pair of pions in a collision event must be searched, in principle.

.. figure:: figures/physics-example.pdf
   :align: center
   :scale: 13%
   
   Example of a particle physics problem requiring combinatorial search: all pairs of pions in a collision event must be tested for compatibility with decay from a kaon. :label:`physics-example`

Physicists employ many other techniques, but most of them involve combinatorial searches like this one. Since the numbers of particles of each type differ from event to event and they carry dozens of attributes that must be referenced in the combinatorial pairs, this would be a hard analysis to do in NumPy.

With Awkward Array, however, it would be

.. code-block:: python

   kaon_masses = ak.combinations(pions[good], 2).mass

where :code:`ak.combinations` is a built-in function, :code:`pions[good]` preselects good pion objects in all events using an array of variable-length lists of booleans, and :code:`.mass` is a user-defined property that computes :math:`m_{K_S}` using NumPy universal functions.

Demonstration with GeoJSON bike routes
--------------------------------------

However, nested data structures are not unique to particle physics, so we present a more complete example using GeoJSON map data. Suppose we want to analyze the following Chicago bike routes [5]_, a dataset with two nested levels of latitude, longitude polylines, string-valued street names, and metadata as a JSON file.

.. code-block:: python

    import urllib.request
    import json

    url = "https://raw.githubusercontent.com/Chicago/" \
          "osd-bike-routes/master/data/Bikeroutes.geojson"
    bikeroutes_json = urllib.request.urlopen(url).read()
    bikeroutes_pyobj = json.loads(bikeroutes_json)

Importing this JSON object as an Awkward Array splits its record-oriented structure into a contiguous buffer for each field, making it ready for columnar operations. Heterogeneous data are split by type, such that each buffer in memory has a single numerical type.

.. code-block:: python

    import awkward1 as ak
    bikeroutes = ak.Record(bikeroutes_pyobj)

Longitude and latitude are in the first two components of fields named :code:`"coordinates"` of fields named :code:`"geometry"` of fields named :code:`"features"`. They can be accessed with NumPy-like slices, including ellipsis, :code:`np.newaxis`, masks, etc.

.. code-block:: python
    
    longitude = bikeroutes["features", "geometry",
                           "coordinates", ..., 0]
    latitude  = bikeroutes["features", "geometry",
                           "coordinates", ..., 1]

The :code:`longitude` and :code:`latitude` arrays both have type :code:`1061 * var * var * float64` (expressed as a Datashape): 1061 routes with a variable number of variable-length polylines.

To compute lengths of each route, we can use NumPy universal functions (like :code:`np.sqrt`) and reducers (like :code:`np.sum`), which are overridden by Awkward-aware functions using NumPy's NEP-13 [6]_ and NEP-18 [7]_ protocols. Distances between points can be computed with :code:`a[:, :, 1:] - a[:, :, :-1]` even though each inner list :code:`a[:, :]` may have a different length.

.. code-block:: python

    km_east = (longitude - np.mean(longitude)) * 82.7
    km_north = (latitude - np.mean(latitude)) * 111.1

    segment_length = np.sqrt(
        (km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
        (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

    route_length = np.sum(segment_length, axis=-1)
    total_length = np.sum(route_length, axis=-1)

The same could be performed with the following, though the vectorized form is shorter and 8 times faster; see Figure :ref:`bikeroutes-scaling`.

.. code-block:: python

    total_length = []
    for route in bikeroutes_pyobj["features"]:
        route_length = []
        for polyline in route["geometry"]["coordinates"]:
            segment_length = []
            last = None
            for lng, lat in polyline:
                km_east = lng * 82.7
                km_north = lat * 111.1
                if last is not None:
                    dx2 = (km_east - last[0])**2
                    dy2 = (km_north - last[1])**2
                    segment_length.append(
                        np.sqrt(dx2 + dy2))
                last = (km_east, km_north)

            route_length.append(sum(segment_length))
        total_length.append(sum(route_length))

.. figure:: figures/bikeroutes-scaling.pdf
   :align: center
   :scale: 45%

   Scaling of Awkward Arrays and pure Python loops for the bike routes calculation shown in the text. :label:`bikeroutes-scaling`

Scope: data types and common operations
---------------------------------------

Awkward Array supports the same suite of abstract data types and features as "typed JSON" serialization formats—Apache Arrow, Parquet, Protobuf, Thrift, Avro, etc.

Namely, there are

* primitive types: numbers and booleans,
* variable-length lists,
* regular-length lists as a distinct type (i.e. tensors),
* records/structs/objects (named, typed fields),
* fixed-width tuples (unnamed, typed fields),
* missing/nullable data,
* mixed, yet specified, types (i.e. union/sum types),
* virtual arrays (functions generate arrays on demand),
* partitioned arrays (for off-core and parallel analysis).

Like Apache Arrow and Parquet, arrays with these features are laid out as columns in memory (more on that below).

Like NumPy, the Awkward Array library contains a primary Python class, :code:`ak.Array`, and a collection of generic operations. Most of these operations change the structure of the data in the array, since NumPy, SciPy [8]_, and others already provide numerical math as universal functions (ufuncs). In each case where an Awkward function generalizes a NumPy function, it is provided with the same interface (corresponds exactly for rectilinear grids).

Awkward functions include

* basic and advanced slices (:code:`__getitem__`) including variable-length and missing data as advanced slices,
* masking, an alternative to slices that maintains length but introduces missing values instead of dropping elements,
* broadcasting of universal functions into structures,
* reducers of and across variable-length lists,
* zip/unzip/projecting free arrays into and out of records,
* flattening and padding to make rectilinear data,
* Cartesian products (cross join) and combinations (self join) at :code:`axis >= 1` (per element of one or more arrays).

Conversions to other formats, such as Arrow, access in third-party libraries, such as Numba [9]_ and Pandas [10]_, methods of building data structures, and customizing high-level behavior are also in the library's scope.

Columnar representation, columnar implementation
------------------------------------------------

Like Arrow, Awkward data structures are not localized in memory. Instead of concentrating all data for one array element in nearby memory (as an "array of structs"), all data for a given field are contiguous, and all data for another field are elsewhere contiguous (as a "struct of arrays"). This favors a pattern of data access in which only a few fields are needed at a time.

Additionally, Awkward operations are performed on columnar data without returning to the record-oriented format. To illustrate, consider a list of variable-length lists, such as longitude points along a bike route,

.. code-block:: python

    [[1.1, 2.2, 3.3], [4.4], [5.5, 6.6], [7.7, 8.8, 9.9]]

Instead of creating objects to represent the four lists, we flatten the :code:`content` and introduce :code:`starts` and :code:`stops` buffers to indicate where each sublist starts and stops.

.. code-block:: python

    starts:  0, 3, 4, 6
    stops:   3, 4, 6, 9
    content: 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9

A list of lists of lists would use these three buffers as the :code:`content` of another node with its own :code:`starts` and :code:`stops`. In general, a hierarchy of columnar array nodes mirrors the hierarchy of the nested data, except that the number of such nodes scales with the complexity of the data type, not the number of elements in the array. Particle physics use-cases require thousands of nodes to describe complex collision events, but billions of events in memory at once. Figure :ref:`example-hierarchy` shows a small example.

.. figure:: figures/example-hierarchy.pdf
   :align: center
   :scale: 60%
   :figclass: w

   Hierarchy for an example data structure: an array of lists of records, in which field :code:`"x"` of the records are numbers and field :code:`"y"` of the records are lists of numbers. This might, for example, represent :code:`[[], [{"x": 1, "y": [1]}, {"x": 2, "y": [2, 2]}]]`, but it also might represent an array with billions of elements (of the same type). The number of nodes scales with complexity, not data volume. :label:`example-hierarchy`

To compute distances in each bike route, we needed to compute :code:`a[:, 1:] - a[:, :-1]`. For :code:`a[:, 1:]`, we only have to add :code:`1` to the :code:`starts`:

.. code-block:: python

    starts:  1, 4, 5, 7
    stops:   3, 4, 6, 9
    content: 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9

This new array represents

.. code-block:: python

    [[     2.2, 3.3], [   ], [     6.6], [     8.8, 9.9]]

and we could reuse the original :code:`content` to construct it. Since :code:`content` is untouched, the slice can be a precompiled routine that treats the :code:`content` as an opaque pointer. The :code:`content` might contain other lists or records, like the example in Figure :ref:`example-hierarchy`. Similarly, :code:`a[:, :-1]` is computed by subtracting :code:`1` from the original :code:`stops`, and it is up to the :code:`-` operation to align the :code:`content` of its arguments before applying :code:`np.subtract`.

Awkward 1.x
-----------

The first widely used version of Awkward Array (0.x) was released in September 2018 as a pure Python module, in which all of the columnar operations were implemented using NumPy. This library was successful, but limited, since some data transformations are difficult or impossible to write without explicit loops.

In August 2019, we began a half-year project to rewrite the library in C++ (1.x), isolating all of the array manipulations in a "CPU kernels" library that can be swapped for "GPU kernels." Apart from the implementation of the "GPU kernels," this project is complete, though users are still transitioning from the original "Awkward 0.x" to the new "Awkward 1.x," which are both available as separate libraries in PyPI.

Figure :ref:`awkward-1-0-layers` shows how Awkward 1.x is organized:

* the high-level interface is in Python,
* the array nodes (managing node hierarchy and ownership/lifetime) are in C++, accessed through pybind11,
* an alternate implementation of array navigation was written for Python functions that are compiled by Numba,
* array manipulation algorithms (without memory management) are independently implemented as "CPU kernels" and "GPU kernels" plugins. The kernels' interface is pure C, allowing for reuse in other languages.

.. figure:: figures/awkward-1-0-layers.pdf
   :align: center
   :scale: 45%

   Components of Awkward Array, as described in the text. :label:`awkward-1-0-layers`

Most array operations are shallow, affecting only one or a few levels of the hierarchy, but even in the worst case, an operation initiated by a Python call steps over at most all of the nodes of an array, which is no more than thousands in particle physics use-cases. The number of elements in the array can be billions (multi-GB memory). The loops over array elements are strictly contained in the kernels, so performance optimizations can focus on this layer.

The C++ layer is therefore not motivated by performance, since that is the responsibility of the kernels. It exists because many particle physics libraries are written in C++ and having a full object model for the arrays in C++ allows us to pass data to and from C++ libraries as Awkward Arrays, avoiding unnecessary conversions. The C++ implementation freely uses dynamic dispatch (virtual methods) and atomic reference counting (shared pointers) to match Python's object model.

Numba for just-in-time compilation
----------------------------------

Some expressions are simpler in "vectorized" form, such as :code:`pions[good]` to select :code:`pions` with a broadcastable array of booleans :code:`good`. However, some operations are more difficult to express in this form, particularly iterative algorithms.

A common case in particle physics is following each particle of a decay tree to a particular type of ancestor, such as a quark. These trees are often expressed as a collection of :code:`particles`, which are records with a field named :code:`parent`—the index of its immediate ancestor.

We can find immediate ancestors in a vectorized expression,

.. code-block:: python

    immediate_ancestors = particles[particles.parent]

but this step must be repeated a different number of times for different elements. The same is true of numerical algorithms that must iterate until they converge.

Iteration is easy to express in imperative Python code:

.. code-block:: python

    def find_quark(particle):
        while not is_quark(particle):
            particle = particles[particle.parent]
        return particle

Doing so, however, gives up on the performance advantage of using arrays. Iteration over Awkward Arrays is even slower than built-in Python objects. Ideally, we want to iterate over the arrays in compiled code, code that involves domain-specific logic and therefore must be written by the user. Users could write their functions in C++, accessing Awkward Array's C++ layer the way a third-party library might, but that would be an unreasonable amount of effort for common analysis tasks.

Instead, we recommend using Numba, a just-in-time compiler for Python. All array nodes except :code:`UnionArray` have been implemented as Numba models, so Awkward Arrays can be used as arguments and return values from compiled Python functions. The function above can be compiled by simply adding a decorator,

.. code-block:: python

    import numba as nb

    @nb.njit
    def find_quark(particle):
        while not is_quark(particle):
            particle = particles[particle.parent]
        return particle

assuming that :code:`is_quark` is similarly defined,

.. code-block:: python

    @nb.njit
    def is_quark(particle):
        return abs(particle.pdg_id) <= 6

Such an implementation would still suffer from poor performance because :code:`find_quark` takes a single particle as input, incurring overhead for each particle in the dataset. Users should write functions that take and return whole datasets, performing the loop inside the compiled block. For this example, we could do that by returning an integer index to use as a slice:

.. code-block:: python

    @nb.njit
    def find_quarks(particles):
        index = np.empty(len(particles), np.int64)
        for i in range(len(particles)):
            index[i] = i
            while not is_quark(particles[index[i]]):
                index[i] = particles[index[i]].parent
        return index

    particles[find_quarks(particles)]

This is fast, but possibly non-intuitive. For more natural user code, we introduced an ArrayBuilder, which is an append-only structure that becomes an Awkward Array when a "snapshot" is taken.

.. code-block:: python

    @nb.njit
    def find_quarks(particles, builder):
        for particle in particles:
            while not is_quark(particle):
                particle = particles[particle.parent]
            builder.append(particle)
        return builder

    find_quarks(particles, ak.ArrayBuilder()).snapshot()

The ArrayBuilder is described in more detail in the next section.

Whereas the C++ implementation uses (relatively) slow runtime objects because the number of nodes touched by a vectorized operation scales with the complexity of the type, not the number elements in the array, a user function written in Numba would walk over the same nodes for each element of the array, and therefore must be more thoroughly optimized.

Each node type is implemented as a Numba type but not as runtime objects. The only runtime object is a lookup table of buffer pointers, and the node types generate specialized code to walk over the lookup table. Since this Numba model is so different from the C++ classes, Awkward's full suite of vectorized functions are not available in the compiled block. However, the following features are supported for imperative programming:

* iteration and :code:`__len__` for arrays,
* simple :code:`__getitem__`: integers indexes, slices, and strings for record fields that are compile-time constants,
* attribute :code:`__getattr__` as an alternative to string-slices.

ArrayBuilder: creating columnar data in-place
---------------------------------------------

Awkward Arrays are immutable; NumPy's ability to assign elements in place is not supported or generalized by the Awkward Array library. (As an exception, users can assign fields to records using :code:`__setitem__` syntax, but this *replaces* the inner tree with one having the new field.) Restricting Awkward Arrays to read-only access allows whole subtrees of nodes to be shared among different versions of an array.

To create new arrays, we introduced ArrayBuilder, an append-only object that accumulates data and cteates :code:`ak.Arrays` by taking a "snapshot" of the current state. New data are attached at various levels of depth through method calls, which also dynamically refines the type of the provisional array.

.. code-block:: python

                       # type of b.snapshot()
    b                  # 0 * unknown
    b.begin_record()   # 0 * {}
    b.field("x")       # 0 * {"x": unknown}
    b.integer(1)       # 0 * {"x": int64}
    b.end_record()     # 1 * {"x": int64}
    b.begin_record()   # 1 * {"x": int64}
    b.field("x")       # 1 * {"x": int64}
    b.real(2.2)        # 1 * {"x": float64}
    b.field("y")       # 1 * {"x": float64, "y": ?unknown}
    b.integer(2)       # 1 * {"x": float64, "y": ?int64}
    b.end_record()     # 2 * {"x": float64, "y": ?int64}
    b.null()           # 3 * ?{"x": float64, "y": ?int64}
    b.string("hello")  # 4 * ?union[{"x": float64,
                       #             "y": ?int64}, string]

In the above example, an initially empty ArrayBuilder :code:`b` has unknown type and zero length. With :code:`begin_record`, its type becomes a record with no fields. Calling :code:`field` adds a field of unknown type, and following that with :code:`integer` sets the field type to an integer. The length of the array is only increased when the record is closed by :code:`end_record`.

In the next record, field :code:`"x"` is filled with a floating point number, which retroactively updates previous integers to floats. Calling :code:`b.field("y")` introduces a field :code:`"y"` to all records, though it has option type because this field is missing for all previous records. The third record is missing (:code:`b.null()`), which refines its type as optional, and in place of a fourth record, we append a string, so the type becomes a union.

Internally, ArrayBuilder maintains a similar hierarchy of nodes as an array, except that all buffers can grow (when the preallocated space is used up, the buffer is reallocated and copied into a buffer 1.5× larger), and :code:`content` nodes can be replaced from specialized types to more general types. Taking a snapshot *shares* buffers with the new array, so it is a lightweight operation.

ArrayBuilder's :code:`append` method dispatches to the other methods based on argument type, and if the argument is an array or record, it includes a preexisting subtree in its accumulated data. This is how the Numba example (previous section) appends :code:`particles`.

Although ArrayBuilder is compiled code and calls into it are specialized by Numba, its dynamic typing has a runtime cost: filling NumPy arrays is faster. ArrayBuilder trades runtime performance for convenience; faster array-building methods would have to be specialized by type.

High-level behaviors
--------------------

One of the most popular features of Awkward 0.x was the ability to create subclasses of array nodes (in Python) that add domain-specific methods to records in an array. This includes "object" methods:

.. code-block:: python

    # distance between pion 5 and pion 6 in event 1000
    events[1000].pions[5].distance(events[1000].pions[6])

and "vectorized" methods:

.. code-block:: python

    # distances between pion 5 and 6 in all events
    events.pions[5].distance(events.pions[6])

This capability has been ported to Awkward 1.x and expanded upon. In Awkward 1.x, records can be named (as part of more general "properties" metadata in C++) and record names are linked to Python classes through an :code:`ak.behavior` dict.

.. code-block:: python

    class Point:
        def magnitude(self):
            return np.sqrt(self.x**2 + self.y**2)

    class PointRecord(Point, ak.Record):
        pass
    class PointArray(Point, ak.Array):
        pass

    ak.behavior["point"] = PointRecord
    ak.behavior["*", "point"] = PointArray

    array = ak.Array([{"x": 1.1, "y": 1},
                      {"x": 2.2, "y": 2},
                      {"x": 3.3, "y": 3}],
                     with_name="point")

    array[2].magnitude()
    # 4.459820624195552

    array.magnitude()
    # <Array [1.49, 2.97, 4.46] type='3 * float64'>

When an operation on array nodes completes and the result is wrapped in a high-level :code:`ak.Array` or :code:`ak.Record` class for the user, the :code:`ak.behavior` is checked for signatures that link records and arrays of records to user-defined subclasses. Only the name :code:`"point"` is stored with the data; methods are all added at runtime, which allows schemas to evolve.

Other kinds of behaviors can be assigned through different signatures in the :code:`ak.behavior` dict, such as overriding ufuncs,

.. code-block:: python

    # link np.absolute("point") to a custom function
    ak.behavior[np.absolute, "point"] = Point.magnitude

    np.absolute(array)
    # <Array [1.49, 2.97, 4.46] type='3 * float64'>

custom broadcasting rules, and Numba extensions (typing and lowering functions).

As a special case, strings are not defined as an array type, but as a parameter label on variable-length lists. Behaviors that present these lists as strings (overriding :code:`__repr__`) and define per-string equality (overriding :code:`np.equal`) are preloaded in the default :code:`ak.behavior`.

Pandas and other third-party libraries
--------------------------------------

Awkward Arrays are Pandas extensions, so they can be used as a :code:`Series` and :code:`DataFrame` column type. NumPy ufuncs on the resulting Pandas objects are correctly passed through to the Awkward Arrays (including behavioral overrides), though many special cases are missing for general use.

Rather than directly embedding complex data structures in Pandas, however, it is often more useful to translate Awkward structures into Pandas structures. Variable-length lists translate naturally into MultiIndex rows:

.. code-block:: python

    ak.pandas.df(ak.Array([[[1.1, 2.2], [], [3.3]],
                           [],
                           [[4.4], [5.5, 6.6]],
                           [[7.7]],
                           [[8.8]]]))
    #                             values
    # entry subentry subsubentry
    # 0     0        0               1.1
    #                1               2.2
    #       2        0               3.3
    # 2     0        0               4.4
    #       1        0               5.5
    #                1               6.6
    # 3     0        0               7.7
    # 4     0        0               8.8

and nested records translate into MultiIndex column names:

.. code-block:: python

    ak.pandas.df(ak.Array([{"I":
                              {"a": _, "b": {"c": _}},
                            "II":
                              {"x": {"y": {"z": _}}}}
                           for _ in range(0, 50, 10)]))
    #         I      II
    #         a   b   x
    #             c   y
    #                 z
    # entry
    # 0       0   0   0
    # 1      10  10  10
    # 2      20  20  20
    # 3      30  30  30
    # 4      40  40  40

If an array contains records with fields of different nested list lengths, however, a single DataFrame cannot losslessly encode the information, though several DataFrames related by a key can.

Other third-party libraries, such as NumExpr, are similarly wrapped to generalize their treatment of NumPy arrays to Awkward Arrays.

GPU backend
-----------

One of the advantages of a vectorized user interface is that it is already optimal for calculations on a GPU. Imperative loops need to be redesigned when porting algorithms to GPUs, but CuPy, Torch, TensorFlow, and JAX demonstrate that array-at-a-time functions can hide the distinction between CPU calculations and GPU calculations.

To allow for a future GPU backend, all instances of reading or writing to an array's buffers were restricted to the "array manipulation" layer of the project (see Figure :ref:`awkward-1-0-layers`). The first implementation of this layer, "CPU kernels," performs all operations that actually access the array buffers, and it is compiled into a physically separate file: :code:`libawkward-cpu-kernels.so`, as opposed to the main :code:`libawkward.so`, Python extension module, and Python code.

In May 2020, we began developing the "GPU kernels" library, provisionally named :code:`libawkward-cuda-kernels.so` (to allow for future non-CUDA versions). Since the main codebase (:code:`libawkward.so`) never dereferences any pointers to its buffers, main memory pointers can be transparently swapped for GPU pointers with additional metadata to identify which kernel to call for a given set of pointers. Thus, the main library does not need to be recompiled to support GPUs and it can manage arrays in main memory and on GPUs in the same process, which could be important, given the limited size of GPU memory. The "GPU kernels" may be deployed as a separate package in PyPI and Conda so that users can choose to install it separately as an "extras" package.

The kernels library contains many functions (428 with an "extern C" interface, 124 independent implementations, as of May 2020) because it defines all array manipulations. All of these must be ported to CUDA for the first GPU implementation. Fortunately, the majority are easy to translate: Figure :ref:`kernels-survey` shows that almost 70% are simple, embarrassingly parallel loops, 25% use a counting index that could be implemented with a parallel prefix sum, and the remainder have loop-carried dependencies or worse (one uses dynamic memory, but there may be alternatives). The kernels were written in a simple style that may be sufficiently analyzable for machine-translation, a prospect we are currently investigating with pycparser.

.. figure:: figures/kernels-survey.pdf
   :align: center
   :scale: 45%

   CPU kernels by algorithmic complexity, as of February 2020. :label:`kernels-survey`

Transition from Awkward 0.x
---------------------------

Awkward Array is one of the most widely used Python packages for particle physics. In part, this is because it is a dependency of Uproot, which interprets ROOT files as arrays in the scientific Python ecosystem. (ROOT is the most widely used software package for particle physics; more than an exabyte of data are stored in ROOT files.) However, Awkward Arrays are increasingly being used apart from Uproot and ROOT, in scientific workflows that communicate via Arrow, Parquet, or HDF5. Figure :ref:`awkward-0-popularity` shows the adoption rate in a common metric with other major particle physics Python packages.

.. figure:: figures/awkward-0-popularity.pdf
   :align: center
   :scale: 58%

   Adoption of Awkward 0.x, measured by PyPI statistics, compared to other popular particle physics packages (root-numpy, iminuit, rootpy) and popular data science packages. :label:`awkward-0-popularity`

Motivated by the success of Awkward 0.x, but learning from its limitations, Awkward 1.x was developed separately from the original library and it includes much needed interface changes as well as refactoring. To avoid disrupting ongoing physics analysis, is still deployed with :code:`awkward1` as its PyPI and Python module name. Although the new library is ready for data analysis, the ecosystem is not: at the time of writing, Uproot is still being updated to present ROOT data as Awkward 1.x arrays. Therefore, :code:`awkward` and :code:`awkward1` will coexist until a point later this year, when PyPI, Conda, Python, and GitHub names will simultaneously transfer :code:`awkward` to :code:`awkward0` and :code:`awkward1` to :code:`awkward`.

As an incentive, the Awkward 1.x project has been heavily documented, with complete docstring and doxygen coverage.

Summary
-------

By providing NumPy-like idioms on JSON-like data, Awkward Array fills a need required by the particle physics community. The inclusion of data structures in array types and operations was an enabling factor in this community's adoption of other scientific Python tools. However, the Awkward Array library itself is not domain-specific and is open to use in other domains.

Acknowledgements
----------------

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Reference
---------

.. [1] Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. *The NumPy Array: A Structure for Efficient Numerical Computation*,
       Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

.. [2] Jim Pivarski, Jaydeep Nandi, David Lange, Peter Elmer. *Columnar data processing for HEP analysis*,
       Proceedings of the 23rd International Conference on Computing in High Energy and Nuclear Physics (CHEP 2018). DOI:10.1051/epjconf/201921406026

.. [3] Jim Pivarski, Peter Elmer, David Lange. *Awkward Arrays in Python, C++, and Numba*,
       CHEP 2019 proceedings, EPJ Web of Conferences (CHEP 2019). arxiv:2001.06307

.. [4] Apache Software Foundation. *Arrow: a cross-language development platform for in-memory data*,
       https://arrow.apache.org

.. [5] City of Chicago Data Portal,
       https://data.cityofchicago.org

.. [6] Pauli Virtanen, Nathaniel Smith, Marten van Kerkwijk, Stephan Hoyer. *NEP 13 — A Mechanism for Overriding Ufuncs*,
       https://numpy.org/neps/nep-0013-ufunc-overrides.html

.. [7] Stephan Hoyer, Matthew Rocklin, Marten van Kerkwijk, Hameer Abbasi, Eric Wieser. *NEP 18 — A dispatch mechanism for NumPy’s high level array functions*,
       https://numpy.org/neps/nep-0018-array-function-protocol.html

.. [8] Pauli Virtanen et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*,
       SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, in press. DOI:10.1038/s41592-019-0686-2

.. [9] Siu Kwan Lam, Antoine Pitrou, Stanley Seibert. *Numba: a LLVM-based Python JIT compiler*,
       LLVM '15: Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC, 7, 1-6 (2015), DOI:10.1145/2833157.2833162


.. [10] Wes McKinney. *Data Structures for Statistical Computing in Python*,
        Proceedings of the 9th Python in Science Conference, 51-56 (2010).
