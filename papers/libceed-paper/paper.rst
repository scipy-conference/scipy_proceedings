:author: Valeria Barra
:email: valeria.barra@colorado.edu
:institution: University of Colorado Boulder

:author: Jed Brown
:email: jed@jedbrown.org
:institution: University of Colorado Boulder

:author: Jeremy Thompson
:email: jeremy.thompson@colorado.edu
:institution: University of Colorado Boulder

:author: Yohann Dudouit
:email: dudouit1@llnl.gov
:institution: Lawrence Livermore National Laboratory

:bibliography: bibliography

----------------------------------------------------------------------------------
High-performance operator evaluations with ease of use: libCEED's Python interface
----------------------------------------------------------------------------------

.. class:: abstract

   libCEED is a new open-source, lightweight matrix-free Finite Element library designed to leverage the next generation exascale machines by allowing a wide variety of applications to share highly optimized kernels. libCEED offers implementations, selectable at runtime, tuned for a variety of computational device types, including CPUs and GPUs. libCEED’s purely algebraic framework can unobtrusively be integrated in new and legacy software to provide performance portable applications. In this work, we present libCEED's newly available Python interface that opens up new strategies for parallelism and scaling in high-performance Python, without having to compromise ease of use.

.. class:: keywords

   High-performance Python, performance portability, scalability, parallelism, high-order finite elements

Introduction
----------------------------------------------------------------------------------

Historically, conventional high-order finite element methods were rarely used for industrial problems because the matrix describing the action of the operatror loses sparsity as the order is increased :cite:`Orszag1980`, leading to unaffordable solve times and memory requirements :cite:`brown2010`. This effect typically limited the order of accuracy to at most quadratic, especially because they are computationally advantageous in terms of floating point operations (FLOPS) per degree of freedom (DOF)---see Fig. :ref:`fig-assembledVsmatrix-free`---, despite the fast convergence and favorable stability properties offered by higher order discretizations. Nowadays, high-order numerical methods, such as the spectral element method (SEM)---a special case of nodal p-Finite Element Method (FEM) which can reuse the interpolation nodes for quadrature---are employed, especially with (nearly) affine elements, because linear constant coefficient problems can be very efficiently solved.

.. figure:: TensorVsAssembly.png
   :align: center
   :scale: 14%
   :figclass: bht

   Comparison of memory transfer and floating point operations per degree of freedom for different representations of a linear operator for a PDE in 3D with :math:`b` components and variable coefficients arising due to Newton linearization of a material nonlinearity. The representation labeled as *tensor* computes metric terms on the fly and stores a compact representation of the linearization at quadrature points. The representation labeled as *tensor-qstore* pulls the metric terms into the stored representation. The *assembled* representation uses a (block) CSR format. :label:`fig-assembledVsmatrix-free`

In Fig. :ref:`fig-assembledVsmatrix-free` we analyze and compare the theoretical costs, of different configurations: assembling the sparse matrix representing the action of the operator (labeled as *assembled*), non assembling the matrix and storing only the metric terms needed as an operator setup-phase (labeled as *tensor-qstore*) and non assembling  the matrix and computing the metric terms on the fly and storing a compact representation of the linearization at quadrature points (labeled as *tensor*). In the right panel, we show the cost in terms of FLOPS/DOF. This metric for computational efficiency made sense historically, when the performance was mostly limited by processors' clockspeed. A more relevant performance plot for current state-of-the-art high-performance machines (for which the bottleneck of performance is mostly in the memory bandwith) is shown in the right panel of Fig. :ref:`fig-assembledVsmatrix-free`, where the memory bandwith is measured in terms of bytes/DOF. We can see that high-order methods, implemented properly with only partial assembly, require optimal amount of memory transfers (with respect to the polynomial order) and near-optimal FLOPs for operator evaluation. Thus, high-order methods in matrix-free representation not only possess favorable properties, such as higher accuracy and faster convergence to solution, but also manifest an efficiency gain compared to their corresponding assembled representations.

Therefore, in recent years, high-order numerical methods have been widely used in Partial Differential Equation (PDE) solvers, but software packages that provide high-performance implementations have often been special-purpose and intrusive. In contrast, libCEED, the library of the Center for Efficient Exascale Discretizations :cite:`CEEDMS10,CEEDMS13,CEEDMS25,CEEDMS34` is light-weight, minimally intrusive, and very versatile. In fact, libCEED offers a purely algebraic interface for matrix-free operator representation and supports run-time selection of implementations tuned for a variety of computational device types, including CPUs and GPUs. libCEED's algebraic interface can unobtrusively be integrated in new and legacy software to provide performance portable interfaces. While libCEED's focus is on high-order finite elements, the approach is algebraic and thus applicable to other discretizations in factored form. libCEED's role, as a lightweight portable library that allows a wide variety of applications to share highly optimized discretization kernels, is illustrated in Fig. :ref:`fig-libCEED-backends`, where a non-exhaustive list of specialized implementations (backends) is provided. libCEED provides a low-level Application Programming Interface (API) for user codes so that applications with their own discretization infrastructure (e.g., those in PETSc :cite:`PETScUserManual`, MFEM :cite:`MFEMlibrary` and Nek5000 :cite:`Nekwebsite`) can evaluate and use the core operations provided by libCEED. GPU implementations are available via pure CUDA :cite:`CUDAwebsite` as well as the OCCA :cite:`OCCAwebsite` and MAGMA :cite:`MAGMAwebsite` libraries. CPU implementations are available via pure C and AVX intrinsics as well as the LIBXSMM :cite:`LIBXSMM` library. libCEED provides a unified interface, so that users only need to write a single source code and can select the desired specialized implementation at run time. Moreover, each process or thread can instantiate an arbitrary number of backends.

.. figure:: libCEEDBackends.png

   The role of libCEED as a lightweight, portable library which provides a low-level API for efficient, specialized implementations. libCEED allows different applications to share highly optimized discretization kernels. :label:`fig-libCEED-backends`

In this work, we first introduce libCEED’s conceptual framework and API, and then illustrate its new Python interface, developed using the C Foreign Function Interface (CFFI) for Python. CFFI allows to reuse most of the C declarations and requires only a minimal adaptation of some of them. The C and Python APIs are mapped in a nearly 1:1 correspondence. For instance, data stored in the CeedVector structure are associated to arrays defined via the NumPy :cite:`NumPy` or Numba :cite:`Numba` packages, for handling host or device memory, when interested in GPU computations with CUDA. In fact, since libCEED heavily relies on pointers and arrays to handle the data, a Python structure that resembles the C arrays is needed. In details, for CPU host memory allocations, NumPy arrays allow this correspondence obtained by passing the NumPy array memory address as pointers to the libCEED C API. Similarly, the CUDA array interface in Numba is used for creation and handling of GPU device memory data.

libCEED's API
----------------------------------------------------------------------------------

When high-order finite/spectral element discretizations are used, the resulting sparse matrix representation of a global operator is computationally expensive, with respect to both the memory transfer and floating point operations needed for its evaluation. libCEED's API provides the local action of an operator (linear or nonlinear) without assembling its sparse representation. The purely algebraic nature of libCEED allows efficient operator evaluations (selectable at runtime) and offers matrix-free preconditioning ingredients. While libCEED’s focus is on high-order finite elements, the approach with which it is designed is algebraic and thus applicable to other discretizations in factored form. This algebraic decomposition also presents the benefit that it can equally represent linear or non-linear finite element operators.

Let us define the global operator as

.. math::
   :label: eq-operator-decomposition

   A = P^T \underbrace{G^T B^T D B G}_{\text{libCEED's scope}} P \, ,

where :math:`P` is the parallel process decomposition operator (external to libCEED, which needs to be managed by the user via external packages, such as ``petsc4py`` :cite:`PETScUserManual,petsc4py`) in which the degrees of freedom (DOFs) are scattered to and gathered from the different compute devices. The operator denoted by :math:`A_L = G^T B^T D B G` gives the local action on a compute node or process, where :math:`G` is a local element restriction operation that localizes DOFs based on the elements, :math:`B` defines the action of the basis functions (or their gradients) on the nodes, and :math:`D` is the user-defined pointwise function describing the physics of the problem at the quadrature points, also called the QFunction (see Fig. :ref:`fig-operator-decomp`). Instead of forming a single operator using a sparse matrix representation, libCEED composes the different parts of the operator described in equation (:ref:`eq-operator-decomposition`) to apply the action of the operator :math:`A_L = G^T B^T D B G` in a fashion that is tuned for the different compute devices, according to the backend selcted at run time.

In libCEED's terminology, the global or total vector is called a T-vector (cf. Fig. :ref:`fig-operator-decomp`). This stores the true degrees of freedom of the problem. In a T-vector, each unknown has exactly one copy, on exactly one processor, or ``rank``. The process decomposition, denoted by :math:`P` in equation (:ref:`eq-operator-decomposition`), is a non-overlapping partitioning. The result of the application of the operator :math:`P` to a T-vector results in an L-vector, or local vector. This stores the data owned by each ``rank``. In an L-vector, each unknown has exactly one copy on each processor that owns an element containing it. This is an overlapping vector decomposition with overlaps only across different processors---there is no duplication of unknowns on a single processor. The nodes adjacent to different elements (at element corners or edges) will be the one that have more than one copy, on different processors. Applying a :math:`G` element restriction operator to an L-vector creates an E-vector. This stores the nodes grouped by the elements they belong to. In fact, in an E-vector each unknown has as many copies as the number of elements that contain it. The application of a basis operatror :math:`B` to an E-vector returns a Q-vector. This has the same layout of an E-vector, but insteed of holding the different unknown values, a Q-vector represents the values at quadrature points, grouped by element.

.. figure:: libCEED.png
   :align: center
   :figclass: bht

   Operator decomposition. :label:`fig-operator-decomp`

The mathematical formulation of QFunctions, described in weak form, is fully separated from the parallelization and meshing concerns. In fact, QFunctions, which can either be defined by the user or selected from a gallery of available built-in functions in the library, are pointwise functions that do not depend on element resolution, topology, or basis degree (selectable at run time). This easily allows :math:`hp`-refinement studies (where :math:`h` commonly denotes the average element size and :math:`p` the polynomial degree of the basis functions in 1D) and :math:`p`-multigrid solvers. libCEED also supports composition of different operators for multiphysics problems and mixed-element meshes (see Fig. :ref:`fig-schematic`). Currently, user-defined QFunctions are written in C and must be precompiled as a foreign function library and loaded via _ctypes_. The single-source C QFunctions allow users to equally compute on CPU or GPU devices, supported by libCEED. The ultimate goal is for users to write only Python code. This will be achieved in the near future by using the Numba high-performance Python compiler or Google's extensible system for composable function transformations, JAX :cite:`jax2018github`.

.. figure:: QFunctionSketch.pdf
   :align: center
   :figclass: bht

   A schematic of element restriction and basis applicator operators for elements with different topology. This sketch shows the independence of QFunctions (in this case representing a Laplacian) element resolution, topology, or basis degree. :label:`fig-schematic`

Source Code Examples
----------------------------------------------------------------------------------
LibCEED for Python is distributed through PyPI :cite:`PyPI` and can be easily installed via

.. code-block:: python

   pip install libceed

or

.. code-block:: python

   python -m pip install libceed

The package can then be simply imported via

.. code-block:: python

   import libceed

The simple declaration of a ``libceed.Ceed`` instance, with default resource (``/cpu/self``) can be obtained as

.. code-block:: python

   ceed = libceed.Ceed()

If libCEED is built with GPU support, you can specify a GPU backend, e.g., ``/gpu/occa`` or ``/gpu/cuda/gen``, with

.. code-block:: python

   ceed = libceed.Ceed('/gpu/cuda/gen')

Next, we show the creation of a ``libceed.Vector`` of a specified size

.. code-block:: python

   n = 10
   x = ceed.Vector(n)

Similarly, this could have been achieved by running

.. code-block:: python

   x = ceed.Vector(size=10)

In the following example, we associate the data stored in a ``libceed.Vector`` with a ``numpy.array`` and use it to set and read the ``libceed.Vector``'s data

.. code-block:: python

   import numpy as np
   import libceed

   ceed = libceed.Ceed()
   x = ceed.Vector(size=3)

   a = np.arange(1, 4, dtype="float64")
   x.set_array(a, cmode=libceed.USE_POINTER)

   with x.array_read() as b:
       print(b)

Similarly, we can set all entries of a ``libceed.Vector`` to the same value (e.g., 10) via

.. code-block:: python

   x.set_value(10)

If the user has installed libCEED with CUDA support and Numba, you can use device memory in your ``libceed.Vector``\s. In the following example, we create a ``libceed.Vector`` with a libCEED contex that supports CUDA, associate the data stored in a ``CeedVector`` with a ``numpy.array``, and get a Numba ``DeviceNDArray`` containing the data on the device.

.. code-block:: python

   ceed_gpu = libceed.Ceed('/gpu/cuda')

   n = 10
   x = ceed_gpu.Vector(n)

   a = np.arange(1, 1 + n, dtype="float64")
   x.set_array(a, cmode=libceed.USE_POINTER)

   with x.array_read(memtype=libceed.MEM_DEVICE) as \
     device_array:
       print(device_array)

Among the Finite Elements objects needed to compose an operator, in the following example we illustrate the creation and applycation of an element restriction, denoted by :math:`G` in equation (:ref:`eq-operator-decomposition`)

.. code-block:: python

   ne = 3

   x = ceed.Vector(ne+1)
   a = np.arange(10, 10 + ne+1, dtype="float64")
   x.set_array(a, cmode=libceed.USE_POINTER)

   ind = np.zeros(2*ne, dtype="int32")
   for i in range(ne):
     ind[2*i+0] = i
     ind[2*i+1] = i+1

   r = ceed.ElemRestriction(ne, 2, 1, 1, ne+1, ind, \
       cmode=libceed.USE_POINTER)

   y = ceed.Vector(2*ne)
   y.set_value(0)

   r.apply(x, y)

An :math:`H^1` Lagrange basis in :math:`d` dimensions can be defined with the following code snippet

.. code-block:: python

   d = 1
   b = ceed.BasisTensorH1Lagrange(
       dim=d,   # topological dimension
       ncomp=1, # number of components
       P=4,     # number of basis functions (nodes)
                # per dimension
       Q=4,     # number of quadrature points
                # per dimension
       qmode=libceed.GAUSS_LOBATTO)

In the following example, we show how to apply a 2D basis opearor, denoted by :math:`B` in equation (:ref:`eq-operator-decomposition`), from an E-vector named ``Ev``, to a Q-vector named ``Qv``, and viceversa, its transpose opeartor :math:`B^T`

.. code-block:: python

   b.apply(1, libceed.EVAL_INTERP, Ev, Qv)
   b.T.apply(1, libceed.EVAL_INTERP, Qv, Ev)

In the following example, we create two QFunctions (for the setup and apply, respectively, of the mass operator in 1D) from the gallery of available built-in QFunctions in libCEED

.. code-block:: python

   qf_setup = ceed.QFunctionByName("Mass1DBuild")
   qf_mass = ceed.QFunctionByName("MassApply")

The setup QFunction, named ``qf_setup`` in the previous example, is the one that defines the formulation of the geometric factors given by the correspondence between deformed finite element coordinates and reference ones. The apply QFunction, named ``qf_mass`` in the previous example, is the one that defines the action of the physics (the spatial discretization of the weak form of the PDE) the user wants to solve for (in this simple example, this is represented by the mass matrix).

Finally, once all ingredients for a ``libceed.Operator`` are defined (i.e., element restriction, basis, and QFunction), one can create and apply an local operator as

.. code-block:: python

   # Define Setup operator
   op_setup = ceed.Operator(qf_setup)
   op_setup.set_field("dx", rx, bx,
                      libceed.VECTOR_ACTIVE)
   op_setup.set_field("weights",
                      libceed.ELEMRESTRICTION_NONE, bx,
                      libceed.VECTOR_NONE)
   op_setup.set_field("qdata", rui,
                      libceed.BASIS_COLLOCATED,
                      libceed.VECTOR_ACTIVE)

   # Define Mass operator
   op_mass = ceed.Operator(qf_mass)
   op_mass.set_field("u", ru, bu,
                     libceed.VECTOR_ACTIVE)
   op_mass.set_field("qdata", rui,
                     libceed.BASIS_COLLOCATED, qdata)
   op_mass.set_field("v", ru, bu,
                     libceed.VECTOR_ACTIVE)

   # Apply Setup operator
   op_setup.apply(x, qdata)

   # Apply Mass operator
   op_mass.apply(u, v)

For all of the illustrated classes of objects (``libceed.Ceed``, ``libceed.Vector``, ``libceed.ElemRestriction``, ``libceed.Basis``, ``libceed.QFunction``, and ``libceed.Operator``), libCEED's Python interface provides a representation method so that they can be viewed/printed by simply typing

.. code-block:: python

   print(x)

These and other examples can be found in the suite of Project Jupyter :cite:`ProjectJupyter` tutorials provided with libCEED in a dedicated Binder :cite:`libCEEDBinder` interactive environment.

Conclusions
----------------------------------------------------------------------------------

We have presented libCEED, a new open-source, lightweight matrix-free Finite Element library, its conceptual framework, and new Python interface. libCEED’s purely algebraic framework can unobtrusively be integrated in new and legacy software to provide performance portable applications. In this work, we have demonstrated the usage of libCEED's Python interface by providing examples of the creation and application of the main classes in libCEED's API: ``libceed.Ceed``, ``libceed.Vector``, ``libceed.ElemRestriction``, ``libceed.Basis``, ``libceed.QFunction``, and ``libceed.Operator``. We have showed how libCEED can be used to open up new strategies for parallelism and scaling in high-performance Python, without having to compromise ease of use.

Acknowledgments
----------------------------------------------------------------------------------

The libCEED library is distributed under a BSD 2-Clause License with Copyright (c) 2017 of the Lawrence Livermore National Security, LLC. The work presented in this paper is supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of two U. S. Department of Energy Organizations (the Office of Science and the National Nuclear Security Administration) responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering and early testbed platforms, in support of the nation’s exascale computing imperative.

References
----------------------------------------------------------------------------------
