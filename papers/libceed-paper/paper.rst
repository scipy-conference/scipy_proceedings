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

Historically, conventional high-order finite element methods were rarely used for industrial problems because the matrix describing the action of the operatror loses sparsity as the order is increased :cite:`Orszag1980`, leading to unaffordable solve times and memory requirements :cite:`brown2010`. This effect typically limited the order of accuracy to at most quadratic, especially because they are computationally advantageous in terms of floating point operations (FLOPS) per degree of freedom (DOF)---see :ref:`fig-assembledVsmatrixfree` ---, despite the fast convergence and favorable stability properties offered by higher order discretizations. Nowadays, high-order numerical methods, such as the spectral element method (SEM)---a special case of nodal p-Finite Element Method (FEM) which can reuse the interpolation nodes for quadrature---are employed, especially with (nearly) affine elements, because linear constant coefficient problems can be very efficiently solved.

.. figure:: TensorVsAssembly.png
   :align: center
   :figclass: bht

   Comparison of memory transfer and floating point operations per degree of freedom for different representations of a linear operator for a PDE in 3D with :math:`b` components and variable coefficients arising due to Newton linearization of a material nonlinearity. The representation labeled as *tensor* computes metric terms on the fly and stores a compact representation of the linearization at quadrature points. The representation labeled as *tensor-qstore* pulls the metric terms into the stored representation. The *assembled* representation uses a (block) CSR format.

In :ref:`fig-assembledVsmatrix-free` we analyze and compare the theoretical costs, of different configurations: assembling the sparse matrix representing the action of the operator (labeled as *assembled*), non assembling the matrix and storing only the metric terms needed as an operator setup-phase (labeled as *tensor-qstore*) and non assembling  the matrix and computing the metric terms on the fly and storing a compact representation of the linearization at quadrature points (labeled as *tensor*). In the right panel, we show the cost in terms of FLOPS/DOF. This metric for computational efficiency made sense historically, when the performance was mostly limited by processors' clockspeed. A more relevant performance plot for current state-of-the-art high-performance machines (for which the bottleneck of performance is mostly in the memory bandwith) is shown in the right panel of :ref:`fig-assembledVsmatrix-free`, where the memory bandwith is measured in terms of bytes/DOF. We can see that high-order methods, implemented properly with only partial assembly, require optimal amount of memory transfers (with respect to the polynomial order) and near-optimal FLOPs for operator evaluation. Thus, high-order methods in matrix-free representation not only possess favorable properties, such as higher accuracy and faster convergence to solution, but also manifest an efficiency gain compared to their corresponding assembled representations.

Therefore, in recent years, high-order numerical methods have been widely used in Partial Differential Equation (PDE) solvers, but software packages that provide high-performance implementations have often been special-purpose and intrusive. In contrast, libCEED is light-weight, minimally intrusive, and very versatile. In fact, libCEED offers a purely algebraic interface for matrix-free operator representation and supports run-time selection of implementations tuned for a variety of computational device types, including CPUs and GPUs. libCEED's algebraic interface can unobtrusively be integrated in new and legacy software to provide performance portable interfaces. While libCEED's focus is on high-order finite elements, the approach is algebraic and thus applicable to other discretizations in factored form. libCEED's role, as a lightweight portable library that allows a wide variety of applications to share highly optimized discretization kernels, is illustrated in :ref:`fig-libCEED-backends`, where a non-exhaustive list of specialized implementations (backends) is provided. libCEED provides a low-level Application Programming Interface (API) for user codes so that applications with their own discretization infrastructure (e.g., those in `PETSc <https://www.mcs.anl.gov/petsc/>`_, `MFEM <https://mfem.org/>`_ and `Nek5000 <https://nek5000.mcs.anl.gov/>`_) can evaluate and use the core operations provided by libCEED. GPU implementations are available via pure `CUDA <https://developer.nvidia.com/about-cuda>`_ as well as the `OCCA <http://github.com/libocca/occa>`_ and `MAGMA <https://bitbucket.org/icl/magma>`_ libraries. CPU implementations are available via pure C and AVX intrinsics as well as the `LIBXSMM <http://github.com/hfp/libxsmm>`_ library. libCEED provides a unified interface, so that users only need to write a single source code and can select the desired specialized implementation at run time. Moreover, each process or thread can instantiate an arbitrary number of backends.

.. _fig-libCEED-backends:

.. figure:: libCEEDBackends.png

   The role of libCEED as a lightweight, portable library which provides a low-level API for efficient, specialized implementations. libCEED allows different applications to share highly optimized discretization kernels.

In this work, we first introduce libCEED’s conceptual framework and C interface, and then illustrate its new Python interface, developed using the C Foreign Function Interface (CFFI) for Python. CFFI allows to reuse most of the C declarations and requires only a minimal adaptation of some of them. The C and Python APIs are mapped in a nearly 1:1 correspondence. For instance, data stored in the CeedVector structure are associated to arrays defined via the NumPy or Numba packages, for handling host or device memory, when interested in GPU computations with CUDA. In fact, since libCEED heavily relies on pointers and arrays to handle the data, a Python structure that resembles the C arrays is needed. In details, for CPU host memory allocations, NumPy arrays allow this correspondence obtained by passing the NumPy array memory address as pointers to the libCEED C API. Similarly, the CUDA array interface in Numba is used for creation and handling of GPU device memory data.

libCEED's API
----------------------------------------------------------------------------------

When high-order finite/spectral element discretizations are used, the resulting sparse matrix representation of a global operator is computationally expensive, with respect to both the memory transfer and floating point operations needed for its evaluation. libCEED's API provides the local action of an operator (linear or nonlinear) without assembling its sparse representation. The purely algebraic nature of libCEED allows efficient operator evaluations (selectable at runtime) and offers matrix-free preconditioning ingredients. While libCEED’s focus is on high-order finite elements, the approach with which it is designed is algebraic and thus applicable to other discretizations in factored form. This algebraic decomposition also presents the benefit that it can equally represent linear or non-linear finite element operators.

Let us define the global operator as

.. math::
   :label: eq-operator-decomposition

   A = P^T \underbrace{G^T B^T D B G}_{\text{libCEED's scope}} P \, ,

where :math:`P` is the parallel process decomposition operator (external to libCEED, which needs to be managed by the user via external packages, such as ``petsc4py`` :cite:`PETScUserManual`, :cite:`petsc4py`) in which the degrees of freedom (DOFs) are scattered to and gathered from the different compute devices. The operator denoted by :math:`A_L = G^T B^T D B G` gives the local action on a compute node or process, where :math:`G` is a local element restriction operation that localizes DOFs based on the elements, :math:`B` defines the action of the basis functions (or their gradients) on the nodes, and :math:`D` is the user-defined pointwise function describing the physics of the problem at the quadrature points, also called the QFunction (see Fig. :ref:`fig-operator-decomp`). Instead of forming a single operator using a sparse matrix representation, libCEED composes the different parts of the operator described in equation (:ref:`eq-operator-decomposition`) to apply the action of the operator :math:`A_L = G^T B^T D B G` in a fashion that is tuned for the different compute devices, according to the backend selcted at run time.

.. figure:: libCEED.png
   :align: center
   :figclass: bht

   Operator decomposition.

The mathematical formulation of QFunctions, described in weak form, is fully separated from the parallelization and meshing concerns. In fact, QFunctions, which can either be defined by the user or selected from a gallery of available built-in functions in the library, are pointwise functions that do not depend on element resolution, topology, or basis degree (selectable at run time). This easily allows :math:`hp`-refinement studies (where :math:`h` commonly denotes the average element size and :math:`p` the polynomial degree of the basis functions in 1D) and :math:`p`-multigrid solvers. libCEED also supports composition of different operators for multiphysics problems and mixed-element meshes (see Fig. :ref:`fig-schematic`). Currently, user-defined QFunctions are written in C and must be precompiled as a foreign function library and loaded via _ctypes_. The single-source C QFunctions allow users to equally compute on CPU or GPU devices, supported by libCEED. The ultimate goal is for users to write only Python code. This will be achieved in the near future by using the Numba high-performance Python compiler or Google's extensible system for composable function transformations, JAX.

.. figure:: QFunctionSketch.pdf
   :align: center
   :figclass: bht

   A schematic of element restriction and basis applicator operators for elements with different topology. This sketch shows the independence of QFunctions (in this case representing a Laplacian) element resolution, topology, or basis degree.

References
----------------------------------------------------------------------------------
