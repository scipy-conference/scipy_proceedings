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

   libCEED is a new open-source, lightweight library designed to leverage the next generation exascale machines by allowing a wide variety of applications to share highly optimized kernels. libCEED offers implementations, selectable at runtime, tuned for a variety of computational device types, including CPUs and GPUs. libCEED’s purely algebraic framework can unobtrusively be integrated in new and legacy software to provide performance portable applications. In this work, we present libCEED's newly available Python interface that opens up new strategies for parallelism and scaling in high-performance Python, without having to compromise ease of use.

.. class:: keywords

   High-performance Python, performance portability, scalability, parallelism

Introduction
----------------------------------------------------------------------------------

libCEED is a new open-source, lightweight library designed to leverage the next generation exascale machines by allowing a wide variety of applications to share highly optimized kernels. Historically, conventional high-order finite element methods were rarely used for industrial problems because the Jacobian rapidly loses sparsity as the order is increased, leading to unaffordable solve times and memory requirements :cite:`brown2010`. This effect typically limited the order of accuracy to at most quadratic, especially because they are computationally advantageous in terms of floating point operations (FLOPS) per degree of freedom (DOF)---see :ref:`fig-assembledVsmatrixfree` ---, despite the fast convergence and favorable
stability properties offered by higher order discretizations. Nowadays, high-order numerical methods are widely used in Partial Differential Equation (PDE) solvers, but software packages that provide high-performance implementations have often been special-purpose and intrusive. In contrast, libCEED is light-weight, minimally intrusive, and very versatile. When high-order finite/spectral element discretizations are used, the resulting sparse matrix representation of a global operator is computationally expensive, with respect to both the memory transfer and floating point operations needed for its evaluation. libCEED's Application Programming Interface (API) provides the local action of an operator (linear or nonlinear) without assembling its sparse representation. The purely algebraic nature of libCEED allows efficient operator evaluations (selectable at runtime) and offers matrix-free preconditioning ingredients. While libCEED’s focus is on high-order finite elements, the approach with which it is designed is algebraic and thus applicable to other discretizations in factored form.

.. figure:: TensorVsAssembly.png
   :align: center
   :figclass: bht

   Comparison of memory transfer and floating point operations per degree of freedom for different representations of a linear operator for a PDE in 3D with :math:`b` components and variable coefficients arising due to Newton linearization of a material nonlinearity. The representation labeled as *tensor* computes metric terms on the fly and stores a compact representation of the linearization at quadrature points. The representation labeled as *tensor-qstore* pulls the metric terms into the stored representation. The *assembled* representation uses a (block) CSR format.

In this work, we first introduce libCEED’s conceptual framework and C interface, and then illustrate its new Python interface, developed using the C Foreign Function Interface (CFFI) for Python. CFFI allows to reuse most of the C declarations and requires only a minimal adaptation of some of them. The C and Python APIs are mapped in a nearly 1:1 correspondence. For instance, data stored in the CeedVector structure are associated to arrays defined via the numpy package. In fact, since libCEED heavily relies on pointers and arrays to handle the data, a Python structure that resembles the C arrays is needed. In details, numpy arrays allow this correspondence obtained by passing the numpy array memory address as pointers to the libCEED C API.

libCEED's API
----------------------------------------------------------------------------------

LibCEED's API provides the local action of the linear or nonlinear operator without assembling its sparse representation. Let us define the global operator as

.. math::
   :label: eq-operator-decomposition

   A = P^T \underbrace{G^T B^T D B G}_{\text{libCEED's scope}} P \, ,

where :math:`P` is the parallel process decomposition operator (external to ``libCEED``, which needs to be managed by the user via external packages, such as ``petsc4py`` :cite:`PETScUserManual`, :cite:`petsc4py`) in which the degrees of freedom (DOFs) are scattered to and gathered from the different compute devices. The operator denoted by :math:`A_L = G^T B^T D B G` gives the local action on a compute node or process, where :math:`G` is a local element restriction operation that localizes DOFs based on the elements, :math:`B` defines the action of the basis functions (or their gradients) on the nodes, and :math:`D` is the user-defined pointwise function describing the physics of the problem at the quadrature points, also called the QFunction (see Fig. :ref:`fig-operator-decomp`).

.. figure:: libCEED.png
   :align: center
   :figclass: bht

   Operator decomposition.

The mathematical formulation of QFunctions, described in weak form, is fully separated from the parallelization and meshing concerns. In fact, QFunctions, which can either be defined by the user or selected from a gallery of available built-in functions in the library, are pointwise functions that do not depend on element resolution, topology, or basis degree (selectable at run time). This easily allows :math:`hp`-refinement studies (where :math:`h` commonly denotes the average element size and :math:`p` the polynomial degree of the basis functions in 1D) and :math:`p`-multigrid solvers. libCEED also supports composition of different operators for multiphysics problems and mixed-element meshes (see Fig. :ref:`fig-schematic`). Currently, user-defined QFunctions are written in C and must be precompiled before running the Python script. The ultimate goal is for users to write only Python code. This will be achieved in the near future by using the Numba high-performance Python compiler.

.. figure:: QFunctionSketch.pdf
   :align: center
   :figclass: bht

   A schematic of element restriction and basis applicator operators for elements with different topology. This sketch shows the independence of QFunctions (in this case representing a Laplacian) element resolution, topology, or basis degree.

References
----------------------------------------------------------------------------------
