:author: Michele Martone
:email: michele.martone@lrz.de
:institution: Leibniz Supercomputing Centre (LRZ), Garching near Munich, Germany

:author: Simone Bacchio
:email: s.bacchio@cyi.ac.cy
:institution: CaSToRC, The Cyprus Institute, Nicosia, Cyprus

-------------------------------------------------------------------
PyRSB: Portable Performance on Multithreaded Sparse BLAS Operations
-------------------------------------------------------------------

.. class:: abstract

  This article introduces **PyRSB**, a Python interface to the **LIBRSB** library.
  LIBRSB is a portable *performance library* offering so called *Sparse BLAS* operations for modern multicore CPUs.
  It is based on the *Recursive Sparse Blocks* (**RSB**) format, which is particularly well suited for matrices of large dimensions.
  PyRSB allows LIBRSB usage with an interface styled after that of **SciPy**'s *sparse matrix* classes, and offers the extra benefit of exploiting multicore parallelism.
  This article introduces the concepts behind the RSB format and LIBRSB, and illustrates usage of PyRSB.
  It concludes with a user-oriented overview of speedup advantage of ``rsb_matrix`` over ``scipy.sparse.csr_matrix`` running general sparse matrix-matrix multiplication on a modern shared-memory computer.

.. class:: keywords
  sparse matrices, PyRSB, LIBRSB, Sparse BLAS


Introduction
------------

Sparse linear systems solution is one of the most widespread problems in numerical scientific computing.
Key to timely solution of sparse linear systems by means of iterative methods resides in fast multiplication of sparse matrices by dense matrices.
More precisely, we mean the update:
:math:`C \leftarrow C + \alpha A B` 
(at the element level, equivalent to :math:`C_{i,k} \leftarrow C_{i,k} + \alpha A_{i,j} B_{j,k}`)
where `B` and `C` are dense rectangular matrices, `A` is a *sparse* rectangular matrix, and `\alpha` a scalar.
If `B` and `C` are vectors (i.e. have one column only) we call this operation `SpMV`; otherwise `SpMM`.

PyRSB 
[PYRSB]_
is a package suited
for problems where:
i) much of the time is spent in SpMV or SpMM,
ii) one wants to exploit multicore hardware, and
iii) sparse matrices are large (i.e. occupy a significant fraction of a computers' memory).

The PyRSB interface is styled after that of the sparse matrix classes in
SciPy
[Virtanen20]_.
Unlike certain similarly scoped projects ([Abbasi18]_, [PyDataSparse]_),
PyRSB is restricted to 2-dimensional matrices only.

Background: LIBRSB 
~~~~~~~~~~~~~~~~~~

LIBRSB
[LIBRSB]_
is a LGPLv3-licensed library written primarily to speed up solution of large sparse linear systems using iterative methods on shared-memory CPUs.
It takes its name from the Recursive Sparse Blocks (RSB) data layout it uses.
The RSB format is geared to execute possibly fast threaded SpMV and SpMM.
LIBRSB is not a solver library, but provides most of the functionality required to build one.
It is usable via several languages:
C, C++, Fortran, GNU Octave [SPARSERSB]_, and now Python, too.
A binding for the Julia language has also been contributed by D.C. Jones [RSB_JL]_.

LIBRSB has been reportedly used for:
Plasma physics
[Stegmeir15]_,
sub-atomic physics
[Klos18]_,
data classification
[Lee15]_,
eigenvalue computations
[Wu16]_,
meteorology
[Browne15T]_,
data assimilation
[Browne15M]_.

It is available in pre-compiled form in popular GNU/Linux distributions like 
Ubuntu
[UBUNTU]_,
Debian
[DEBIAN]_,
OpenSUSE
[OPENSUSE]_.
One may install it via a source-based distribution like
Spack
[SPACK]_,
or EasyBuild
[EASYBUILD]_,
and
GUIX
[GUIX]_: using a version built it with an optimizing compiler shall give better results than with one from a binary distributions.
LIBRSB has minimal dependencies, so building it stand-alone is trivial.

PyRSB [PYRSB]_ is a thin
wrapper around LIBRSB based on 
Cython [Behnel11]_.
It aims at bringing native 
LIBRSB performance and most of its functionality at minimal overhead.

Basic Sparse Matrix Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The explicit (**dense**) way to represent any numerical matrix is that by listing each of its numerical entries, whatever their value.
This can be done in Python using e.g.
``scipy.matrix``.

.. code-block:: python

   >>> from scipy import matrix
   >>>
   >>> A = matrix([[11., 12.], [ 0., 22.]])
   matrix([[11., 12.],
           [ 0., 22.]])
   >>> A.shape
   (2, 2)

This matrix has two rows and two columns; it contains three non-zero elements and one zero element on the second row.
Many scientific problems give rise to systems of linear equations with many (e.g. millions) of unknowns, but relatively few coefficients which are different than zero (e.g. `<1%`) in their matrix-form representation.
Then it is usually the case that representing these zeroes in memory and processing them in linear algebraic operations does not impact the results, but takes `compute time` nevertheless.
In these cases the matrix is usually referred as **sparse**, and appropriate **sparse data structures** and algorithms are sought.

The most straightforward sparse data structure for a numeric matrix is the one listing each of the non-zero elements, along with its `coordinate` location, by means of three arrays.
This is called **COO**.
It's one of the classes in ``scipy.sparse``; see the following listing, also illustrating conversion to dense:

.. code-block:: python

   >>> from scipy.sparse import coo_matrix
   >>>
   >>> V = [11.0, 12.0, 22.0]
   >>> I = [0, 0, 1]
   >>> J = [0, 1, 1]
   >>> A = coo_matrix((V, (I, J)))
   <2x2 sparse matrix of type '<class 'numpy.float64'>'
        with 3 stored elements in COOrdinate format>
   >>> B = A.todense()
   >>> B
   matrix([[11., 12.],
           [ 0., 22.]])
   >>> A.shape
   (2, 2)


Even if yielding same results, the algorithms beneath differ considerably.
To carry out the 
:math:`C_{i,k} \leftarrow C_{i,k} + \alpha A_{i,j} B_{j,k}` updates
the ``scipy.coo_matrix`` implementation will get the matrix coefficients from the ``V`` array, its coordinates from ``I`` and ``J`` arrays, and use those (notice the **indirect access**) to address the operand's elements.

In contrast to that, a dense implementation like ``scipy.matrix`` does not use any index array: the location of each numerical value (including zeroes) is in bidirectional correspondence with its row and column indices.

Beyond the ``V,I,J`` arrays, COO has no extra structure.
COO serves well as an exchange format, and allows expressing many operations.

The second most straightforward format is CSR (Compressed Sparse Rows).
In CSR, non-zero matrix elements and their column indices are laid consecutively row after row, in the respective arrays ``V`` and ``J``.
Differently than in COO, the row index information is compressed in a *row pointers* array ``P``,
dimensioned one plus rows count.
For each row index ``i``, ``P[i]`` is the count of non-zero elements (`nonzeroes`) on preceding rows.
The count of nonzeroes at each row ``i`` is therefore ``P[i+1]-P[i]``, with ``P[0]==0``.
SciPy offers CSR matrices via ``scipy.csr_matrix``:

.. code-block:: python

   >>> import scipy
   >>> from scipy.sparse import csr_matrix
   >>>
   >>> V = [11.0, 12.0, 22.0]
   >>> P = [0, 2, 3]
   >>> J = [0, 1, 1]
   >>> A = csr_matrix((V, J, P))
   >>> A.todense()
   matrix([[11., 12.],
           [ 0., 22.]])
   >>> A.shape
   (2, 2)


CSR's ``P`` array allows direct access of each `sparse row`.
This helps expressing row-oriented operations.
In the case of the SpMV operation, CSR encourages accumulation of partial results on a per-row basis.

Notice that indices' occupation with COO is strictly proportional to the non-zeroes count of a matrix;
in the case of CSR, only the ``J`` indices array.
Consequently, a matrix with more nonzeroes than rows (as usual for most problems) will use less index space if represented by CSR.
But in the case of a particularly sparse block of such a matrix, that may not be necessarily true.
These considerations back the usage choice of COO and CSR within the RSB layout, described in the following section.

From RSB to PyRSB
-----------------

Recursive Sparse Blocks in a Nutshell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Recursive Sparse Blocks (RSB) format in LIBRSB
[Martone14]_
represents sparse matrices by 
exploiting a hierarchical data structure.
The matrix is recursively subdivided in halves until the individual submatrices (also: *sparse blocks* or simply *blocks*) occupy approximately the amount of memory contained in the CPU caches.
Each submatrix is then assigned the most appropriate format: COO if very sparse, CSR otherwise.

.. figure:: bayer02--D-N-1--base.pdf
   :scale: 35%

   Rendering of an RSB instance of classical matrix ``bayer02``
   (sized :math:`14k \times 14k` with `64k` nonzeroes, from the SuiteSparse Matrix Collection [SSMC]_);
   each sparse block is labeled with its own format (the 'H' prefix indicating use of a shorter integer type);  
   each block's effectively non-empty rectangle is shown, in colour;
   greener blocks have fewer nonzoeroes than average; rosier ones have more.
   Blocks' rows and columns ranges are evidenced (respectively magenta and green) on the blocks' sides.
   Note that larger blocks (like ``"9/9"``) may have fewer nonzeroes than smaller ones (like ``"4/9"``).
   :label:`bayer02`

Any operation on an RSB matrix is effectively a `polyalgorithm`, i.e. 
each block's contribution will use an algorithm specific to its format, and the intermediate results will be combined.
For a more detailed description, please consult 
[Martone14]_
and further references from there.

The above details are useful to understand, but not necessary to use PyRSB.
To create an ``rsb_matrix`` object one proceeds just as with e.g. ``coo_matrix``:

.. code-block:: python

   >>> from pyrsb import rsb_matrix
   >>>
   >>> V = [11.0, 12.0, 22.0]
   >>> I = [0, 0, 1]
   >>> J = [0, 1, 1]
   >>> A = rsb_matrix((V, (I, J)))
   >>> A.todense()
   matrix([[11., 12.],
           [ 0., 22.]])
   >>> A.shape
   (2, 2)

Direct conversion from ``scipy.sparse`` classes is also supported.
Instancing an RSB structure is computationally more demanding than with COO or CSR (in both memory and time).
Exploiting multiple cores and the savings from faster SpMM's shall make the extra construction time negligible.


Multi-threaded Sparse Matrix-Vector Multiplication with RSB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following sequence of pictures schematizes eight states of a two-threaded SpMV on an RSB matrix consisting of four (non-empty sparse) blocks.
At any moment, up to two blocks are being object of concurrent SpMV (`active`).
Here each active block has a gray background; its rows and column ranges are evidenced.
Left of the matrix, a (out-of-horizontal-scale) result vector is depicted.
For each of the active blocks, the corresponding `active range` (corresponding to the rows) is evidenced on the vector.
Similarly, right of the matrix, the (out-of-horizontal-scale) operand vector is shown; 
its active ranges (corresponding to each blocks' column range) are evidenced.

.. figure:: spmv.pdf
   :scale: 100%
   :alt: alternate text

   SpMV goes through steps leading to the following states:
   1) upper left block becomes active;
   2) lower left block becomes active;
   3) upper left block is done (not active anymore);
   4) upper right block becomes active;
   5) upper right block is done;
   6) lower left block is done;
   7) lower right block is now active;
   8) lower right block is done.
   :label:`spmv`


The idea behind the algorithm is that a thread won't write to a portion of the result array which is currently being updated by another thread.
Beyond that, there is no further synchronization of threads.

This algorithm applies to square as well as non-square matrices.
It supports transposed operation (in which case the ranges of each block are swapped).
Symmetric operation is supported, too; in this case, an additional `transposed` contribution is considered for each block.

As depicted in the first RSB illustration (Fig. :ref:`bayer02`), the order of the sparse blocks in memory proceeds along a *space-filling curve*.
That order of processing the individual blocks can help delivering data from the memory to the cores faster; therefore it is prioritized.

To have enough work for each thread, RSB arranges to have more blocks than threads.
For this and other trade-offs involved,
as well for a formal description of the multiplication algorithm,
see [Martone14]_ and further literature about RSB listed there.

The SpMV algorithm sketched above is what happens `under the hood` in PyRSB.
In practice,
``rsb_matrix`` is used in SpMV just as with ``scipy.sparse`` classes seen earlier:


.. code-block:: python

   >>> from numpy import ones
   >>> B = ones([2], dtype=A.dtype)
   >>> C = A * B

Multi-threaded Sparse Matrix-Matrix Multiplication with RSB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With multiple column operands (in jargon, `multiple right hand sides`), the operation result is equivalent to that of performing correspondingly many SpMVs.

In these cases it comes naturally to lay the columns one after the other (consecutively) in memory, and have the resulting *rectangular dense matrix* as operand to the SpMM.
Also here the same notation of the previous section is supported;
see this example with 2 right hand sides:

.. code-block:: python

   >>> from numpy import ones
   >>> B = ones([2,2], dtype=A.dtype)
   >>> C = A * B

Let's look at how to deal with this when using the RSB layout.
As anticipated, the individual right hand sides may lay after each other, as columns of a rectangular dense matrix.
See Fig. :ref:`forder`, where a broken line follows the two operands' layout in memory, also `by columns`.

.. figure:: rsb-spmv-frame-0000-F2.pdf
   :scale: 25%
   :alt: alternate text

   A Matrix and its SpMM operands, in **columns-major** order. Matrix consisting of four sparse blocks, of which one evidenced. Left hand side and right hand side operands consist of two vectors each. These are stored one column after the other (memory follows blue line). Consequently, the two column portions operands pertaining a given sparse block are not contiguous.
   :label:`forder`

A straightforward SpMM implementation may run two individual SpMV over the entire matrix, one column at a time.
That would have the entire matrix (with all its blocks) being read once per column.

A first RSB-specific optimization would be to run all the per-column SpMVs at a block level.
That is, given a block, repeat the SpMVs over all corresponding column portions.
This would increase chance of reusing cached matrix elements as the operands are visited.
This reuse mechanism is being exploited by LIBRSB-1.2.
The `by columns` layout (or `order`) is the recommended one for SpMM there.

The most convenient thing though, would be to read the entire matrix only once.
That is the case for LIBRSB-1.3 (scheduled for release in summer 2021): for small column counts, block-level SpMM goes through all the columns while reading a block exactly once.

The aforementioned SpMM algorithm is to be regarded as LIBRSB-specific internals, with not much user-level control over it.

But there is another factor instead, that plays a certain role in the efficiency of SpMM, where the PyRSB user has a choice:
the layout of the SpMM operands.

SpMM with different Operands Layout 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **by-columns** layout described earlier and shown in Fig. :ref:`forder` appears to be the most natural one if one thinks of the columns as laid in successive **multiple arrays**.
However, one may instead opt to choose a **by-rows** layout instead, shown in figure :ref:`corder`. 

.. figure:: rsb-spmv-frame-0000-C2.pdf
   :scale: 25%
   :alt: alternate text

   :label:`corder`
   A Matrix and its SpMM operands, in **rows-major order**. Matrix consisting of four sparse blocks, of which one evidenced. Left hand side and right hand side operands consist of two vectors each, interspersed (memory follows blue line). Consequently, the two column portions operands pertaining a given sparse blocks are contiguous.

A by-rows layout can be thought as interspersing all the columns, one index at a time.
Here in the figure, the blue line follows their **order in memory**.
At SpMM time, given one of the input columns, an element at a given index is multiplied by nonzeroes located at that column index.
Similarly, given one of the output columns, an element at a given index receives a contribution from the nonzeroes located at that row coordinate.
With a by-rows layout of the operands, SpMM may proceed by reading a nonzero once, read all right hand sides at that row index (they are adjacent), and then update the corresponding left hand sides' elements (which are also adjacent).
On current cache- and register- based CPUs, the locality induced by this layout leads often to a slightly faster operation than with a by-columns layout.

The by-columns and by-rows layouts go by the respective names of Fortran (``'F'``) and C (``'C'``) order.
A user can choose which dense layout to use when creating operands for SpMM.
Their physical layouts differ, but NumPy makes their results are interoperable; see e.g.:

.. code-block:: python

   >>> import scipy, numpy, rsb
   >>> 
   >>> size = 1000
   >>> density = 0.01
   >>> nrhs = 10
   >>> 
   >>> A = scipy.sparse.random(size, size, density)
   >>> A = rsb.rsb_matrix(A)
   >>> 
   >>> B = numpy.random.rand(size, nrhs)
   >>> 
   >>> B_c = numpy.ascontiguousarray(B)
   >>> B_f = numpy.asfortranarray(B)
   >>> 
   >>> assert B.flags.c_contiguous
   >>> assert B_c.flags.c_contiguous
   >>> assert B_f.flags.f_contiguous
   >>> 
   >>> C = A * B
   >>> C_c = A * B_c
   >>> C_f = A * B_f

While both layouts are supported, the ``'C'`` layout is the recommended one for SpMM operands when using PyRSB with LIBRSB-1.3.
Also notice that SpMV is a special case of SpMM with one left-hand side and one right-hand side, so the two layouts are equivalent here.
In the following, we will often refer to **right-hand sides count** as by **NRHS**.

Using PyRSB: Environment Setup and Autotuning
---------------------------------------------

Usage of PyRSB requires no knowledge beyond its documentation.
However, the underlying LIBRSB library can be configured in a variety of ways, and this affects PyRSB.
To begin using PyRSB, a distribution-provided installation shall suffice.
To expect best performance results, a *native* LIBRSB build is recommended.
The next section comments some basic facts to control LIBRSB and make the most out of PyRSB.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

PyRSB does not use any environment variable directly; it is affected via underlying LIBRSB and Python.
By default, LIBRSB it is built with shared-memory parallelism enabled via OpenMP [OPENMP]_.
As a consequence, a few dozen OpenMP environment variables (all prefixed by ``OMP_``) apply to LIBRSB as well.
Of these, the most important is the one setting the active threads count: ``OMP_NUM_THREADS``.
Administrators of HPC (High Performance Computing) systems customarily set this variable to recommended values.
Even if unset, chances are good the OpenMP runtime will guess the right value for this.
Most other OpenMP variables will be of less use to PyRSB, except one:
setting ``OMP_DISPLAY_ENV=TRUE`` will get current defaults printed at program start (very useful when debugging a configuration).

In addition to the above, there are environment variables affecting specifically LIBRSB.
All of those are prefixed by ``RSB_``, so to avoid any clash.
One recommended to end users is ``RSB_USER_SET_MEM_HIERARCHY_INFO``, and is used to override cache hierarchy information detected at runtime or `hardcoded` at build time.
Essentially, one can use it to force a finer or coarser blocking.
For its usage, and for verification of further LIBRSB defaults, please see its documentation (accessible from [LIBRSB]_).
Modifying the variables mentioned in this section will be mostly useful on very new or not fully configured systems, or for tuning a bit over the defaults.


RSB Autotuning Procedure for SpMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:label:`sec:at`

Cores count, cache sizes, operands data layout, and matrix structure all play a role in RSB performance.
The default blocks layout chosen when assembling an RSB instance may not be the most efficient for the particular SpMM to follow.
In practice, given an RSB instance and an SpMM context (vector and scalar operands info, transposition parameter, run-time threads count), 
it may be the case that a better-performing layout can be found by 
exploring slightly `coarser` or `finer` blockings, 
An automated (`autotuning`) procedure for this exists and is accessible via ``autotune``.
The following example shows how to use it on matrix ``audikw_1`` from  [SSMC]_.

.. code-block:: python

   >>> import sys, rsb, numpy
   >>> dtype=numpy.float32
   >>> 
   >>> A = rsb.rsb_matrix("audikw_1.mtx",dtype=dtype)
   >>> print(A) # original blocking printed out
   >>> sf = A.autotune(verbose=False)
   >>> print("autotune speedup for SpMV  : %.2e x" %sf )
   >>> print(A) # updated blocking printed out
   >>>
   >>> A = rsb.rsb_matrix("audikw_1.mtx",dtype=dtype)
   >>> print(A) # original blocking printed out
   >>> sf = A.autotune(verbose=False, transA='N', 
   >>>       order='C', nrhs=8)
   >>> print("autotune speedup for SpMM-8: %.2e x" %sf )
   >>> print(A) # updated blocking printed out

In scenarios where SpMM is to be iterated many times, time spent autotuning an instance shall amortize over the now faster iterations.
See the comments of instances of autotuning on
Fig. :ref:`audikw-1-S-tuned-C-1`,
Fig. :ref:`audikw-1-S-tuned-C-2`.
and
Fig. :ref:`audikw-1-S-tuned-C-8` for realistic use cases.

The reader impatient to see further speedup figures achievable by ``autotune`` can already peek at
Fig. :ref:`bench:autotuning:speedup:vs:matrix`.


.. figure:: audikw_1-S-tuned-C-1.pdf
   :scale: 35%

   Rendering of an RSB instance matrix ``audikw_1`` (for this and other matrices, see table) as ``dtype=numpy.float32`` (or S) after ``autotune(order='C',nrhs=1)`` on our setup.
   Autotuning merged an initial 766 blocks guess into 295, bringing a :math:`1.56\times` speedup to ``rsb_matrix`` SpMV time.
   With ``rsb_matrix`` it now takes 1/34th of (1-threaded) ``csr_matrix`` time; before autotuning, it took 1/22th.
   Autotuning itself took the time of 1.5 ``csr_matrix`` SpMV iterations, or 34 pre-autotuning ``rsb_matrix`` SpMV iterations.
   :label:`audikw-1-S-tuned-C-1`


.. figure:: audikw_1-S-tuned-C-2.pdf
   :scale: 35%

   Same matrix as Fig. :ref:`audikw-1-S-tuned-C-1`, but autotuned with ``nrhs=2``.
   Here the initial 766 blocks have been merged into 406, with :math:`1.14\times` speedup.
   Before autotuning, it took 1/22th of a (1-threaded) ``csr_matrix`` time; now it's  1/31th.
   Here too, it took the time of 1.5 ``csr_matrix`` SpMM iterations, or 34 with the pre-autotuning ``rsb_matrix`` instance.
   :label:`audikw-1-S-tuned-C-2`


.. figure:: audikw_1-S-tuned-C-8.pdf
   :scale: 35%

   Differently than with ``nrhs=1`` or ``nrhs=2``, ``autotune(nrhs=8)`` did not find a better blocking than the original 766 blocks.
   Still, the procedure costed the time of 11 ``csr_matrix`` SpMM's, or 234 ``rsb_matrix`` ones.
   Though not autotuned, (threaded) RSB takes merely 1/22th the time of CSR here.
   :label:`audikw-1-S-tuned-C-8`



.. [PYRSB] *PyRSB*. (2021, May). Retrieved May 28, 2021, https://github.com/michelemartone/pyrsb
.. [LIBRSB] *LIBRSB*. (2021, May). Retrieved May 28, 2021, https://librsb.sf.net
.. [Martone14] Michele Martone. "Efficient multithreaded untransposed, transposed or symmetric sparse matrix-vector multiplication with the Recursive Sparse Blocks format". Parallel Comput. 40(7): 251-270 (2014)
.. [Virtanen20] P.Virtanen, R.Gommers, T.Oliphant, et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python". Nat Methods 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2
.. [Behnel11] S.Behnel, R.Bradshaw, C.Citro, L.Dalcin, D.S.Seljebotn and K.Smith. "Cython: The Best of Both Worlds", in Computing in Science & Engineering, vol. 13, no. 2, pp. 31-39, March-April 2011, doi: 10.1109/MCSE.2010.118.
.. [RSB_JL] *RecursiveSparseBlocks.jl*, (2021, April 08). Retrieved April 08, 2021, from https://github.com/dcjones/RecursiveSparseBlocks.jl.git
.. [Abbasi18] H.Abbasi, "Sparse: A more modern sparse array library", Proceedings of the 17th Python in Science Conference (SciPy 2018), July 9-15, 2018, Austin, Texas, USA.  http://conference.scipy.org/proceedings/scipy2018/hameer_abbasi.html
.. [PyDataSparse] *PyDataSparse.jl*, (2021, April 08). Retrieved April 08, 2021, from https://github.com/pydata/sparse.
.. [Lee15] M.Lee, W.Chiang and C.Lin, "Fast Matrix-Vector Multiplications for Large-Scale Logistic Regression on Shared-Memory Systems," 2015 IEEE International Conference on Data Mining, Atlantic City, NJ, USA, 2015, pp. 835-840, doi: 10.1109/ICDM.2015.75.
.. [Stegmeir15] A.Stegmeir (Jan 2015). "GRILLIX: A 3D turbulence code for magnetic fusion devices based on a field line map". Available from INIS: http://inis.iaea.org/search/search.aspx?orig_q=RN:46119630
.. [Klos18] P.Klos, S.König, H.-W.Hammer, J.E. Lynn, and A.Schwenk. "Signatures of few-body resonances in finite volume". Phys. Rev. C 98, 034004 – Published 24 September 2018
.. [Wu16] L.Wu. "Algorithms for Large Scale Problems in Eigenvalue and Svd Computations and in Big Data Applications" (2016). Dissertations, Theses, and Masters Projects. Paper 1477068451.  http://doi.org/10.21220/S2S880
.. [Browne15T] P.A. Browne, P.J. van Leeuwen. "Twin experiments with the equivalent weights particle filter and HadCM3". Quarterly Journal of the Royal Meteorological Society, vol. 141, no. 693, pp. 3399-3414, https://doi.org/10.1002/qj.2621
.. [Browne15M] P.A. Browne, S. Wilson. "A simple method for integrating a complex model into an ensemble data assimilation system using MPI". Environmental Modelling & Software, vol. 68, pp. 122-128, https://doi.org/10.1016/j.envsoft.2015.02.003
.. [SPACK] *Spack*. (2021, May). Retrieved May 28, 2021, https://spack.io
.. [EASYBUILD] *EasyBuild*. (2021, May). Retrieved May 28, 2021, https://easybuild.io
.. [DEBIAN] *Debian*. (2021, May). Retrieved May 28, 2021, http://www.debian.org
.. [UBUNTU] *Ubuntu*. (2021, May). Retrieved May 28, 2021, http://www.ubuntu.com
.. [OPENSUSE] *OpenSUSE*. (2021, May). Retrieved May 28, 2021, from https://www.opensuse.org
.. [GUIX] *GuixHPC*. (2021, May). Retrieved May 28, 2021, from https://hpc.guix.info/
.. [SPARSERSB] *SparseRSB*, (2021, April 09). Retrieved April 09, 2021, from https://octave.sourceforge.io/sparsersb/ 
.. [SSMC] *SuiteSparse Matrix Collection*, (2021, May 28). Retrieved May 28, 2021, from https://sparse.tamu.edu/
.. [OPENMP] *OpenMP*, (2021, May). Retrieved May 28, 2021, from https://www.openmp.org/

