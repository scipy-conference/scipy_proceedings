---
title: "GPU-Accelerated Awkward Arrays with CUDA Python"
abstract: |
  Awkward Array is a Python library for manipulating nested,
  variable-length ("ragged") data structures with NumPy-like idioms,
  widely used in High-Energy Physics (HEP) and beyond. Accelerating
  these analyses on GPUs has, until now, required hand-written CUDA
  C++ kernels (compiled at runtime with CuPy): code that is hard to
  write and maintain, difficult to tune for peak performance across
  successive GPU architectures, and awkward to build and package
  because it mixes C++ with Python. This paper reports on a
  collaboration between the Awkward Array and NVIDIA teams to rebuild
  Awkward's GPU backend on `cuda.compute`, a Python library that
  brings the building blocks of NVIDIA's CUDA C++ libraries (CUB and
  Thrust, which provide state-of-the-art, composable parallel
  primitives such as reductions, scans, and sorts) directly to
  Python. The new backend expresses GPU algorithms as compositions of
  these primitives so contributors write ordinary Python instead of
  CUDA C++. The large majority of GPU kernels now run through
  `cuda.compute`, and on realistic physics-analysis benchmarks the new
  backend matches or substantially outperforms the hand-written
  kernels at scale, letting physicists obtain CUDA C++-level GPU
  performance from pure Python.
---

## Introduction

Awkward Array [@awkward; @awkward-scipy2020] is a Python library for manipulating
nested, variable-length data structures (lists of differing lengths, records,
missing values, and unions) using the same vectorized, NumPy-like [@numpy] idioms
scientists already know. It grew out of the High-Energy Physics (HEP) community,
where a single collision event contains a variable number of particles, each with a
variable number of measurements [@awkward-numba], and it is now a foundation of the
Scikit-HEP ecosystem, interoperating with ROOT [@root], Uproot [@uproot], Numba,
Dask [@dask-awkward], JAX, and Apache Arrow.

Modern data analysis increasingly runs on GPUs, but the dominant GPU array and
tensor frameworks are built around dense, rectangular arrays. They offer little for
ragged data: forcing variable-length lists into a rectangular shape requires padding
and masking, and, more fundamentally, the operations a physicist needs (per-event
reductions, combinatorics, selections that depend on the jagged structure itself)
are not expressible as a handful of dense-tensor calls. What is needed is not
another tensor library, but a way to build custom, structure-aware GPU algorithms
while staying in Python.

Awkward has supported GPUs for several years through hand-written CUDA kernels,
compiled at runtime with CuPy [@cupy]. While effective, this approach carries three
persistent costs. First, the kernels are CUDA C++: writing and maintaining them
demands GPU-programming expertise that most contributors (physicists and data
scientists) do not have. Second, hand-written kernels are difficult to keep
performant across successive GPU architectures, each of which introduces new
hardware features; vendor libraries are re-tuned for every generation, but a
project's own kernels are not. Third, mixing C++ with Python complicates building,
packaging and deploying the library. A further missed opportunity is *kernel
fusion* (combining steps so intermediate results stay on-chip), which is key
to obtaining good performance.

This paper reports on a collaboration between the Awkward Array and
NVIDIA teams to rebuild Awkward's GPU backend on `cuda.compute`, a
Python library that brings NVIDIA's CUDA C++ building blocks to
Python. `cuda.compute` exposes the algorithms of CUB and Thrust (the
same battle-tested, architecture-tuned primitives that power
production GPU software) as ordinary Python callables, and composes
them to build a fused kernel that is tuned to provide the best
possible performance for any given GPU architecture. The result lets a
physicist write idiomatic Python and obtain the performance of
expert-written CUDA.

(sec-background)=
## Background

### Awkward Array Layout

An Awkward array stores ragged data as flat one-dimensional buffers rather than as a
rectangular block. A list-of-lists, for example, is represented by a *content*
buffer holding all of the values contiguously and an *offsets* array marking where
each sublist begins and ends ({numref}`fig-layout`). This keeps the data compact (no
padding is wasted on short lists), but it means every operation has to be written in
terms of these buffers rather than a simple multidimensional shape. That requirement
is exactly what made hand-written GPU kernels laborious.

```{figure} layout.png
:label: fig-layout
:align: center
:width: 85%

Awkward represents a ragged list-of-lists as a flat `content` buffer plus an
`offsets` array that delimits each sublist (here the empty middle list spans no
elements). The same idea, applied recursively, encodes arbitrarily nested data.
```

### The Cost of Hand-Written CUDA Kernels

Awkward's GPU backend has been a dictionary of hand-written CUDA C++ kernels,
compiled and cached on first use through CuPy. Consider `ak.min`, the minimum over
each ragged sublist. Its hand-written GPU implementation takes *three* separate
kernels: one to initialize a scratch buffer, one to perform a within-block reduction
using shared memory and explicit thread synchronization, and one to copy the result
out. Each kernel is launched separately and communicates with the next through GPU
global memory ({numref}`fig-akmin`), and a Python dispatcher selects and
parameterizes the right specialization on top of that.

```{figure} akmin_passes.png
:label: fig-akmin
:align: center
:width: 92%

The hand-written `ak.min` runs as three separately launched kernels that pass
partial results through global memory, with a synchronization point between each. A
contributor must reason about CUDA threads, synchronization, and the ragged buffers
all at once.
```

A contributor fixing a subtle bug in this path must reason simultaneously about CUDA
thread indexing, synchronization, and the ragged buffer arithmetic, and must be
fluent in CUDA C++ to begin with. Multiplied across the backend's more than one
hundred GPU kernels, this is a substantial and specialized maintenance burden, and
it is the burden the new backend is designed to remove.

## cuda.compute

`cuda.compute` is a Python library that brings NVIDIA's CUDA C++ parallel-algorithm
building blocks, CUB and Thrust, directly to Python [@cuda-compute-docs]. A reduction
that would otherwise be hand-written in CUDA C++ becomes a single call, with the
reduction operator written as a plain Python function:

```python
import cupy as cp, numpy as np
from cuda.compute import reduce_into

d_in:  cp.ndarray = cp.arange(1_000_000, dtype=cp.float64)
d_out: cp.ndarray = cp.empty(1, dtype=cp.float64)

def add(a: float, b: float) -> float:
    return a + b

reduce_into(d_in, d_out, add, len(d_in),
            h_init=np.array([0.0]))   # d_out -> 499999500000.0
```

Its key features:

- **Pure Python.** Both the algorithms and the operators they take (here `add`) are
  written in Python; no CUDA C++ is involved.
- **Built on CUB and Thrust.** The underlying implementations are the
  state-of-the-art, composable primitives (reductions, scans, sorts, transforms) used
  throughout production GPU software, and they are specialized and re-tuned for each
  GPU architecture as new hardware features (such as the Tensor Memory Accelerator)
  appear [@cub].
- **JIT-compiled.** User operators are compiled to device code with Numba CUDA and
  cached, so the compilation cost is paid once per specialization.
- **Composable.** Algorithms accept *iterators* that compute their elements lazily
  during execution, fusing steps that would otherwise require separate CUDA kernel
  launches.

### Kernel Fusion

Iterators are what make kernel fusion possible: composing an algorithm with an
iterator folds an extra map, gather, or post-processing step *into* the algorithm's
own pass, so intermediate values stay on-chip instead of being written out and read
back. Two examples from the new backend illustrate the pattern.

A reduction over a *function* of the input fuses the map into the reduction. Awkward's
`min_range` kernel (the smallest sublist length across all rows) pairs the `stops` and
`starts` buffers with a `ZipIterator`, subtracts them (lazily) with a
`TransformIterator`, and reduces, so the per-row lengths are never materialized:

```python
import numpy as np
from cuda.compute import (
    OpKind, TransformIterator, ZipIterator, reduce_into,
)

def diff(pair) -> int:
    return pair.field_0 - pair.field_1          # stop - start

lengths: TransformIterator = TransformIterator(ZipIterator(stops, starts), diff)
reduce_into(lengths, out, OpKind.MINIMUM, n,
            h_init=np.array([np.iinfo(dtype).max], dtype=dtype))
```

Locating the position of an extremum (`argmin`/`argmax`) fuses the per-row search and
the conversion to a global index into one pass: a `CountingIterator` enumerates the
rows, and a single transform maps each row to the global index of its extremum, so no
intermediate index array is ever built:

```python
import numpy as np
from cuda.compute import CountingIterator, unary_transform

def row_argmax(row: int) -> int:
    lo, hi = starts[row], stops[row]
    if lo == hi:                                # empty list -> sentinel
        return -1
    return np.argmax(content[lo:hi]) + lo       # local argmax -> global index

unary_transform(CountingIterator(index_dtype(0)), result, row_argmax, nrows)
```

In both cases several logical steps collapse into a single kernel: exactly the kind of
fusion that is tedious to write and maintain by hand.

## The New Awkward GPU Backend

The new backend leaves Awkward's user-facing API unchanged and replaces the
hand-written CUDA layer with `cuda.compute` calls ({numref}`fig-architecture`).
Operations that previously required bespoke CUDA C++
are now written in Python, and the work of generating efficient device code (including
fusion and per-architecture tuning) is delegated to `cuda.compute`.

```{figure} architecture.png
:label: fig-architecture
:align: center
:width: 85%

The dispatcher behind Awkward's unchanged `ak.Array` API routes GPU work to the new
pure-Python `cuda.compute` backend instead of the legacy hand-written CUDA C++ path.
There, the user-supplied operators and iterators are JIT-compiled to LTO-IR while
`cuda.compute` independently compiles its CUB/Thrust kernels to LTO-IR; the two are
JIT-linked into a single fused CUDA kernel that runs on the GPU.
```

### From CUDA C++ to Python: `ak.min`

The three-kernel `ak.min` of {numref}`fig-akmin` becomes a few lines of Python:

```python
import cupy as cp, numpy as np
from cuda.compute import OpKind, segmented_reduce

def awkward_reduce_min(
    toptr: cp.ndarray, fromptr: cp.ndarray, offsets: cp.ndarray,
    outlength: int, identity: float,
) -> None:

    toptr[:outlength] = identity
    segmented_reduce(
        fromptr, toptr, offsets[:-1], offsets[1:],
        OpKind.MINIMUM, np.asarray(identity, dtype=fromptr.dtype), outlength)
```

There is no CUDA C++, no manual synchronization, and no scratch-buffer
bookkeeping: the library handles boundary conditions, temporary storage, and fusion.
The same pattern applies across the backend, and the operator (`min_op`) is an
ordinary Python function that can be read and tested without a GPU.

### Replacing C++ with Python

Counting the code that must actually be maintained (the CUDA C++ kernels and the Python
that drives them), the GPU backend is both smaller and largely contains pure Python
({numref}`fig-loc`). A release predating `cuda.compute` carried about 8,300 lines of
hand-written CUDA C++ with only a few hundred lines of Python glue. In the latest
Awkward the hand-written CUDA C++ still in use is about 2,700 lines (the structural
kernels not yet migrated), while the `cuda.compute` reimplementations and dispatch are
about 4,000 lines of Python. The CUDA C++ a contributor must read and maintain has
fallen by roughly two-thirds, and the GPU backend is now mostly Python: readable,
reviewable, and testable without GPU expertise.

```{figure} loc_comparison.png
:label: fig-loc
:align: center
:width: 64%

GPU kernel code that must be maintained, by language and Awkward version. Before `cuda.compute` the
backend was almost entirely CUDA C++ (~8,300 lines); in the latest Awkward the C++ to
maintain has dropped by roughly two-thirds and the backend is mostly Python.
```

## Results

### Kernel Coverage

The migration of the GPU backend to `cuda.compute` is tracked publicly in the
project's issue tracker [@awkward-issue-3793] and is largely complete. In the latest
Awkward, the backend has on the order of 130 GPU kernels; roughly 80 of them (about
60%) now run through `cuda.compute`. These include every reduction (`sum`, `prod`,
`min`, `max`, `argmin`, `argmax`, `count`) and the sort, which have no hand-written
CUDA implementation remaining at all. The kernels still hand-written are structural
operations (some jagged indexing and union-array manipulations) and are being migrated
incrementally.

```{table} GPU kernel coverage in the latest Awkward: of the kernels with a CUDA implementation, the large majority now run through cuda.compute, with the reductions and sort running exclusively on it.
:label: tbl-coverage

| Category | Count |
|---|---|
| GPU kernels with a CUDA implementation | ~130 |
| Running through `cuda.compute` | ~80 |
| Still hand-written only | ~50 |
```

### End-to-End Analysis Benchmarks

To gauge the impact on realistic workloads rather than microbenchmarks, we ran the
ADL (Analysis Description Language) benchmark queries [@adl-benchmarks] on CMS 2012
open data (Run2012B `SingleMu`), using the GPU port of the queries from
[@columnar-gpu]. These eight canonical HEP analysis tasks range from simple spectra
(missing transverse energy, jet $p_T$) to combinatoric reconstructions (opposite-sign
di-muon invariant mass, trijet selection). We compare a released Awkward predating
`cuda.compute` (the hand-written CUDA backend) against the `cuda.compute` backend, on
an NVIDIA RTX 6000 Ada, measuring the GPU compute stage at three event counts.

```{figure} benchmark_speedup.png
:label: fig-adl
:align: center
:width: 78%

GPU compute-stage speedup (hand-written ÷ `cuda.compute`) versus event count for the
ADL benchmark queries. The combination-heavy queries Q5 (di-muon mass) and Q6
(trijet) pull away from parity as the problem grows; Q5 reaches roughly 84× at 10M
events because its `cuda.compute` time stays nearly flat while the hand-written time
grows super-linearly. Q3 and Q8 are omitted because the hand-written backend fails on
them while `cuda.compute` succeeds.
```

Two effects stand out ({numref}`fig-adl`). First, the advantage of `cuda.compute`
*grows with problem size*. The combination-heavy di-muon query (Q5) moves from rough
parity at 100k events to roughly 9× at 1M and 84× at 10M: its `cuda.compute`
execution time stays nearly constant with event count while the hand-written
implementation scales super-linearly. At small sizes the GPU is under-utilized and
per-launch overhead dominates, so the two are comparable; the gap opens once the work
is large enough to be bandwidth-bound. Second, `cuda.compute` is more *robust*: the
hand-written backend fails outright on two queries (a scale-dependent error on Q3 at
≥1M events, and an illegal memory access in `argmin` over jagged data with empty
sublists on Q8), whereas `cuda.compute` completes all eight at every size. On the
light, elementwise-dominated queries (Q4, Q7) the two are comparable, as expected.

The comparison isolates the GPU compute stage; data ingestion uses an identical
path for both backends and is excluded. The trijet query (Q6) was evaluated in fixed
event batches at the largest size to fit device memory, a change that leaves results
unchanged. This is a reminder that in such analyses it is the combinatoric
intermediates, not the reductions, that dominate memory.

## Discussion

### Why `cuda.compute`?

Other routes to GPU code from Python exist: hand-written CUDA C++, kernel-generation
frameworks, and domain-specific compilers among them. `cuda.compute` fits Awkward
particularly well for three reasons.

- **Maintenance.** It is maintained by NVIDIA and its underlying primitives are
  re-tuned for each new GPU architecture. Awkward inherits good performance on new
  hardware almost as soon as that hardware ships, without any change to Awkward's own
  code.
- **Ecosystem.** The underlying CUDA C++ libraries, CUB and Thrust, are used
  pervasively across GPU-accelerated software, so Awkward builds on a heavily
  exercised, well-tested foundation rather than a bespoke one.
- **Fit.** Awkward's GPU operations map cleanly onto the composable primitives
  `cuda.compute` provides, and the library performs the kernel fusion that would
  otherwise have to be written and maintained by hand.

### Lowering the Barrier to Contribution

A stated goal of the Awkward Array project is to let physics and data analysts write
high-performance code in Python without GPU expertise. The hand-written backend
required contributors to understand CUDA thread hierarchies, atomics, and
shared-memory behavior to add or fix a kernel. The new backend asks only for a scalar
operator written in Python and a call to the appropriate `cuda.compute` primitive.
That the GPU code is now *pure Python* (readable, reviewable, and testable by the
domain scientists who use the library) is as significant as any single performance
number.

### Limitations

`cuda.compute` is still maturing, and a portion of Awkward's structural kernels remain
hand-written for now. Just-in-time compilation also adds latency on the first use of a
given specialization; for short interactive sessions this can be noticeable, and
ahead-of-time specialization of common cases is an area of ongoing work.

## Future Work

Several directions extend naturally from this work. The most immediate is **finishing
the migration** of the remaining structural kernels so that the GPU backend is
entirely free of hand-written CUDA C++. For the small inner lists common in HEP data
(a handful of elements per list), **warp- and block-level cooperative algorithms** can
be more efficient than device-wide ones; `cuda.coop` [@cuda-coop] brings exactly these
cooperative primitives to Python and is a natural next step. **Ahead-of-time
specialization** of common dtype and operator combinations would hide the
first-use compilation latency for interactive analysis. Finally, **fusion across
operations** (capturing a chain of high-level Awkward calls and lowering it into a
single fused pipeline) would extend the within-operation fusion demonstrated here to
whole analysis expressions.

## Conclusion

We have presented a collaboration between the Awkward Array and NVIDIA teams that
rebuilds Awkward's GPU backend on `cuda.compute`. The change replaces hand-written
CUDA C++ kernels with compositions of Python-callable primitives drawn from CUB and
Thrust, letting the library handle fusion and per-architecture tuning. The large
majority of GPU kernels now run through `cuda.compute` (every reduction and sort
exclusively so), the GPU code is increasingly pure Python rather than CUDA C++, and on
realistic physics
benchmarks the new backend matches the hand-written kernels on light workloads and
outperforms them by large and growing margins as data sizes grow, while also
completing queries the old backend could not. Most importantly, the GPU code is now
pure Python: high-level abstraction and hardware-class performance need not be in
tension when the underlying library understands both the hardware and the problem.

## Acknowledgements

This work was supported in part by NSF grants OAC-1450377, OAC-1836650,
OAC-2103945, PHY-2121686, and PHY-2323298. The authors thank the `cuda.compute`
and CUB/Thrust developers at NVIDIA, and the Scikit-HEP community.

Portions of this work were assisted using a generative AI tool (Claude, by
Anthropic). The tool was used for drafting and refining text and for code and
figure assistance. All outputs were reviewed, verified, and revised by the
authors, who take full responsibility for the accuracy and integrity of the final
content.
