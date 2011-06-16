
the perf gap

libraries are usual, but can't use when:
  - dont know platform at runtime
  - significant speedups possible if know problem - can pick strategy
    and/or tuning
  - higher order function: awkward to bridge languages

our approach: 
  - provide class methods that encapsulate specific computations
  - when called, they (a) combine source code tempaltes authored by
    human expert that impelment a strategy, and optionally (b) lower
    Python code into the target lang to inline the H.O. func
  - result is a source file implementing the computation for that
    particular problem instance
  - then feed to compiler and link to python

insight: because each class specializes one type of computation only, we
can apply optimizations at the level of problm instance rather than
limit ourselves to optimizations that would work generally.

for example, consider stencil.  when iterating over neighbors in a grid,
optimizations like loop unrolling or loop transposition can be applied,
because semantics of stencil means we can embarrassingly parallelize.

at a high level, our approach is to intercept the first call to a
specializable method, grab the AST of the python code at call site, and
immediately transform it to a domain-specific DAST.  That is, we
immediately move the computation into a domain where problem-specific
optimiations and knowledge can be applied, by applying transformations
to the DAST.  The specific transformations depend on the type of problem
and probably on the hardware platform, eg GPU vs. multicore.  The DAST
is then turned into ELL code; the ELL code comes partially from
templates supplied by a human expert and partially by code-generating
from the DAST.  Finally, the generated source code is compiled by an
appropriate downstream compiler (gcc, cudac, proprietary compilers, etc)
and the binary is called to do the computation.  The usual code caching
strategies avoid the cost of this process on subsequent calls.

[need diagram showing human expert, strategy consisting of templates and
AST transformation rules (for each of N platforms), app writer, Asp,
generated code]

Features:

- Leverage downstream compilers; indeed, can emit code that's easier for
  them to optimize

- Not a magic compiler/parallelizer: relies on human expert providing a
  specialization strategy packages as templates + AST transformation
  rules

- separates concerns by hiding complexity: in ELL code, even "simple"
  computation like stencil expands 10x when the extra code necessary to
  get performance (loop unrolling, software pipelining, parallel
  annotations, etc) is added to the main application logic

- performance portability: same Python source can be used to
  JIT-specialize to radically different platforms, so source code has
  nothing in it that ties its performance to a specific platform

Open issues:

- when to respecialize
