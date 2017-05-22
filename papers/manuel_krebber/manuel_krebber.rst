:author: Manuel Krebber
:email: manuel.krebber@rwth-aachen.de
:institution: RWTH Aachen University, AICES, HPAC Group

:author: Henrik Bartels
:email: barthels@aices.rwth-aachen.de
:institution: RWTH Aachen University, AICES, HPAC Group

:bibliography: literature

.. latex::
   \providecommand*\DUrolebuiltin[1]{\PY{n+nb}{#1}}
   \providecommand*\DUrolepython[1]{\footnotesize{#1}}
   \renewcommand\DUrolestring[1]{\PY{l+s+s1}{#1}}

.. role:: py(code)
   :language: python

-----------------------------------
MatchPy: A Pattern Matching Library
-----------------------------------

.. class:: abstract

   Pattern matching is a powerful tool for symbolic computations, based on the well-defined theory of term rewriting systems.
   Application domains include algebraic expressions, abstract syntax trees, and XML and JSON data.
   Unfortunately, no implementation of pattern matching as general and flexible as in Mathematica exists for Python.
   Therefore, we created the open source module MatchPy_, which offers similar pattern matching functionality in Python.
   In addition, we implemented a novel algorithm which finds matches for large pattern sets more efficiently by exploiting similarities between patterns.

.. class:: keywords

   pattern matching, symbolic computation, discrimination nets, term rewriting systems

Introduction
------------

Pattern matching is a powerful tool which is part of many functional programming languages as well as computer algebra systems such as Mathematica.
It is useful for many applications including symbolic computation, term simplification, term rewriting systems, automated theorem proving, and model checking.
Term rewriting systems rely on pattern matching to find matches for the rewrite rules.
In functional programming languages, pattern matching enables a more readable and intuitive expression of algorithms.

Among the existing systems, Mathematica offers the most expressive pattern matching.
Its pattern matching offers similar expressiveness as regular expressions in Python, but for symbolic tree structures instead of strings.
Patterns are used widely in Mathematica, e.g. in function definitions or for manipulating expressions.
It is possible to define custom function symbols which can be associative and/or commutative.
Mathematica also offers sequence variables which can match a sequence of expressions instead of a single one.
They are especially useful when working with variadic function symbols.

There is currently no open source alternative to Mathematica with comparable pattern matching capabilities.
In particular, we are interested in similar pattern matching for an experimental linear algebra compiler written in Python.
Unfortunately, Mathematica is proprietary and there are no publications on the underlying pattern matching algorithm.

Previous work predominantly covers syntactic pattern matching, i.e. associative/commutative/variadic
function symbols are not supported. Specifically, no existing work allows function symbols
which are either commutative or associative but not both. However, there are domains where
functions have those properties, e.g. matrix multiplication in linear algebra.
Most of the existing pattern matching libraries for Python only support syntactic patterns.
While the pattern matching in SymPy_ can work with associative/commutative functions, it is limited to finding a single match and does not support sequence variables.
However, we are also interested in finding all possible matches for a pattern.

In many applications, a fixed set of patterns will be matched repeatedly against different subjects.
The simultaneous matching of multiple patterns is called many-to-one matching, as opposed to
one-to-one matching which denotes matching with a single pattern.
Many-to-one matching can be sped up by exploiting similarities between patterns.
This has already been the subject of research for both syntactic and AC pattern matching, but not with
the full feature set described above.
Discrimination nets are the state-of-the-art solution for many-to-one matching.
Our goal is to generalize this approach to support all the aforementioned features.

We implemented pattern matching with sequence variables and associative/commutative function symbols
as an open-source library for Python called MatchPy_. In addition to standard one-to-one matching,
this library also includes an efficient many-to-one matching algorithm that uses generalized discrimination nets.
In our experiments we observed significant speedups of the many-to-one matching over one-to-one matching.

Usage Overview
--------------

MatchPy can be installed using ``pip install matchpy`` and all necessary classes can be imported with
:py:`from matchpy import *`. Expressions in MatchPy consist of symbols and operations.
For patterns, wildcards can also be used as placeholders. Similarly to the
`notation <https://reference.wolfram.com/language/guide/Patterns.html>`_ that
Mathematica uses, we will append an underscore to wildcard names to
distinguish them from constant symbols. Optionally, patterns can have further
constraints that restrict what they can match. You can use MatchPy with native Python types
such as lists and ints:

.. code-block:: pycon

    >>> x_ = Wildcard.dot('x')
    >>> next(match([0, 1], Pattern([x_, 1])))
    {'x': 0}

You can also define custom operations by creating a subclass of the ``Operation`` class:

.. code-block:: python

    class MyOp(Operation):
      name = 'MyOp'
      arity = Arity.variadic
      associative = True
      commutative = True
      one_identity = True

These can also be associative and/or commutative which influences what a pattern can match.
Nested associative operations have to be variadic and are automatically flattened:

.. code-block:: pycon

    >>> print(MyOp(0, MyOp(1, 2)))
    MyOp(0, 1, 2)

The argument of commutative operations are automatically sorted:

.. code-block:: pycon

    >>> print(MyOp(2, 1, 3))
    MyOp(1, 2, 3)

Sequence wildcards can match a sequence of arguments:

.. code-block:: pycon

    >>> y___ = Wildcard.star('y')
    >>> next(match([1, 2, 3], Pattern([x_, y___])))
    {'x': 1, 'y': (2, 3)}

Note that patterns containing multiple sequence variables or patterns with commutative operations
can have multiple matches:

.. code-block:: pycon

    >>> z_ = Wildcard.dot('z')
    >>> pattern = Pattern(MyOp(x_, z_))
    >>> list(match(MyOp(1, 2), pattern))
    [{'x': 2, 'z': 1}, {'x': 1, 'z': 2}]

We can use the ``CustomConstraint`` class to create a constraint that checks whether
the substitution for ``a`` is smaller than the one for ``b``:

.. code-block:: python

    a_ = Wildcard.dot('a')
    b_ = Wildcard.dot('b')
    h___ = Wildcard.star('h')
    t___ = Wildcard.star('t')
    a_lt_b = CustomConstraint(lambda a, b: a < b)

With this constraint we can define a replacement rule that basically describes bubble sort:

.. code-block:: pycon

    >>> pattern = Pattern([h___, b_, a_, t___], a_lt_b)
    >>> rule = ReplacementRule(pattern,
                    lambda a, b, h, t: [*h, a, b, *t])

This replacement rule can be used to sort a list when applied repeatedly with ``replace_all``:

.. code-block:: pycon

    >>> replace_all([1, 4, 3, 2], [rule])
    [1, 2, 3, 4]

More examples can be found in `MatchPy's documentation <https://matchpy.readthedocs.io/latest/>`_.

Example Domain: Linear Algebra
------------------------------

As an example, we will create the classes necessary to construct linear algebra expressions.
These expressions consist of scalars, vectors, and matrices, as well as multiplication, addition,
transposition, and inversion. The following Python code defines the classes:

.. code-block:: python

    class Scalar(Symbol):
        pass

    class Vector(Symbol):
        pass

    class Matrix(Symbol):
      def __init__(self, name, properties=[]):
        super().__init__(name)
        self.properties = frozenset(properties)

    Times = Operation.new(
      '*', Arity.variadic, 'Times',
      associative=True, one_identity=True,
      infix=True)
    Plus = Operation.new('+', Arity.variadic, 'Plus',
      one_identity=True, infix=True,
      commutative=True, associative=True)

    class PostfixUnaryOperation(Operation):
      arity = Arity.unary
      def __str__(self):
        return '({}){}'.format(self.operands[0],
          self.name)

    class Transpose(PostfixUnaryOperation):
      name = '^T'

    class Inverse(PostfixUnaryOperation):
      name = '^-1'

Matrix symbols have a set of properties which can be checked by constraints on the patterns. For
``Plus`` and ``Times``, the ``Operation.new`` convenience function is used to quickly create the classes.
If ``one_identity`` is ``True``, :math:`op(x) = x` holds and and occurences of the operation with a
single argument are simplified. ``infix`` has just cosmetic effects and makes the string
representation of the operation use infix instead of prefix notation. For the unary operations,
custom classes are implemented in order to override their string representation.

Application: Finding matches for a BLAS kernel
..............................................

Lets assume we want to find all subexpressions of some linear algebra expression which we can compute efficiently with
the `?TRMM`_ BLAS_ routine. These all have the form :math:`\alpha \times op(A)  \times B` or :math:`\alpha  \times B  \times op(A)` where
:math:`op(A)` is either the identity function or transposition, and :math:`A` is a triangular matrix.
For this example, we will leave out all variants where :math:`\alpha \neq 1`. We can construct the
patterns using sequence variables to capture the remaining operands of the multiplication:

.. code-block:: python

    A_ = Wildcard.symbol('A', Matrix)
    B_ = Wildcard.symbol('B', Matrix)
    before_ = Wildcard.star('before')
    after_ = Wildcard.star('after')
    A_is_triangular = CustomConstraint(
      lambda A: 'triangular' in A.properties)

    trmm_patterns = [
      Pattern(Times(before_, A_, B_, after_),
        A_is_triangular),
      Pattern(Times(before_, Transpose(A_), B_, after_),
        A_is_triangular),
      Pattern(Times(before_, B_, A_, after_),
        A_is_triangular),
      Pattern(Times(before_, B_, Transpose(A_), after_),
        A_is_triangular),
    ]

As an example, we can find all matches for the first pattern using ``match``:

.. code-block:: pycon

    >>> expr = Times(Transpose(M3), M1, M3, M2)
    >>> print(next(match(expr, trmm_patterns[0])))
    {A -> M3, B -> M2, after -> (), before -> ((M3)^T, M1)}

.. _`?TRMM`: https://software.intel.com/en-us/node/468494
.. _BLAS: http://www.netlib.org/blas/

Challenges
----------

While there are plenty of implementations of syntactic matching and the algorithms are well known,
the pattern matching in MatchPy has several more challenging features.

Associativity/Sequence variables
................................

Associativity enables arbitrary grouping of arguments for matching: For example, :math:`1 + a + b`
matches :math:`1 + \pmb{x}` with :math:`\{ \pmb{x} \mapsto a + b \}`, because we can group the
arguments as :math:`1 + (a + b)`. Basically, when regular
variables are arguments of an associative function, they behave like sequence variables.
Both can result in multiple distinct matches for a single pattern. In constrast, for syntactic
patterns there is always at most one match. This means that the matching algorithm needs to be
non-deterministic to explore all potential matches. We employ backtracking with the help of Python
generators to enable this. Associative matching is NP-complete :cite:`Benanav1987`.

Commutativity
.............

Matching commutative functions is difficult, because matches need to be found independant of the
argument order. Commutative matching has been shown to be NP-complete :cite:`Benanav1987`.
It is possible to solve this by matching all permutations of the subjects arguments
against all permutations of the pattern arguments. However, with this naive approach, a total of
:math:`n!m!` combinations have to be matched where :math:`n` is the number of subject arguments
and :math:`m` the number of pattern arguments. Most of these combinations will likely not match
or yield redunant matches.

Instead, we interpret the arguments as a multiset, i.e. an orderless collection that allows
repetition of elements. Also, we use the following order for matching a commutative term:

1. Constant arguments
2. Matched variables, i.e. variables that already have a value assigned in the current substitution
3. Non-variable arguments
4. Repeat step 2
5. Regular variables
6. Sequence variables

Each of those steps reduces the search space for successive steps. This also means that if one step
finds no match, the remaining steps do not have to be performed. Note that steps 3, 5 and 6 can
yield multiple matches and backtracking is employed to check every combination. This can speed up
matching significantly. Because step 6 is the most involved, it is described in more detail in the
next section.

Sequence Variables in Commutative Functions
...........................................

The distribution of :math`n` subjects subterms onto :math`m` sequence variables within a
commutative function symbol can yield up to :math`m^n` distict solutions. Enumerating all of the
is accomplished by generating and solving several linear Diophantine equations. As an example,
lets assume we want to match :math:`f(a, b, b, b)` with
:math:`f(\pmb{x}^{\pmb{*}}, \pmb{y}^{\pmb{+}}, \pmb{y}^{\pmb{+}})` where :math:`f` is commutative.
This means that the possible distributions are given by the non-negative integer solutions of
these equations:

.. math::
    :type: eqnarray

    1 &=& x_a + 2 y_a \\
    3 &=& x_b + 2 y_b

Because :math:`\pmb{y}^{\pmb{+}}` requires at least one term, we have the additional constraint
:math:`y_a + y_b \geq 1`. The only possible solution :math:`x_a = 1, x_b = 1, y_a = 0, y_b = 1`
corresponds to the match substitution :math:`\{\pmb{x}^{\pmb{*}} \mapsto (a, b), \pmb{y}^{\pmb{+}} \mapsto (b) \}`.

Extensive research has been done on solving linear Diophantine equations and linear Diophantine
equation systems :cite:`Weinstock1960,Bond1967,Lambert1988,Clausen1989,Aardal2000`. In our case
the equations are actually independant expect for requiring at least one term for plus variables.
Also, the non-negative solutions can be found more easily. We use an adaptation of the
algorithm used in SymPy_ which recursively reduces any linear Diophantine equation to equations
of the form :math:`ax + by = d`. Those can be solved efficiently with the Extended Euclidian algorithm
:cite:`Menezes1996`. Then the solutions for those can be combined into a solution for the original
equation.

All coefficients in those equations are likely very small, because they correspond to the multiplicity
of sequence variables. Similarly, the number of variables in the equations is usually small as they
map to sequence variables. The constant is the multiplicity of a subject term and hence also
usually small. Overall, the number of distict equations that are solved is small and the
solutions are cached. This reduces the impact of the sequence variables on the overall run time.


Many-to-one Matching
--------------------

Since most applications for pattern matching will repeatedly match a fixed set of patterns against
multiple subjects, we implemented many-to-one matching for MatchPy. We will give a brief overview
over the underlying algorithms. Full details can be found in the master thesis :cite:`thesis` that MatchPy is
based on.

MatchPy also includes two additional algorithms for matching: ``ManyToOneMatcher`` and
``DiscriminationNet``. Both enable matching multiple pattern against a single subject
much faster than matching each pattern individually using ``match``. The later can only be used
for syntactic patterns, i.e. patterns without associative/commutative operations and sequence
variables. Both are based on discrimination nets which are a data structure similar to a
decision tree used to speed up many-to-one matching :cite:`Christian1993,Graef1991,Nedjah1997`.
The ``ManyToOneMatcher`` uses a non-deterministic discrimination net with
backtracking, while the ``DiscriminationNet`` is deterministic.

.. figure:: dn.pdf

   Example Discrimination Net. :label:`fig:dn`

In Figure :ref:`fig:dn`, an example for a non-deterministic discrimination net is shown.
It contains three patterns that match Python lists: One matches the list that consists of a single 1,
the second one matches a list with exactly two elements where the last element is 0, and the third pattern
matches any list where the first element is 1. Note, that these patterns can also match nested lists,
e.g. the second pattern would also match :math:`[[2, 1], 0]`.

Matching starts at the root and proceeds along the transitions.
Simultaneously, the subject is traversed in preorder and each symbol is check against the
transitions. Only transitions matching the current subterm can be used. Once a final state is
reached, its label gives a list of matching patterns. For non-deterministic discrimination nets,
all possibilities need to be explored via backtracking. The discrimination net allows to
reduce the matching costs, because common parts of different pattern only need to be matched once.
For non-matching transitions, their whole subtree is pruned and all the patterns are excluded
at once, further reducing the match cost.

In Figure :ref:`fig:dn`, for the subject :math:`[1, 0]`:, there are two paths and therefore two
matching patterns: :math:`[\pmb{y}, 0]` matches with :math:`\{ \pmb{y} \mapsto 1 \}` and
:math:`[1, \pmb{x}^{\pmb{*}}]` matches with :math:`\{ \pmb{x}^{\pmb{*}} \mapsto 0 \}`. Both the
:math:`\pmb{y}`-transiton and the 1-transition can be used in the second state to match a 1.

Compared to existing discrimination net variants, we added transitions for the end of a compound term
to support variadic functions. Furthermore, we added support for both associative function symbols
and sequence variables. Finally, our discrimination net supports transitions restricted to
symbol classes (i.e. ``Symbol`` subclasses) in addition to the ones that match just a specific symbol.
We decided to use a non-deterministic discrimination net instead of a deterministic one, since
the number of states of the later would grow exponentially with the number of patterns. While
the ``DiscriminationNet`` also has support for sequence variables, in practice the net became to large
to use with just a dozen patterns.

Commutative Many-to-one Matching
--------------------------------

Many-to-one matching for commutative terms is more involved. We use a nested ``CommutativeMatcher``
which in turn uses another ``ManyToOneMatcher`` to match the subterms. Our approach is similar to
the one used by Bachmair and Kirchner in their respecitive works :cite:`Bachmair1995,Kirchner2001`.
We match all the subterms of the commutative function in the subject with a many-to-one matcher
constructed from the subpatterns of the commutative function in the pattern (except for sequence
variables, which are handled separately). The resulting matches
form a bipartite graph, where one set of nodes consists of the subject subterms and the other
contains all the pattern subterms. Two nodes are connected by an edge iff the pattern matches the
subject. Such an edge is also labeled with the match substitution(s). Finding an overall match is then
accomplished by finding a maximum matching in this graph. However, for the matching to be valid, all the
substitutions on its edges must be compatible, i.e. they cannot have contradicting replacements for
the same variable. We use the Hopcroft-Karp algorithm :cite:`Hopcroft1973` to find an initial
maximum matching. However, since we are also interested in all matches and the inital matching might
have incompatible substitutions, we use the algorithm described by Uno, Fukuda and Matsui
:cite:`Fukuda1994,Uno1997` to enumerate all maximum matchings.

We want to avoid yielding redundant matches, therefore we extended the bipartite graph by introducing
a total order over its two node set. This enables to determine whether the edges of a matching
maintain the order induced by the subjects or if some of the edges "cross". Formally,
for all edge pairs :math:`(p, s), (p', s') \in M` we require
:math:`(s \equiv s' \wedge p > p') \implies s > s'` to hold where :math:`M` is the matching,
:math:`s, s'` are subjects, and :math:`p, p'` are patterns.
An example of this is given in Figure :ref:`fig:bipartite2`. The order of the nodes is indicated by
the numbers next to them. The only two maximum matchings for this particular match graph are
displayed. In the left matching, the edges with the same subject cross and hence this matching is
discarded. The other matching is used because it maintains the order. This ensures only unique
matches are yielded.

Once a matching for the subpatterns is obtained, the remaining

.. figure:: bipartite2.pdf

   Example for Order in Bipartite Graph. :label:`fig:bipartite2`

Experiments
-----------

To evaluate the performance of MatchPy, we performed several experiments. All experiments were
conducted on an Intel Core i5-2500K 3.3 GHz CPU with 8GB of RAM.

Linear Algebra
..............

The operations for the linear algebra problem are shown in Table :ref:`tbl:laop`. The patterns
all match BLAS_ kernels similar to the example pattern which was previously described. The pattern
set consists of 199 such patterns. Out of those, 61 have an addition as outermost operation, 135
are patterns for products, and 3 are patterns for single matrices. A lot of these patterns only
differ in terms of constraints, e.g. there are ten distinct patterns matching :math:`A \times B`
with different constraints on the two matrices. By removing the sequence variables from the product
patterns, these pattern can be made syntactic when ignoring the multiplication's associativity.
In the following, we refer to the set of patterns with sequence variables as ``LinAlg``
and the set of syntactic product patterns as ``Syntactic``.

.. table This is the caption for the materials table. :label:`mtable`
   :class: w
   +-----------------------------+-----------------+----------+--------------------------+
   | Operation                   | Symbol          | Arity    | Properties               |
   +=============================+=================+==========+==========================+
   | Multiplication              | :math:`\times`  | variadic | associative              |
   +-----------------------------+-----------------+----------+--------------------------+
   | Addition                    | :math:`+`       | variadic | associative, commutative |
   +-----------------------------+-----------------+----------+--------------------------+
   | Transposition               | :math:`{}^T`    | unary    |                          |
   +-----------------------------+-----------------+----------+--------------------------+
   | Inversion                   | :math:`{}^{-1}` | unary    |                          |
   +-----------------------------+-----------------+----------+--------------------------+
   | Inversion and Transposition | :math:`{}^{-T}` | unary    |                          |
   +-----------------------------+-----------------+----------+--------------------------+


.. latex::
    :usepackage: booktabs

    \begin{table}
        \centering
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{l c c p{1.5cm}}
            \toprule
            \textbf{Operation} & \textbf{Symbol} & \textbf{Arity} & \textbf{Properties} \\
            \midrule
            Multiplication & $\times$ & variadic & associative \\
            Addition & $+$ & variadic & associative,\newline commutative \\
            Transposition & ${}^T$ & unary & \\
            Inversion & ${}^{-1}$ & unary & \\
            Inversion and Transposition & ${}^{-T}$ & unary & \\
            \bottomrule
        \end{tabular}
        \caption{Linear Algebra Operations}
    \label{tbl:laop}
    \end{table}

The subjects were randomly generated such that matrices had random properties and each factor could
randomly be transposed/inverted. The number of factors was chosen according to a normal
distribution with :math:`\mu = 5`. The total subject set consisted of 70 random products and 30 random sums.
Out of the pattern set, random subsets were used to examine the influence of the pattern set size on
the matching time. Across multiple subsets and repetitions per subject, the mean match and setup
times were measured. Matching was performed both with the ``match`` function and the
``ManyToOneMatcher`` (MTOM). The results are displayed in Figure :ref:`fig:linalgtime`.

.. figure:: linalg_times.pdf

   Timing Results for ``LinAlg``. :label:`fig:linalgtime`

As expected, both setup and match times grow with the pattern set size. The growth of the
many-to-one match time is much slower than the one for one-to-one matching. This is also expected,
because the simultaneous matching is more efficient. However, the growth of setup time for the
many-to-one matcher beckons the question whether the speedup of the many-to-one matching is worth it.

.. figure:: linalg_speed.pdf

   Comparison for ``LinAlg``. :label:`fig:linalgspeed`

Figure :ref:`fig:linalgspeed` depicts both the speedup and the break even point for many-to-one
matching for ``LinAlg``. The first graph indicates that the speedup of many-to-one matching
increases with larger pattern sets. But in order to fully profit from that speedup, the setup
cost of many-to-one matching must be amortized. Therefore, the second graph shows the break even
point for many-to-one matching in terms of number of subjects. If for a given number of patterns and
subjects the corresponding point is above the line, then many-to-one matching is overall faster.
In this example, when matching more than eight times, many-to-one matching is overall always faster
than one-to-one matching.

Syntactic
'''''''''

For the syntactic product patterns we compared the ``match`` function, the ``ManyToOneMatcher``
(MTOM) and the ``DiscriminationNet`` (DN). Again, randomly generated subjects were used. The
resulting speedups and break even points are displayed in Figure :ref:`fig:syntacticspeed`.

.. figure:: syntactic_speed.pdf

   Comparison for ``Syntactic``. :label:`fig:syntacticspeed`

In this case, the discrimination net is the fastest overall reaching a speedup of up to 60.
However, because it also has the highest setup time, it only outperforms the many-to-one matcher
after about 100 subjects for larger pattern set sizes. In practice, the discrimination net is likely
the best choice for syntactic patterns, as long as the discrimination net does not grow to large.
In the worst case, the size of the discrimination net can grow exponentially in the number of patterns.

Abstract Syntax Trees
.....................

Python includes a tool to convert code from Python 2 to Python 3.
It is part of the standard library package ``lib2to3`` which has a collection of "fixers" that each convert one of the incompatible cases.
To find matching parts of the code, those fixers use pattern matching on the abstract syntax tree (AST).
Such an AST can be represented in the MatchPy data structures.
We converted some of the patterns used by ``lib2to3`` both to demonstrate the generality of MatchPy and to evaluate the performance of many-to-one matching.
Because the fixers are applied one after another and can modify the AST after each match,
it would be difficult to use many-to-one matching for ``lib2to3`` in practice.

The following is an example of such a pattern:

.. code-block:: python

    power<
        'isinstance'
        trailer< '(' arglist< any ',' atom< '('
            args=testlist_gexp< any+ >
        ')' > > ')' >
    >

It matches an ``isinstance`` expression with a tuple as second argument. Its tree structure is
illustrated in Figure :ref:`fig:ast`. The corresponding fixer cleans up duplications generated by previous
fixers. For example :py:`isinstance(x, (int, long))` would be converted by another fixer into
:py:`isinstance(x, (int, int))`, which in turn is then simplified to :py:`isinstance(x, int)` by this fixer.

.. figure:: ast.pdf
   :scale: 80 %

   AST of the ``isinstance`` pattern. :label:`fig:ast`

Out of the original 46 patterns, 36 could be converted to MatchPy patterns. Some patterns could not
be converted, because they contain features that MatchPy does not support yet.
The features include negated subpatterns (e.g. :py:`not atom<'(' [any] ')'>`)
or subpatterns that allow an aritrary number of repetitions (e.g. :py:`any (',' any)+`).

Furthermore, some of the AST patterns contain alternative or optional subpatterns, e.g.
:py:`power<'input' args=trailer<'(' [any] ')'>>`. These features are also not directly supported
by MatchPy, but they can be replicated by using multiple patterns.
For those ``lib2to3`` patterns, all combinations of the alternatives were generated and added as invividual patterns.
This resulted in about 1200 patterns for the many-to-one matcher that completely cover the original 36 patterns.

For the experiments, we used a file that combines the examples from the unittests of ``lib2to3``
with about 900 non-empty lines. We compared the set of 36 patterns with the original matcher and
the 1200 patterns with the many-to-one matcher. A total of about 560 matches are found.
Overall, on average, our many-to-one matcher takes 0.7 seconds to find
all matches, while the matcher from ``lib2to3`` takes 1.8 seconds. This yields a speedup of
approximately 2.5. However, the construction of the many-to-one matcher takes 1.4
seconds on average. This time needs to be amortized before many-to-one matching pays off.
This is achieved once the AST gets sufficiently large, because at some point the speedup outweighs the setup cost.
The setup time can also mostly be eliminated by saving the many-to-one matcher to disk and loading it once required.

Compared the one-to-one matching implementation in MatchPy, the many-to-one matching achieves a speedup of about 60.
This is due to the fact that for any given subject less than 1% of patterns match.
When taking into account the setup time of the many-to-one matcher, this means that the break even point for it is at about 200 subjects.

..  setup 1.397398018220357
    matchpy 0.7200570708846341
    lib2to3 1.803501565011998
    Matches: 561
    Matcher patterns: 1203
    Converted: 36
    Original: 46

Conclusions
-----------

We have presented MatchPy, which is a pattern matching library for Python with support for sequence variables and associative/commutative functions.
This library includes algorithms and data structures for both one-to-one and many-to-one matching.
Because non-syntactic pattern matching is NP-hard, in the worst case the pattern matching will take exponential time.
Nonetheless, our experiments on real world examples indicate that many-to-one matching can give a significant speedup over one-to-one matching.
However, the employed discrimination nets come with a one-time construction cost.
This needs to be amortized before using them is faster than one-to-one matching.
In our experiments, the break even point for many-to-one matching was always reached well within the typical number of subjects for the respective application.
Therefore, many-to-one matching is likely to result in a compelling speedup in practice.

For syntactic patterns, we also compared the syntactic discrimination net with the many-to-one matcher.
As expected, discrimination nets are faster at matching, but also have a significantly higher setup time.
Furthermore, their number of states can grow exponentially with the number of patterns, making them unsuitable for some pattern sets.
Overall, if applicable, discrimination nets offer better performance than a many-to-one matcher.

Which pattern matching algorithm is the fastest for a given application depends on many factors.
Hence, it is not possible to give a general recommendation.
Yet, the more subjects are matched against the same pattern set, the more likely it is that many-to-one matching pays off.
A higher number of patterns seems to increase the speedup of the many-to-one matching.
In terms of the size of the many-to-one matcher, the growth of the net seems to be sublinear in practice.
The efficiency of using many-to-one matching also heavily depends on the actual pattern set, i.e. the degree of similarity and overlap between the patterns.

Future Work
-----------

We plan on extending MatchPy with more powerful pattern matching features to make it useful for an even wider range of applications.
The greatest challenge with additional features is likely to implement them for many-to-one matching.
In the following, we will discuss some possibilities for extending the library.

Additional pattern features
...........................

In the future, we plan to implement similar functionality to the ``Repeated``, ``Sequence``, and ``Alternatives`` functions from Mathematica.
These provide another level of expressive power which cannot be replicated with the current feature set of MatchPy's pattern matching.
Another useful feature are context variables as described by Kutsia :cite:`Kutsia2006`.
They allow matching subterms at arbitrary depths which is especially useful for structures like XML.
With context variables, MatchPy's pattern matching would be as powerful as XPath_ or `CSS selectors`_ for such structures.
Similarly, function variables that can match any function symbol would also be useful for those applications.

.. _XPath: https://www.w3.org/TR/2017/REC-xpath-31-20170321/
.. _`CSS selectors`: https://www.w3.org/TR/2017/NOTE-css-2017-20170131/

Integration
...........

Currently, in order to use MatchPy, any data structures must be adapted to inherit from the MatchPy expression classes.
Where that is not possible, for example because the data structures are provided by a third party library, translation functions need to be applied.
This also means that native Python data structures like lists or tuples cannot be used directly for the pattern matching.
In general, the inheritance-based pattern matching makes the integration of MatchPy into existing projects difficult.
Therefore, it would be useful, to have an abstraction that allows users to use their existing data structures with MatchPy.

In particular, easy integration with SymPy_ is an important goal, because it is a popular tool for working with symbolic mathematics.
SymPy already implements `a form of pattern matching <http://docs.sympy.org/0.7.2/tutorial.html#pattern-matching>`_ which is less powerful than MatchPy.
It lacks support for sequence variables, symbol wildcards and constraints.
While SymPy has predefined properties for symbols (e.g. a symbol can be an integer, non-negative, etc.),
it is not possible to add custom properties to symbols (e.g. matrix properties such as symmetric, triangular, etc.).
On the other hand, those properties in SymPy allow each invidual constant symbols to be commutative or non-commutative instead of everything within a certain function symbol.
One benefit of this approach is easier modeling of linear algebra multiplication, where matrices and vectors do not commute, but scalars do.
Better integration of MatchPy with SymPy would provide the users of SymPy with more powerful pattern matching tools.
However, Matchpy would required selective commutativity to be fully compatible with SymPy.

Performance
...........

If pattern matching is a major part of an application, its running time can significantly impact the overall speed.
Reimplementing parts of MatchPy as a C module would likely result in a substantial speedup.
Alternatively, adapting part of the code to Cython_ could be another option to increase the speed.
Furthermore, generating source code for a pattern set similar to parser generators for formal grammars could improve matching performance.
While code generation for syntactic pattern matching has been the subject of various works
:cite:`Augustsson1985,Fessant2001,Maranget2008,Moreau2003`, its application with the extended
feature set of MatchPy is another potential area of future research.

Functional pattern matching
...........................

Since Python does not have pattern matching as a language feature, MatchPy could be
extended to provide a syntax similar to other functional programming languages.
However, without a switch statement as part of the language, there is a limit to the syntax of this pattern expression.
The following is an example of what such a syntax could look like:

.. code-block:: python

   with match(f(a, b)):
       if case(f(x_, y_)):
           print("x={}, y={}".format(x, y)))
       elif case(f(z_)):
           ....

There are already several libraries for Python which implement such a functionality for syntactic
patterns and native data structures (e.g. MacroPy_, patterns_ or PyPatt_).
However, the usefulness of this feature needs further evaluation.

.. _MatchPy: https://github.com/HPAC/matchpy
.. _Cython: http://cython.org/
.. _SymPy: http://www.sympy.org/
.. _MacroPy: https://github.com/lihaoyi/macropy#pattern-matching
.. _patterns: https://github.com/Suor/patterns
.. _PyPatt: https://pypi.python.org/pypi/pypatt