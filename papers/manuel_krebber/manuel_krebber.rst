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
   Application domains include algebraic expressions, abstract syntax trees or XML and JSON data.
   Unfortunately, no implementations of pattern matching as general and flexible as in Mathematica exists for Python.
   Therefore, we created the open source module MatchPy_, which offers similar pattern matching functionality in Python.
   In addition, we implemented a novel algorithm which finds matches for large pattern sets more efficiently by exploiting similarities between patterns.

.. class:: keywords

   pattern matching, symbolic computation, discrimination nets, term rewriting systems

Introduction
------------

Pattern matching is a powerful tool which is part of many functional programming languages as well as computer algebra systems such as Mathematica.
It is useful for many applications including symbolic computation, term simplification, term rewriting systems, automated theorem proving, and model checking.
Term rewriting systems can be used with pattern matching to find matches for the rewrite rules and transform terms.
In functional programming languages, pattern matching enables a more readable and intuitive expression of algorithms.

Among the existing systems, Mathematica offers the most expressive pattern matching.
Its pattern matching offers similar expressiveness as regular expressions in Python, but for symbolic tree structures instead of strings.
Patterns are used widely in Mathematica, e.g. in function definitions or for manipulating expressions.
Users can define custom function symbols which can also be associative and/or commutative.
Mathematica also offers sequence variables which can match a sequence of expressions instead of a single expression.
They are especially useful when working with variadic function symbols.

There is currently no open source alternative to Mathematica with comparable pattern matching capabilities.
In particular, we are interested in similar pattern matching for an experimental linear algebra compiler written in Python.
Unfortunately, Mathematica is proprietary and nothing has been published on the underlying pattern matching algorithm.

Previous work predominantly covers syntactic pattern matching, i.e. associative/commutative/variadic
function symbols are not supported. Specifically, no existing work allows function symbols
which are either commutative or associative but not both. However, there are domains where
functions have those properties, e.g. matrix multiplication in linear algebra.
Most of the existing pattern matching libraries for Python only support syntactic patterns.
While the pattern matching in SymPy_ can work with associative/commutative functions, it is limited to finding a single match and does not support sequence variables.
However, we are interested in finding all possible matches for a pattern.

In many applications, a fixed set of patterns will be matched repeatedly against different subjects.
The simultaneous matching of multiple patterns is called many-to-one matching, as opposed to
one-to-one matching which denotes matching with a single pattern.
Many-to-one matching can be sped up by exploiting similarities between patterns.
This has already been the subject of research for both syntactic and AC pattern matching, but not with
the full feature set described above.
Discrimination nets are the state-of-the-art solution for many-to-one matching.
Our goal is to generalize this approach to support the aforementioned full feature set.

We implemented pattern matching with sequence variables and associative/commutative function symbols
as an open-source library for Python called MatchPy_. In addition to regular one-to-one matching,
this library also includes an efficient many-to-one matching algorithm that uses generalized discrimination nets.
In our experiments we observed significant speedups of the many-to-one matching over one-to-one matching.


Experiments
-----------

To evaluate the performance of MatchPy, we performed several experiments. All experiments were
conducted on an Intel Core i5-2500K 3.3 GHz CPU with 8GB of RAM.

Linear Algebra
..............

TODO

Syntactic
'''''''''

TODO

Abstract Syntax Trees
.....................

Python includes a tool to convert code from Python 2 to Python 3.
It is part of the standard library package ``lib2to3`` which has a collection of "fixers" that each convert one of the incompatible cases.
To find matching parts of the code, those fixers use pattern matching on the abstract syntax tree (AST).
Such an AST can be represented in the MatchPy data structures.
We converted some of the patterns used by ``lib2to3`` both to demonstrate the generality of MatchPy and to evaluate the performance of many-to-one matching.

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
Because the fixers are applied one after another and can modify the AST after each match,
it would be difficult to use many-to-one matching for ``lib2to3`` in practice.

..  setup 1.397398018220357
    matchpy 0.7200570708846341
    lib2to3 1.803501565011998
    Matches: 561
    Matcher patterns: 1203
    Converted: 36
    Original: 46

Conclusions and Future Work
---------------------------

Conclusions
...........

We have presented MatchPy, which is a pattern matching library for Python with support for sequence variables and associative/commutative functions.
This library includes algorithms and data structures for both one-to-one and many-to-one matching.
Because non-syntactic pattern matching is NP-hard, in the worst case the pattern matching will take exponential time.
Nonetheless, our experiments on real world examples indicate that many-to-one matching can give a significant speedup over one-to-one matching.
However, the employed discrimination nets come with a one-time construction cost which needs to be amortized before using them is faster than one-to-one matching.
In our experiments, the break even point for many-to-one matching was always reached well within the typical number of subjects for the respective application.
Therefore, many-to-one matching is likely to result in a compelling speedup in practice.

For syntactic patterns, we also compared the syntactic discrimination net with the many-to-one matcher.
As expected, discrimination nets are faster at matching, but also have a significantly higher setup time.
Furthermore, their number of states can grow exponentially with the number of patterns, making them unsuitable for some pattern sets.
Overall, for the limited cases where they are applicable, discrimination nets offer better performance than a many-to-one matcher.

Which pattern matching algorithm is the fastest for a given application depends on many factors.
Hence, it is not possible to give a general recommendation.
Yet, the more subjects are matched against the same pattern set, the more likely it is that many-to-one matching pays off.
A higher number of patterns seems to increase the speedup of the many-to-one matching.
In terms of the size of the many-to-one matcher, the growth of the net seems to be sublinear in practice.
The efficiency of using many-to-one matching also heavily depends on the actual pattern set, i.e. the degree of similarity and overlap between the patterns.

Future Work
...........

We plan on extending MatchPy with more powerful pattern matching features to make it useful for an even wider range of applications.
The greatest challenge with additional features is likely to implement them for many-to-one matching.
In the following, we will discuss some possibilities for extending the library.

Additional pattern features
'''''''''''''''''''''''''''

In the future, we plan to implement similar functionality to the ``Repeated``, ``Sequence``, and ``Alternatives`` functions from Mathematica.
These provide another level of expressive power which cannot be replicated with the current feature set of MatchPy's pattern matching.
Another useful feature are context variables as described by Kutsia :cite:`Kutsia2006`.
They allow matching subterms at arbitrary depths which is especially useful for structures like XML.
With context variables, MatchPy's pattern matching would be as powerful as XPath :cite:`Robie2017` or CSS selectors :cite:`Rivoal2017` for such structures.
Similarly, function variables that can match any function symbol would also be useful for those applications.

Integration
'''''''''''

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
'''''''''''

If pattern matching is a major part of an application, its running time can significantly impact the overall speed.
Reimplementing parts of MatchPy as a C module would likely result in a substantial speedup.
Alternatively, adapting part of the code to Cython_ could be another option to increase the speed.
Furthermore, generating source code for a pattern set similar to parser generators for formal grammars could improve matching performance.
While code generation for syntactic pattern matching has been the subject of various works
:cite:`Augustsson1985,Fessant2001,Maranget2008,Moreau2003`, its application with the extended
feature set of MatchPy is another potential area of future research.

Functional pattern matching
'''''''''''''''''''''''''''

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

There are already several libraries for Python which implement such a functionality for syntacitc
patterns and native data structures (e.g. MacroPy_, patterns_ or PyPatt_).
However, the usefulness of this feature needs further evaluation.

.. _MatchPy: https://github.com/HPAC/matchpy
.. _Cython: http://cython.org/
.. _SymPy: http://www.sympy.org/
.. _MacroPy: https://github.com/lihaoyi/macropy#pattern-matching
.. _patterns: https://github.com/Suor/patterns
.. _PyPatt: https://pypi.python.org/pypi/pypatt