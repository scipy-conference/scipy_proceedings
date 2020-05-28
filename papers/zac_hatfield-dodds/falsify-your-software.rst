:author: Zac Hatfield-Dodds
:email: zac.hatfield.dodds@anu.edu.au
:institution: Australian National University

:bibliography: references


-----------------------------------------------------------------------------
Falsify your Software: validating scientific code with property-based testing
-----------------------------------------------------------------------------

.. class:: abstract

    Where traditional example-based tests check software using manually-specified
    input-output pairs, property-based tests exploit a general description of
    valid inputs and program behaviour to automatically search for falsifying
    examples.  Given that Python has excellent property-based testing tools,
    such tests are often *easier* to work with and routinely find serious bugs
    that all other techniques had missed.

    I present four categories of properties relevant to most scientific projects,
    demonstrate how each found real bugs in Numpy and Astropy, and propose that
    property-based testing should be adopted more widely across the SciPy ecosystem.


.. class:: keywords

   property-based testing, validation, verification



Introduction
------------

Much research now depends on software for data collection, analysis, and reporting;
including on software produced and maintained by researchers.  This has empowered
us enormously: it is hard to imagine an analysis that was possible *at all* a
generation ago which could not be accomplished quickly by a graduate student today.

Unfortunately, neither raw computation nor the available software guarantee that
our results are meaningful [#]_.  Indeed, :cite:`Soergel2015` estimates that

    Any reported scientific result could very well be wrong if data have passed
    through a computer, and these errors may remain largely undetected [#]_. It is
    therefore necessary to greatly expand our efforts to validate scientific software.

This paper argues that property-based testing :cite:`PraisePBT` is considerably more
effective for validation of Python programs than using only traditional example-based
tests [#]_, without demanding new skills or workflows, and supports this argument
with a variety of examples from well-known scientific Python projects.


.. [#] There is a saying that any fool can make a mistake - but to make billions of
        mistakes per second, you need a computer.
.. [#] and indeed :cite:`NMRbug` reported such a bug, casting doubt on the results of
        around 160 papers.
.. [#] e.g. common workflows using ``pytest``.  If you have no automated tests at all,
        fix that first, but your second test could reasonably use Hypothesis.


What is property-based testing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where example-based tests check for an exact expected output, property-based tests
make *less precise* but *more general* assertions.  By giving up hand-specification
of the expected output, we gain tests that can be run on a wide range of inputs.

This generality also guides our tests to the right level of abstraction, and gives
clear design feedback: where example-based tests map one input to one output regardless
of complexity, every special case or interacting feature has to be addressed.  Clean
abstractions which allow you to say "for all ...", without caveats, stand in clear
contrast and are a pleasure to test.

Tests which use random data are usually property-based, but using a library like
Hypothesis :cite:`MacIver2019` which is designed for the task has several advantages:

- a concise and expressive interface to describe the range of allowed inputs
- test failures are cached and replayed - never lose a bug, even from other machines
- automatic shrinking, to present a minimal failing example for each distinct error

Automating these tedious tasks makes finding bugs considerably faster, and debugging
both easier and more fun.  When Hypothesis world-class shrinker :cite:`MacIver2020` [#]_
hands you a failing example, you *know* that every feature is relevant - if any integer
could be smaller (without the test passing) it would be; if any list could be shorter or
sorted (ditto) it would be, and so on.


.. [#] full paper and citation to be publically available soon



Why is this so effective?
~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples I wouldn't think of reveal bugs I didn't know were possible.**
It turns out that general descriptions of your data have several advantages over
writing out specific examples, and that these are even stronger for research code.

**Input descriptions are concise.**
Writing out a Hypothesis "strategy" [#]_ describing objects like a Numpy array or
a Pandas dataframe is often less code than a single instance, and clearly expresses
to readers what actually matters.  The sheer tedium of writing out representative
test cases, without resorting to literally random data, is a substantial deterrent
to testing data analysis code; with property-based testing you don't have to.

**The system is designed to find bugs.**
Hypothesis also comes packed with well-tuned heuristics and tools for finding bugs
which uniformly random data would almost never find - literal 'edge cases' in the
space of possible inputs.  In one memorable case a new user was testing coordinate
transformation logic, which fails for singular matrices (a set of measure zero).
Hypothesis knows nothing at all about matrices, or even the topic of the test, but
promptly generated a failing example anyway.

**Describe once, test everywhere.**
In a codebase with :math:`M` variations on the core data structures and :math:`N`
features, example-based tests have :math:`M \times N` tests to write - and it's
all too easy to forget to test the interaction of lesser-known features [#]_.
With property-based tests those :math:`M` variations can be designed into the
strategies which describe your data, scaling as :math:`M + N` and ensuring that
nothing is forgotten [#]_.

This scaling effect makes effective testing much easier for new contributors,
who do not need to consider all possible feature interactions - they will arise
naturally from the shared input descriptions.  For example, Hypothesis' Numpy
extension includes tools to describe arrays, array shapes including broadcasting
and generalised ufunc signatures, scalar and structured dtypes, and both basic
and advanced indexing.  Thinking carefully about what inputs *should* be supported
is usually a valuable exercise in itself!

**We face multiple sources of uncertainty.**
When experimental results come out weird, unpicking the unexpected behaviour of
your research domain from the possibility of equipment error or software bugs is
hard enough already.  Property-based tests let you verify more general behaviours
of your code, and focus on the domain rather than implementation details.


.. [#] Hypothesis calls its descriptions of valid inputs 'strategies' for historical reasons
.. [#] e.g. signalling NaNs, zero-dimensional arrays, structured Numpy dtypes with
        field titles in addition to names, etc.
.. [#] test 'fixture' systems scale similarly, but can only be as effective as the
        explicit list of inputs provided to them.



Properties and Case Studies
---------------------------

In this section I present four categories of properties.  While not an exhaustive
list, they are relevant to a very wide range of scientific software - and when
tested often uncover serious errors.

I also present case studies of real-world bugs [#]_ from the SciPy stack, especially
from foundational libraries like Numpy :cite:`oliphant2006guide` and Astropy
:cite:`astropy:2013` :cite:`astropy:2018`.  While seriously under-resourced
given their importance to essentially all research in their fields :cite:`NumpyFunding`
:cite:`AstropyProblem`, they are well-engineered and no more defect-prone than any
comparable software.  *If it can happen to them, it can certainly happen to you.*

The bugs presented below were each discovered, reported, and fixed within a few days
thanks to a community-driven and open source development model; and projects from
Astropy to Xarray - via Numpy and Pandas - have begun to adopt property-based tests.

.. [#] preferring those which can be demonstrated and explained in only a few lines,
        though we have found plenty more which cannot.



Outputs within expected bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many functions, the simplest property to check is that their output is within
some expected bound.  This can be physical, such as the bounds on temperature or
velocity, or it may be a logical or computational bound - such as probabilities
which must be in the range :code:`[0 1]`.

In cases such as the :code:`softmax` function, which is often used to convert a
vector of real numbers into a probability distribution, we also know that the
sum should always be (approximately) equal to one.  Let's implement this, as
described by the SciPy documentation [#]_:

.. code-block:: python

    from hypothesis import given, strategies as st
    import hypothesis.extra.numpy as npst
    finite = st.floats(allow_nan=False, allow_infinity=False)

    def softmax(x):
        return np.exp(x) / np.exp(x).sum()

    assert softmax(np.arange(5)).sum() == 1  # passes

    @given(npst.arrays(float, npst.array_shapes(), elements=finite))
    def test_softmax_property(arr):
        np.testing.assert_almost_equal(1, softmax(arr).sum())

While our example-based test passes for small arrays of small integers, the naive
algorithm is numerically unstable!  Our property-based test fails almost instantly,
showing us the minimal example of overflow with :code:`np.exp([710.])`.  If we
instead use :code:`np.exp(x - x.max())`, the test passes.

I will not argue that this kind of testing can substitute for numerical analysis,
but rather that it can easily be applied to routines which would otherwise not be
analysed at all.


A more sophisticated example of bounds testing comes from recent work in Astropy
[#]_, using Hypothesis to check that conversions between different time scales did
not unexpectedly lose precision [#]_.  Astropy contributors wrote custom strategies
to incorporate bias towards leap-seconds (unrepresentable in :code:`datetime.datetime`),
and an :code:`assert_almost_equal` helper which uses :code:`hypothesis.target()`
to guide the search process towards larger errors.

These tests found that round-trip conversions could be off by up to twenty
microseconds over several centuries due to loss of precision in
:code:`datetime.timedelta.total_seconds()`.  This effort also contributed to
improved error reporting around the 'threshold problem' [#]_, where a minimal
failing example *only just* fails and suggests that serious bugs may be subtle.


.. [#] e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
.. [#] merged in https://github.com/astropy/astropy/pull/10373
.. [#] as background, Python's builtin :code:`datetime.datetime` type represents time as a
        tuple of integers for year, month, ..., seconds, microseconds; and assumes UTC and
        the current Gregorian calendar extended in both directions.  By contrast
        :code:`astropy.time.Time` represents time with a pair of 64-bit floats; supports a
        variety of civil, geocentric, and barycentric time scales; and maintains sub-nanosecond
        precision over times spanning the age of the universe - bugs excepted!
.. [#] see https://hypothesis.works/articles/threshold-problem/ and
        https://github.com/HypothesisWorks/hypothesis/pull/2393



Round-trip properties
~~~~~~~~~~~~~~~~~~~~~

Whenever you have a pair of inverse functions, think of round-trip testing.
If you save and then load data, did you lose any?  If you convert between two
formats, or coordinate systems, can you convert back?

These properties are remarkably easy to test, vitally important, and often catch
subtle bugs due to the complex systems interactions.  It is often worth investing
considerable effort to describe *all* valid data, so that examples can be generated
with very rare feature combinations.

If you write only one test based on this paper, *try to save and load any valid data*.

I have consistently found testing IO round-trips to be among the easiest and most
rewarding property tests I write.  My own earliest use of Hypothesis came after almost
a month trying to track down data corruption issues in multi-gigabyte PLY files.
Within a few hours I wrote a strategy to generate PLY objects, executed the test,
and discovered that our problems were due to mishandling of whitespace in the file
header [#]_.

Even simple tests are highly effective though - consider as an example

.. code-block:: python

    @given(st.text(st.characters()).map(lambda s: s.rstrip("\x00")))
    def test_unicode_arrays_property(string):
        assert string == np.array([string])[0]

This is a more useful test that it might seem: after working around null-termination
of strings, we can still detect a variety of issues with length-aware dtypes, Unicode
version mismatches, or string encodings.  A very similar test did in fact find an
encoding error [#]_, which was traced back to a deprecated - and promptly removed -
compatibility workaround to support 'narrow builds' of Python 2.


.. [#] https://github.com/dranjan/python-plyfile/issues/9
.. [#] https://github.com/numpy/numpy/issues/15363



Differential testing
~~~~~~~~~~~~~~~~~~~~

Running the same input through your code and through a trusted - or simply
different - implementation is another widely applicable property: any difference
in the outputs indicates that *at least* one of them has a bug.  Common sources
of alternative implementations include:

**Another project or language.** If you aim to duplicate functionality from an
existing project, you can check that your results are identical for whatever
overlap exists in the features of the two projects.  This might involve
cross-language comparisons, or be as simple as installing an old version of
your code from before a significant re-write.

**A toy or brute-force implementation** which only works for small inputs might
be out of the question for 'production' use, but can nonetheless be useful for
testing.  This is especially useful when there is a technique so simple that
there are "obviously no bugs", but an optimised version has "no obvious bugs".

**Varying irrelevant parameters** such as performance hints which are not expected
to affect the calculated result.  Combining this and the previous tactic, try
comparing single-threaded vs. multi-threaded mode - while some care is required
to ensure determinism it is often worth the effort.


As our demonstration, consider the :code:`numpy.einsum` function and two tests.
The example-based test comes from the Numpy test suite; and the property-based
test is a close translation - it still requires two-dimensional arrays, but
allows the shapes and contents to vary.  Note that both are differential tests!

.. code-block:: python

    def test_einsum_example():
        p, q = np.ones(shape=(10, 2)), np.ones(shape=(1, 2))
        assert_array_equal(
            np.einsum("ij,ij->j", p, q, optimize=True),
            np.einsum("ij,ij->j", p, q, optimize=False)
        )

    @given(
        data=st.data(),
        dtype=npst.integer_dtypes() | npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=2, max_dims=2),
    )
    def test_einsum_property(data, dtype, shape):
        p = data.draw(npst.arrays(dtype, shape))
        q = data.draw(npst.arrays(dtype, shape))
        assert_array_equal( ... )  # as above

When an optimisation to avoid dispatching to :code:`numpy.tensordot` over a
dimension of size one was added, the example-based test kept passing - despite the
bug :code:`if 1 in operands[n]` instead of :code:`if 1 in operands[n].shape` [#]_.
This bug could only be triggered with :code:`optimize=True` and an input array
with a dimension of size one, *xor* containing the integer :code:`1`.
This kind of interaction is where property-based testing really shines.


There's another twist to this story though: the bug was actually identified
downstream of Numpy, when Ryan Soklaski was testing that :code:`Tensor`\ s from
his auto-differentiation library :code:`MyGrad` :cite:`MyGrad` were in fact
substitutable for Numpy arrays [#]_.  He later said of property-based tests that [#]_

    It would have been impossible for me to implement a trustworthy autograd library
    for my students to learn from and contribute to if it weren't for Hypothesis.


.. [#] https://github.com/numpy/numpy/issues/10930
.. [#] making :code:`test_einsum_property` a differential test derived from a
        derivative auto-differentiator.
.. [#] https://github.com/HypothesisWorks/hypothesis/issues/1641



Metamorphic properties
~~~~~~~~~~~~~~~~~~~~~~

A serious challenge when testing research code is that the correct result may
be genuinely unknown - and running the shiny new simulation or analysis code is
the only way to get any result at all.  One very powerful solution is to compare
several input-output pairs, instead of attempting to analyse one in isolation:

    A test oracle determines whether a test execution reveals a fault, often by
    comparing the observed program output to the expected output. This is not always
    practical... Metamorphic testing provides an alternative, where correctness is
    not determined by checking an individual concrete output, but by applying a
    transformation to a test input and observing how the program output “morphs”
    into a different one as a result.
    :cite:`MTsurvey`


Let's return to :code:`softmax` as an example.  We can state general properties
about a single input-output pair such as "all elements of the output are between
zero and one", or "the sum of output elements is approximately equal to one" [#]_.
A metamorphic property we could test is scale-invariance: multiplying the input
elements by a constant factor should leave the output approximately unchanged.

.. code-block:: python

    @given(arr=..., factor=st.floats(-1000, 1000))
    def test_softmax_metamorphic_property(arr, factor):
        result, scaled = softmax(arr), softmax(arr * factor)
        np.testing.assert_almost_equal(result, scaled)


Astropy's tests for time precision include metamorphic as well as round-trip properties:
several assert that given a :code:`Time`, adding a tiny timedelta then converting it to
another time scale is almost equal to converting and then adding.

Metamorphic properties based on domain knowledge are particularly good for testing
"untestable" code.  In bioinformatics, :cite:`MTbioinformatics` presents testable
properties for gene regulatory networks and short sequence mapping [#]_, and found a bug
attributable to the *specification* - not just implementation errors.
METTLE :cite:`METTLE` proposes eleven generic metamorphic properties for unsupervised
machine-learning sytems [#]_, and studies their use as an aid to end-users selecting
an appropriate algorithm for their domain (LIDAR, DNA sequencing, etc.).


.. [#] or even :code:`np.argsort(arr) == np.argsort(softmax(arr))`
.. [#] e.g. K-means clustering
.. [#] for which :code:`hypothesis-bio` :cite:`hypothesis_bio` provides many
        useful data-generation strategies



Conclusion
----------

Example-based tests provide anecdotal evidence for validity, in that the software
behaves as expected on a few known and typically simple inputs.  Property-based
tests require a precise description of possible inputs and a more general
specification, but then automate the search for falsifying counter-examples.
They are quick to write, convenient to work with, and routinely find serious
bugs that all other techniques had missed.

I argue that this Popperian approach is superior to the status quo of using only
example-based tests, and hope that the property-based revolution comes quickly.


Acknowledgements
----------------

Thanks to David MacIver and the many others who have contributed to Hypothesis;
to Ryan Soklaski and Kathy Reid for their comments on an early draft of this paper;
to Anne Archibald for her work with threshold tests; and to the many mantainers
of the wider Python ecosystem.  None of this would be possible without you.
