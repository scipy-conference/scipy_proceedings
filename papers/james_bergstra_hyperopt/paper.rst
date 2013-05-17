:author: James Bergstra
:email: james.bergstra@uwaterloo.ca
:institution: University of Waterloo

:author: Dan Yamins
:email: yamins@mit.edu
:institution: Massachusetts Institute of Technology

:author: David D. Cox
:email: davidcox@fas.harvard.edu
:institution: Harvard University


-------------------------------------------------------------------------------------------
Hyperopt: A Python Library for Optimizing the Hyperparamters of Machine Learning Algorithms
-------------------------------------------------------------------------------------------

.. class:: abstract

    Model-based optimization (i.e. based on Bayesian optimization) is one of the most efficient
    methods for optimizing expensive functions over awkward spaces.
    The Hyperopt library (Hyperopt) provides model-based optimization in Python.
    Hyperopt provides an implementation of the TPE algorithm, which has proven to be an effective strategy for optimizing neural networks, deep networks, and computer vision systems.
    Hyperopt supports parallel and asynchronous evaluation of hyperparameter configurations via its MongoDB backend.
    This paper describes the usage and architecture of hyperopt.

.. class:: keywords

    Bayesian optimization, hyperparameter optimization, model selection


Introduction
------------

Sequential model-based optimization (SMBO, also known as Bayesian optimization) is a general technique for function optimization that includes some of the most
call-efficient (in terms of function evaluations) optimization methods currently available.
Originally developed for experiment design (and oil exploration, [Mockus78]_) SMBO methods are generally applicable to scenarios in which a user wishes to minimize some scalar-valued function :math:`f(x)` that is costly to evaluate, often in terms of time or money.
Compared with standard optimization strategies such as conjugate gradient descent methods, model-based optimization algorithms invest more time between function evaluations in order to reduce the number of function evaluations overall.

The advantages of SMBO are that it:

* leverages smoothness without analytic gradient,

* handles real-valued, discrete, and conditional variables,

* handles parallel evaluations of :math:`f(x)`,

* copes with hundreds of variables, even with budget of just a few hundred function evaluations.


Many widely-used machine learning algorithms take a significant amount of time to train from data.
At the same time, these same algorithms must be configured prior to training.
These configuration variables are called *hyperparameters*.
For example, Support Vector Machines (SVMs) have hyperparameters that include the regularization strength (often :math:`C`) the scaling of input data
(and more generally, the preprocessing of input data), the choice of similarity kernel, and the various parameters that are specific to that kernel choice.
Decision trees are another machine learning algorithm with hyperparameters related to the heuristic for creating internal nodes, and the pruning strategy for the tree after (or during) training.
Neural networks are a classic type of machine learning algorithm but they have so many hyperparameters that they have been considered too troublesome for inclusion in the sklearn library.

Generally speaking, no matter what algorithm has been chosen (i.e. SVM, decision tree, neural network, or something else), hyperparameters have a significant
effect on the success of the training algorithm.
A poorly-configured SVM may perform no better than chance, while a well-configured one may achieve state-of-the-art prediction accuracy.
To experts and non-experts alike, adjusting hyperparameters to optimize end-to-end performance can be a daunting task.
Hyperparameters come in many varieties--continuous-valued ones with and without bounds, discrete ones that are either ordered or not, and conditional ones that do not even always apply
(e.g., the parameters of an optional pre-processing stage).
Because of this variety, conventional continuous and combinatorial optimization algorithms either do not directly apply,
or else operate without leveraging valuable structure in the configuration space.
Common practice for the optimization of hyperparameters is
(a) for algorithm developers to tune them by hand on representative problems to get good rules of thumb and default values,
and (b) for algorithm users to tune them manually for their particular prediction problems, perhaps with the assistance of [multiresolution] grid search.
However, when dealing with more than a few hyperparameters (e.g. 5) this standard practice of manual search with grid refinement is not guaranteed to work well;
in such cases even random search has been shown to be competitive with domain experts [BB12]_.

Hyperopt [Hyperopt]_ provides algorithms and software infrastructure for carrying out hyperparameter optimization for machine learning algorithms.
Hyperopt provides an optimization interface that distinguishes a *configuration space* and an *evaluation function* that assigns real-valued
*loss values* to points within the configuration space.
Unlike the fmin interface in [SciPy]_ or [Matlab]_, Hyperopt's fmin interface requires users to specify the configuration space as a probability distribution.
Specifying a probability distribution rather than just bounds and hard constraints allows domain experts to encode more of their intuitions
regarding which values are plausible for various hyperparameters.
Like SciPy's new fmin interface, hyperopt makes the SMBO algorithm itself an interchangeable component, so it is easy for a user to search a specific
space using any of the provided SMBO implementations. Currently just two algorithms are provided --random search and Tree-of-Parzen-Estimators (TPE) --
but more algorithms are planned: [SMAC]_, [ROAR]_, and Gaussian-process-based ones such as [Brochu10]_ and [SLA13]_.

We are motivated to make hyperparameter optimization more reliable for four reasons:

**Reproducibile research**
    Hyperopt formalizes the practice of model evaluation, so that benchmarking experiments can be reproduced at later dates, and by different people.

**Empowering users**
    Learning algorithm designers can deliver flexible fully-configurable implementations to non-experts (e.g. deep learning systems), so long as they also provide a corresponding Hyperopt driver.

**Designing better algorithms**
    As algorithm designers, we appreciate Hyperopt's capacity to find successful configurations that we might not have considered.

**Fuzz testing**
    As algorithm designers, we appreciate Hyperopt's capacity to find failure modes via configurations that we had not considered.

This paper describes the usage and architecture of Hyperopt, for both sequential and parallel optimization of expensive functions.
Hyperopt can in principle be used for any SMBO problem, but our development and testing efforts have been limited so far to the optimization of
hyperparameters for neural networks [XXX]_, deep networks [XXX]_, and computer vision systems for object recognition [XXX]_.


Getting Started with Hyperopt
-----------------------------

This section introduces basic usage of the ``hyperopt.fmin`` function, which is hyperopt's basic optimization driver. 
We will look at how to write an objective function that ``fmin`` can optimize, and how to describe a configuration space that ``fmin`` can search.

Hyperopt shoulders the responsibility of finding the best value of a scalar-valued,
possibly-stochastic function over a set of possible arguments to that function.
Whereas most optimization packages assume that these inputs are drawn from a vector space,
Hyperopt encourages you, the user, to describe your configuration space in more detail.
Hyperopt is typically aimed at very difficult search settings, especially ones with many hyperparameters and a small budget for function evaluations.
By providing more information about where your function is defined, and where you think the best values are,
you allow algorithms in hyperopt to search more efficiently.

The way to use hyperopt is to describe:

* the objective function to minimize
* the space over which to search
* a trials database [optional]
* the search algorithm to use [optional]

This section will explain how to describe the objective function, configuration space, and optimization algorithm.
Section XXX below will explain how to use a non-default trials database to analyze the results of a search,
and to make parallel search possible.


Step 1: define an objective function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperopt provides a few levels of increasing flexibility / complexity when it comes to specifying an objective function to minimize.
In the simplest case, an objective function is a Python function that accepts a single argument that stands for :math:`x` (which can be an arbitrary object),
and returns a single scalar value that represents the *loss* (:math:`f(x)`) incurred by that argument.

So for a trivial example, if we want to minimize a quadratic function :math:`q(x, y) := x^2 + y^2` then we could define our objective ``q`` as follows:

.. code-block:: python

    def q(args):
        x, y = args
        return x ** 2 + y ** 2

Although hyperopt accepts objective functions that are more complex in both the arguments they accept and their return value,
we will use this simple calling and return convention for the next few sections that introduce configuration spaces, optimization algorithms, and basic usage
of the fmin interface.
Later, as we explain how to use the Trials object to analyze search results, and how to search in parallel with a cluster,
we will introduce different calling and return conventions.

Step 2: define a configuration space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *configuration space* object describes the domain over which hyperopt is allowed to search.
If we want to search :math:`q` over values of :math:`x \in [0, 1]`, and values of :math:`y \in {\mathbb R}` ,
then we can write our search space as:

.. code-block:: python

    from hyperopt import hp

    space = [hp.uniform('x', 0, 1), hp.normal('y', 0, 1)]

Note that for both :math:`x` and :math:`y` we have specified not only the hard bound constraints, but also
we have given hyperopt an idea of what range of values for :math:`y` to prioritize. 


Step 3: choose a search algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choosing the search algorithm is currently as simple as passing ``algo=hyperopt.tpe.suggest`` or ``algo=hyperopt.rand.suggest``
as a keyword argument to ``hyperopt.fmin``.
To use random search to our search problem we can type:

.. code-block:: python

    from hyperopt import hp, fmin, rand, tpe, space_eval
    best = fmin(q, space, algo=rand.suggest)
    print best
    # =>  XXX
    print space_eval(space, best)
    # =>  XXX

    best = fmin(q, space, algo=tpe.suggest)
    print best
    # =>  XXX
    print space_eval(space, best)
    # =>  XXX


The search algorithms are global functions which may generally have extra keyword arguments
that control their operation beyond the ones used by ``fmin`` (they represent hyper-hyper-parameters!).
The intention is that these hyper-hyper-parameters are set to default that work for a range of configuration problems,
but if you wish to change them you can do it like this:

.. code-block:: python

    from functools import partial
    from hyperopt import hp, fmin, tpe
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(q, space, algo=algo)
    print best
    # =>  XXX


In a nutshell, these are the steps to using hyperopt.
Implement an objective function that maps configuration points to a real-valued loss value,
define a configuration space of valid configuration points,
and then call ``fmin`` to search the space to optimize the objective function.
The remainder of the paper describes
(a) how to describe more elaborate configuration spaces,
especially ones that enable more efficient search by expressing *conditional variables*,
(b) how to analyse the results of a search as stored in a ``Trials`` object,
and (c) how to use a cluster of computers to search in parallel.



Configuration Spaces
--------------------

Part of what makes Hyperopt a good fit for optimizing machine learning hyperparameters is that
it can optimize over general Python objects, not just e.g. vector spaces.
Consider the simple function ``w`` below, which optimizes over dictionaries with "type" and either "x" and "y" keys:

.. code-block:: python

    def w(pos):
        if pos['use_var'] == 'x':
            return pos['x'] ** 2
        else:
            return math.exp(pos['y'])

To be efficient about optimizing ``w`` we must be able to
(a) describe the kinds of dictionaries that ``w`` requires and
(b) correctly associate ``w``'s return value to the elements of ``pos`` that actually contributed to that return value.
Hyperopt's configuration space description objects address both of these requirements.
This section describes the nature of configuration space description objects,
and how the description language can be extended with new expressions,
and how the ``choice`` expression supports the creation of *conditional variables* that support
efficient evaluation of structured search spaces of the sort we need to optimize ``w``.


Configuration space primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A search space is a stochastic expression that always evaluates to a valid input argument for your objective function.
A search space consists of nested function expressions.
The stochastic expressions are the hyperparameters.
(Random search is implemented by simply sampling these stochastic expressions.)

The stochastic expressions currently recognized by hyperopt's optimization algorithms are in the ``hyperopt.hp`` module.
The simplest kind of search spaces are ones that are not nested at all.
For example, to optimize the simple function ``q`` (defined above) on the interval :math:`[0, 1]`, we could type
``fmin(q, space=hp.uniform('a', 0, 1))``.

The first argument to ``hp.uniform`` here is the *label*. Each of the hyperparameters in a configuration space must be labeled like this
with a unique string.  The other hyperparameter distributions at our disposal as modelers are as follows:

``hp.choice(label, options)``
    Returns one of the options, which should be a list or tuple.  The elements of `options` can themselves be [nested] stochastic expressions.  In this case, the stochastic choices that only appear in some of the options become *conditional* parameters.

``hp.pchoice(label, options, probs)``
    Return one of the options according to the probabilities listed in ``probs`` (which should sum to 1).

``hp.uniform(label, low, high)``
    Draws uniformly between ``low`` and ``high``.
    When optimizing, this variable is constrained to a two-sided interval.

``hp.quniform(label, low, high, q)``
    Drawn by ``round(uniform(low, high) / q) * q``,
    Suitable for a discrete value with respect to which the objective is still somewhat smooth.

``hp.loguniform(label, low, high)``
    Drawn by ``exp(uniform(low, high))``.
    When optimizing, this variable is constrained to the interval :math:`[e^{\text{low}}, e^{\text{high}}]`.

``hp.qloguniform(label, low, high, q)``
    Drawn by ``round(exp(uniform(low, high)) / q) * q``.
    Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the increasing size of the value.

``hp.normal(label, mu, sigma)``
    Draws a normally-distributed real value.
    When optimizing, this is an unconstrained variable.

``hp.qnormal(label, mu, sigma, q)``
    Drawn by ``round(normal(mu, sigma) / q) * q``.
    Suitable for a discrete variable that probably takes a value around mu, but is technically unbounded.

``hp.lognormal(label, mu, sigma)``
    Drawn by ``exp(normal(mu, sigma))``.
    When optimizing, this variable is constrained to be positive.

``hp.qlognormal(label, mu, sigma, q)``
    Drawn by ``round(exp(normal(mu, sigma)) / q) * q``.
    Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is non-negative.

``hp.randint(label, upper)``
    Returns a random integer in the range :math:`[0, upper)`.
    In contrast to ``quniform``
    optimization algorithms should assume *no* additional correlation in the loss function between nearby integer values,
    as compared with more distant integer values (e.g. random seeds).


Structure in configuration spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search spaces can also include lists, and dictionaries.
Using these containers make it possible for a search space to include multiple variables (hyperparameters).
The following code fragment illustrates the syntax:

.. code-block:: python

    from hyperopt import hp

    list_space = [
        hp.uniform('a', 0, 1),
        hp.loguniform('b', 0, 1)]

    tuple_space = (
        hp.uniform('a', 0, 1),
        hp.loguniform('b', 0, 1))

    dict_space = {
        'a': hp.uniform('a', 0, 1),
        'b': hp.loguniform('b', 0, 1)}

There should be no functional difference between using list and tuple syntax to describe a sequence of elements in a configuration space,
but both syntaxes are supported for everyone's convenience.

Creating list, tuple, and dictionary spaces as illustrated above is just one example of nesting. Each of these container types can be nested
to form deeper configuration structures:

.. code-block:: python

    nested_space = [
        [ {'case': 1, 'a': hp.uniform('a', 0, 1)},
          {'case': 2, 'b': hp.loguniform('b', 0, 1)}],
        'extra literal string',
        hp.randint('r', 10) ]

There are no requirement that list elements have some kind of similarity, each element can be any valid configuration expression.
Note that Python values (e.g. numbers, strings, and objects) can be embedded in the configuration space.
These values will be treated as constants from the point of view of the optimization algorithms, but they will be included
in the configuration argument objects passed to the objective function.


Sampling from a configuration space by hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous few code fragments have defined various configuration spaces.
These spaces are not objective function arguments yet, they are simply a description of *how to sample* objective function arguments.
You can use the routines in ``hyperopt.pyll.stochastic`` to sample values from these configuration spaces.

.. code-block:: python

    from hyperopt.pyll.stochastic import sample

    print sample(list_space)
    # => [0.13, .235]

    print sample(nested_space)
    # => [[{'case': 1, 'a', 0.12}, {'case': 2, 'b': 2.3}],
    #     'extra_literal_string',
    #     3]

Note that the labels of the random configuration variables have no bearing on the sampled values themselves,
the labels are only used internally by the optimization algorithms.
Later when we look at the ``trials`` parameter to fmin we will see that the labels are used for analyzing
search results too.
For now though, simply note that the labels are not for the objective function.



Deterministic expressions in configuration spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to include deterministic expressions within the description of a configuration space.
For example, we can write

.. code-block:: python

    from hyperopt.pyll import scope

    def foo(x):
        return str(x) * 3

    expr_space = {
        'a': 1 + hp.uniform('a', 0, 1),
        'b': scope.minimum(hp.loguniform('b', 0, 1), 10),
        'c': scope.call(foo, args=(hp.randint('c', 5),)),
        }

The ``hyperopt.pyll`` submodule implements an expression language that stores
this logic in a symbolic representation.
Significant processing can be carried out by these intermediate expressions.
In fact, when you call ``fmin(f, space)``, your arguments are quickly combined into
a single objective-and-configuration evaluation graph of the form:
``scope.call(f, space)``.
Feel free to move computations between these intermediate functions and the final
objective function as you see fit in your application.

You can add new functions to the ``scope`` object with the ``define`` decorator:

.. code-block:: python

    from hyperopt.pyll import scope

    @scope.define
    def foo(x):
        return str(x) * 3

    # -- this will print "000"; foo is called as usual.
    print foo(0)

    expr_space = {
        'a': 1 + hp.uniform('a', 0, 1),
        'b': scope.minimum(hp.loguniform('b', 0, 1), 10),
        'c': scope.foo(hp.randint('cbase', 5)),
        }

    # -- this will draw a sample by running foo(x)
    #    on a random integer x.
    print sample(expr_space)

Read through ``hyperopt.pyll.base`` and ``hyperopt.pyll.stochastic`` to see the
functions that are available, and feel free to add your own.
One important caveat is that functions used in configuration space descriptions
must be picklable in order to be compatible with parallel search (discussed below).


Defining conditional variables with ``choice`` and ``pchoice``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having introduced nested configuration spaces, it is worth coming back to the ``hp.choice`` and ``hp.pchoice`` hyperparameter types.
An ``hp.choice(label, options)`` hyperparameter *chooses* one of the options that you provide, where the ``options`` must be a list.
We can use ``choice`` to define an appropriate configuration space for the ``w`` objective function (introduced pg XXX).

.. code-block:: python

    w_space = hp.choice('case', [
        {'use_var': 'x', 'x': hp.normal('x', 0, 1)},
        {'use_var': 'y', 'y': hp.uniform('y', 1, 3)}])

    print sample(w_space)
    # ==> {'use_var': 'x', 'x': -0.89}

    print sample(w_space)
    # ==> {'use_var': 'y', 'y': 2.63}

Recall that in ``w``, the "y" key of the configuration is not used if the "use_var" value is "x".
Similarly, the "x" key of the configuration is not used if the "use_var" value is "y". 
The use of ``choice`` in the ``w_space`` search space reflects the conditional usage of keys "x" and "y" in the ``w`` function.
We have used the ``choice`` variable to define a space that never has more variables than is necessary.

The choice variable here plays more than the a cosmetic role, it can make optimization much more efficient.
In terms of ``w`` and ``w_space``, the choice node prevents ``y`` for being *blamed* for poor performance when "use_var" is "x",
or *credited* for good performance when "use_var" is "x".
The choice variable creates a special node in the expression graph that prevents the conditionally un-necessary part of the
expression graph from being evaluated at all.
During optimization, similar special-case logic prevents any association between the return value of the objective function
and irrelevant hyperparameters (ones that were not chosen, and hence not involved in the creation of the configuration passed to the objective function).

The ``hp.pchoice`` hyperparameter constructor is similar to ``choice`` except that we can provide a list of probabilities
corresponding to the options, so that random sampling chooses some of the options more often than others.

.. code-block:: python

    w_space_with_probs = hp.pchoice('case', [
        (0.8, {'use_var': 'x',
               'x': hp.normal('x', 0, 1)}),
        (0.2, {'use_var': 'y',
               'y': hp.uniform('y', 1, 3)})])

Using the ``w_space_with_probs`` configuration space expresses to fmin that we believe the first case (using "x") is five times as likely to yield an optimal configuration that the second case.
If your objective function only uses a subset of the configuration space on any given evaluation, then you should
use ``choice`` or ``pchoice`` hyperparameter variables to communicate that pattern of inter-dependencies to ``fmin``.


Sharing a configuration variable across choice branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using choice variables to divide a configuration space into many mutually exclusive possibilities,
it can be natural to re-use some configuration variables across a few of those possible branches.
Hyperopt's configuration space supports this in a natural way, by allowing the objects to appear in multiple places within
a nested configuration expression. For example, if we wanted to add a ``randint`` choice to the returned dictionary
that did not depend on the "use_var" value, we could do it like this:

.. code-block:: python

    c = hp.randint('c', 10)

    w_space_c = hp.choice('case', [
        {'use_var': 'x',
         'x': hp.normal('x', 0, 1),
         'c': c},
        {'use_var': 'y',
         'y': hp.uniform('y', 1, 3),
         'c': c}])


Optimization algorithms in hyperopt would see that ``c`` is used regardless of the outcome of the ``choice`` value,
so they would correctly associate ``c`` with all evaluations of the objective function. 



Configuration Example: ``sklearn`` classifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see several configuration space description techniques in action,
let's look at how one might go about describing the space of hyperparameters of classification algorithms in [sklearn]_.

.. code-block:: python

    from hyperopt import hp
    space = hp.choice('classifier_type', [
        {
            'type': 'naive_bayes',
        },
        {
            'type': 'svm',
            'C': hp.lognormal('svm_C', 0, 1),
            'kernel': hp.choice('svm_kernel', [
                {'ktype': 'linear'},
                {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
                ]),
        },
        {
            'type': 'dtree',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dtree_max_depth',
                [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
            'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
        },
        ])


At the top level we have a ``choice`` between


Advanced Configuration Spaces
-----------------------------

XXX Advanced:
The hyperparameter optimization algorithms work by replacing normal "sampling" logic with
adaptive exploration strategies, which make no attempt to actually sample from the distributions specified in the search space.



2.3 Adding Non-Stochastic Expressions with pyll
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Using a non-Python Evaluation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are basically two ways to interface hyperopt with other languages: 

1. you can write a Python wrapper around your cost function that is not written in Python, or 
2. you can replace the `hyperopt-mongo-worker` program and communicate with MongoDB directly using JSON.

The easiest way to use hyperopt to optimize the arguments to a non-python function, such as for example an external executable, is to write a Python function wrapper around that external executable. Supposing you have an executable `foo` that takes an integer command-line argument `--n` and prints out a score, you might wrap it like this:

.. code-block:: python

    import subprocess
    def foo_wrapper(n):
        # Optional: write out a script for the external executable
        # (we just call foo with the argument proposed by hyperopt)
        proc = subprocess.Popen(['foo', '--n', n], stdout=subprocess.PIPE)
        proc_out, proc_err = proc.communicate()
        # <you might have to do some more elaborate parsing of foo's output here>
        score = float(proc_out)
        return score

Of course, to optimize the `n` argument to `foo` you also need to call hyperopt.fmin, and define the search space. I can only imagine that you will want to do this part in Python.

.. code-block:: python

    from hyperopt import fmin, hp, random

    best_n = fmin(foo_wrapper, hp.quniform('n', 1, 100, 1), algo=random.suggest)

    print best_n

When the search space is larger than the simple one here, you might want or need the wrapper function to translate its argument into some kind of configuration file/script for the external executable.

This approach is perfectly compatible with MongoTrials.


The Trials Object
-----------------

The simplest protocol for communication between hyperopt's optimization
algorithms and your objective function, is that your objective function
receives a valid point from the search space, and returns the floating-point
*loss* (aka negative utility) associated with that point.


.. code-block:: python

    from hyperopt import fmin, tpe, hp
    best = fmin(fn=lambda x: x ** 2,
        space=hp.uniform('x', -10, 10),
        algo=tpe.suggest,
        max_evals=100)
    print best


This protocol has the advantage of being extremely readable and quick to
type. As you can see, it's nearly a one-liner.
The disadvantages of this protocol are
(1) that this kind of function cannot return extra information about each evaluation into the trials database, and
(2) that this kind of function cannot interact with the search algorithm or other concurrent function evaluations.
You will see in the next examples why you might want to do these things.


1.2 Attaching Extra Information via the Trials Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your objective function is complicated and takes a long time to run, you will almost certainly want to save more statistics
and diagnostic information than just the one floating-point loss that comes out at the end.
For such cases, the fmin function is written to handle dictionary return values.
The idea is that your loss function can return a nested dictionary with all the statistics and diagnostics you want.
The reality is a little less flexible than that though: when using mongodb for example,
the dictionary must be a valid JSON document.
Still, there is lots of flexibility to store domain specific auxiliary results.

When the objective function returns a dictionary, the fmin function looks for some special key-value pairs
in the return value, which it passes along to the optimization algorithm.
There are two mandatory key-value pairs:
* `status` - one of the keys from `hyperopt.STATUS_STRINGS`, such as 'ok' for successful completion, and 'fail' in cases where the function turned out to be undefined.
* `loss` - the float-valued function value that you are trying to minimize, if the status is 'ok' then this has to be present.

The fmin function responds to some optional keys too:

* `attachments` -  a dictionary of key-value pairs whose keys are short strings (like filenames) and whose values are potentially long strings (like file contents) that should not be loaded from a database every time we access the record. (Also, MongoDB limits the length of normal key-value pairs so once your value is in the megabytes, you may *have* to make it an attachment.)
* `loss_variance` - float - the uncertainty in a stochastic objective function
* `true_loss` - float - When doing hyper-parameter optimization, if you store the generalization error of your model with this name, then you can sometimes get spiffier output from the built-in plotting routines.
* `true_loss_variance` - float - the uncertainty in the generalization error

Since dictionary is meant to go with a variety of back-end storage
mechanisms, you should make sure that it is JSON-compatible.  As long as it's
a tree-structured graph of dictionaries, lists, tuples, numbers, strings, and
date-times, you'll be fine.

**HINT:** To store numpy arrays, serialize them to a string, and consider storing
them as attachments.

Writing the function above in dictionary-returning style, it
would look like this:

.. code-block:: python

    import pickle
    import time
    from hyperopt import fmin, tpe, hp, STATUS_OK

    def objective(x):
        return {'loss': x ** 2, 'status': STATUS_OK }

    best = fmin(objective,
        space=hp.uniform('x', -10, 10),
        algo=tpe.suggest,
        max_evals=100)

    print best

1.3 The Trials Object
~~~~~~~~~~~~~~~~~~~~~

To really see the purpose of returning a dictionary,
let's modify the objective function to return some more things,
and pass an explicit `trials` argument to `fmin`.

.. code-block:: python

    import pickle
    import time
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    def objective(x):
        return {
            'loss': x ** 2,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            # -- attachments are handled differently
            'attachments':
                {'time_module': pickle.dumps(time.time)}
            }
    trials = Trials()
    best = fmin(objective,
        space=hp.uniform('x', -10, 10),
        algo=tpe.suggest,
        max_evals=100,
        trials=trials)

    print best

In this case the call to fmin proceeds as before, but by passing in a trials object directly,
we can inspect all of the return values that were calculated during the experiment.

So for example:
* `trials.trials` - a list of dictionaries representing everything about the search
* `trials.results` - a list of dictionaries returned by 'objective' during the search
* `trials.losses()` - a list of losses (float for each 'ok' trial)
* `trials.statuses()` - a list of status strings

This trials object can be saved, passed on to the built-in plotting routines,
or analyzed with your own custom code.

The *attachments* are handled by a special mechanism that makes it possible to use the same code
for both `Trials` and `MongoTrials`.

You can retrieve a trial attachment like this, which retrieves the 'time_module' attachment of the 5th trial:
```python
msg = trials.trial_attachments(trials.trials[5])['time_module']
time_module = pickle.loads(msg)
```

The syntax is somewhat involved because the idea is that attachments are large strings,
so when using MongoTrials, we do not want to download more than necessary.
Strings can also be attached globally to the entire trials object via trials.attachments,
which behaves like a string-to-string dictionary.


**N.B.** Currently, the trial-specific attachments to a Trials object are tossed into the same global trials attachment dictionary, but that may change in the future and it is not true of MongoTrials.



Hyperopt with a Cluster
-----------------------

The Ctrl Object for Realtime Communication with MongoDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible for `fmin()` to give your objective function a handle to the mongodb used by a parallel experiment. This mechanism makes it possible to update the database with partial results, and to communicate with other concurrent processes that are evaluating different points.
Your objective function can even add new search points, just like `random.suggest`.

The basic technique involves:

* Using the `fmin_pass_expr_memo_ctrl` decorator
* call `pyll.rec_eval` in your own function to build the search space point
  from `expr` and `memo`.
* use `ctrl`, an instance of `hyperopt.Ctrl` to communicate with the live
  trials object.

It's normal if this doesn't make a lot of sense to you after this short tutorial,
but I wanted to give some mention of what's possible with the current code base,
and provide some terms to grep for in the hyperopt source, the unit test,
and example projects, such as [hyperopt-convnet](https://github.com/jaberg/hyperopt-convnet).
Email me or file a github issue if you'd like some help getting up to speed with this part of the code.


To Organize
~~~~~~~~~~~

Hyperopt is designed to support different kinds of trial databases.
The default trial database (`Trials`) is implemented with Python lists and dictionaries.
The default implementation is a reference implementation and it is easy to work with,
but it does not support the asynchronous updates required to evaluate trials in parallel.
For parallel search, hyperopt includes a `MongoTrials` implementation that supports asynchronous updates.

To run a parallelized search, you will need to do the following (after [installing mongodb](Installation-Notes)):

1. Start a mongod process somewhere network-visible.

#. Modify your call to `hyperopt.fmin` to use a MongoTrials backend connected to that mongod process.

#. Start one or more `hyperopt-mongo-worker` processes that will also connect to the mongod process,
    and carry out the search while `fmin` blocks.

1. Start a mongod process
~~~~~~~~~~~~~~~~~~~~~~~~~

Once mongodb is installed, starting a database process (mongod) is as easy as typing e.g.

.. code-block:: bash

    mongod --dbpath . --port 1234
    # or storing each db its own directory is nice:
    mongod --dbpath . --port 1234 --directoryperdb --journal --nohttpinterface
    # or consider starting mongod as a daemon:
    mongod --dbpath . --port 1234 --directoryperdb --fork --journal --logpath log.log --nohttpinterface

Mongo has a habit of pre-allocating a few GB of space (you can disable this with --noprealloc) for better performance, so think a little about where you want to create this database.
Creating a database on a networked filesystem may give terrible performance not only to your database but also to everyone else on your network, be careful about it.

Also, if your machine is visible to the internet, then either bind to the loopback interface and connect via ssh or read mongodb's documentation on password protection.

The rest of the tutorial is based on mongo running on **port 1234** of the **localhost**.

2. Use MongoTrials
~~~~~~~~~~~~~~~~~~

Suppose, to keep things really simple, that you wanted to minimize the `math.sin` function with hyperopt.
To run things in-process (serially) you could type things out like this:

.. code-block:: python

    import math
    from hyperopt import fmin, tpe, hp, Trials

    trials = Trials()
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest)

To use the mongo database for persistent storage of the experiment, use a `MongoTrials` object instead of `Trials` like this:

.. code-block:: python

    import math
    from hyperopt import fmin, tpe, hp
    from hyperopt.mongoexp import MongoTrials

    trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)

The first argument to MongoTrials tells it what mongod process to use, and which *database* (here 'foo_db') within that process to use.
The second argument (`exp_key='exp_1'`) is useful for tagging a particular set of trials *within* a database.
The exp_key argument is technically optional.

**N.B.** There is currently an implementation requirement that the database name be followed by '/jobs'.

Whether you always put your trials in separate databases or whether you use the exp_key mechanism to distinguish them is up to you.
In favour of databases: they can be manipulated from the shell (they appear as distinct files) and they ensure greater independence/isolation of experiments.
In favour of exp_key: hyperopt-mongo-worker processes (see below) poll at the database level so they can simultaneously support multiple experiments that are using the same database.


3. Run `hyperopt-mongo-worker`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run the code fragment above, you will see that it blocks (hangs) at the call fmin.
MongoTrials describes itself internally to fmin as an *asynchronous* trials object, so fmin
does not actually evaluate the objective function when a new search point has been suggested.
Instead, it just sits there, patiently waiting for another process to do that work and update the mongodb with the results.
The `hyperopt-mongo-worker` script included in the `bin` directory of hyperopt was written for this purpose.
It should have been installed on your `$PATH` when you installed hyperopt.

While the `fmin` call in the script above is blocked, open a new shell and type

.. code-block:: bash

    hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1

It will dequeue a work item from the mongodb, evaluate the `math.sin` function, store the results back to the database.
After the `fmin` function has tried enough points it will return and the script above will terminate.
The `hyperopt-mongo-worker` script will then sit around for a few minutes waiting for more work to appear, and then terminate too.

We set the poll interval explicitly in this case because the default timings are set up for jobs (search point evaluations) that take at least a minute or two to complete.

MongoTrials is a Persistent Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run the example above a second time,

.. code-block:: python

    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)

you will see that it returns right away and nothing happens.
That's because the database you are connected to already has enough trials in it; you already computed them when you ran the first experiment.
If you want to do another search, you can change the database name or the `exp_key`.
If you want to extend the search, then you can call fmin with a higher number for `max_evals`.
Alternatively, you can launch other processes that create the MongoTrials specifically to analyze the results that are already in the database. Those other processes do not need to call fmin at all.



Hyperopt Architecture
---------------------


Hyperopt provides serial and parallelizable HOAs via a Python library [2, 3].
Fundamental to its design is a protocol for communication between
(a) the description of a hyperparameter search space,
(b) a hyperparameter evaluation function (machine learning system), and
(c) a hyperparameter search algorithm.
This protocol makes it possible to make generic HOAs (such as the bundled "TPE" algorithm) work for a range of specific search problems.
Specific machine learning algorithms (or algorithm families) are implemented as hyperopt *search spaces* in related projects:
Deep Belief Networks [4],
convolutional vision architectures [5],
and scikit-learn classifiers [6].
My presentation will explain what problem hyperopt solves, how to use it, and how it can deliver accurate models from data alone, without operator intervention.


Adding Optimization Algorithms 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Adding Hyperparameter Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Stuff from Website
------------------


2.4 Adding New Kinds of Hyperparameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding new kinds of stochastic expressions for describing parameter search spaces should be avoided if possible.
In order for all search algorithms to work on all spaces, the search algorithms must agree on the kinds of hyperparameter that describe the space.
As the maintainer of the library, I am open to the possibility that some kinds of expressions should be added from time to time, but like I said, I would like to avoid it as much as possible.
Adding new kinds of stochastic expressions is not one of the ways hyperopt is meant to be extensible.


Basic usage of the hyperopt library is illustrated by the following code.

.. code-block:: python

    # define an objective function
    def objective(args):
        case, val = args
        if case == 'case 1':
            return val
        else:
            return val ** 2

    # define a search space
    from hyperopt import hp
    space = hp.choice('a',
        [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print best
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print hyperopt.space_eval(space, best)
    # -> ('case 2', 0.01420615366247227}




Communicating with MongoDB Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to interface more directly with the search process (when using MongoTrials) by communicating with MongoDB directly, just like `hyperopt-mongo-worker` does. It's beyond the scope of a tutorial to explain how to do this, but Hannes Schultz (@temporaer) got hyperopt working with his MDBQ project, which is a standalone mongodb-based task queue:

https://github.com/temporaer/MDBQ/blob/master/src/example/hyperopt_client.cpp

Have a look at that code, as well as the contents of [hyperopt/mongoexp.py](https://github.com/jaberg/hyperopt/blob/master/hyperopt/mongoexp.py) to understand how worker processes are expected to reserve jobs in the work queue, and store results back to MongoDB.


This code illustrates hyperopt's `fmin` function.
The `fmin` function is the main interface for both synchronous and asynchronous
(parallel, including across hosts)
execution.

* fmin
* configuration language
* returning more than the loss function
* Trials
* MongoDB
* Parallel/Asynchronous optimization
* MongoTrials
* algorithms
* pyll
* vectorization?
* TPE



Ongoing and Future Work
------------------------

Drivers for other systems: 
* Jasper Snoek's "spearmint" package for Gaussian process-based Bayesian optimization
* Frank Hutter's SMAC and ROAR algorithms, as implemented in XXX.

[hp-dbn]_
[hp-sklearn]_
[hp-convnet]_


Acknowedgements
---------------

NSF grant, NSERC Banting Fellowship program.
Nicolas Pinto for design advice.
Hristijan Bogoevski for the `pchoice` function and ongoing work on an sklearn driver.

References
----------
.. [BB12] J. Bergstra  and Y. Bengio. *Random Search for Hyperparameter Optimization* J. Machine Learning Research, XXX:XX, 2012. http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
.. [BBBK11]  XXX http://www.eng.uwaterloo.ca/~jbergstr/files/pub/11_nips_hyperopt.pdf
.. [Brochu10] XXX
.. [Hyperopt] github link XXX
.. [hp-dbn] github link XXX https://github.com/jaberg/hyperopt-dbn) - optimize Deep Belief Networks
.. [hp-sklearn] github link XXX https://github.com/jaberg/hyperopt-sklearn
.. [hp-convnet] github link XXX https://github.com/jaberg/hyperopt-convnet optimize convolutional architectures for image classification used in Bergstra, Yamins, and Cox in (ICML 2013).
.. [MATLAB] XXX
.. [Mockus78] Mockus. *XXX*, XXX, 1978.
.. [ROAR] http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software
.. [sklearn] http://scikit-learn.org
.. [SLA13]  XXX
.. [Spearmint] http://www.cs.toronto.edu/~jasper/software.html Gaussian-process SMBO in Python.
.. [SMAC] http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software Sequential Model-based Algorithm Configuration (based on regression trees)
.. [SciPy] XXX
.. [XXX] XXX

