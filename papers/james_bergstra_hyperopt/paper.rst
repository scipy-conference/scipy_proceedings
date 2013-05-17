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
or else operate without leveraging valuable structure in the search space.
Common practice for the optimization of hyperparameters is
(a) for algorithm developers to tune them by hand on representative problems to get good rules of thumb and default values,
and (b) for algorithm users to tune them manually for their particular prediction problems, perhaps with the assistance of [multiresolution] grid search.
However, when dealing with more than a few hyperparameters (e.g. 5) this standard practice of manual search with grid refinement is not guaranteed to work well;
in such cases even random search has been shown to be competitive with domain experts [BB12]_.

Hyperopt [Hyperopt]_ provides algorithms and software infrastructure for carrying out hyperparameter optimization for machine learning algorithms.
Hyperopt provides an optimization interface that distinguishes a *search space* and an *evaluation function* that assigns real-valued
*loss values* to points within the search space.
Unlike the fmin interface in [SciPy]_ or [Matlab]_, Hyperopt's fmin interface requires users to specify the search domain as a probability distribution.
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


Getting Started
---------------

This page is a tutorial on basic usage of `hyperopt.fmin()`.
It covers how to write an objective function that fmin can optimize, and how to describe a search space that fmin can search.

Hyperopt's job is to find the best value of a scalar-valued, possibly-stochastic function over a set of possible arguments to that function.
Whereas many optimization packages will assume that these inputs are drawn from a vector space,
Hyperopt is different in that it encourages you to describe your search space in more detail.
By providing more information about where your function is defined, and where you think the best values are, you allow algorithms in hyperopt to search more efficiently.

The way to use hyperopt is to describe:

* the objective function to minimize
* the space over which to search
* the database in which to store all the point evaluations of the search
* the search algorithm to use

This (most basic) tutorial will walk through how to write functions and search spaces,
using the default `Trials` database, and the dummy `random` search algorithm.
Section (1) is about the different calling conventions for communication between an objective function and hyperopt.
Section (2) is about describing search spaces.

Parallel search is possible when replacing the `Trials` database with
a `MongoTrials` one;
there is another wiki page on the subject of [using mongodb for parallel search](Parallelizing-Evaluations-During-Search-via-MongoDB).

Choosing the search algorithm is as simple as passing `algo=hyperopt.tpe.suggest` instead of `algo=hyperopt.random.suggest`.
The search algorithms are actually callable objects, whose constructors
accept configuration arguments, but that's about all there is to say about the
mechanics of choosing a search algorithm.


Define a function to minimize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperopt provides a few levels of increasing flexibility / complexity when it comes to specifying an objective function to minimize.
The questions to think about as a designer are
* Do you want to save additional information beyond the function return value, such as other statistics and diagnostic information collected during the computation of the objective?
* Do you want to use optimization algorithms that require more than the function value?
* Do you want to communicate between parallel processes? (e.g. other workers, or the minimization algorithm)

The next few sections will look at various ways of implementing an objective
function that minimizes a quadratic objective function over a single variable.
In each section, we will be searching over a bounded range from -10 to +10,
which we can describe with a *search space*:

.. code-block:: python

    space = hp.uniform('x', -10, 10)

Below, Section 2, covers how to specify search spaces that are more complicated.


2. Defining a Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~

A search space consists of nested function expressions, including stochastic expressions.
The stochastic expressions are the hyperparameters.
Sampling from this nested stochastic program defines the random search algorithm.
The hyperparameter optimization algorithms work by replacing normal "sampling" logic with
adaptive exploration strategies, which make no attempt to actually sample from the distributions specified in the search space.

It's best to think of search spaces as stochastic argument-sampling programs. For example

.. code-block:: python

    from hyperopt import hp
    space = hp.choice('a',
        [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

The result of running this code fragment is a variable `space` that refers to a graph of expression identifiers and their arguments.
Nothing has actually been sampled, it's just a graph describing *how* to sample a point.
The code for dealing with this sort of expression graph is in `hyperopt.pyll` and I will refer to these graphs as *pyll graphs* or *pyll programs*.

If you like, you can evaluate a sample space by sampling from it.

.. code-block:: python

    import hyperopt.pyll.stochastic
    print hyperopt.pyll.stochastic.sample(space)

This search space described by `space` has 3 parameters:
* 'a' - selects the case
* 'c1' - a positive-valued parameter that is used in 'case 1'
* 'c2' - a bounded real-valued parameter that is used in 'case 2'

One thing to notice here is that every optimizable stochastic expression has a *label* as the first argument.
These labels are used to return parameter choices to the caller, and in various ways internally as well.

A second thing to notice is that we used tuples in the middle of the graph (around each of 'case 1' and 'case 2').
Lists, dictionaries, and tuples are all upgraded to "deterministic function expressions" so that they can be part of the search space stochastic program.

A third thing to notice is the numeric expression `1 + hp.lognormal('c1', 0, 1)`, that is embedded into the description of the search space.
As far as the optimization algorithms are concerned, there is no difference between adding the 1 directly in the search space
and adding the 1 within the logic of the objective function itself.
As the designer, you can choose where to put this sort of processing to achieve the kind modularity you want.
Note that the intermediate expression results within the search space can be arbitrary Python objects, even when optimizing in parallel using mongodb.
It is easy to add new types of non-stochastic expressions to a search space description, see below (Section 2.3) for how to do it.

A fourth thing to note is that 'c1' and 'c2' are examples what we will call *conditional parameters*.
Each of 'c1' and 'c2' only figures in the returned sample for a particular value of 'a'.
If 'a' is 0, then 'c1' is used but not 'c2'.
If 'a' is 1, then 'c2' is used but not 'c1'.
Whenever it makes sense to do so, you should encode parameters as conditional ones this way,
rather than simply ignoring parameters in the objective function.
If you expose the fact that 'c1' sometimes has no effect on the objective function (because it has no effect on the argument to the objective function) then search can be more efficient about credit assignment.


2.1 Parameter Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~

The stochastic expressions currently recognized by hyperopt's optimization algorithms are:

`hp.choice(label, options)`
   Returns one of the options, which should be a list or tuple.  The elements of `options` can themselves be [nested] stochastic expressions.  In this case, the stochastic choices that only appear in some of the options become *conditional* parameters.

`hp.randint(label, upper)`
   Returns a random integer in the range [0, upper). The semantics of this distribution is that there is *no* more correlation in the loss function between nearby integer values, as compared with more distant integer values.  This is an appropriate distribution for describing random seeds    for example.  If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either `quniform`, `qloguniform`, `qnormal` or `qlognormal`.

`hp.uniform(label, low, high)`
   Returns a value uniformly between `low` and `high`.  When optimizing, this variable is constrained to a two-sided interval.

`hp.quniform(label, low, high, q)`
    Returns a value like round(uniform(low, high) / q) * q Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below.

`hp.loguniform(label, low, high)`
    * Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed.
    * When optimizing, this variable is constrained to the interval [exp(low), exp(high)].

`hp.qloguniform(label, low, high, q)`
    * Returns a value like round(exp(uniform(low, high)) / q) * q
    * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

`hp.normal(label, mu, sigma)`
    * Returns a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

`hp.qnormal(label, mu, sigma, q)`
    * Returns a value like round(normal(mu, sigma) / q) * q
    * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

`hp.lognormal(label, mu, sigma)`
    * Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed.
        When optimizing, this variable is constrained to be positive.

`hp.qlognormal(label, mu, sigma, q)`
    * Returns a value like round(exp(normal(mu, sigma)) / q) * q
    * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

2.2 A Search Space Example: scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see all these possibilities in action, let's look at how one might go about describing the space of hyperparameters of classification algorithms in scikit-learn.
(I think that there's room for a library that actually does this, possibly even bundled with hyperopt itself in the future, but for now it's just an idea.)

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


2.3 Adding Non-Stochastic Expressions with pyll
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use such nodes as arguments to pyll functions (see pyll).
File a github issue if you want to know more about this.

In a nutshell, you just have to decorate a top-level (i.e. pickle-friendly) function so
that it can be used via the `scope` object.

.. code-block:: python

    import hyperopt.pyll
    from hyperopt.pyll import scope

    @scope.define
    def foo(a, b=0):
         print 'runing foo', a, b
         return a + b / 2

    # -- this will print 0, foo is called as usual.
    print foo(0)

    # In describing search spaces you can use `foo` as you
    # would in normal Python. These two calls will not actually call foo,
    # they just record that foo should be called to evaluate the graph.

    space1 = scope.foo(hp.uniform('a', 0, 10))
    space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1)

    # -- this will print an pyll.Apply node
    print space1

    # -- this will draw a sample by running foo()
    print hyperopt.pyll.stochastic.sample(space1)


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

