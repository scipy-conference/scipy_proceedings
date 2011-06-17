:author: Mark Dewing
:email: markdewing@gmail.com
:institution: Intel

--------------------------------------------
Constructing scientific programs using Sympy
--------------------------------------------

.. class:: abstract

We describe a method for constructing scientific programs where Sympy is
used to model the mathematical steps in the derivation.  With this workflow,
each step in the process can be checked by machine, from the derivation of
the equations to the generation of the source code.  We present an example
based on computing the partition function integrals in statistical mechanics.

.. class:: keywords

   Sympy, code generation, metaprogramming

Introduction
------------

Writing correct scientific programs is difficult, largely manual process.
The steps in the process include
deriving of the constituent equations, translating those into source code, and testing
the result to ensure correctness.   The final step of testing is challenging because
it is hard to untangle the set of possible errors from testing the final code.
For example, if the result appears incorrect, it is hard to determine whether the problem
is with the algorithm, or a mistake was made in the derivation, or a simple transcription error
in writing the source code.
We wish to look at this process of writing a scientific code and see what steps, if any, can be checked
by the computer.


Our approach is to operate on a symbolic representation of the scientific program, and then programmatically
transform it into the target system.  The specifications for scientific software are expressed
largely as equations, and ideally suited for a symbolic mathematics package.
We use Sympy, a symbolic mathematics package written in Python, for this part of the process.


The target system is likely a source code representation (C, Fortran, Python, etc), but could encompass
more than that.
For instance, C might be used to target the CPU or GPU, but the source code might look
quite different in those cases.
Or the user may call different libraries for the same function to compare performance characteristics.



Modeling Derivations
--------------------
The goal is to model on the computer a similar set of steps that one would use when manually performing the derivation.
Deriving the equations using in scientific software is similar to a proof where there is a series
of logically justified steps connecting each expression until the final result is reached.

The OpenMathDocuments (OMDoc) project [OMDoc]_ is a representation for mathematics that works at a
higher level than expressions in MathML.  For instance, it has representations for proofs and lemmas.
Similarly, for scientific computation we need to represent structures a higher level.   One major 
difference between proofs and derivations in scientific software is that some steps are approximations.

The steps can be categorized as exact transformations, approximations, or specializations.
An exact transform leaves the equality satisfied.  Some types of exact transforms are rearranging terms,
multiplying by factors, and identities (which operates only on one side of the equation).
A specialization is specifying a physical or model parameter, such as the number of spatial
dimensions, the number of particles, interaction potential, etc.

Finally, the results can be displayed in rendered mathematics (we use MathML or MathJax in web pages)
to make the operation of each step and the results clearly visible.

Implementations
---------------

Modeling Derivations
^^^^^^^^^^^^^^^^^^^^
For the implementation, there is class named ``derivation``.  The constructor takes an initial equation (lhs and rhs).  The primary method is ``add_step``, which takes an operation (or list of operations) to perform
and a textual description
of the operation(s).  There are series of classes for various operations, such ``approx_lhs``, which replaces the left-hand side of the equation with a new value.  Also there is ``add_term``, which adds the same term to
both sides of the equation.

Code Generation
^^^^^^^^^^^^^^^
It is easy to start generating code by simply printing the statements of the
target language.  However, this will eventually be unsatisfying and we will use 
a model of the target language.  Currently this work has (incomplete) language models for Python and C.

At the lowest level of transforming expressions, we developed a pattern-matching syntax that can
concisely capture some the the Sympy idioms.


The ``Match`` object matches a Sympy expression.  The  ``__call__`` method matches the first argument
as the type of the expression.  Subsequent arguments are variables to be bound to 
arguments.  If the argument is a tuple, it is matched recursively on that argument.  In this way
the pattern for a tree structure can be built up concisely.

Variables than can match (for later binding) are members of an ``AutoVar`` class.  This class
creates member variables upon first access, and they are bound when the match succeeds.
 
Here is an example fragment of part of the Sympy to Python expression transformation.
Sympy normalizes subtraction
as adding two expressions where the subtractand is multiplied by negative one.

.. code-block:: python

    class sympy_to_py(object):
     def __call__(self, e):
       v = AutoVar()
       m = Match(e)

       if m(Add, (Mul, S.NegativeOne, v.e1), v.e2):
         return py_expr(py_expr.PY_OP_MINUS,
            self(v.e2), self(v.e1))

       if m(Add, v.e1, v.e2):
            return py_expr(py_expr.PY_OP_PLUS,
                self(v.e1), self(v.e2))

       if m(Pow, v.e2, S.NegativeOne):
            return py_expr(py_expr.PY_OP_DIVIDE,
                py_num(1.0), self(v.e2))




Examples
--------

Simple derivation
^^^^^^^^^^^^^^^^^

The Euler method is the simplest method for solving a differential equation.
The steps involve a finite difference approximation to the derivative, rearranging terms, and the 
result is

.. math::

    f_1 = f_0 + h*2*x

The derivation is the following code:

.. code-block:: python

    from sympy import *
    f = Function('f')
    x = Symbol('x')
    df = diff(f(x),x)
    d = derivation(df,2*x)

    d.add_step(approx_lhs(fd),
        'Approximate derivative with finite difference')
    d.add_step(mul_factor(h),'Multiply by h')
    d.add_step(add_term(f0),'Move f_0 term to left side')

This can be output to MathML (or MathJax) for display in a web browser, which looks
approximately like the following:

.. math::

  \frac{\partial}{\partial x} \operatorname{f}\left(x\right) = 2*x

Approximate derivative with finite difference

.. math::

  \frac{f_{1} - f_{0}}{h} = 2*x

Multiply by h

.. math::

  f_{1} - f_{0} = 2*x h

Move f_0 term to left side to get the final result

.. math::

    f_{1} = f_{0} + 2*x h




Quadrature
^^^^^^^^^^
For one of the simplest quadrature formulas, we use the trapezoidal rule [Trapezoid]_.
The derivation part consists
of starting from the rule for single interval, and extending it to a series of intervals. (The rules for 
a single interval can be derived from interpolating polynomials, but we didn't start there)

The starting point for the derivation in Python is to define all the symbols, and the initial expression,
then manipulate the expression so the function evaluation of each point is used only once.

.. code-block:: python

 i = Symbol('i',integer=True)
 n = Symbol('n',integer=True)

 I = Symbol('I')
 f = Function('f')
 h = Symbol('h')
 x = IndexedBase('x')

 trap = derivation(I, Sum(h/2*(f(x[i])+f(x[i+1])), (i,1,n)))
 trap.add_step(identity(split_sum),'Split sum')
 trap.add_step(identity(adjust_limits),'Adjust limits')
 trap.add_step(identity(peel_terms),'Peel terms')

The LaTeX representation for the steps was copied from the generated output. (There is still room for
some improvements in the notation.)

Start with a sum of single interval formulas

.. math::

  I = \sum_{i=1}^{n} \frac{1}{2} h \left(\operatorname{f}\left(x[i]\right) + \operatorname{f}\left(x[1 + i]\right)\right)

Split into two sums ('Split sum')

.. math::

  I = \sum_{i=1}^{n} \frac{1}{2} h \operatorname{f}\left(x[i]\right) + \sum_{i=1}^{n} \frac{1}{2} h \operatorname{f}\left(x[1 + i]\right) 


Adjust the limits so the functions in the sum have compatible indices ('Adjust limits')

.. math::

  I = \sum_{i=0}^{-1 + n} \frac{1}{2} h \operatorname{f}\left(x[i]\right) + \sum_{i=1}^{n} \frac{1}{2} h \operatorname{f}\left(x[i]\right)

Peel of some terms to the sum limits match, and combine the sums.  ('Peel terms')

.. math::

  I = 2 \sum_{i=1}^{-1 + n} \frac{1}{2} h \operatorname{f}\left(x[i]\right) + \frac{1}{2} h \operatorname{f}\left(x[0]\right) + \frac{1}{2} h \operatorname{f}\left(x[n]\right)


Now we have the final expression and can move to the transformation step.  The approach to multiple
dimensional integrals will be iterated one-dimensional integrals.



Partition Function
^^^^^^^^^^^^^^^^^^
We start with the configuration integral from statistical mechanics [Partition]_.
The dimensionality rises with the number of particles. The complexity the convergence of grid-based methods is exponential in the number of dimensions, and they quickly become overwhelmed.
The convergence of Monte Carlo methods is independent of dimension, and so are commonly used to compute
these integrals.
However, it would be still be useful to use a grid method for a small number of particles as a way to
check the Monte Carlo algorithms.

The derivation starts as follows:

.. code-block:: python

  partition_function =
     derivation(Z,Integral(exp(-V/(k*T)),R))

Once again, the LaTeX has been copied from the output (although some steps have been combined to
for space)

.. math::

  Z = \int e^{- \frac{V}{T k}}\,dR


Insert the definition of :math:`\beta =kT` and specialize to two particles

.. math::

  Z = \int\int e^{- \beta \operatorname{V}\left(r_{1},r_{2}\right)}\,dr_{1} dr_{2}

Change variables and switch to a potential that depends only on the magnitude of the interparticle distance

.. math::

  Z = \int\int e^{- \beta \operatorname{V}\left(\lvert{r_{12}}\rvert\right)}\,dr_{12} dr_{cm}


Integrate out the center of mass (or fixed coordinate) (This step could be performed by Sympy, but isn't right now)

.. math::

  Z = \Omega \int e^{- \beta \operatorname{V}\left(\lvert{r_{12}}\rvert\right)}\,dr_{12}


Decompose into vector components and specify limits

.. math::

 Z = \Omega \int_{-L/2}^{L/2}\int_{-L/2}^{L/2} e^{- \beta \operatorname{V}\left(\sqrt{r_{12 x}^{2} + r_{12 y}^{2}}\right)}\,dr_{12 x} dr_{12 y}


Specialize to the Lennard-Jones potential

.. math::

  \operatorname{V}\left(r\right) = \frac{4}{r^{12}} - \frac{4}{r^{6}}

And get

.. math::

 Z = \Omega \int_{- \frac{1}{2} L}^{\frac{1}{2} L}\int_{- \frac{1}{2} L}^{\frac{1}{2} L} e^{- \beta \left(\frac{4}{\left(r_{12 x}^{2} + r_{12 y}^{2}\right)^{6}} - \frac{4}{\left(r_{12 x}^{2} + r_{12 y}^{2}\right)^{3}}\right)}\,dr_{12 x} dr_{12 y}


Insert numerical values for the box size and temperature.

.. math::

 Z = 4.0 \int_{-1.0}^{1}\int_{-1.0}^{1} e^{- 4.0 \frac{1}{\left(r_{12 x}^{2} + r_{12 y}^{2}\right)^{6}} + 4.0 \frac{1}{\left(r_{12 x}^{2} + r_{12 y}^{2}\right)^{3}}}\,dr_{12 x} dr_{12 y}

Now we have an integral that is completely specified numerically [1]_.  It can be evaluated by an existing
quadrature routine in Sympy, by another another package (``scipy.quadrature.dblquad``), or by the trapezoidal
rule code we derived earlier.


Code Generation
^^^^^^^^^^^^^^^

As an example of the language model, the classic 'Hello World' program in python is

.. code-block:: python

 from sympy.prototype.codegen.lang_py import *

 body = py_stmt_block()

 hello_func = py_function_def('hello')
 hello_func.add_statement(
    py_print_stmt(py_string("Hello, World")))
 body.add_statements(hello_func)
 main = py_if(
     py_expr(py_expr.PY_OP_EQUAL,
         py_var('__name__'), py_string('__main__')))
 main.add_true_statement(
    py_expr_stmt(py_function_call('hello')))
 body.add_statements(main)

 f = open('hello_py.py','w')
 f.write(body.to_string())
 f.close()


The generated output is

.. code-block:: python

  def hello():
     print "Hello, World"
  if __name__ == "__main__":
     hello()


For C, the program is

.. code-block:: python

  from sympy.prototype.codegen.lang_c import *

  body = c_block()
  body.add_statement(pp_include('stdio.h'))
  main_body = c_block()

  main = c_function_def(
    c_func_type(c_int('main')), main_body)

  main_body.add_statement(
    c_stmt(c_function_call("printf",
            c_string("Hello, World\\n"))))

  main_body.add_statement(c_return(c_num(0)))
  body.add_statement(main)

  f = open('hello_c.c','w')
  f.write(body.to_string())
  f.close()

The generated program is

.. code-block:: c 

  #include <stdio.h>
  int main(){
    printf("Hello, World\n");
    return 0;
  }



The code and examples described here can be found in my sympy fork on GitHub,
in the derivation_modeling branch, in the ``prototype`` directory:
https://github.com/markdewing/sympy/tree/derivation_modeling/sympy/prototype



Discussion
----------
The example derivations presented here are all straightforward and linear.
In reality, the connections form a more general
graph.  For instance, one is often interested in multiple properties 
(energy, pressure, distribution functions) that may branch off the original derivation or have a
separate thread of steps, but eventually, for efficiency they should all be evaluated
in the same integral.

The pattern-matching style makes the lower levels of expression translation fairly clear, but 
the the translations at the next level up (combining the source code statements) is not very transparent
yet.  An important future step is enhancing debugging by making the connections between the
code generator and the generated code clearer.


Other Work
----------

.. Structured derivations is a tightly specified, formal method for performing a proof used for teaching
.. high school mathematics - it is of interest because each step is similar

There are a number of systems with the same general features under various generic names, such as 'Automated Scientific Computing' [Terrel11]_, [FEniCS]_ and 'Software Automation'. 
For solving partial differential equations, there is 
FEniCS [FEniCS]_ and the SAGA (Scientific computing with Algebraic and Generative Abstractions)
project [SAGA]_ .

Ignition [Ignition]_ (also described in [Terrel11]_) is a library that provides support for writing and combining DSL's (Domain Specific Languages) for describing problems (or aspects of problems)

Pivot [Pivot]_ is a project for modeling C++.  CodeBoost [CodeBoost]_ is the code transformation portion of the SAGA system.
PyCUDA [PyCUDA]_ is a potential target system, and it also has an associated model of C and CUDA for generation of code [CodePy]_


Conclusions
-----------
We've described a snapshot of some work on some blocks necessary for a system of scientific computing,
including modeling a derivation, transforming to a source code representation, and code generation.



References
----------
.. [CodeBoost] http://codeboost.org/

.. [CodePy] http://mathema.tician.de/software/codepy

.. [FEniCS] http://www.fenicsproject.org

.. [Ignition] http://andy.terrel.us/ignition/ 

.. [OMDoc] http://www.omdoc.org

.. [Partition] http://en.wikipedia.org/wiki/Partition_function_%28statistical_mechanics%29

.. [Pivot] http://parasol.tamu.edu/pivot/ 

.. [PyCUDA] http://mathema.tician.de/software/pycuda

.. [Terrel11] A. Terrel. *From Equations to Code: Automated Scientific Computing*
                Computing in Science and Engineering 13(2):78-982, March 2011

.. [Trapezoid] See http://en.wikipedia.org/wiki/Trapezoidal_rule or any numerical analysis textbook

.. [SAGA] http://www.ii.uib.no/saga/


.. [1] There is a division-by-zero error at :math:`r=0` that must be avoided, either by offsetting one limit
       slightly, or better, by capping the potential for small :math:`r`.  This latter step has not been
       added to the definition of the potential yet.  
