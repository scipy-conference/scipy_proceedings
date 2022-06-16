:author: Nadia Udler
:email: nadiakap@optonline.net
:institution: University of Connecticut (Stamford)


---------------------------------------------------------------
Global optimization software library for research and education
---------------------------------------------------------------

.. class:: abstract

Machine learning models are often represented by functions given by computer programs. Optimization
of such functions is a challenging task because traditional derivative based
optimization methods with guaranteed convergence properties cannot be used.. This software
allows to create new optimization methods with desired properties, based on basic modules.
These basic modules are designed in accordance with approach for constructing global optimization
methods based on potential theory [KAP]_. These methods do not use derivatives of objective function
and as a result work with nondifferentiable functions (or functions given by computer programs,
or black box functions), but have guaranteed convergence. The software helps to understand
principles of learning algorithms. This software may be used by researchers to design their own
variations or hybrids of known heuristic optimization methods. It may be used by students to
understand how known heuristic optimization methods work and how certain parameters affect the behavior of the method.




.. class:: keywords

   global optimization, black-box functions, algorithmically defined functions, potential functions


Introduction
------------
Optimization lies at the heart of machine learning and data science.
One of the most relevant problems in machine learning is automatic selection of the algorithm depending on
the objective. This is necessary in many applications such as robotics, simulating biological or chemical
processes, trading strategies optimization, to name a few.
We developed a library of optimization methods as a first step for self-adapting algorithms. Optimization
methods in this library work with all objectives including very onerous ones, such as black box functions
and functions given by computer code, and the convergences of methods is guaranteed. This library allows
to create customized derivative free learning algorithms with desired properties  by combining building
blocks from this library or other Python libraries.

The library is intended primarily for educational
purposes and its focus is on transparency of the methods rather than on efficiency of implementation.

Minpy can be used by researches to design optimization methods with desired properties by varying parameters of the general algorithm. Such design step is very often required in newly developing industry projects, where objective function has unknown behavior, Algorithm designing stage is necessary for two reasons: It helps to understand objective function behavior, and it is needed for function optimization. Very often one method would work best for studying function behavior, and another method - for function optimization.

One typical situation is when different algorithm behavior is required when being far from optimal point, and when being close to optimal point.

As an example, consider variant of simulated annealing proposed in [FGSB]_ where different values of parameters ( Boltzman distribution parameters, step size, etc.) are used depending of the distance to optimal point. In this paper the basic SA algorithm is used as a starting point. We can offer more basic module as a starting point ( and by specifying distribution as 'exponential' get the variant of SA) thus achieving more flexible design opportunities for custom optimization algorithm. Note that convergence of the newly created hybrid algorithm does not need to be verified when using minpy basic modules, whereas previously mentioned SA-based hybrid has to be verified separately ( see [GLUQ]_)

Testing functions are included in the library. They represent broad range of use cases covering above
mentioned difficult functions. In this paper we describe the approach underlying these optimization methods.
The distinctive feature of these methods is that they are not heuristic in nature. The algorithms are derived
based on potential theory [KAP]_, and their convergence is guaranteed by their derivation method [KPP]_.
Recently potential theory was applied to prove convergence of well known heuristic methods, for example
see [BIS]_ for convergence of PSO, and to re prove convergence of well known gradient based methods, in particular,
first order methods   - see  [NBAG]_ for convergence of gradient descent and [ZALO]_ for mirror descent.
For potential functions approach for stochastic first order optimization methods see [ATFB]_.


Outline of the approach
-----------------------

The approach works for non-smooth or algorithmically defined functions.  For detailed description of the approach see [KAP]_, [KP]_.
In this approach the original optimization problem is replaced with a randomized problem, allowing the use of Monte-Carlo methods for calculating integrals.
This is especially important if objective function is given by its values (no analytical formula) and derivatives
are not known. The original problem is restated in the framework of gradient (sub gradient) methods, employing the
standard theory (convergence theorems for gradient (sub gradient) methods), whereas no derivatives of the objective
function are needed. At the same time, the method obtained is a method of nonlocal search unlike other gradient methods.
It will be shown, that instead of measuring the gradient of the objective function we can measure the gradient of the
potential function at each iteration step  , and the value of the gradient can be obtained using values of objective
function only, in the framework of Monte Carlo methods for calculating integrals. Furthermore, this value does not have
to be precise, because it is recalculated at each iteration step. It will also be shown that well-known zero-order
optimization methods ( methods that do not use derivatives of objective function but its values only) are generalized
into their adaptive extensions. The generalization of zero-order methods (that are heuristic in nature) is obtained
using standardized methodology, namely, gradient (sub gradient) framework.
We consider the unconstrained optimization problem

.. math::
   :label: fx
   
   f(x_1, x_2,..x_n)\to\min_{x \in R_n }


By randomizing we get

.. math::
   :label: rfx
   
   F(X) = E(f(x))\to\min_{x \in R_n }

where  X is a set of random vectors with values from :math:`R^n` .

The solution of :ref:`rfx` is a random vector from  {X} that optimizes a functional F(X) .

Note that :ref:`fx` and :ref:`rfx` are equivalent (see [KAP]_ for proof) and :ref:`rfx` is the stochastic optimization problem of the functional F(X) .

To study the gradient nature of the solution algorithms for problem :ref:`rfx`, a variation of objective functional  F(X)  will be considered.

The suggested approach makes it possible to obtain optimization methods in systematic way, similar to methodology adopted in smooth optimization. 
Derivation includes randomization of the original optimization problem, finding directional derivative for the randomized problem and choosing moving direction Y based on the condition that directional derivative in the direction of Y is being less or equal to 0.

Because of randomization, the expression for directional derivative doesn't contain the differential characteristics of the original function. We obtain the condition for selecting the direction of search Y in terms of its characteristics - conditional expectation. Conditional expectation is a vector function (or vector field) and can be decomposed (following the theorem of decomposition of the vector field) into the sum of the gradient of scalar function P and a function with zero divergence. P is called a potential function. As a result the original problem is reduced to optimization of the potential function, furthermore, the potential function is specific for each iteration step. Next, we arrive at partial differential equation that connects P and the original function.
To define computational algorithms it is necessary to specify the dynamics of the random vectors. For example, the dynamics can be expressed in a form of densities. For certain class of distributions, for example normal distribution, the dynamics can be written in terms of expectation and covariance matrix. It is also possible to express the dynamics in mixed characteristics.


Expression for directional derivative
-------------------------------------


Derivative of objective functional F(X) in the direction of the random vector Y at the point :math:`X^0` (Gateaux derivative) is:


 :math:`\delta _Y F(X^0 )=\frac{d}{d \epsilon} F(X^0+\epsilon Y) _{\epsilon=0}=\frac{d}{d \epsilon} F(X^\epsilon) dx_{\epsilon=0}=\frac{d}{d \epsilon} \int f(X) p_{x^\epsilon}(x) _{\epsilon=0}`

where density function of the random vector :math:`X^\epsilon=X^0+\epsilon Y` may be expressed in terms of joint density function :math:`p_{{X^0},Y} (x,y)` of :math:`X^0` and Y as follows:

.. math::
   :label: ddrv
   
   p_{x^ \epsilon} (x) = \int_{R^n} p_{x^ \epsilon} (x - \epsilon y,y) dy


The following relation (property of divergence) will be needed later

.. math::
   :label: pdiv
   
   \frac{d}{d \epsilon} p_{x^ \epsilon} (x - \epsilon y,y) =(-\nabla_x  p_{x^ \epsilon} (x,y), y ) = -div_x ( p_{x^ \epsilon} (x,y) y )


where ( , ) defines dot product.

Assuming differentiability of the integrals (for example, by selecting the appropriate :math:`p_{x^ \epsilon} (x,y)` and using :ref:`ddrv`, :ref:`pdiv` we get

.. math::

   \delta _Y F(X^0 ) = [\frac{d}{d \epsilon} \int_{R^n}   \int_{R^n} f(x) p_{x^ \epsilon} (x - \epsilon y,y) dx dy] _{\epsilon=0}=


:math:`= [\frac{d}{d \epsilon} \int_{R^n} f(x)  \int_{R^n} p_{x^ \epsilon} (x - \epsilon y,y) dx dy] _{\epsilon=0}= [ \int_{R^n} f(x) ( \frac{d}{d \epsilon} \int_{R^n} p_{x^ \epsilon} (x - \epsilon y,y) dy )dx] _{\epsilon=0}=`


:math:`= \int_{R^n} f(x)(  \int_{R^n} [\frac{d}{d \epsilon}  p_{x^ \epsilon} (x - \epsilon y,y)] _{\epsilon=0} dy) dx=- \int_{R^n} f(x)(  \int_{R^n} [div_x ( p_{x^ \epsilon} (x,y) y )]  dy) dx=`


.. math::

   - \int_{R^n} f(x) div_x [  \int_{R^n} ( p_{x^ \epsilon} (x,y) y )  dy] dx


Using formula for conditional distribution :math:`p_{Y/X^0=x} (y)=\frac {p_{x^ \epsilon y} (x,y)}{p_{x^ \epsilon} (x) )}` ,

where  :math:`p_{x^ \epsilon}(x) =  \int_{R^n} p_{x^ \epsilon y} (x,u) du`

we get :math:`\delta _Y F(X^0 )= - \int_{R^n} f(x) div_x [ p_{x^ \epsilon}(x) \int_{R^n}  p_{Y/X^0=x} (y) y dy] dx`

Denote :math:`\overline {y}(x) = \int_{R^n} yp_{Y/X^0=x} (y) dy=E[Y/X^0=x]`

Taking into account normalization condition for density we arrive at the following expression for directional derivative:

.. math::

   \delta _Y F(X^0 )= - \int_{R^n} (f(x)-C) div_x [ p_{x^0}(x)\overline y(x)]dx


where C is arbitrary chosen constant

Considering solution to :math:`\delta _Y F(X^0 )\to\min_Y` allows to obtain gradient-like alggorithms for optimization that use only objective function values ( do not use derivatives of objective function)


Potential function as a solution to Poisson's equation
------------------------------------------------------
Decomposing vector field :math:`p_{x^0}(x)\overline y(x)`  into potential field :math:`\nabla \varphi_0 (x)` and divergence-free component :math:`W_0 (x)`:

.. math::

   p_{x^0}(x)\overline y(x)= \nabla \phi_0 (x) +W_0 (x)


we arrive at Poisson's equation for potential function:

.. math::

   \nabla \varphi_0 (x) = -L [f(x)-C]p_u (x)

where L is a constant

Solution to Poisson's equation approaching 0 at infinity may be written in the following form

.. math::

   \varphi_0 (x)=  \int_{R^n} E(x,\xi)  [f(\xi) - C] p_u (\xi)d\xi


where :math:`E(x,\xi)` is a fundamental solution to Laplace's equation.

Then for potential component :math:`\Delta \varphi_0 (x)`  we have


.. math::

   \Delta \varphi_0 (x) = -L E[\Delta_x E(x,u)(f(x)-C)]


To conclude, the representation  for gradient-like direction is obtained. This direction maximizes directional derivative of the objective functional F(X). Therefore, this representation can be used for computing the gradient of the objective function f(x) using only its values.
Gradient direction of the objective function f(x) is determined by the gradient of the potential function :math:`\varphi_0 (x)`, which, in turn,  is determined by Poisson's equation.

Practical considerations
------------------------
The dynamics of the expectation of objective function may be written in the space of random vectors as follows:

.. math::

   X_{N+1} = X_{N}+ \alpha_{N+1}Y_{N+1}


where N - iteration number, :math:`Y^{N+1}` - random vector that defines direction of move at ( N+1)th iteration, :math:`\alpha_{N+1}` -step size on (N+1)th iteration.
:math:`Y^{N+1}`  must be feasible at each iteration, i.e. the objective functional should decrease: :math:`F(X^{N+1})<(X^{N})`.
Applying expection to (12) and presenting :math:`E[Y_{N+1}` asconditional expectation :math:`E_x E[Y/X]` we get:

.. math::

   X_{N+1} =E[ X_{N}]+ \alpha_{N+1}E_{X^N} E[Y^{N+1}/X^N]


Replacing mathematical expectations :math:`E[ X_{N}]` and :math:`Y_{N+1}]`  with their estimates :math:`\overline E ^{ N+1}` and  :math:`\overline y (X^N)` we get:

.. math::

   \overline E  ^{ N+1} = \overline E  ^{ N}+ \alpha_{N+1} \overline E  _{X^N} [ \overline y (X^N)]


Note that expression for  :math:`\overline y (X^N)` was obtained in the previos section up to certain parameters. By setting parameters to certain values
we can obtain stochastic extensions of well known heuristics such as Nelder and Mead algorithm or Covariance Matrix Adaptation Evolution Strategy.
In minpy library we use several common building blocks to create different algorithms. Customized algorithms may be defined by combining these
common blocks and varying their parameters.

Main building blocks include computing center of mass of the sample points and finding newtonian potential. 


       
Stochastic extention of Nelder and Mead algorithm
-------------------------------------------------

1. Initialize the search by generating :math:`K \geq n`  separate realizations of  :math:`u_0^i`,i=1,..K of the random vector :math:`U_0`.

Set :math:`m_0=\frac{1}{K} \sum_{i=0}^{K} u_0^i`

2. On step k = 1, 2, ...

Compute the mean level :math:` c_{k-1}=\frac{1}{K} \sum_{i=1}^K f(u_{k-1}^i )`

Calculate a new set of vertices:

:math:`u_k^i= m_{k-1}+\epsilon_{k-1} (f(u_{k-1}^i)-c_{k-1})\frac{  m_{k-1} -u_{k-1}^i}  {||m_{k-1} -u_{k-1}^i ||^n }`

Set  :math:`m_k=\frac{1}{K} \sum_{i=0}^K u_k^i`

Adjust the step size :math:`\epsilon_{k-1}` so that :math:`f(m_k)<f(m_{k-1})`.

If approximate :math:`\epsilon _{k-1}` cannot be obtained within the specified number of trails, then set :math:`m_k=m_{k-1}`

Use the sample standard deviation as the termination criterion: :math:`D_k=(\frac{1}{K-1} \sum_{i=1}^K (f(u_k^i)-c_k)^2)^{1/2}`

References
----------
.. [KAP] Kaplinskij, A.I.,Pesin, A.M.,Propoj, A.I.(1994). Analysis of search methods of optimization based on potential theory. I: Nonlocal properties. Automation and Remote Control. N.9, pp.97-105
.. [KP] Kaplinskiĭ, A. I. Propoĭ, A.I , First-order nonlocal optimization methods that use potential theory, Automation and Remote Control,1994
.. [KPP] Kaplinskij, A.I., Pesin, A.M.,Propoj, A.I. (1994). Analysis of search methods of optimization based on potential theory. III: Convergence of methods. Automation and Remote Control.
.. [NBAG] Nikhil Bansal, Anupam Gupta, Potential-function proofs for gradient methods, Theory of Computing, 2019
.. [ATFB] Adrien Taylor, Francis Bach, Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions, 2019
.. [ZALO] Zeyuan Allen-Zhu and Lorenzo Orecchia, Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent, Innovations in Theoretical Computer Science Conference (ITCS), 2017, pp. 3:1-3:22.
.. [BIS] Berthold Immanuel Schmitt, Convergence Analysis for Particle Swarm Optimization, Dissertation, 2015
.. [FGSB] FJuan Frausto-Solis, Ernesto Liñán-García, Juan Paulo Sánchez-Hernández, J. Javier González-Barbosa, Carlos González-Flores, Guadalupe Castilla-Valdez, Multiphase Simulated Annealing Based on Boltzmann and Bose-Einstein Distribution Applied to Protein Folding Problem,  Advances in Bioinformatics, 2016
.. [GLUQ] Gong G., Liu, Y., Qian M, Simulated annealing with a potential function with discontinuous gradient on :math:`R^d`,  Ici. China Ser. A-Math. 44, 571-578, 2001
