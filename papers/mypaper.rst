:author: Nadia Udler
:email: nadiakap@optonline.net
:institution: University of Connecticut (Stamford)
:orcid: 0101-0101-0101-0103

------------------------------------------------
Global optimization software library for research and education
------------------------------------------------

.. class:: abstract

Machine learning models are often represented by functions given by computer programs. Optimization 
of such functions is a challenging task because traditional derivative based 
optimization methods with guaranteed convergence properties cannot be used.. This software 
allows to create new optimization methods with desired properties, based on basic modules. 
These basic modules are designed in accordance with approach for constructing global optimization 
methods based on potential theory[KAP1]_. These methods do not use derivatives of objective function 
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
blocks from this library or other Python libraries. The library is intended primarily for educational 
purposes and its focus is on transparency of the methods rather than on efficiency of implementation. 
Testing functions are included in the library. They represent broad range of use cases covering above 
mentioned difficult functions. In this paper we describe the approach underlying these optimization methods.
The distinctive feature of these methods is that they are not heuristic in nature. The algorithms are derived 
based on potential theory [KAP1], and their convergence is guaranteed by their derivation method. 
Recently potential theory was applied to prove convergence of well known heuristic methods, for example 
see [BIS] for convergence of PSO, and to re prove convergence of well known gradient based methods, in particular, 
first order methods   - see  [NBAG] for convergence of gradient descent and [ZALO] for mirror descent. 
For potential functions approach for stochastic first order optimization methods see [ATFB].


Derivation of the approach
--------------

The approach works for non-smooth or algorithmically defined functions. In this approach the original optimization 
problem is replaced with a randomized problem, allowing the use of Monte-Carlo methods for calculating integrals. 
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

   f(x_1,x_2,..x_n)\to\min_{x \in R_n }


Stochastic extention of Nelder and Mead algorithm
--------------

1.Initialize the search by generating $K \geq n$  separate realizations of  $u_0^i $,i=1,..K of the random vector $ U_0$.  
	
Set $m_0=\frac{1}{K} \sum_{i=0}^{K} u_0^i$

2.On step k = 1, 2, ...

Compute the mean level $ c_{k-1}=\frac{1}{K} \sum_{i=1}^K f(u_{k-1}^i )$

Calculate a new set of vertices:

 $u_k^i= m_{k-1}+\epsilon_{k-1} (f(u_{k-1}^i)-c_{k-1})\frac{  m_{k-1} -u_{k-1}^i}  {||m_{k-1} -u_{k-1}^i ||^n }$

Set  $ m_k=\frac{1}{K} \sum_{i=0}^K u_ k^i $

Adjust the step size $\epsilon_{k-1}$ so that $f(m_k)<f(m_{k-1})$. 

If approximate $\epsilon _{k-1}$ cannot be obtained within the specified number of trails, then set $m_k=m_{k-1}$

 Use the sample standard deviation as the termination criterion: $D_k=(\frac{1}{K-1} \sum_{i=1}^K (f(u_k^i)-c_k)^2)^{1/2} $

References
----------
.. [KAP1] Kaplinskij, A.I. & Pesin, A.M. & Propoj, A.I.. (1994). Analysis of search methods of optimization based on potential theory. I: Nonlocal properties. Automation and Remote Control. N.9, pp.97-105 
.. [KAP2] A.I. Kaplinskiĭ, A. I. Propoĭ, First-order nonlocal optimization methods that use potential theory, Automation and Remote Control,1994
.. [NBAG] Nikhil Bansal, Anupam Gupta, Potential-function proofs for gradient methods, Theory of Computing, 2019
.. [ATFB] Adrien Taylor, Francis Bach, Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions, 2019
.. [ZALO] Zeyuan Allen-Zhu and Lorenzo Orecchia, Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent, Innovations in Theoretical Computer Science Conference (ITCS), 2017, pp. 3:1–3:22.
.. [BIS] Berthold Immanuel Schmitt, Convergence Analysis for Particle Swarm Optimization, Dissertation, 2015

