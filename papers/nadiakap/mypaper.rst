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
blocks from this library or other Python libraries. 

The library is intended primarily for educational 
purposes and its focus is on transparency of the methods rather than on efficiency of implementation. 

Minpy can be used by researches to design optimization methods with desired properties by varying parameters of the general algorithm. Such design step is very often required in newly developing industry projects, where objective function has unknown behavior, Algorithm designing stage is necessary for two reasons: It helps to understand objective function behavior, and it is needed for function optimization. Very often one method would work best for studying function behavior, and another method - for function optimization.

One typical situation is when different algorithm behavior is required when being far from optimal point, and when being close to optimal point.

As an example, consider variant of simulated annealing proposed in [FGSB]_ where different values of parameters ( Boltzman distribution parameters, step size, etc.) are used depending of the distance to optimal point. In this paper the basic SA algorithm is used as a starting point. We can offer more basic module as a starting point ( and by specifying distribution as 'exponential' get the variant of SA) thus achieving more flexible design opportunities for custom optimization algorithm. Note that convergence of the newly created hybrid algorithm does not need to be verified when using minpy basic modules, whereas previously mentioned SA-based hybrid has to be verified separately ( see [GLUQ]_)

Testing functions are included in the library. They represent broad range of use cases covering above 
mentioned difficult functions. In this paper we describe the approach underlying these optimization methods.
The distinctive feature of these methods is that they are not heuristic in nature. The algorithms are derived 
based on potential theory [KAP1]_, and their convergence is guaranteed by their derivation method [KAP3]_. 
Recently potential theory was applied to prove convergence of well known heuristic methods, for example 
see [BIS]_ for convergence of PSO, and to re prove convergence of well known gradient based methods, in particular, 
first order methods   - see  [NBAG]_ for convergence of gradient descent and [ZALO]_ for mirror descent. 
For potential functions approach for stochastic first order optimization methods see [ATFB]_.

References
----------
.. [KAP1] Kaplinskij, A.I.,Pesin, A.M.,Propoj, A.I.. (1994). Analysis of search methods of optimization based on potential theory. I: Nonlocal properties. Automation and Remote Control. N.9, pp.97-105 
.. [KAP2] A.I. Kaplinskiĭ, A. I. Propoĭ, First-order nonlocal optimization methods that use potential theory, Automation and Remote Control,1994
.. [KAP3] Kaplinskij, A.I., Pesin, A.M.,Propoj, A.I.. (1994). Analysis of search methods of optimization based on potential theory. III: Convergence of methods. Automation and Remote Control. 
.. [NBAG] Nikhil Bansal, Anupam Gupta, Potential-function proofs for gradient methods, Theory of Computing, 2019
.. [ATFB] Adrien Taylor, Francis Bach, Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions, 2019
.. [ZALO] Zeyuan Allen-Zhu and Lorenzo Orecchia, Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent, Innovations in Theoretical Computer Science Conference (ITCS), 2017, pp. 3:1–3:22.
.. [BIS] Berthold Immanuel Schmitt, Convergence Analysis for Particle Swarm Optimization, Dissertation, 2015
.. [FGSB] FJuan Frausto-Solis, Ernesto Liñán-García, Juan Paulo Sánchez-Hernández, J. Javier González-Barbosa, Carlos González-Flores, Guadalupe Castilla-Valdez, Multiphase Simulated Annealing Based on Boltzmann and Bose-Einstein Distribution Applied to Protein Folding Problem,  Advances in Bioinformatics, 2016
.. [GLUQ] Gong G., Liu, Y., Qian M, Simulated annealing with a potential function with discontinuous gradient on $R^d$,  Ici. China Ser. A-Math. 44, 571–578, 2001
