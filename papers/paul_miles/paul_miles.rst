:author: Paul R. Miles
:email: prmiles@ncsu.edu
:institution: Department of Mathematics, North Carolina State University, Raleigh, NC 27695

:author: Ralph C. Smith
:email: rsmith@ncsu.edu
:institution: Department of Mathematics, North Carolina State University, Raleigh, NC 27695
:corresponding:

:bibliography: mybib


--------------------------------------------------------
Parameter Estimation Using the Python Package pymcmcstat
--------------------------------------------------------

.. class:: abstract

   Metropolis algorithms have greatly expanded our ability to estimate
   parameter distributions.  In this talk we introduce pymcmcstat
   :cite:`pymcmcstat2018v1.6.0`, which utilizes the Delayed Rejection
   Adaptive Metropolis (DRAM) algorithm :cite:`haario2006dram`,
   :cite:`haario2001adaptive` to perform Markov Chain Monte Carlo (MCMC)
   simulations.  The user interface provides a straight forward environment
   for experienced and new Python users to quickly compare their models
   with data.  Furthermore, the package provides a wide variety of diagnostic
   tools for visualizing uncertainty propagation.  This package has been
   utilized in a wide array of scientific and engineering problems, including
   radiation source localization and constitutive model development of smart
   material systems.

.. class:: keywords

   Markov Chain Monte Carlo (MCMC), Delayed Rejection Adaptive Metropolis (DRAM), Parameter Estimation, Bayesian Inference


Introduction
------------

The Python package pymcmcstat :cite:`pymcmcstat2018v1.6.0` provides a robust
platform for a variety of engineering inverse problems.  Bayesian
statistical analysis is a powerful tool for parameter estimation,
and many algorithms exist for numerical approaches that utilize
Markov Chain Monte Carlo (MCMC) methods :cite:`smith2014uncertainty`.
   
In pymcmcstat, the user is provided with a suite of Metropolis based
algorithms, with the primary approach being Delayed Rejection Adaptive
Metropolis (DRAM) :cite:`haario2006dram`, :cite:`haario2001adaptive`.  A simple
procedure of adding data, defining model parameters and settings, and
setting up simulation options provides the user with a wide variety of
computational tools for considering inverse problem.  This approach to
inverse problems utilizes data to provide insight into model limitations
and provide accurate estimation of the underlying model and observation
uncertainty. 

As many Python packages currently exist for performing MCMC simulations,
we had several goals in developing this code.  To our knowledge, no
current package contains the $n$-stage delayed rejection algorithm,
so pymcmcstat was intended to fill this gap.  Furthermore, many
researchers in our community have extensive experience using the MATLAB
toolbox mcmcstat.  Our implementation provides a similar user environment,
while exploiting Python structures.  We hope to decrease dependence on
MATLAB in academic communities by advertising comparable tools in Python.

This package has been applied to a wide variety of engineering problems,
including radiation source localization as well as constitutive model
development of smart material systems.  This is not an exhaustive listing
of scientific problems that could be analyzed using pymcmcstat, and more
details regarding the program methodology can be found via the project
homepage: https://github.com/prmiles/pymcmcstat/wiki.

Localization of special nuclear material in urban environments poses a
very important task with many challenges.  Accurate representation of
radiation transport in a three-dimensional domain that includes various
forms of construction materials presents many computational challenges.
For a representative domain in Ann Arbor, Michigan we can construct
surrogate models using machine learning algorithms based on Monte Carlo
N-Particle (MCNP) simulations.  The surrogate models provide a
computationally efficient approach for subsequent inverse model
calibration, where we consider the source location (:math:`x, y, z`) as our
model parameters.  We will demonstrate the viability of using pymcmcstat
for localization problems of this nature.

Many smart material systems depend on robust constitutive relations for
applications in robotics, flow control, and energy harvesting.  To fully
characterize the material or system behavior, uncertainty in the model must
be accurately represented.  By using experimental data in conjunction with
pymcmcstat, we can estimate the model parameter distributions and visualize
how that uncertainty propagates through the system.  We will consider
specific examples in viscoelastic modeling of dielectric elastomers and
also continuum approximations of ferroelectric monodomain crystal
structures.


References
----------
