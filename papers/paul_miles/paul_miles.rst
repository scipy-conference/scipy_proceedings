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

The Python package pymcmcstat :cite:`pymcmcstat2018v1.6.0` provides a robust platform for a variety of engineering inverse problems.  Bayesian statistical analysis is a powerful tool for parameter estimation, and many algorithms exist for numerical approaches that utilize Markov Chain Monte Carlo (MCMC) methods :cite:`smith2014uncertainty`.
   
In pymcmcstat, the user is provided with a suite of Metropolis based algorithms, with the primary approach being Delayed Rejection Adaptive Metropolis (DRAM) :cite:`haario2006dram`, :cite:`haario2001adaptive`.  A simple procedure of adding data, defining model parameters and settings, and setting up simulation options provides the user with a wide variety of computational tools for considering inverse problem.  This approach to inverse problems utilizes data to provide insight into model limitations and provide accurate estimation of the underlying model and observation uncertainty. 

As many Python packages currently exist for performing MCMC simulations, we had several goals in developing this code.  To our knowledge, no current package contains the :math:`n`-stage delayed rejection algorithm, so pymcmcstat was intended to fill this gap.  Furthermore, many researchers in our community have extensive experience using the MATLAB toolbox `mcmcstat <https://mjlaine.github.io/mcmcstat/>`_.  Our implementation provides a similar user environment, while exploiting Python structures.  We hope to decrease dependence on MATLAB in academic communities by advertising comparable tools in Python.

This package has been applied to a wide variety of engineering problems, including radiation source localization as well as constitutive model development of smart material systems.  This is not an exhaustive listing of scientific problems that could be analyzed using pymcmcstat, and more details regarding the program methodology can be found via the `project homepage <https://github.com/prmiles/pymcmcstat/wiki>`_.

Localization of special nuclear material in urban environments poses a very important task with many challenges.  Accurate representation of radiation transport in a three-dimensional domain that includes various forms of construction materials presents many computational challenges. For a representative domain in Ann Arbor, Michigan we can construct surrogate models using machine learning algorithms based on Monte Carlo N-Particle (MCNP) simulations.  The surrogate models provide a computationally efficient approach for subsequent inverse model calibration, where we consider the source location (:math:`x, y, z`) as our model parameters.  We will demonstrate the viability of using pymcmcstat for localization problems of this nature.

Many smart material systems depend on robust constitutive relations for applications in robotics, flow control, and energy harvesting.  To fully characterize the material or system behavior, uncertainty in the model must be accurately represented.  By using experimental data in conjunction with pymcmcstat, we can estimate the model parameter distributions and visualize how that uncertainty propagates through the system.  We will consider specific examples in viscoelastic modeling of dielectric elastomers and also continuum approximations of ferroelectric monodomain crystal structures.

Methodology
-----------
The goal of Bayesian inference is to estimate the posterior densities :math:`\pi(q|F^{obs}(i))`, which quantify the probability of parameter values given a set of observations.  From Bayes' relation

.. math::
    :label: eqnbayes

    \pi(q|F^{obs}(i)) = \frac{\mathcal{L}(F^{obs}(i)|q)\pi_0(q)}{\int_{\mathbb{R}^p}\mathcal{L}(F^{obs}(i)|q)\pi_0(q)dq},

we observe that the posterior is proportional to the likelihood and prior functions.  The function :math:`\mathcal{L}(F^{obs}(i)|q)` describes the likelihood of the model given a parameter set, and any information known *a priori* about the parameters is defined in the prior distribution :math:`\pi_0(q)`.  The denominator ensures the posterior integrates to unity.

Direct evaluation of (:ref:`eqnbayes`) is often computationally untenable due to the integral in the denominator.  To avoid the issues that arise due to quadrature, we alternatively employ Markov Chain Monte Carlo (MCMC) methods.  In MCMC we use sampling based Metropolis algorithms whose stationary distribution is the posterior density :math:`\pi(q|F^{obs}(i))`.

The pymcmcstat package is designed to work with statistical models of the form

.. math::

    F^{obs}(i) = F(i; q) + \epsilon_i,\; \text{where}\; \epsilon_i\sim\mathit{N}(0, \sigma^2).

We expect the observations :math:`F_i^{obs}` (experimental data or high-fidelity simulations) to equal the model response :math:`F(i; q)` plus independent and identically distributed error :math:`\epsilon_i` with mean zero and observation error variance :math:`\sigma^2`.  A direct result of assuming a statistical model of this nature is that the likelihood function becomes

.. math::
    :label: eqnlikelihood

    \mathcal{L}(F^{obs}(i)|q) = \exp\Big(-\frac{SS_q}{2\sigma^2}\Big),

where :math:`SS_q=\sum_{i=1}^N[F^{obs}(i) - F(i, q)]^2` is the sum-of-squares error.  This is consistent with the observations being independent and normally distributed with :math:`F^{obs}(i)\sim\mathit{N}(F(i;q), \sigma^2)`.  As the observation error variance :math:`\sigma^2` is unknown in many cases, we will often include it as part of the inference process.

There are a wide variety of Metropolis algorithms that may be used within MCMC.  In an ideal case one can adapt the proposal distribution as information is learned about the posterior distribution from accepted candidates.  This is referred to as adaptive Metropolis (AM) and it is implemented in pymcmcstat using the algorithm presented in :cite:`haario2001adaptive`.  Another desirable feature in Metropolis algorithms is to include delayed rejection (DR), which helps to stimulate mixing within the sampling chain.  This has been implemented using the algorithm presented in :cite:`haario2006dram`.  A summary of the Metropolis algorithms available inside pymcmcstat is presented in Table :ref:`tabmetalg`.

.. table:: Metropolis algorithms available in pymcmcstat. :label:`tabmetalg`

   +----+--------------------------+
   |    | Algorithm                |
   +====+==========================+
   | MH | Metropolis-Hastings      |
   +----+--------------------------+
   | AM | Adaptive Metropolis      |
   +----+--------------------------+
   | DR | Delayed Rejection        |
   +----+--------------------------+
   |DRAM| DR + AM                  |
   +----+--------------------------+

Radiation Source Localization
-----------------------------
Efficient and accurate localization of special nuclear material (SNM) in urban environments is a vitally important task to national security and presents many unique computational challenges. A realistic problem requires accounting for radiation transport in 3D, using representative nuclear cross-sections for construction materials, and simulating the expected interaction with a network of detectors.  The details of this research are discussed elsewhere :cite:`miles2019radiation`, and the basic.  For the purpose of this proceeding, we will simply focus on the MCMC implementation.

We can utilize MCMC methods in inferring the source location and intensity, providing us with the posterior estimates.  Given the challenges of modeling the radiation transport physics in 3D, it is extremely useful to visualize the potential source locations in light of the underlying uncertainty.

.. figure:: figures/annarbor_python.png

   Simulated downtown area of Ann Arbor, MI. :label:`figurbanenv`

.. figure:: figures/x_vs_y.eps

   Marginal posteriors from MCMC simulation.  It is observed that several
   regions of high probability were inferred, several of which are reasonably
   close to the true source location. :label:`figxymarg`

.. figure:: figures/x_vs_z.eps

   Marginal posteriors from MCMC simulation.  It is observed that several
   regions of high probability were inferred, several of which are reasonably
   close to the true source location. :label:`figxzmarg`

.. code-block:: python

    def radiation_ssfun(theta, data):
        x, y, z, source_activity = theta
        udo = data.user_defined_object[0]
        GPR = udo['model']
        background = udo['background']
        dwell = udo['dwell']
        dvolume = udo['detector_volume']
        ndet = data.ydata[0].shape[1]
        # evaluate Gaussian Process
        GPout = np.zeros([ndet,])
        for ii in range(ndet):
            loc = np.array([x, y, z])
            tmp1 = np.exp(-GPR[ii].predict(loc))
            tmp2 = souce_activity * dwell * dvolume
            GPout[ii] = tmp1 * tmp2 + background
        # compute residual
        GPres = data.ydata[0] - GPout
        # compute sum-of-squares error
        return (GPres**2).sum(axis=0)

.. code-block:: python

    # Initialize MCMC Object
    mcstat = MCMC()
    # Define simulation options
    mcstat.simulation_options.define_simulation_options(
            savedir=savedir,
            nsimu=5e4,
            updatesigma=True,
            method='dram',
            savesize=1000,
            save_to_json=False,
            save_to_bin=True,
            verbosity=0,
            waitbar=0,
            )
    # Define model settings
    mcset.model_settings.define_model_settings(
            sos_function=ssfun,
            sigma2=observations.mean(axis=0),
        )
    # setup data structure for dram
    mcstat.data.add_data_set(
        x=np.zeros(observations.shape),
        y=observations,
        user_defined_object=dict(
            model=model,
            background=nback * dwell,
            dwell=dwell,
            detector_volume=2098.
            ),
    )
    mcstat.parameters.add_model_parameter(
        name='$x$',
        theta0=(XMAX-XMIN)/2+XMIN,
        minimum=XMIN,
        maximum=XMAX,
    )
    mcstat.parameters.add_model_parameter(
        name='$y$',
        theta0=(YMAX-YMIN)/2+YMIN,
        minimum=YMIN,
        maximum=YMAX,
    )
    mcstat.parameters.add_model_parameter(
        name='$z$',
        theta0=(ZMAX-ZMIN)/2+ZMIN,
        minimum=ZMIN,
        maximum=ZMAX
    )
    mcstat.parameters.add_model_parameter(
        name='$I$',
        theta0=(IMAX-IMIN)/2+IMIN,
        minimum=IMIN,
        maximum=IMAX,
    )
    # Run simulation
    mcstat.run_simulation()

Viscoelastic Modeling of Dielectric Elastomers
----------------------------------------------

Dielectric elastomers as part of adaptive structures provide unique capabilities for control of a structure's shape, stiffness, and damping :cite:`smith2005smart`.  Many of these materials exhibit viscoelastic behavior which varies significantly with the rate of deformation :cite:`rubinstein2003polymer`.  Figure :ref:`figfinalcycles` shows uni-axial experimental data for the elastomer Very High Bond (VHB) 4910, which highlights how the hysteretic behavior increases with the rate of deformation.  For more details regarding the experimental procedure, the reader is referred to :cite:`miles2015bayesian`.

.. figure:: figures/final_cycle_for_each_rate.png
   
    Experimental data for VHB 4910. :label:`figfinalcycles`

.. code-block:: python

    # Setup MCMC object
    mcstat = MCMC()
    # add data
    mcstat.data.add_data_set(
            x=time,
            y=stress,
            user_defined_object=dict(
                    stretch_function=stretch_function,
                    num=400,
                    quadset=None,
                    num_cores=multiprocessing.cpu_count())
            )
    # define model parameters
    mcstat.parameters.add_model_parameter(
            name='$G_c$',
            theta0=lstheta0['Gc'],
            minimum=bounds['Gc'][0],
            maximum=bounds['Gc'][1])
    mcstat.parameters.add_model_parameter(
            name='$G_e$',
            theta0=lstheta0['Ge'],
            minimum=bounds['Ge'][0],
            maximum=bounds['Ge'][1])
    mcstat.parameters.add_model_parameter(
            name='$\\lambda_{max}$',
            theta0=lstheta0['lam_max'],
            minimum=bounds['lam_max'][0],
            maximum=bounds['lam_max'][1])
    mcstat.parameters.add_model_parameter(
            name='$\\eta$',
            theta0=lstheta0['eta'],
            minimum=bounds['eta'][0],
            maximum=bounds['eta'][1])
    mcstat.parameters.add_model_parameter(
            name='$\\gamma$',
            theta0=lstheta0['gamma'],
            minimum=bounds['gamma'][0],
            maximum=bounds['gamma'][1],
            sample=False)
    mcstat.parameters.add_model_parameter(
            name='$\\beta$',
            theta0=lstheta0['beta'],
            minimum=bounds['beta'][0],
            maximum=bounds['beta'][1])
    mcstat.parameters.add_model_parameter(
            name='$\\alpha$',
            theta0=lstheta0['alpha'],
            minimum=bounds['alpha'][0],
            maximum=bounds['alpha'][1])
    # Setup simulation options
    mcstat.simulation_options.define_simulation_options(
            nsimu=nsimu,
            updatesigma=True,
            save_to_bin=True,
            method='dram',
            savedir=outputdir,
            savesize=int(nsimu/10),
            )
    # Define model settings
    mcstat.model_settings.define_model_settings(
            sos_function=dtbx.ssfun,
            sigma2=mse)
    # Execute MCMC simulation
    mcstat.run_simulation()    

Monodomain Crystal Structure Modeling in Ferroelectric Ceramics
---------------------------------------------------------------

Ferroelectric materials are used in a wide variety of engineering applications :cite:`smith2005smart`, necessitating methodologies that can account for uncertainty across multi-scale physics models.  Bayesian statistics allow us to quantify model parameter uncertainty associated with approximating lattice strain and full-field electron density from density functional theory calculations as a homogenized, electromechanical continuum.

Consider the 6th order Landau function, :math:`u(q, {\bf P})`, where :math:`q = [\alpha_{1},\alpha_{11}, \alpha_{111},\alpha_{12},\alpha_{112},\alpha_{123}]`. The Landau energy is a function of 3-dimensional polarization space, :math:`{\bf P}=[P_1, P_2, P_3]`. For the purpose of this example, we consider the case where :math:`P_1 = 0`.  Often times we are interested in using information calculated from Density Functional Theory (DFT) calculations in order to inform our continuum approximations, such as our Landau function. For this example, we will assume we have a set of energy calculations corresponding to different values of :math:`P_2` and :math:`P_3` which were found using DFT. For more details regarding this type of research, the reader is referred to :cite:`miles2018analysis` and :cite:`leon2018analysis`.

.. figure:: figures/monodomain_pairwise.png

   Pairwise correlation from MCMC simulation.  Strong correlation is observed
   between several of the Landau parameters, which supports the results
   from the sensitivity analysis discussed in :cite:`leon2018analysis`. :label:`figmonodomainpairs`
 
Concluding Remarks
------------------
In this paper we have demonstrated two distinct areas of scientific study where MCMC methods provide enhanced understanding of the underlying physics.  The pymcmcstat package presents a robust platform from which to perform a wide array of Bayesian inverse problems.  Several Metropolis algorithms are available, including Delayed Rejection Adaptive Metropolis (DRAM).

With regarding to radiation transport, the resulting posterior distributions illuminate potential source locations with higher probability. In practice, isolating a source location to within the span of a few buildings is a very useful result which can be used to motivate future detector placement.

In considering viscoelasticity models, we calibrated the parameters and propagated the uncertainty through the model to generate credible and prediction intervals.  This provided insight regarding model limitations and led to the implementation of the fractional-order approach.

The pymcmcstat is currently limited to Gaussian likelihood and prior functions.  To improve the overall usefulness of the code will require expanding its functionality to allow for user-defined likelihood and prior functions.  We designed the package to serve as a Python replacement for the MATLAB toolbox `mcmcstat`, so it is important to maintain the features of the original user interface for ease of transition from one platform to another.

Acknowledgments
---------------

This research was supported by the Department of Energy National Nuclear Security Administration (NNSA) under the Award Number DE-NA0002576 through the Consortium for Nonproliferation Enabling Capabilities (CNEC).

References
----------
