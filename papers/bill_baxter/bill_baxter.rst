:author: G William Baxter
:email: gwb6@psu.edu
:institution: Physics, School of Science, Penn State Erie - The Behrend College


----------------------------------------------------------------
Scientific Computing with SciPy for Undergraduate Physics Majors
----------------------------------------------------------------

.. class:: abstract

The physics community is working to improve the undergraduate curriculum to include computer skills that graduates will need in the workforce.  We have added computational tools to our Junior/Senior physics laboratory, PHYS421w Research Methods.  The course uses Linux, LaTeX, Python and SciPy and focuses on software tools for data analysis rather than programming.  The course motivates the learning of these tools through the use of real experiments in which these tools are used.  

.. class:: keywords

   laboratory, computing, software tools, experiment

Introduction
------------

There is a growing debate within the physics community as to what skills a graduate must have to be successful in industry or academia [Chonacky2008]_, [Landau2006]_.  Traditionally many undergraduate physics degrees required one or more courses in programming, currently C++ programming.  Many programs require little or no computer skills from their graduates [Fuller2006]_.  Computing could be taught by way of a new course [Kaplan2004]_, [Spencer2005]_, by adding the material to an existing course [Caballero2014]_, [Timberlake2008]_, [Serbanescu2011]_, [Nakroshis2013]_ or both [Taylor2006]_.  Many degree programs are limited by their total number of credits so that adding a computing course would require removing an existing course.  Easier is to add the material to an existing course.  We added the material to our advanced laboratory, PHYS 421w Research Methods [Hanif2009]_.  Also, in those majors that include computation, the emphasis is often on either simulation or programming numerical techniques.  A strong case can be made that a student would benefit from less programming and more *computing with software tools*, by which we mean analysing data or solving problems with software which might be created by someone else.  We focus on computing with software tools

Research Methods Laboratory
---------------------------

.. table:: Partial list of available experiments for PHYS421w. :label:`experiment`

   ====================================================  =========
   Experiment                                              Type
   ====================================================  =========
   Charge to Mass Ratio of the Electron                     Exp
   Cavendish Measurement of G                               Exp
   Millikan Oil Drop Measurement of e                       Exp
   Harmonic Motion and Torsion Oscillators                  Exp
   Domains and Bubbles in a Magnetic Film                   Exp
   Two-slit Interference, One Photon at a Time              Exp
   Earth's Field Nuclear Magnetic Resonance                 Exp
   Vibrations of a Vertical Chain Fixed at One End          Exp
   Video Microscop of Brownian Motion                       Exp
   Diffusivity in Liquids                                   Exp
   Percolation                                              Sim
   Scaling Properties of Random Walks                       Sim
   Critical Temperature of a Two Dimensiona Ising Model     Sim
   ====================================================  =========


PHYS 421w Research Methods is a 3 credit lab in which each student does three experiments.  These experiments could be classic or modern.  Most are physical experiments, but students are allowed to do a computer simulation for their thrd experiment.  A partial list of experiments is shown in Table :ref:`experiment`.

The course's guiding principles are: 

- Experiments should be as close as possible to the way physics is really done including submitting LaTeX papers for peer-reviewed grading.  
- Emphasis is on software tools for analysis and publication rather than on numerical techniques or programming.
- Software tools presented will be needed in the experiments.  
- All software will be free and open-source.  

The emphasis is on a *realistic* experimental experience.  Students have 4 weeks to setup, perform, analyze, and write up the results of each experiment.  Papers must adhere to the standards for publication in an American Physical Society (APS) 
Physical Review journal.  Papers are reviewed by a set of anomymous reviewers who comment on each paper.  Authors have the opportunity to rewrite their paper and submit it for a final review.  Papers must be written in LaTeX with the APS RevTeX extensions.  


The course is taken by Junior and Senior level physics majors and some engineering students seeking a minor in physics.  Students entering the course typically have had one or more computer programming classes using C++.  They have some familiarity with 
Microsoft Word and Excel.  On average, their programming skills are poor.  Knowing only Excel, they are very poorly prepared for the manipulation and analysis of experimental data.  

The course begins with two lectures introducing the Unix/Linux operating system.  I continue with 4 lectures on LaTeX and BibTeX.  Each lecture is followed by a homework assignment which allows the student to practice the day's topic.  Then a series of lectures on Scientific Python tools follow as shown in Table:ref:`tools`.  All the software tools are necessary for one or more of the experiments so students know that they will be using the tools soon.  Students get custom documentation [1]_ on each topic with many examples showing the commands and the results as well as homework on each topic for practice.  

.. [1] Materials are available for download from `box.psu.edu/gwb6 <http://box.psu.edu/gwb6/>`_.  

.. table:: Software tool categories. :label:`tools`

   ========================== ===============
   Software Tools Topic       Covered?
   ========================== ===============
   Visualization              Always   
   Fitting                    Always
   Array Math                 Always
   Statistics and Uncertainty Always
   Image Processing           As Needed
   Frequency Space            As Needed
   Differential Equations     Always
   [Monte Carlo Techniques]   As Needed (new)
   ========================== ===============


We begin with plotting and visualization.  Viewing data is the first step to determining what to do with it.  Students often have little experience with 
errorbars and no experience with when or how to use logarithmic scales.  This topic also includes reading and writing of data files.  We follow 
this with a discussion and exercises on fitting.  Students are given five noisy data sets.  With no additional information on each, they must determine the correct functional form with uncertainties on all parameters and plot the fitted curve through the data.  "Guessing" the functional form is difficult for many students, but they are strongly motivated by the fact that they know they will have to use this skill in their upcoming experiments.  Examples of the data sets and fitted curves are shown in figure :ref:`fitting`.  

.. figure:: two_fits_b.png

   Examples of two data sets used for fitting practice.  Students are given only a simple data file with no additional information.  They must decide on the appropriate function and the necessary fit parameters.  In **(a)**, :math:`y(x)=8.0e^{-0.5x}\cos{(5.0x)}+0.25` and in **(b)** :math:`y(x)=3.3e^{-2.5(x-2.0)^2} + 0.30x` .  :label:`fitting`

Notice that there is little discussion of the numerical technique.  We are choosing to treat this as a *tool* and save discussions of the details of the numerical technique for a numerical analysis course, an optional course in our major but not a requirement.  After plotting the data, students must determine the appropriate 
function, the necessary parameters required to describe the data, and appropriate initial conditions.  This is very difficult for students who have no prior experience.  Other topics are introduced as needed depending on which experiments students have chosen.  The differential equations material is introduced so that it can be used in a Junior/Senior classical mechanics class offered the following semester.  


Discussion
----------

We have no formal assessment in place; however, anecdotal evidence is positive.  Returning graduates have specifically cited the material on fitting as valuable in 
graduate school.  Faculty have said they value research students who have learned to plot data in this course.  Students display a greater interest in learning software tools when they know they will need to use them shortly.  Nevertheless, it remains a challenge to convince students that they need to know more than Microsoft Excel.  Students and physics faculty alike prefer to stick with what they already know.  Therefore, any course following PHYS421w should reinforce these skills by also requiring students to use these computer tools.  Unfortunately, other than classical mechanics, this is seldom the case.  


References
----------
.. [Caballero2014] M. Caballero and S. Pollock, *A model for incorporating computation without changing the course: An example from middle-division classical mechanics*, American Journal of Physics 82 (2014) pp231-237.

.. [Chonacky2008] N. Chonacky and D. Winch, *Integrating computation into the undergraduate curriculum: A vision and guidelines for future developments*, American Journal of Physics, 76(4&5) (2008) pp327-333.

.. [Fuller2006] R. Fuller, *Numerical Computations in US Undergraduate Physics Courses*, Computing in Science and Engineering, September/October 2006, pp16-21.

.. [Hanif2009] M. Hanif, P. H. Sneddon, F. M. Al-Ahmadi, and R. Reid, *The perceptions, views and opinions of university students about physics learning during undergraduate laboratory work*, Eur J. Phys, 30, 2009, pp85-96.

.. [Kaplan2004] D. Kaplan, *Teaching computation to undergraduate scientists*, SIGSCE, Norfolk, VA, March 3-7, 2004.

.. [Landau2006] R. Landau, *Computational Physics: A better model for physics education?*, Computing in Science and Engineering, September/October 2006, pp22-30.

.. [Nakroshis2013] P. Nakroshis, *Introductory Computational Physics Using Python*, unpublished course notes, 2013.

.. [Serbanescu2011] R. Serbanescu, P. Kushner, and S. Stanley, *Putting computation on a par with experiments and theory in the undergraduate physics curriculum*, American Journal of Physics, 79 (2011), pp919-924.

.. [Spencer2005] R. Spencer, *Teaching computational physics as a laboratory sequence*, 73, (2005), pp151-153.

.. [Taylor2006] J. Taylor and B. King, *Using Computational Methods to Reinvigorate an Undergraduate Physics Curriculum*, Computing in Science and Engineering, September/October 2006, pp38-43.

.. [Timberlake2008] T. Timberlake and J. Hasbun, *Computation in classical mechanics*, American Journal of Physics, 76 (2008), pp334-339.

