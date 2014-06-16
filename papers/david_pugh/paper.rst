:author: David R. Pugh
:email: pugh@maths.ox.ac.uk
:institution: Oxford Mathematical Institute, INET@Oxford

-------------------------------------------------------
Python for research and teaching economics: a manifesto 
-------------------------------------------------------

.. class:: abstract
   
   Together with theory and experimentation, computational modeling and simulation has become a “third pillar” of scientific enquiry. I am developing a curriculum for a three part, graduate level course on computational methods designed to increase the exposure of graduate students and researchers in the College of Humanities and Social Sciences at the University of Edinburgh to basic techniques used in computational modeling and simulation using the Python programming language. My course requires no prior knowledge or experience with computer programming or software development and all current and future course materials will be made freely available online via GitHub.

.. class:: keywords

   python, computational economics, dynamic economic models, numerical methods

Introduction
------------
In this paper, I discuss the goals, objectives, and pedagogical choices that I have made in designing and teaching a Python-based course on numerical methods to first-year graduate students in the Scottish Graduate Programme in Economics (SGPE) at the University of Edinburgh. This course would not have been possible without the generous funding and support from the SGPE, the Scottish Insitute for Research in Economics (SIRE), the School of Economics at the University of Edinburgh, and the Challenge Investment Fund (CIF).

The idea for my computational modeling and simulation course emerged out of my PhD research agenda during the summer of 2012. I originally conceived of the computational methods course as a way to partially fill what I felt was a significant gap in the training of economics post-graduate students at the University of Edinburgh. The first iteration of the course consisted of roughly six Python-based laboratory sessions and ran during the 2012-2013 academic year. 

The course was a huge success and I am now working to develop a more extensive curriculum for a three part course on computational methods. The first part of the course is a suite of Python-based laboratory sessions designed to expose students to the basics of computer programming in Python. 

The second part of the course is a week-long intensive computational methods “boot camp.”  The boot camp curriculum focuses on deepening students’ computer programming skills using the Python programming language and teaching important software design principles that are crucial for generating high-quality, reproducible scientific research using computational methods. 

The final part of the course focuses on applying key computational science techniques to economic problems via a series of interactive lectures and tutorials. The curriculum for this part of the course will derive from 

Software
--------
In this section I discuss my decisions regarding software.

Python vs. MATLAB
~~~~~~~~~~~~~~~~~
Discussion of MatLab vs Python. Why Python rocks comapred with MatLab.

The School of Economics at the University of Edinburgh, like most economics departments in the U.S. in Europe has a site license for MATLAB.

Miranda does not have a desire to turn his students into computer programmers. Probably explains why he uses MATLAB! I on the other hand believe that it is important to teach good programming practices to students from the beginning. Too many papers using computational methods (typically MATLAB code) are being published where the code used to generate the results is poorly written and insufficiently documented. This makes results difficult replicate, and even if they can be replicated it is often difficult to understand how the results are being obtained (i.e., what is the code really doing?). Python is an excellent programming langauge in this regard.

Which Python distribution to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First year I taught the course I used Sage. Needed a Python distro that would work on all platforms (even within a VM). Web-based notebooks were really useful for teaching purposes.Sage worked OK, but the creation and rapid advancement of the IPython notebook basically killed Sage for me. For IS reasons really only had a choice between Enthought Canopy and Continuum Analytics Anaconda distributions. I settled on Enthought Canopy for legacy reasons (I had used EPD heavily in my PhD research). Considering switching to Anaconda in the near future in order to make use of the Accelerate add-ons for multi-core and GPU enabled computations.

Which text editor to use?
~~~~~~~~~~~~~~~~~~~~~~~~~
Many possibilities: Typical student is familiar with MS Word and notepad (Texit if a Mac user). This rules out high-performance editors like Vim and Emacs (learning curve is too high). I went with Sublime.  

Version control: Git.
~~~~~~~~~~~~~~~~~~~~~
Never even considered using anything else. Existence of GitHub, particularly now that you can register an academic email and get free private repositories, makes Git the only real choice for version control software. 

Motivating the use of numerical methods in economics
----------------------------------------------------
The typical economics student enters graduate school with great faith in the analytical mathematical tools that he or she was taught as an undergraduate. In particular this student is under the impression that virtually all economic models have closed-form solutions. At worst the typical student believes that if he or she were to encounter an economic model without a close-form solution, then simplifying assumptions could be made that would render the model analytically tractable without sacrificing important economic content. 

The typical economics student is, of course, wrong about general existence of closed-form solutions to economic models. In fact the opposite is true: most economic models, particular dynamic, non-linear models with meaningful constraints (i.e., most any *interesting* model) will fail to have an analytic solution. In order to demonstrate this fact and thereby motivate the use of numerical methods in economics, I begin my course with a laboratory session on the Solow model of economic growth [solow1956]_. 

Economics graduate student are very familiar with the Solow growth model. For many students, the Solow model will have been one of the first macroeconomic models taught to them as undergraduates. Indeed, the dominant macroeconomics textbook for first and second year undergraduates, [mankiw2010]_, devotes two full chapters to motivating and deriving the Solow model. The first few chapters of [romer2011]_, one of the most widely used final year undergraduate and first-year graduate macroeconomics textbook, are also devoted to the Solow growth model and its descendants.

The Solow growth model
~~~~~~~~~~~~~~~~~~~~~~
The Solow model boils down to a single non-linear differential equation and associated initial condition describing the time evolution of capital stock per effective worker, :math:`k(t)`.

.. math::
    \dot{k}(t) = sf(k(t)) - (n + g + \delta)k(t),\ k(t) = k_0

The parameter :math:`0 < s < 1` is the fraction of output invested and the parameters :math:`n, g, \delta` are the rates of population growth, technological progress, and depreciation of physical capital. The intensive form of the production function :math:`f` is assumed to be to be strictly concave with 

.. math::
   f(0) = 0,\ lim_{k\rightarrow 0}\ f' = \infty,\ lim_{k\rightarrow \infty}\ f' = 0. 

A common choice for the function :math:`f` which satisfies the above conditions is known as the Cobb-Douglas production function.

.. math::
   f(k) = k^{\alpha}

Assuming a Cobb-Douglas functional form for :math:`f` also makes the model analytically tractable (and thus contributes to the typical economics student's belief that all such models "must" have an analytic solution). [sato1963]_ showed that the solution to the model under the assumption of Cobb-Douglas production is

.. math::
   :type: eqnarray

   k(t) &=& \Bigg[\bigg(\frac{s}{n+g+\delta}\bigg)\bigg(1 - e^{-(n+g+\delta)(1-\alpha)t}\bigg)+ \notag \\
   &&k_0e^{-(n+g+\delta)(1-\alpha)t}\Bigg]^{\frac{1}{1-\alpha}}.

A notable property of the Solow model with Cobb-Douglas production is that the model predicts that the shares of real income going to capital and labor should be constant. Denoting capital's share of income as :math:`\alpha_K(k)`, the model predicts that 

.. math::
   \alpha_K(k) \equiv \frac{\partial \ln\ f(k)}{\partial \ln\ k} = \alpha

Unfortunately, from figure :ref:`figure1` it is clear that the prediction of constant factor shares is strongly at odds with the empirical data for most countries. Fortunately, there is a simple generalization of the Cobb-Douglas production function, known as the constant elasticity of subsitution (CES) function, that is capable of generating the variable factor shares observed in the data.

.. math::
   f(k) = \bigg[\alpha k^{\rho} + (1-\alpha)\bigg]^{\frac{1}{\rho}}

where :math:`\rho = \frac{\sigma-1}{\sigma}` and :math:`0 < \sigma < \infty` is the elasticity of substitution between capital and effective labor in production. Note that 
   
.. math::
   \lim_{\rho\rightarrow 0} f(k) = k^{\alpha}

and that the CES production function nests the Cobb-Douglas functional form as a special case. To see that the CES production function also generates variable factor shares note that 

.. math::
   \alpha_K(k) \equiv \frac{\partial \ln\ f(k)}{\partial \ln\ k} = \frac{\alpha k^{\rho}}{\alpha k^{\rho} + (1 - \alpha)}

which varies with :math:`k`.

.. figure:: labor-shares.png
   :align: center
   :figclass: w

   Labor's share of real GDP has been declining, on average, for much of the post-war period. For many countries, such as India, China, and South Korea, the fall in labor's share has been dramatic. :label:`figure1`

This seemingly simple generalization of the Cobb-Douglas production function, which is necessary in order for the Solow model generate variable factor share, an economically important feature of the post-war growth experience in most countries, renders the Solow model analytically intractable. To make progress solving a Solow growth model with CES production one needs to resort to computational methods.

Numerically solving the Solow model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A computational solution to the Solow model allows me to demonstrate a number of numerical techniques that students will find generally useful in their own research. 

First and foremost, solving the model requires efficiently and accurately approximating the solution to a non-linear ordinary differential equation (ODE) with a given initial condition (i.e., an non-linear initial value problem). Finite-difference methods are commonly employed to solve such problems. Typical input to such alorithms is the Jacobian matrix of partial derivatives of the system of ODEs. Solving the Solow growth model allows me to demonstrate the use of finite difference methods as well as how to compute Jacobian matrices of non-linear systems of ODEs.  

Much of the empirical work based on the Solow model focuses on the model's predictions concerning the long-run or steady state equilibrium of the model. Solving for the steady state of the Solow growth model requires solving for the roots of a non-linear equation. Root finding problems, which are equivalent to solving systems of typically non-linear equations, are one of the most widely encountered computational problems in economic applications. Typical input to root-finding alorithms is the Jacobian matrix of partial derivatives of the system of non-linear equations. Solving for the steady state of the Solow growth model allows me to demonstrate the use of various root finding algorithms as well as how to compute Jacobian matrices of non-linear systems of equations.

Finally, given some data, estimation of the model's structural parameters (i.e., :math:`\alpha,\ \delta,\ \sigma,\ n,\ g,\ s`) requires solving a non-linear, constrained optimization problem. Typical input to algorithms for solving non-linear programs is the Jacobian of the objective function with respect to the parameters being estimated. The Hessian of the objective function is also needed for computing standard errors of the parameter estimates. Structural estimation of the parameters of the model allows me to demonstrate the use of non-linear optimization algorithms as well as how to compute the Jacobian and Hessian matrices of the objective function. 

In addition to the various generic numerical methods that can be demonstrated in a computational solution to the Solow growth model, solving the model computationally allows me to demonstrate a particular Python workflow that I have found to be useful in a wide variety of scientific computing applications:

1. Specify the original problem symbolically using SymPy. Use Sympy to compute relevant Jacobians and Hessians and then convert them into vectorized functions that are NumPy aware.
2. Solve the functional equations, root-finding, and optimization problems using various SciPy routines.
3. Finally, conduct data analysis and generate publication-ready graphics using Matplotlib, Pandas, and Statsmodels.

Course outline
----------------------
Having motivated the need for computational methods in economics, in this section I outline the three major components of my computational methods course: laboratory sessions, an intensive week-long Python boot camp, and an advanced PhD training course. The first two components are already up and running (thanks to funding support from the SGPE, SIRE, and the CIF). I am still looking to secure funding to develop the advanced training course component.

Laboratory sessions
~~~~~~~~~~~~~~~~~~~
The first part of the course is a suite of Python-based laboratory sessions that run concurrently as part of the core macroeconomics sequence. There are 8 labs in total: two introductory sessions, three labs covering computational methods for solving models that students are taught in macroeconomics I (fall term), three labs covering computational methods for solving models taught in macroeconomics II (winter term).

The material for the two introductory labs draws heavily from parts I and II of `Quantitative Economics`_ by Thomas Sargent and John Stachurski. The material for the remaining 6 labs is designed to complement the core macroeconomic sequence and thus varies a bit from year to year. The purpose of the lab sessions designed to expose students to the basics of scientific computing using Python in a way that reinforces the material covered in the lectures. The laboratory sessions make use of the excellent IPython notebooks. Examples of the laboratory notebooks can be found on GitHub.

* `Initial value problems <http://nbviewer.ipython.org/urls/raw.github.com/davidrpugh/numerical-methods/master/labs/lab-1/lab-1.ipynb>`_ 
* `Boundary value problems <http://nbviewer.ipython.org/urls/raw.github.com/davidrpugh/numerical-methods/master/labs/lab-2/lab-2.ipynb>`_
* `Numerical dynamic programming <http://nbviewer.ipython.org/urls/raw.github.com/davidrpugh/numerical-methods/master/labs/lab-3/lab-3.ipynb)>`_
* `RBC and DSGE models using dynare++ <http://nbviewer.ipython.org/urls/raw.github.com/davidrpugh/numerical-methods/master/labs/lab-4/lab-4.ipynb)>`_

Labs to be included in next years course: DSGE monetary policy models, DSGE models with financial frictions. Labor search. Some of the additional labs are likely to be based around the MSc dissertations of students from this years cohort. I like the idea of getting students directly involved in determining the direction of future iterations of the course.

.. _`Quantitative Economics`: http://quant-econ.net

Python boot camp
~~~~~~~~~~~~~~~~
Whilst the laboratory sessions expose students to some of the basics of programming in Python as well as numerous applications of computational methods in economics, these lab sessions are inadequate preparation for those students wishing to apply such methods as part of their MSc dissertations or PhD theses. 

In order to provide interested students with the skills needed to appy computational methods in their own research I have developed a week-long intensive computational methods “boot camp.” The boot camp requires no prior knowledge or experience with computer programming or software development and all current and future course materials are made freely available online.

Each day of the boot camp is split into morning and afternoon sessions. The morning sessions are designed to develop attendees Python programming skills while teaching important software design principles that are crucial for generating high-quality, reproducible scientific research using computational methods. The syllabus for the morning sessions closely follows `Think Python`_ by Allen Downey.

In teaching Python programming during the boot camp I subscribe to the principle of "learning by doing." As such my primary objective on day one of the Python boot camp is to get attendees up and coding as soon as possible. The goal for the first morning session is to cover the first four chapters of *Think Python*. 

* `Chapter 1`_: The way of the program;
* `Chapter 2`_: Variables, expressions, and statements; 
* `Chapter 3`_: Functions; 
* `Chapter 4`_: Case study on interface design. 

The material in these introductory chapters is clearly presented and historically students have generally had no trouble interactively working through the all four chapters before the lunch break.  Most attendees break for lunch on the first day feeling quite good about themselves. Not only have they covered a lot of material, they have managed to write some basic computer programs. Maintaining student confidence is crucially important. As long as student are confident and feel like they are progressing, they will remain focused on continuing to build their skills. If students get discouraged, perhaps because they are unable to solve a certain exercise or decipher a cryptic error traceback, they will lose their focus and fall behind. 

The second morning session covers the next three chapters of `Think Python`:

* `Chapter 5`_: Conditionals and recursion;
* `Chapter 6`_: Fruitful functions; 
* `Chapter 7`_: Iteration. 

At the start of the session I make a point to emphasize that the material being covered in chapters 5-7 is substantially more difficult than the introductory material covered in the previous morning session and that I do not expect many students to make it through the all of material before lunch. The idea is to manage student expectations by continually reminding them that the course is designed in order that they can learn at their own pace  

The objective of for the third morning session is the morning session of day three the stated objective is for students to work through the material in chapters 8-10 of `Think Python`_.

* `Chapter 8`_: Strings;
* `Chapter 9`_: A case study on word play; 
* `Chapter 10`_: Lists.

The material covered in `chapter 8`_ and `chapter 10`_ is patricularly important as these chapters cover two commonly used Python data types: strings and lists. As a way of drawing attention to the importance of chapters 8 and 10, I encourage students to work through both of these chapters before returning to `chapter 9`_. 

The fourth morning session covers the next four chapters of `Think Python`:

* `Chapter 11`_: Dictionaries;
* `Chapter 12`_: Tuples; 
* `Chapter 13`_: Case study on data structure selection;
* `Chapter 14`_: Files.

The morning session of day four is probably the most demanding. Indeed many students take two full session to work through this material. Chapters 11 and 12 cover two more commonly encoutered and important Python data types: dictionaries and tuples. `Chapter 13`_ is an important case study that demonstrates the importance of thinking about data structures when writing library code. 

The final morning session is designe to cover the remaining five chapters of `Think Python`_ on object-oriented programming (OOP):

* `Chapter 15`_: Classes and Objects;
* `Chapter 16`_: Classes and Functions;
* `Chapter 17`_: Classes and Methods;
* `Chapter 18`_: Inheritance;
* `Chapter 19`_: Case Study on Tkinter.

While this year a few students managed to get through at least some of the OOP chapters, the majority of students managed only to get through chapter 13 over the course of the five, three-hour morning sessions. Those students who did manage to reach the OOP chapters in general failed to grasp the point of OOP and did not see how they might apply OOP ideas in their own research. I see this as a major failing of my teaching. I find OOP ideas extremely intutitive and make use of them to varying degrees in almost all code that I write. I need to find a way to better motivate/present OOP concepts!

.. _`Chapter 1`: http://www.greenteapress.com/thinkpython/html/thinkpython002.html
.. _`Chapter 2`: http://www.greenteapress.com/thinkpython/html/thinkpython003.html
.. _`Chapter 3`: http://www.greenteapress.com/thinkpython/html/thinkpython004.html
.. _`Chapter 4`: http://www.greenteapress.com/thinkpython/html/thinkpython005.html
.. _`Chapter 5`: http://www.greenteapress.com/thinkpython/html/thinkpython006.html
.. _`chapter 6`: http://www.greenteapress.com/thinkpython/html/thinkpython007.html
.. _`chapter 7`: http://www.greenteapress.com/thinkpython/html/thinkpython008.html
.. _`chapter 8`: http://www.greenteapress.com/thinkpython/html/thinkpython009.html
.. _`chapter 9`: http://www.greenteapress.com/thinkpython/html/thinkpython010.html
.. _`chapter 10`: http://www.greenteapress.com/thinkpython/html/thinkpython011.html
.. _`Chapter 11`: http://www.greenteapress.com/thinkpython/html/thinkpython012.html
.. _`chapter 12`: http://www.greenteapress.com/thinkpython/html/thinkpython013.html
.. _`chapter 13`: http://www.greenteapress.com/thinkpython/html/thinkpython014.html
.. _`Chapter 14`: http://www.greenteapress.com/thinkpython/html/thinkpython015.html
.. _`Chapter 15`: http://www.greenteapress.com/thinkpython/html/thinkpython016.html
.. _`Chapter 16`: http://www.greenteapress.com/thinkpython/html/thinkpython017.html
.. _`Chapter 17`: http://www.greenteapress.com/thinkpython/html/thinkpython018.html
.. _`Chapter 18`: http://www.greenteapress.com/thinkpython/html/thinkpython019.html
.. _`Chapter 19`: http://www.greenteapress.com/thinkpython/html/thinkpython020.html

While the morning sessions focus on building the foundations of the Python programming language, the afternoon sessions are devoted to exploring the Python scientific computing stack: IPython, Matplotlib, NumPy, Pandas, SciPy, and SymPy. The afternoon curriculum is built around the `Scientific Programming in Python`_ lecture series, parts I and II of `Quantitative Economics`_ by Thomas Sargent and John Stachurski, and supplemented with specific use cases from my own research.  

.. _`Think Python`: http://www.greenteapress.com/thinkpython
.. _`Scientific Programming in Python`: http://scipy-lectures.github.io

During the afternoon session on day one I motivate the use of Python in scientific computing and spend considerable time getting students set up with a suitable Python environment. This includes a quick tutorial on the Enthought Canopy distribution; discussing the importance of working with a high quality text editor and making sure that student have been able to install Sublime; discussing the importance of using version control in scientific computing and making sure that students have installed Git; making sure that students have installed relevant Sublime plug-ins (i.e., for Git and LaTeX integration, code linting and PEP 8 checking, etc); finally covering the various flavours of IPython interpreter: basic IPython terminal, IPython QTconsole, and the IPython notebook. Objective for the afternoon session is to set up a Python environment for scientific computing and to demonstrate basic scientific work flow. 

I do not teach Git, but rather demonsrate the usefulness of Git to students first as a convenient file sharing technology (an alternative to DropBox). Whilst mentioning the importance of distributed version control. 

Advanced course in numerical methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final part of the course (for which we are still looking for funding to develop!) is a six week course (with 3 lecture hours per week) that focuses on applying key computational science techniques to economic problems via a series of interactive lectures and tutorials.

Teaching material will be based on parts 3 and 4 of `Quantitative Economics`_ with additional material drawn from [judd1998]_.

Conclusion
----------
This is the second year that I have run the boot camp. The first year I did not advertise the course outside of the SGPE. Small group of students (mostly MSc students, but we also have a few economics PhD students from Edinburgh and Glasgow).  This year I decided to advertise the course via SGPE and SIRE. Almost 50 students registered interest in attending this year's Python boot camp.

* Undergraduate economics students from University of Edinburgh.
* SGPE MSc students as well as MSc students from other University of Edinburgh schools (i.e., maths and physics).
* PhD students from at least 5 Scottish Universities.
* PhD students from at least 2 English Universities.
* Faculty members from at least 2 Scottish Universities.
* Faculty members from one English University. 

Of the 50 students that registered interest almost 40 completed the course. 40 students represents a rougly 400% increase in attendance from last year suggesting that there is significant demand amongst UK economists for the type of training that I am providing. 

There is an increasing demand for both applied and theoretical economists interested in inter-disciplinary collaboration. The key to developing and building the capacity for inter-disciplinary research is effective communication using a common language. Historically that common language has been mathematics. Increasingly however this language is becoming computation. Economists and other social sciences can greatly benefit from scientific collaboration and the use of the numerical techniques used across disciplines such as mathematics, physics, biology, computer science and informatics. 

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [judd1998] K. Judd. *Numerical Methods for Economists*, MIT Press, 1998.
.. [mankiw2010] N.G. Mankiw. *Intermediate Macroeconomics, 7th edition*, Worth Publishers, 2010. 
.. [romer2011] D. Romer. *Advanced Macroeconomics, 4th edition*, MacGraw Hill, 2011.
.. [sato1963] R. Sato. *Fiscal policy in a neo-classical growth model: An analysis of time required for equilibrating adjustment*, Review of Economic Studies, 30(1):16-23, 1963.
.. [solow1956] R. Solow. *A contribution to the theory of economic growth*, Quarterly Journal of Economics, 70(1):64-95, 1956.


