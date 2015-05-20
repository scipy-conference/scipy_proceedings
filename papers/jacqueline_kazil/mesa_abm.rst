:author: David Masad
:email: david.masad@gmail.com
:institution: Department of Computational Social Science, George Mason University

:author: Jacqueline Kazil
:email: jackies@email.net
:institution: Department of Computational Social Science, George Mason University

----------------------------------------
Mesa: An Agent-Based Modeling Framework
----------------------------------------

.. class:: abstract
    
    Mesa is a new ABM framework for Python and it's gonna be the best. Some more words will go here. Words words words about what Mesa is and how cool it is. Seriously, pretty cool, even if it isn't really done yet.

.. class:: keywords

        agent based modeling, complexity, simulation


Introduction
------------

*Talk a little about what ABMs are in general, who uses them, and why we should care*

There are currently several tools and frameworks in wide use for agent-based modeling. NetLogo [Wilensky99] provides a complete environment for designing and running models, including its own programming language and a drag-and-drop user interface creator. NetLogo's key features are both strengths and weaknesses: since it is a self-contained environment, it makes it simple to load and run someone else's model or to begin designing a new one. However, the tool does not allow for detailed analysis of model outputs, and they must be exported for analysis elsewhere. NetLogo's custom scripting language is helpful for novices, but is sufficiently specialized as to make it difficult to transition between it and more common general-purpose languages. At the other end of the spectrum, MASON [Luke05] is a powerful Java library for modeling and visualizing multi-agent systems, which exhibits a steep learning curve and requires a large volume of boilerplate code to create even a simple simulation. As with NetLogo, MASON has no built-in facilities to collect and analyze model outputs, requiring developers to write their own data exporter and analyze the output in a different environment. This is also a weakness in RePast [North13], another modeling and simulation toolkit which falls somewhere between NetLogo and MASON in power and ease-of-use.

Unlike Java, which most of the above frameworks are written in, Python is intended for interactive use. In fact, much of the growing scientific Python ecosystem is built around such interactive analysis. 

Architecture
-------------

**Overview**

**Scheduler**

**Data Collection**

**Batch Runner**

Visualization
--------------

Sample Application
-------------------

References
-----------
.. [Wilensky99] Wilensky, Uri. NetLogo. Evanston, IL: Center for Connected Learning and Computer-Based Modeling, Northwestern University, 1999.
.. [North13] North, Michael J., Nicholson T. Collier, Jonathan Ozik, Eric R. Tatara, Charles M. Macal, Mark Bragen, and Pam Sydelko. “Complex Adaptive Systems Modeling with Repast Simphony.” Complex Adaptive Systems Modeling 1, no. 1 (March 13, 2013): 3. doi:10.1186/2194-3206-1-3.
.. [Luke05] Luke, Sean, Claudio Cioffi-Revilla, Liviu Panait, Keith Sullivan, and Gabriel Balan. “Mason: A Multiagent Simulation Environment.” Simulation 81, no. 7 (2005): 517–27.
