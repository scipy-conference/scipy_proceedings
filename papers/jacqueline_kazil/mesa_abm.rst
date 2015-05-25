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

The guiding princinple of Mesa's architecture is modularity. Mesa makes minimal assumptions about the form a model will take. For example, while many models have spatial components, many others -- particularly economic models, such as [] -- do not; while some models may involve multiple separate spaces. Similarly, visualizations which display each step of a model may be a critical component of some models and completely unneccessary for others. Thus Mesa aims to offer a set of components that can be easily combined and extended to build different kinds of models.

We divide the modules into three overall categories: modeling, analysis and visualization. The modeling components are the core of what's needed to build a model: a **Model** class to store model-level parameters and serve as a container for the rest of the components; one or more **Agent** classes which describe the model agents; most likely a **scheduler** which controls the agent activation regime, and handles time in the model in general, and components describing the **space** and/or **network** the agents are situated in. The analysis components are the **data collectors** used to record data from each model run, and **batch runners** for automating multiple runs and parameter sweeps. Finally, the visualization components are used to map from a model object to one or more visual representations, either as plain-text or via a server interface to a browser window.

[[ Notional UML-ish diagram of what a model looks like, how everything fits together]]

**Scheduler**

The scheduler is a model component which deserves special attention. Unlike systems dynamics models, and dynamical systems more generally, time in agent-based models is almost never continuous; ABMs are, at bottom, discrete-event simulations. Thus, scheduling the agents' activation is particularly important. First, some terminology. Many models distinguish between a step or tick of the model, and an activation of a single agent, with multiple agent activations in each step of the model. There are numerous possible scheduling regimes used in agent-based modeling, including:
	* Synchronious or simultaneous activation, where all agents act simultaneously. In practice, this is generally implemented by recording each agent's decision one at a time, but not altering the state of the model until all agents have decided.
	* Uniform activation, where all agents are activated in the same order each step of the model. 
	* Random activation, where each agent is activated each step of the model, but the order in which they are activated is randomized for each step.
	* Random interval activation, where the interval between each activation is drawn from a random distribution (most often Poisson). In this regime, there is no set model step; instead, the model maintains an internal 'clock' and schedule which determines which agent will be activated at which time on the internal clock. 
	* More exotic activation regimes may be used as well, such as agents needing to spend resources to activate more frequently.

The activation regime can have a substantial effect on the behavior of a simulation [CITE], yet many ABM frameworks do not make it easy to change. For example, NetLogo defaults to a random activation system, while MASON's scheduler is uniform by default. By separating out the scheduler into a separate, extensible class, Mesa both requires modelers to specify their choice of activation regime, and makes it easy to change and observe the results. Additionally, the scheduler object serves as the model's storage struture for active agents.

All scheduler classes share a few standard method conventions, in order to make them both simple to use and seamlessly interchangable. Schedulers are instantiated with the model object they belong to. Agents are added to the schedule using the **add_agent** method, and removed using **remove_agent**. Agents can be added at the very beginning of a simulation, or any time in the middle -- e.g. as they are born from other agents' reproduction. 

The **step** method runs one step of the *model*, activating agents accordingly. It is here that the schedulers primarily differ from one another. For example, the uniform **BaseScheduler** simply loops through the agents in the order they were added, while **RandomActivation** shuffles their order prior to looping.

Each agent is assumed to have a **step()** method, which receives the model state as its sole argument. This is the method that the scheduler calls in order to activate each agent.

The scheduler maintains two variables determining the model clock. **steps** counts how many steps of the model have occured, while **time** tracks the model's simulated clock time. Many models will only utilize **steps**, but a model using Poisson activation, for example, will track both separately, with steps counting individual agent activations and **time** the scheduled model time of the most recent activation. Some models may implement particular schedules simulating real time: for example, **time** may attempt to simulate real-world time, where agent activations simulate them as they engage in different activities of different durations based on the time of day.

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
