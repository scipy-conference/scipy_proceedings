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

There are currently several tools and frameworks in wide use for agent-based modeling. NetLogo [Wilensky99] provides a complete environment for designing and running models, including its own programming language and a drag-and-drop user interface creator. NetLogo's key features are both strengths and weaknesses: since it is a self-contained environment, it makes it simple to load and run someone else's model or to begin designing a new one. However, the tool does not allow for detailed analysis of model outputs, and they must be exported for analysis elsewhere. NetLogo's custom scripting language is helpful for novices, but is sufficiently specialized as to make it difficult to transition between it and more common general-purpose languages. At the other end of the spectrum, MASON [Luke05] is a powerful Java library for modeling and visualizing multi-agent systems, which has a steep learning curve and requires a large volume of boilerplate code to create even a simple simulation. As with NetLogo, MASON has no built-in facilities to collect and analyze model outputs, requiring developers to write their own data exporter and analyze the output in a different environment. This is also a weakness in RePast [North13], another modeling and simulation toolkit which falls somewhere between NetLogo and MASON in power and ease-of-use.

Unlike Java, which most of the above frameworks are written in, Python is intended for interactive use. In fact, much of the growing scientific Python ecosystem is built around such interactive analysis, particularly using the IPython / Jupyter [CITE] Notebook system. Mesa was written from the ground up to interface with these tools, making it possible to build models, and particularly run and analyze them, interactively within a single environment. 

.. figure:: ipython_screenshot.png

    A Mesa model run and analyzed inside of an IPython Notebook. :label:`fig1`

.. figure:: browser_screenshot.png

    A Mesa model visualized in a browser window. :label:`fig2`

Direct visualization is nevertheless an important part of agent-based modeling: both for debugging, and for developing an intuition of the dynamics that emerge from the model. Mesa facilitiates such live visualization as well. It avoids issues of system-specific GUI dependencies by using the browser as a front-end, giving framework and model developers access to the full range of modern JavaScript data visualization tools.

Architecture
-------------

**Overview**

The guiding princinple of Mesa's architecture is modularity. Mesa makes minimal assumptions about the form a model will take. For example, while many models have spatial components, many others -- particularly economic models, such as [] -- do not; while some models may involve multiple separate spaces. Similarly, visualizations which display each step of a model may be a critical component of some models and completely unneccessary for others. Thus Mesa aims to offer a set of components that can be easily combined and extended to build different kinds of models.

We divide the modules into three overall categories: modeling, analysis and visualization. The modeling components are the core of what's needed to build a model: a **Model** class to store model-level parameters and serve as a container for the rest of the components; one or more **Agent** classes which describe the model agents; most likely a **scheduler** which controls the agent activation regime, and handles time in the model in general, and components describing the **space** and/or **network** the agents are situated in. The analysis components are the **data collectors** used to record data from each model run, and **batch runners** for automating multiple runs and parameter sweeps. Finally, the visualization components are used to map from a model object to one or more visual representations, either as plain-text or via a server interface to a browser window.

.. figure:: mesa_diagram.png

   Simplified UML diagram of Mesa architecture. :label:`fig3`

**Scheduler**

The scheduler is a model component which deserves special attention. Unlike systems dynamics models, and dynamical systems more generally, time in agent-based models is almost never continuous; ABMs are, at bottom, discrete-event simulations. Thus, scheduling the agents' activation is particularly important. First, some terminology. Many models distinguish between a step or tick of the model, and an activation of a single agent, with multiple agent activations in each step of the model. There are numerous possible scheduling regimes used in agent-based modeling, including:

    * Synchronious or simultaneous activation, where all agents act simultaneously. In practice, this is generally implemented by recording each agent's decision one at a time, but not altering the state of the model until all agents have decided.
    * Uniform activation, where all agents are activated in the same order each step of the model. 
    * Random activation, where each agent is activated each step of the model, but the order in which they are activated is randomized for each step.
    * Random interval activation, where the interval between each activation is drawn from a random distribution (most often Poisson). In this regime, there is no set model step; instead, the model maintains an internal 'clock' and schedule which determines which agent will be activated at which time on the internal clock. 
    * More exotic activation regimes may be used as well, such as agents needing to spend resources to activate more frequently.

The activation regime can have a substantial effect on the behavior of a simulation [CITE], yet many ABM frameworks do not make it easy to change. For example, NetLogo defaults to a random activation system, while MASON's scheduler is uniform by default. By separating out the scheduler into a separate, extensible class, Mesa both requires modelers to specify their choice of activation regime, and makes it easy to change and observe the results. Additionally, the scheduler object serves as the model's storage struture for active agents.

All scheduler classes share a few standard method conventions, in order to make them both simple to use and seamlessly interchangable. Schedulers are instantiated with the model object they belong to. Agents are added to the schedule using the ``add_agent`` method, and removed using **remove_agent**. Agents can be added at the very beginning of a simulation, or any time in the middle -- e.g. as they are born from other agents' reproduction. 

The **step** method runs one step of the *model*, activating agents accordingly. It is here that the schedulers primarily differ from one another. For example, the uniform **BaseScheduler** simply loops through the agents in the order they were added, while **RandomActivation** shuffles their order prior to looping.

Each agent is assumed to have a **step()** method, which receives the model state as its sole argument. This is the method that the scheduler calls in order to activate each agent.

The scheduler maintains two variables determining the model clock. **steps** counts how many steps of the model have occured, while **time** tracks the model's simulated clock time. Many models will only utilize **steps**, but a model using Poisson activation, for example, will track both separately, with steps counting individual agent activations and **time** the scheduled model time of the most recent activation. Some models may implement particular schedules simulating real time: for example, **time** may attempt to simulate real-world time, where agent activations simulate them as they engage in different activities of different durations based on the time of day.

**Data Collection**

An agent-based model is not particularly useful if there is no way to see the behaviors and outputs it produces. Generally speaking, there are two ways of extracting these: visualization, which allows for observation and qualitative examination (and which we will discuss below), and quantitative data collection. In order to facilitate the latter option, we provide a generic **Data Collector** class, which can store and export data from most models without needing to be subclassed.

The data collector stores three categories of data: *model-level* variables, *agent-level variables*, and *tables* which are a catch-all for everything else. Model- and agent-level variables are added to the data collector along with a function for collecting them. Model-level collection functions take a model object as an input, while agent-level collection functions take an agent object as an input; both then return a value computed from the model or each agent at their current state. When the data collector's **collect** method is called, with a model object as its argument, it applies each model-level collection function to the model, and stores the results in a dictionary, associating the current value with the current step of the model. Similarly, the method applies each agent-level collection function to each agent currently in the schedule, associating the resulting value with the step of the model, and the agent's unique ID. The Data Collector may be placed within the model class itself, with the collect method running as part of the model step; or externally, with additional code calling it every step or every $N$ steps of the model. 

The third category, *tables*, is used for logging by the model or the agents rather than fixed collection by the data collector itself. Each table consists of a set of columns, stored as dictionaries of lists. The model or agents can then append records to a table according to their own internal logic. This can be used to log specific events (e.g. every time an agent is killed), and data associated with them (e.g. agent lifespan at destruction), particularly when these events do not necessarily occur every step. 

Internally, the data collector stores all variables and tables in Python's standard dictionaries and lists. This reduces the need for external dependencies, and allows the data to be easily exported to JSON or CSV. However, one of the goals of Mesa is facilitating integration with Python's larger scientific and data-analysis ecosystems, and thus the data collector also includes methods for exporting the collected data to pandas [CITE] DataFrames. This allows rapid, interactive processing of the data, easy charting, and access to the full range of statistical and machine-learning tools that are compatible with pandas.

**Batch Runner**

Since most ABMs are stochastic, a single model run gives us only one particular realization of the process the model describes. Furthermore, the questions we want to use ABMs to answer are often about how a particular parameter drives the behavior of the entire system -- requiring multiple model runs with multiple parameter values. In order to facilitate this, Mesa provides the **BatchRunner** class. Like the DataCollector, it does not need to be subclassed in order to conduct parameter sweeps on most models.

The Batch Runner is instantiated with a model class, and a dictionary mapping names of model parameters to either a single value, or a list or range of values. Like the Data Collector, it is also instantiated with dictionaries mapping model- and agent-level variable names to functions used to collect them. The Batch Runner uses the *product* combination generator included in Python's *itertools* library to generate all possible combinations of the parameter values provided. For each combination, the batch collector instantiates a model instance with those parameters, and runs the model until it terminates or a set number of steps has been reached. Once the model terminates, the batch collector runs the reporter functions, collecting data on the model run and storing it along with the relevant parameters. Like the Data Collector, the batch runner can then export the resulting datasets to pandas dataframes.

Visualization
--------------

Mesa uses a browser window to visualize its models. This avoids both the developers and the users needing to deal with cross-system GUI programming; more importantly, perhaps, it gives us access to the universe of advanced JavaScript-based data visualization tools. The entire visualization system is divided into two parts: the server side, and the client side. The server runs the model, and at each step extracts data from it to visualize, which it sends to the client as JSON via a WebSocket connection. The client receives the data, and uses JavaScript to actually draw the data onto the screen for the user.

The actual visualization is done by the visualization modules. Conceptually, each module consists of a server-side and a client-side element. The server-side element is a Python object implementing a ``render`` method, which takes a model instance as an argument and returns a JSON object with the information needed to visualize some part of the model. This might be as simple as a single number representing some model-level statistic, or as complicated as a list of JSON objects, each encoding the position, shape, color and size of an agent on a grid. 

The client-side element is a JavaScript class, which implements a ``render`` method of its own. This method receives the JSON data created by the Python element, and renders it in the browser. This can be as simple as updating the text in a particular HTML paragraph, or as complicated as drawing all the shapes described in the aforementioned list. The object also implements a ``reset`` method, used to reset the visualization element when the model is reset. Finally, the object creates the actual necessary HTML elements in its constructor, and does any other initial setup necessary.

Obviously, the two sides of each visualization must be designed in tandem. They result in one Python class, and one JavaScript ``.js`` file. The path to the JavaScript file is a property of the Python class, meaning that a particular object does not need to include it separately. Mesa includes a variety of pre-built elements, and they are easy to extend or add to.

The ``ModularServer`` class manages the various visualization modules, and is meant to be generic to most models and modules. A visualization is created by instantiating a ``ModularServer`` object with a model class, one or more ``VisualizationElement`` module objects, and model parameters (if necessary). The ``launch()`` method then launches a Tornado server, using templates to insert the JavaScript code specified by the modules to create the client page. The application uses Tornado's coroutines to run the model in parallel with the server itself, so that the model running does not block the serving of the page and the WebSocket data. For each step of the model, each module's ``render`` method extracts the visualization data and stores it in a list. That list item is then sent to the client via WebSocket when the request for that step number is received.


Sample Application
-------------------

References
-----------
.. [Wilensky99] Wilensky, Uri. NetLogo. Evanston, IL: Center for Connected Learning and Computer-Based Modeling, Northwestern University, 1999.
.. [North13] North, Michael J., Nicholson T. Collier, Jonathan Ozik, Eric R. Tatara, Charles M. Macal, Mark Bragen, and Pam Sydelko. “Complex Adaptive Systems Modeling with Repast Simphony.” Complex Adaptive Systems Modeling 1, no. 1 (March 13, 2013): 3. doi:10.1186/2194-3206-1-3.
.. [Luke05] Luke, Sean, Claudio Cioffi-Revilla, Liviu Panait, Keith Sullivan, and Gabriel Balan. “Mason: A Multiagent Simulation Environment.” Simulation 81, no. 7 (2005): 517–27.
