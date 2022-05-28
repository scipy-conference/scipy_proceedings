:author: Justin C. Fisher
:email: fisher@smu.edu
:institution: Southern Methodist University, Department of Philosophy
:corresponding:

-------------------------------------------------------------
A New Python API for Webots Robotics Simulations
-------------------------------------------------------------

.. class:: abstract

   Webots is a popular open-source package for 3D robotics simulations. It can also be used as a 3D interactive environment for other physics-based modeling, virtual reality, teaching or games. Webots provided a simple API allowing Python programs to control robots and/or the simulated world, but this API was inefficient and did not provide many "pythonic" conveniences. A new Python API for Webots is presented that is more efficient and provides a more intuitive, easily usable, and "pythonic" interface.
   
.. class:: keywords

   Webots, Python, Robotics, Robot Operating System (ROS), Open Dynamics Engine (ODE), 3D Physics Simulation

1. Introduction
---------------

Webots is a popular open-source package for 3D robotics simulations. It can also be used as a 3D interactive environment for other physics-based modeling, virtual reality, teaching or games. Webots has historically provided a simple Python API, allowing Python programs to control individual robots or the simulated world. This Python API was a thin wrapper over a C++ API, which itself was a wrapper over Webots’ core C API.  These nested layers of API-wrapping were inefficient. Furthermore, this API was not very "pythonic" and did not provide many of the conveniences that help to make development in Python be fast, intuitive, and easy to learn.  This new Python API more efficiently interfaces directly with the Webots C API and provides a more intuitive, easily usable, and "pythonic" interface for controlling Webots robots and simulations.

In qualitative terms, the old API felt like one is awkwardly using Python to call C and C++ functions, whereas the new API feels much simpler, much easier, and like it was fully intended for Python.  Here is a representative (but far from comprehensive) list of examples:

* Unlike the old API, the new API contains helpful Python type annotations and docstrings.
* Webots employs many vectors, e.g., for 3D positions, 4D rotations, and RGB colors.  The old API typically treated these as lists or integers (24-bit colors).  In the new API these are Vector objects,with conveniently addressable components (e.g. `vector.x` or `color.red`), conveneint helper methods like `vector.magnitude` and `vector.unit_vector`, and overloaded vector arithmetic operations, akin to (and interoperable with) NumPy arrays.
* The new API also provides easy interfacing between high-resolution Webots sensors (like cameras and Lidar) and Numpy arrays, to make it much more convenient to use Webots with popular python packages like Numpy, Scipy, PIL or OpenCV.  For example, converting a Webots camera image to a NumPy array is now as simple as `camera.array` and this now allows the array to share memory with the camera, making this extremely fast regardless of image size.
* The old API often required that all function parameters be given explicitly in every call, whereas the new API gives many parameters commonly used default values, allowing them often to be omitted, and keyword arguments to be used where needed.
* Most attributes are now accessible (and alterable, when applicable) by pythonic properties like `motor.velocity`.
* Many devices now have python methods like `__bool__` overloaded in intuitive ways.  E.g., you can now use `if bumper` to detect if a bumper has been pressed, rather than the old `if bumper.getValue()`.
* Pythonic container-like interfaces are now provided.  You may now use `for target in radar` to iterate through the various targets a radar device has detected or `for packet in receiver` to iterate through communication packets that a receiver device has received (and it now automatically handles a wide variety of python objects, not just strings).
* The old API required supervisor controllers to use a wide variety of separate functions to traverse and interact with the simulation’s scene tree, including different functions for different VRML datatypes (like `SFVec3f` or `MFInt32`). The new API automatically invisibly handles these datatypes and translates intuitive python syntax (like dot-notation and square-bracket indexing) to the Webots equivalents.  E.g., you can now move a particular crate 1 meter in the x direction using a command like `world.CRATES[3].translation += [1,0,0]`. Under the old API, this would have required numerous function calls (calling `getNodeFromDef` to find the CRATES node, `getMFNode` to find the child with index 3, `getSFField` to find its translation field, and `getSFVec3f` to retrieve that field’s value, then some list manipulation to alter the x-component of that value, and finally a call to `setSFVec3f` to set the new value).

As another example illustrating how much easier the new API is to use, here are two lines from Webots' sample `supervisor_draw_trail`, as it would appear in the old Python controller.

  root_children_field = supervisor.getField(supervisor.getRoot(), "children")

  root_children_field.importMFNodeFromString(-1, trail_string)

And here is how that looks written in the new controller:

  world.children.append(trail_string)

The new API is mostly backwards-compatible with the old Python Webots API, and provides an option to display deprecation warnings with helpful advice for changing to the new API.

The new Python API is planned for inclusion in an upcoming Webots release, to replace the old one.  In the meantime, an early-access version of it is available, distributed with the same liberal Apache open-source license as Webots itself.

In what follows, the history and motivation for this new API is discussed, including its use in teaching an interdisciplinary undergraduate Cognitive Science course called Minds, Brains and Robotics.  And some of the design decisions for the new API are discussed, which will not only aid in understanding it, but also have broader relevance to parallel dilemmas that face many other software developers.

2. History and Motivation.
--------------------------

Much of this new API was developed by the author in the course of teaching an interdisciplinary University Undergraduate Cognitive Science course entitled Minds, Brains and Robotics.  Before the Covid pandemic, this course had involved lab activities where students build and program physical robots. The pandemic forced these activities to become virtual.  It turned out, though, that Webots simulations actually have many advantages over physical robots, including not requiring any specialized hardware (beyond a decent personal computer), making much more interesting uses of altitude rather than having the robots confined to a safely flat surface, allowing robots to engage in dangerous or destructive activities that would be risky or expensive with physical hardware, allowing a much broader array of sensors including high-resolution cameras, and enabling full-fledged neural network and computational vision simulations.

This interdisciplinary class draws students with diverse backgrounds, and programming skills. Accomodating those with fewer skills required simplifying many of the complexities of the old Webots API.  It also required setting up tools to use Webots "supervisor" powers to help manipulate the simulated world, e.g. to provide students easier customization options for their robots.  The old Webots API made the use of such supervisor powers tedious and difficult, even for expereinced coders, so this practically required developing new tools to streamline this process.  These factors led to the development of an interface that would be much easier for novice students to adapt to, and that would make it much easier for an experienced programmer to make much use of supervisor powers to manipulate the simulated world. Discussion of this with the core Webots development team then led to the decision to incorporate these improvements into Webots, where they can be of benefit to a much broader community.

[Not sure whether to include more examples or images here of ways in which the new API were relevant to this class?]

3. Design Decisions.
--------------------
This section discusses some design decisions that arose in developing this API, and discusses the factors that drove these decisions. This may help give the reader a better understanding of this API, and also of relevant considerations that would arise in many other development scenarios.

3.1. Shifting from functions to properties.
===========================================
The old Python API for Wbots consisted largely of methods like `motor.getVelocity()` and `motor.setVelocity(new_velocity)`.  In the new API these have quite uniformly been changed to Python properties, so these purposes are now accomplished with `motor.velocity` and `motor.velocity = new_velocity`.

Reduction of wordiness and punctuation helps to make programs easier to read and to understand, and it reduces the cognitive load on coders.  However, there are also drawbacks.

One drawback is that, properties can give the mistaken impression that some attributes are computationally cheap to get or set. In cases where this impression would be misleading, more traditional method calls were often retained and/or the comparative expense of the operation was clearly documented.

Two other drawbacks are related.  One is that inviting ordinary users to assign properties might lead them to assign other attributes that could cause problems. Since Python lacks true privacy protections, it has always faced this sort of worry, but this worry becomes even worse when users start to feel familiar moving beyond just using defined methods to interact with an object.

Relatedly, Python debugging provides much more direct feedback in cases where a user misspells `motor.setFoo(v)` than when someone mispells 'motor.foo = v`.  E.g. if a user inadvertently types `motor.setFool(v)` they will get an `AttributeError` noting that `motor` lacks a `setFool` attribute.  But if a user inadvertently types `motor.fool = v`, then Python will silently create a new `fool` attribute for `motor` and the user will often have no idea what has gone wrong.

These drawbacks are related in that they both involve users setting an attribute they shouldn't: either an attribute that has another purpose, or one that doesn't.  Defenses against the first include "hiding" important attributes behind a leading "_", or protecting them with a Python property, which can also help provide useful do-strings.  Unfortunately it's much harder to protect against misspellings in this piece-meal fashion.

This led to the decision to have robot devices like motors and cameras employ a blanket `__setattr__` that will generate warnings if non-property attributes of devices are set from outside the module.  So the user who inadvertently types `motor.fool = v` will immediately be warned of their mistake. This does incur a performance cost, but that cost is often worthwhile when it saves development time and frustration. For cases when performance is crucial, and/or a user wants to live dangerously and meddle inside of API objects, this layer of protection can be deactivated.

3.2 Backwards Compatibility.
============================
The new API offers many new ways of doing things, ways that are often better by most objective metrics, with the main drawback being just that they differ from old ways.  The possibility of making a clean break from the old API was considered, but that would stop old code from working, alienate veteran users, and risk causing a schism akin to the deep one between Python 2 and Python 3 communities.

Another option would have been to refrain from adding a new-and-better feature to avoid introducing backward incompatibilities. But that has obvious drawbacks too.

Instead, a compromise was typically adopted: to provide both the new-and-better way and the worse-old way.  This duplication was eased by shifting from `getFoo`/`setFoo` methods to properties, and from `CamelCase` to Pythonic `snake_case`, which reduced the number of name collisions between old and new.   Employing the worse old way leads to a deprecation warning that includes helpful advice regarding shifting to the new-and-better way of doing things.  This may help users to transition more gradually to the new ways, or they can shut these warnings off to help preserve good will, and hopefully avoid a schism like Python2 vs Python3 communities.

3.1 Separating `robot` and `world`.
===================================
In Webots there is a distinction between "ordinary robots" whose capabilities are generally limited to using the robot's own devices, and "supervisor robots" who share those capabilities, but also have virtual omniscience and omnipotence over most aspects of the simulated world.  In the old API, controller programs would import a `Supervisor` subclass of `Robot`, but typically would still call this unusually powerful robot `robot`, which has led to many confusions.

In the new API these two sorts of powers are strictly separated.  Importing `robot` provides an object that can be used to control the devices in the robot itself. Importing `world` provides an object that can be used to observe and enact changes anywhere in the simulated world (presuming that the controller has such permissions, of course).  In many real use cases, god-like supervisor robots don't actually have bodies and devices of their own, and just use their supervisor powers incorporeally, so all they will need is `world`.  In the case where a robot's controller wants to exert both forms of control, it can import both `robot` to control its own body, and `world` to control the rest of the world.

This distinction helps to make things more intuitively clear.  It also frees `world` from having all the properties and methods that `robot` has, which in turn reduces the risk of name-collisions as `world` takes on the role of serving as the root of the proxy scene tree.  In the new API, `world.children` refers to the `children` field of the root of the scene tree which contains (almost) all of the simulated world, `world.WorldInfo` refers to one of these children, a `WorldInfo` node, and `world.ROBOT2` dynamically returns a node within the world whose Webots DEF-name is "ROBOT2".  These uses of `world` would have been much less intuitive if users thought of `world` as being a special sort of robot, rather than as being their handle on controlling the simulated world.  Other sorts of supervisor functionality also are very intuitively associated with world, like `world.save(filename)` to save the state of the simulated world, or `world.mode = "PAUSE"`.

Having `world.attributes` dynamically fetch nodes and fields from the scene tree did come with some drawbacks.  There is a risk of name-collisions, though these are rare since Webots field-names are known in advance, and nodes are typically sought by ALL-CAPS DEF-names, which won't collide with `world` 's lower-case and MixedCase attributes.  Linters like Pycharm also are confused by such dynamic references, which is unfortunate, but does not stop such dynamic references from being extremely useful.

4. More Information
-------------------
Lengthy discussion of the new API and its planned inclusion in Webots is here: https://github.com/cyberbotics/webots/pull/3801

A working early-access version of the new API and a variety of sample programs are available here: https://github.com/Justin-Fisher/new_python_api_for_webots

Webots home page, including freee download of Webots: https://cyberbotics.com/