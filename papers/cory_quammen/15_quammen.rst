:author: Cory Quammen
:email: cory.quammen@kitware.com
:institution: Kitware, Inc.

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

----------------------------------------------------------------
Scientific Data Analysis and Visualization with VTK and ParaView
----------------------------------------------------------------

.. class:: abstract

   In this paper, we provide an overview of Python integration in VTK
   and ParaView and give some concrete examples of usage. We also
   provide a roadmap for additional Python integration in VTK and
   ParaView in the future.

.. class:: keywords

   visualization, VTK, ParaView

Introduction
------------

The Visualization Toolkit (VTK) is an open-source, freely available
software system for 3D computer graphics and visualization. It
consists of a set of C++ class libraries and bindings for Python along
with several other languages. VTK supports a wide variety of
visualization algorithms for 2D and 3D scalar, vector, tensor,
texture, and volumetric data, as well as advanced modeling techniques
such as implicit modeling, polygon reduction, mesh smoothing, cutting,
contouring, and Delaunay triangulation. VTK has an extensive
information visualization framework and a suite of 3D interaction
widgets. The toolkit supports parallel processing and integrates with
various GUI toolkits such as Qt and Tk. Python bindings expose nearly
all VTK classes and class methods, making it possible to write full
VTK-based applications in Python. VTK also includes interfaces to
popular Python modules such as NumPy and matplotlib. Support for
writing custom VTK algorithms in Python is also available.

Based on VTK, ParaView is a scalable visualization tool that runs on a
variety of platforms ranging from PCs to some of the largest
supercomputers in the world. ParaView consists of a suite of
executables for generating data visualizations using the techniques
available in VTK. ParaView executables interface with Python in a
number of ways: data sources, filters, and integrated plots can be
defined via Python code, data can be queried with Python expressions,
and several executables can be controlled interactively with Python
commands in an integrated Python shell. Batch processing via Python
scripts that are either written by hand or generated as a trace of
events during an interactive visualization session is available for
offline visualization generation.

Python Language Bindings for VTK
--------------------------------

Since 1997, VTK has provided language bindings for Python. Over the
years, the Python binding support has evolved so that today nearly
every semantic feature of C++ used by VTK has a direct semantic analog
in Python. C++ classes from VTK are wrapped into Python equivalents
and provided in a single module named ``vtk``. The few classes that
are not wrapped are typically limited to abstract base classes that
cannot be instantiated or classes that are meant for internal use in
VTK.

Python Wrapping Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python classes for VTK classes and special types are generated using a
shared lex/yacc-based parser and code generation utilities tailored
for VTK programming conventions. VTK is organized into a number of C++
modules. When built with shared libraries enabled, a library
containing C++ classes is generated at build time for each module.
Each Python-wrapped source file is likewise compiled into a shared
library corresponding to the C++ module. All VTK C++ modules are
provided in a single ``vtk`` Python module.

Importing VTK in Python
~~~~~~~~~~~~~~~~~~~~~~~

VTK is provided as a single ``vtk`` module that can be imported like
any other module. When building VTK from source, if Python support is
enabled, the build process will produce an executable named
``vtkpython``. This is the standard Python executable with environment
variables set to make it simple to import the ``vtk`` module. It is
also possible to use the same ``python`` executable from the Python
installation by setting the PYTHONPATH to the location of VTK's shared
libraries.

VTK Usage
~~~~~~~~~

To access VTK classes, you simply import ``vtk``:

.. code-block:: python

   import vtk

Creation of VTK objects is straightforward:

.. code-block:: python

   contourFilter = vtk.vtkContourFilter()

Each Python object references an underlying VTK object.  Objects in
VTK are reference counted and automatically deleted when no longer
used. The wrapping interface updates the underlying VTK object's
reference count and alleviates the need for explicit memory
management.

One particularly nice semantic equivalence between VTK's C++ and
Python interfaces involves methods that accept a pointer to a C++
array. Such methods are common in VTK to, for instance, set 3D
Cartesian coordinates as properties of classes. In Python, the
corresponding method accepts a tuple or list object. This works well
as long as the list or tuple has the expected number of elements.

.. code-block:: python

   sphere = vtk.vtkSphereSource()
   sphere.SetCenter([0, 1, 0])
   sphere.SetCenter((0, 1, 0))

Methods that return pointers to arrays with a fixed number of elements
are also supported. Such methods require a hint to the wrapping
infrastructure indicating how many elements are in the tuple that is
returned.

.. code-block:: python

   center = sphere.GetCenter()
   print center
   (0, 1, 0)

For VTK classes that have operators ``<``, ``<=``, ``==``, ``>=``, ``>``
defined, equivalent Python operators are provided.

Some class methods in VTK return information via parameters passed by
reference. For example, in the following code block, the parameter
``t`` is a return parameter from the method ``IntersectWithLine``.

.. code-block:: c++

   double t, x[3]
   plane->IntersectWithLine(point1, point2, t, x);

In Python, the equivalent is

.. code-block:: python

   t = mutable(0.0)
   plane.IntersectWithLine(point1, point2, t, x)

Class and method documentation is processed by the wrapping
infrastructure to make it available via the standard ``docstring``
mechanism.

.. code-block:: python

   help(vtk.vtkSphereSource)

The above shows the full documentation of the ``vtkSphereSource``
class (too extensive to list here), while the code below produces help
for only the ``SetCenter`` method.

.. code-block:: python

   help(vtk.vtkSphereSource.SetCenter)

   Help on built-in function SetCenter:

   SetCenter(...)
       V.SetCenter(float, float, float)
       C++: void SetCenter(double, double, double)
       V.SetCenter((float, float, float))
       C++: void SetCenter(double a[3])

Some additional mappings between C++ and Python semantics are
described in the file ``VTK/Wrapping/Python/README_WRAP.txt`` in the
VTK source code repository in versions 4.2 and above.

Integration with NumPy
~~~~~~~~~~~~~~~~~~~~~~

Since 2008, a low-level interface layer between VTK and NumPy has been
added to VTK. In VTK, data associated with points or cells in a data
structure (EXPLAIN THIS SOMEWHERE) is stored in an instance of a
subclass of a ``vtkAbstractArray``. There are limited functions within
VTK itself to process or analyze these arrays. This interface layer
can be used to map VTK arrays to NumPy arrays, enabling the full power
of NumPy operations on those arrays to be used. Suppose we have a data
set from a computation fluid dynamics simulation that we can load with
a VTK reader class that has a point-associated array representing
pressure. We can find several properties of this array using NumPY,
e.g.

.. code-block:: python

   import numpy as np
   import vtk.util.numpy_support as nps

   # Load data with VTK reader
   reader.Update()

   ds = reader.GetOutput()
   pd = ds.GetPointData()
   pressure = pd.GetArray('pressure')
   np_pressure = nps.vtk_to_numpy(pressure)

   min_pressure = np.min(np_pressure)
   max_pressure = np.max(np_pressure)

This interface can also be used to add data arrays to loaded data
sets that can be handed of to VTK for visualization:

.. code-block:: python

   norm_pressure = (np_pressure - min_pressure) / \
      (max_pressure - min_pressure)
   vtk_norm_pressure = np.numpy_to_vtk(norm_pressure, 1)
   vtk_norm_pressure.SetName('normalized pressure')
   pd.AddArray(vtk_norm_pressure)

The second argument to ``np.numpy_to_vtk`` indicates that the NumPy
array should be deep copied to the VTK array. This is necessary if no
reference to the NumPy array will otherwise be kept. If a reference to
the numpy array will be kept, then the second argument can be omitted
and the NumPy array will be shallow copied instead, saving memory and
time for copying.

More recently, a higher-level NumPy-like interface layer has been
added to VTK. This ``numpy_interface`` was designed to combine the
ease of use of NumPy with the distributed memory parallel computing
capabilities and broad data set type support of VTK. The
straightforward interface between VTK data set arrays and NumPy
described above works only when the entire data set is available on
one node. However, data sets in VTK may be distributed across
different computational nodes in a parallel computer using MPI
[Sni99]. In this scenario, global reduction operations using NumPy are
not possible. For this reason, a NumPy-like interface has been added
to VTK that properly handles distributed data sets [Aya14].

A key feature in VTK's ``numpy_interface`` is a set of classes that
wrap VTK data set objects.

.. code-block:: python

   import vtk
   from vtk.numpy_interface import dataset_adapter as dsa

   reader = vtk.vtkXMLPolyDataReader()
   reader.SetFileName(filename)
   reader.Update()
   ds = dsa.WrapDataObject(reader.GetOutput())
   
In this code, ``ds`` is an instance of a ``dataset_adapter.PolyData``
class returned by the ``WrapDataObject`` function because the
``vtkXMLPolyDataReader`` produces a ``vtkPolyData`` data set.  The
wrapper class provides a more Pythonic way of accessing data stored in
VTK arrays.

.. code-block:: python

   ds.PointData.keys()
   ['pressure']

   pressure = ds.PointData['pressure']

Note the the ``pressure`` array here is an instance of ``VTKArray``
rather than a ``vtkAbstractArray``. ``VTKArray`` is a wrapper around
the VTK array object that inherits from ``numpy.ndarray``. All the
standard ``ndarray`` operations on this wrapped array, e.g.,

.. code-block:: python

   >>> pressure[0]
   0.112

   >>> pressure[1:3]
   VTKArray([34.2432, 47.2342, 38.1211], dtype=float32)

   >>> pressure[1:3] + 1
   VTKArray([35.2432, 48.2342, 39.1211], dtype=float32)

   >>> pressure[pressure > 40]
   VTKArray([48.2342], dtype=float32)

The ``numpy_interface.algorithms`` module provides additional
functionality beyond the array interface.

.. code-block:: python

   import vtk.numpy_interface.algorithms as algs

   >>> algs.min(pressure)
   VTKArray(0.1213)

   >>> algs.where(pressure > 40)
   (array(1))

In addition to most of the ufuncs provided by NumPy, the
``algorithms`` interface provides some functions to access quantities
that VTK can compute in the wide variety of data set types (e.g.,
surface meshes, unstructured grids, uniform grids) available in
VTK. This can be used to compute the total volume of cells in an
unstructured grid, for instance,

.. code-block:: python

   cell_volumes = algs.volume(ds)
   algs.sum(cell_volumes)

This example illustrates nicely the power of combining a NumPy-like
interface with VTK's uniform API for computing various quantities on
different types of data sets.

Another distinct advantage of the ``numpy_interface.algorithms``
module is that all operations are supported in parallel when data sets
are distributed across computational nodes. [Aya14] describes this
functionality in more detail.

Integration with matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK excels at interactive 3D rendering of scientific data. Matplotlib
excels at producing publication-quality plots. VTK leverages each
toolkit's strengths in two ways.

As we described earlier, convenience functions for exposing VTK data
arrays as NumPy arrays are provided in the ``vtk.util.numpy_support``
and ``numpy_interface.algorithms`` modules. These arrays can be passed
to matplotlib plotting functions to produce publication-quality plots.

VTK itself incorporates some of matplotlib's rendering capabilities
directly in some cases. When VTK Python wrapping is enabled and
matplotlib is available, VTK use's the ``matplotlib.mathtext`` module
to render LaTeX math expressions to either ``vtkImageData`` objects
that can be displayed as images or to paths that may be rendered to a
``vtkContextView`` object, VTK's version of a canvas.

Qt applications with Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python support in VTK is robust enough to create full-featured
applications without writing a single line of C++ code. PySide (or
PyQt) provide Python bindings for Qt. A simple example is provided below:

VTK filters defined in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While VTK sources and filters are available in Python, they cannot be
subclassed to create new sources or filters because the virtual
function table defined in C++ do not know about methods defined in
Python. Instead, one can subclass from a special ``VTKAlgorithm``
class defined in ``vtk.util.vtkAlgorithm``. This class specifies the
interface for classes that interact with ``vtkPythonAlgorithm``, a
C++ class that delegates the primary VTK data update methods to
the Python class. By doing this, it is possible to implement complex
new sources and filters using Python alone.

Python integration in VTK tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python has become so integral to VTK development that 26% of tests
(544 out of 2046) are written in Python. This outnumbers the number of
Tcl-based tests that were actively added in VTK's early history.


Python and ParaView
-------------------

While ParaView may be built without supporting it, Python is
integrated into ParaView in a number of ways. This section provides an
overview on this integration.

Python Console
~~~~~~~~~~~~~~

ParaView includes a Python console available under the Tools -> Python
Console menu item. This console is a fully-featured Python console with
the environment set up so that the ``vtk`` module is available as well as
a number of modules from ParaView itself. When first started, the command

.. code-block:: python

   from paraview.simple import *

is executed to import the ``paraview.simple`` module. This module
provides a simplified layer of Python functions to execute common
commands in ParaView such as file reading, filter creation, and
chaining filters together to produce data transformations and
visualizations. This layer is described in more detail later.

Running commands in ParaView's Python console is identical to running
commands in other Python consoles. The key difference is that commands
can be used to change the state of the ParaView application. This
gives a similar experience to using a Python console to change
matplotlib plots.

Another way to interact with the Python console is by loading a Python
script with ParaView commands to be executed. This feature is ideal
for Python script development for ParaView. It is also possible to
execute Python scripts from command-line invocations of ParaView by
supplying the Python script as an argument::

   paraview MyScript.py


Simple Layer
~~~~~~~~~~~~

ParaView can be run on several distinct computing resources in a
number of configurations. In a number of configurations, the client
software running on a local workstation connects to a remote process
running on a high-performance computing resource. In most cases, VTK
objects of the same type, such as a filter, exist on all processes in
the overall ParaView application. Because VTK classes for the most
part do not know how to communicate among themselves, ParaView wraps
designated VTK classes in proxy classes that are able to communicate
with each other among distributed processes. This proxy layer is
exposed in the ``paraview.servermanager`` Python module.

The ``paraview.servermanager`` module provides direct access to a
proxy manager class. It can be used to create sources and filters

.. code-block:: python

   pm = paraview.servermanager.ProxyManager()
   ss = pm.NewProxy('sources', 'SphereSource')
   pm.RegisterProxy('sources', 'SphereSource1', ss)
   radius = ss.GetProperty('Radius')
   radius.SetElement(0, 2.0)

   rv = pm.GetProxy('views', 'RenderView1')
   rep = rv.SMProxy.CreateDefaultRepresentation(np, 0)

   # FINISH THIS EXAMPLE

Creating a new data source, a representation for it (how it is
rendered), and adding the representation to the view (where it is
rendered), is an involved process. The ``paraview.simple`` layer
simplifies this process with a set of high-level functions that
take care of most of the tedium. The same example above expressed
in ``paraview.simple`` functions is reduced to

.. code-block:: python

   ss = Sphere(Radius=2.0)
   sd = Show(ss, rv)

Python State Files
~~~~~~~~~~~~~~~~~~

ParaView is able to  supports saving the current state of data, filters, and
rendering parameters to a Python source file that, when executed,
recreates the currrent state in ParaView. The Python state file is
generated in terms of ``paraview.simple`` module functions.

Python Tracing
~~~~~~~~~~~~~~

In addition to saving a snapshot of ParaView's state, live tracing of
user interactions with the ParaView user interface is also supported.
Each time a user performs an interaction that modifies ParaView's
state, Python code is generated that captures the event. This is
implemented via instrumenting the ParaView application at event
handlers. The tracing mechanism can record either the entire state of
proxies or just modifications of state to non-default values to reduce
the trace size. It is also possible to show the trace code as it is
being generating, which can be a useful way to learn Python scripting
in ParaView.

One of ParaView's strenghts is the ability to connect data sources and
filters together into a workflow to perform some action. For example,
in scientific computing it is not uncommon to convert one file format
to another. If ParaView can read the source file format and write the
desitnation file format, it is easy to perform the conversion manually
with the ParaView user interface. For a large list of files, though, a
more automated approach is useful. The Python tracing mechanism
provides a way to generate a conversion script by performing actions in
the user interface, generating a trace, and then modifying the trace to
apply to a series of files.

Python Programmable Source
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Create new sources of data with Python scripts

Python Programmable Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Create new filters with Python scripts

Unified Server Bindings
~~~~~~~~~~~~~~~~~~~~~~~

* In client-server configuration, send Python commands rather than
  using custom message protocols. Reduces size of executables/shared
  libraries.

pvpython and pvbatch
~~~~~~~~~~~~~~~~~~~~


With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }
 
The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.


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
.. [Aya14] U. Ayachit, B. Geveci, *Scientific data analysis and visualization at scale in VTK/ParaView with NumPy*,
           4th Workshop on Python for High Performance and Scientific Computing PyHPC 2014, November, 2014.

.. [Aya15] U. Ayachit, *The ParaView Guide: A Parallel Visualization Application*,
           Kitware, Inc. 2015, ISBN 978-1930934306.

.. [Sch04] W. Schroeder, K. Martin, and B. Lorensen, *The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics*,
           4th ed. Kitware, Inc., 2004, ISBN 1-930934-19-X.

.. [Sni99] M. Snir, S. Otto, S. Huss-Lederman, D. Walker, and J. Dongarra, *MPI - The Complete Reference: Volume 1, The MPI Core*,
           2nd ed., MIT Press, 1999, ISBN 0-262-69215-5.
