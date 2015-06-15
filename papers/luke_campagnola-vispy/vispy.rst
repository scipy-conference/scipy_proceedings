:author: Luke Campagnola
:email: luke.campagnola@gmail.com
:institution: University of North Carolina at Chapel Hill

------------------------------------------------------------
VisPy: Harnessing The GPU For Fast, High-Level Visualization
------------------------------------------------------------

.. class:: abstract

   The growing availability of large, multidimensional data sets has created
   demand for high-performance, interactive visualization tools. VisPy 
   leverages the GPU to provide fast, interactive, and beautiful visualizations
   in a high-level API. Here we introduce the main features,
   architecture, and techniques used in VisPy.

.. class:: keywords

   graphics, visualization, plotting, performance, interactive, opengl 


Motivation
----------

Despite the rapid growth of the scientific python stack, one conspicuously absent element is a standard package for high-performance visualization. The de-facto standard for plotting is Matplotlib; however, this package is designed for publication graphics and is not optimized for visualizations that require realtime, interactive performance, or that incorporate very large data volumes. Several smaller packages have apeared in the python ecosystem to fill this gap (vtk, chaco, pyqtgraph, visvis, galry, glumpy, etc.), but none has yet emerged as a serious candidate standard package. Consequently, these projects have expended much effort re-solving the same problems, and the ecosystem as a whole suffers from the lack of a focused, collaborative effort.

In recognition of this problem and the potential benefit to the Python community, VisPy was created as a collaborative effort to succeed several of these projects (visvis, galry, glumpy, and the visualization components of pyqtgraph). VisPy has quickly grown an active community of developers and is approaching beta status.


What is VisPy
-------------

VisPy is a pure-Python graphics library for scientific visualization based on OpenGL. VisPy aims to provide publication-quality graphics, a high-level 2D and 3D plotting API, and web publishing capabilities. By making use of the modern, shader-based OpenGL pipeline, most of the graphical rendering cost is offloaded to the graphics processor (GPU), thereby allowing both interactive framerates and high data throughput. This also relieves VisPy of the need for CPU-optimized code or numerous external dependencies; VisPy depends only on NumPy and a suitable GUI library. 



VisPy's Architecture
--------------------

VisPy's functionality is divided into a layered architecture, with each new layer providing higher-level primitives. The top layers provide a and powerful system for quickly and easily visualizing data, while the lower layers provide greater flexibility and control over OpenGL's features.


Layer 1: Object-Oriented GL
'''''''''''''''''''''''''''

The OpenGL API, although very powerful, is also somewhat verbose and unwieldy. VisPy's lowest-level layer, ``vispy.gloo``, provides an object-oriented wrapper that provides a clean, compact, and Pythonic alternative to traditional OpenGL programming. Objects such as textures, vertex buffers, frame buffers, and shader programs that typically require several GL calls to instantiate are instead encapsulated in simple Python classes (e.g. Table 1).

.. table:: Example comparing ``vispy.gloo`` API to OpenGL API. Using ``vispy.gloo`` is compact, clean, and Pythonic.
   :class: w

   +-----------------------------------------------+------------------------------------------------------------------+
   |            ``vispy.gloo``                     |            ``pyopengl``                                          |
   +===============================================+==================================================================+
   |                                               |                                                                  |
   |.. code-block:: python                         |.. code-block:: python                                            |
   |                                               |                                                                  |
   |   # create a shader program and assign        |   prg = glCreateProgram()                                        |
   |   # a value to the 'color' variable           |   vsh = glCreateShader(GL_VERTEX_SHADER)                         |
   |   program = Program(vert_code, frag_code)     |   glShaderSource(vsh, vert_code)                                 |
   |   program['color'] = (1, 0.5, 0, 1)           |   fsh = glCreateShader(GL_FRAGMENT_SHADER)                       |
   |                                               |   glShaderSource(fsh, vert_code)                                 |
   |                                               |   for shader in (vsh, fsh):                                      |
   |                                               |       glCompileShader(shader)                                    |
   |                                               |       assert glGetShaderParameter(shader, GL_COMPILE_STATUS) = 1 |
   |                                               |       glAttachShader(prg, shader)                                |
   |                                               |                                                                  |
   |                                               |   glLinkProgram(prg)                                             |
   |                                               |   assert glGetProgramParameter(prg, GL_LINK_STATUS) == 1         |
   |                                               |   nunif = glGetProgramParameter(prg, GL_ACTIVE_UNIFORMS)         |
   |                                               |   uniforms = {}                                                  |
   |                                               |   for i in range(nunif):                                         |
   |                                               |       name, id, typ = glGetActiveAttrib(prg, i)                  |
   |                                               |       uniforms[name] = id                                        |
   |                                               |   glUseProgram(prg)                                              |
   |                                               |   glUniform4fv(uniforms['color'], 1, (1.0, 0.5, 0.0, 1.0))       |
   +-----------------------------------------------+------------------------------------------------------------------+

   

OpenGL commands cannot be invoked until a context (usually provided by the GUI toolkit) has been created and activated. This requirement imposes program design limitations that can make OpenGL programs more awkward. To circumvent this restriction, ``vispy.gloo`` uses a context management system that queues all OpenGL commands until the appropriate context has become active. The direct benefit is that the end user is free to interact with ``vispy.gloo`` however makes sense for their program. Most notably, ``vispy.gloo`` objects can be instantiated as the program starts up, before any context is available.

The command queues used by ``vispy.gloo`` are also designed to be serializable such that commands generated in one process or thread can be executed in another. In this way, a stream of GL commands could be sent to a web browser (as in the case of IPython notebook), recorded to disk and replayed later, or shared between processes to take advantage of multi-core systems.

Another purpose of ``vispy.gloo`` is to hide many of the differences between various versions and implementations of OpenGL. We currently target OpenGL versions 2.1 (desktop) and ES2.0 (embedded and WebGL), which is available on virtually all commodity hardware today. A closely related system, ``vispy.app``, also abstracts the differences between the various supported GUI backends, which include PyQt4/5, PySide, IPython, SDL, GLFW, and several others. This support, combined with VisPy's pure-python and low-dependency approach, helps to ensure that VisPy will run on most platforms with minimal effort from users and developers alike.


Layer 2: Visuals
''''''''''''''''

The core of VisPy is its library of ``Visual`` classes that provide the primitive graphical objects used to build more complex visualizations. These objects range from very simple primitives (lines, points, triangles) to more powerful primitives (text, volumes, images), to high-level visualization tools (histograms, surface plots, spectrograms, isosurfaces). 

Internally, visuals upload their data to graphics memory and implement a shader program [ref GLSL] that is executed on the GPU. This allows the most computationally intensive operations to run in compiled, parallelized code without adding any build dependencies (because all OpenGL implementations since 2.1 include a GLSL compiler). Visuals can be reconfigured and updated in real time by simply uploading new data or shaders to the GPU.

Visuals may also be modified by applying arbitrary filters (such as opacity, clipping, and lighting) and coordinate transformations. To support this flexibility, it is necessary to be able to recombine smaller chunks of shader code. VisPy implements a shader management system that allows independent GLSL functions to be attached together in a single shader program. This allows [reword] the insertion of arbitrary coordinate transformations and color modification into each visual's shader program.

VisPy implements a collection of coordinate transformation classes that are used to map between a visual's raw data and its output coordinate system (screen, image, svg, etc.). By offloading coordinate transformations to the GPU alon with drawing operations, VisPy makes it possible to stream data directly from its source to the GPU without any modification in Python. Most transforms affect the location, orientation, and scaling of visuals and can be chained together to produce more complex adjustments. Transforms may also be nonlinear, as in logarithmic, polar, and mercator projections, and custom transforms can be implemented easily given the forward and inverse mapping functions in both Python and GLSL.

.. figure:: image_transforms.png

   One image viewed using four different coordinate transformations. VisPy supports linear transformations such as scaling, translation, and affine matrix multiplication (bottom left) as well as nonlinear transformations such as logarithmic (top left) and polar (top right). Custom transform classes are also easy to construct (bottom right).


Layer 3: Scenegraph
'''''''''''''''''''

The majority of VisPy's graphical features can be accessed by working directly with its Visual classes. However, managing the visuals, coordinate transforms, and filters for a complex scene can be confusing and tedious. To automate this process, VisPy implements a scenegraph |---| a standard data structure used in computer graphics that organizes visuals into a hierarchy. Each node in the hierarchy inherits coordinate transformations and filters from its parent. VisPy's scenegraph allows visuals to be easily arranged in a scene and, in automating control of the system of transformations, it is able to handle some common interactive visualization requirements:

* Picking: mouse and touch events are delivered to the objects in the scene that are clicked on. This works by rendering the scene using unique colors for each visual; thus the otherwise expensive ray casting computation is carried out on the GPU.
* Interactive viewports: allow the user to interactively pan, scale, and rotate data within the view
* Cameras: library of camera classes, each implementing a different mode of user interaction. For example, `PanZoomCamera` allows panning and scaling for 2D plot data, whereas `ArcballCamera` allows data to be rotated in 3D like a trackball.
* Lighting: user may add lights to the scene and shaded objects will react automatically.
* Export: any portion of the scene may be rendered to an image at any resolution. In the future, the scenegraph will also support exporting to SVG.
* Layouts: automatically partition window space into grids.
* High-resolution displays -- the scenegraph automatically corrects for high-resolution displays to ensure visuals are scaled correctly on all devices.

.. code-block:: python

   import vispy.scene as vs
   
   # Create a window with a grid layout inside
   window = vs.SceneCanvas()
   grid = window.central_widget.add_grid()
   
   # Create a view with a 2D line plot
   view1 = grid.add_view(row=0, col=0, camera='panzoom')
   plot = vs.PlotLine(data1, parent=view1.scene)
   
   # Create a second view with a 3D surface plot
   view2 = grid.add_view(row=0, col=1, camera='turntable')
   axes = vs.SurfacePlot(data2, parent=view2.scene)
   
   # start UI event loop
   window.app.run()



Layer 4: Plotting
'''''''''''''''''

VisPy's highest level API allows quick and easy access to data visualization, similar to `matplotlib.pyplot` [ref]
Intended for simple analysis scripts and command line / ipython use, but generates scenegraph structures allowing lower level control over the visual output. 

[code example]

Also includes `mpl_plot`, which uses `mplexporter` [ref mpld3] to convert any matplotlib visualization to vispy (however this approach is not expected to have the same performance benefits as using the native vispy.plot API).



Conclusion
----------

