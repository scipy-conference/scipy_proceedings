:author: Nick Bray
:email: ncbray@google.com
:institution: Google

------------------------------------------------
PyStream: Compiling Python onto the GPU
------------------------------------------------

.. class:: abstract

PyStream is a static compiler that can radically transform Python code and run it on a Graphics Processing Unit (GPU).  Python compiled to run on the GPU is ~100,000x faster than when interpreted on the CPU.  The PyStream compiler is specially designed to simplify the development of real-time rendering systems by allowing the entire rendering system to be written in a single, highly productive language.  Without PyStream, GPU-accelerated real-time rendering systems must contain two separate code bases written in two separate languages: one for the CPU and one for the GPU.  Functions and data structures are not shared between the code bases, and any common functionality must be redundantly written in both languages.  PyStream unifies a rendering system into a single, Python code base, allowing functions and data structures to be transparently shared between the CPU and the GPU.  A single, unified code base makes it easy to create, maintain, and evolve a high-performance GPU-accelerated application.

.. class:: keywords

   pystream, compiling python, gpu


Introduction
------------

High-performance computer hardware can be difficult to program because ease of programming is often traded for raw performance.  For example, graphics processing units (GPUs) pack so many floating point units into a single chip that overall performance can easily be limited by the memory subsystem's ability to provide data to the functional units.  GPUs have traditionally been programmed either in languages that restrict memory access or in languages that explicitly expose management of the memory hierarchy to the programmer.  The OpenGL Shading Language (GLSL) is an example of the former, and OpenCL is an example of the latter.

GPU-specific languages are specialized for performance-critical numeric computations and are not suitable for writing every part of an application.  For example, they cannot load data from disk or provide a graphical user interface.  Instead, GPU languages typically provide APIs so they can interoperate with a different, more general-purpose language.  By design, using a GPU-specific language results in a system with two code bases, each with distinct semantics.  "Glue code" is required to overcome the impedance mismatch between these code bases.  Glue code is code that remaps and transfers data structures from one code base to another.  Glue code is also used to invoke functions across the language boundary.

.. figure:: cow-fog-incorrect.png

   Naïve background color. :label:`cow-fog-incorrect`


.. figure:: cow-fog-correct.png

   Background color calculation using shared code. :label:`cow-fog-correct`

Constructing an application in this manner has a number of software engineering costs.  For example, any data transferred between the CPU and the GPU must have its structure defined in both languages and have glue code to remap and transfer the data.  Functions may also need to be duplicated in each code base.  For example, Figure :ref:`cow-fog-incorrect` shows a rendering system where the background color, specified on the CPU, does not take into account the color processing performed on the GPU.  Figure :ref:`cow-fog-correct` shows the same rendering system, but with the color-processing code duplicated on the CPU and applied to the background color.  Ultimately, software systems that incorporate GPU-specific languages can be difficult to maintain and modify because of the duplicated code that must be kept in sync.


PyStream
--------------

PyStream [Bra10]_ is a static source-to-source compiler that can translate Python code into GLSL for use in real-time rendering systems.  PyStream also generates the glue code necessary to seamlessly invoke the generated GLSL code from Python.  PyStream allows a GPU-accelerated real time rendering systems to be written as a single, unified Python code base.  This allows the high productivity of the Python language to be used while also gaining the performance of GPU acceleration.  PyStream eliminates the problems that arise from having two code bases, which would otherwise diminish the productivity gains of using Python.  The PyStream compiler allows many of Python's features to be used on the GPU, including object-oriented programming and polymorphic functions.  Most GPU-specific languages do not support these features.  Python is also much more concise.  In practice, Python code is roughly five to seven times more terse than the corresponding GLSL code.


Language Restrictions
---------------------

PyStream allows a restricted subset of Python to run on the GPU.  Restrictions are necessary to make the compilation process tractable.  PyStream's restricted subset provides, at minimum, the functionality of GLSL but with the syntax, semantics, and abstraction mechanisms of Python.  PyStream requires the following of any code that with be translated onto the GPU:

* A closed world
* No global side effects
* No recursive function calls
* Bounded memory usage

To statically compile a Python program, a closed world must be created.  If an unknown function is called, the function can have arbitrary side effects: rewriting globals, classes, and other data structures.  Static analysis cannot prove anything in such a situation.  All code in a PyStream program must be known at compile time.  Dynamic code compilation and execution is completely disallowed.  Import statements are assumed to always import the code found during compilation.

GLSL has a few hard restrictions that are adopted by PyStream so that it can generate GLSL code.  GLSL programs are constrained to have no global side effects.  PyStream programs must do the same.  GLSL is designed to run on hardware without a call stack and it does not allow recursive function calls.  This restriction is adopted by PyStream.  Similarly, GLSL is designed to run in an environment where memory is statically allocated for each processor.  PyStream requires its programs have bounded memory usage so that the compiler can statically allocate memory when generating GLSL code.

Most of these restrictions are applied after the program has been optimized.  For example, a highly polymorphic function may initially appear to be recursive, but this recursion can disappear once the function has been duplicated and specialized for each polymorphic context.  As will be discussed later, PyStream uses a novel approach for representing Python programs. This approach treats the Python interpreter as part of the program being compiled.  There are often recursive calls through the interpreter, such as when the addition of a vector type is implemented in terms of the addition of its scalar elements.  This pattern is pervasive enough that recursive calls cannot be disallowed up front.  Disallowing recursive calls after compilation requires that a programmer understand how the compiler behaves.  Although this is undesirable, it is unclear how Python could be mapped onto the GPU without making this conceit.

PyStream currently does not support a few Python features, including exceptions and closures.  These features will be supported in the future.

PyStream in Practice
--------------------

.. figure:: ao-complete.png

   An image produced by the example rendering system. :label:`ao-complete`


A real-time rendering system was developed in tandem with the PyStream compiler to validate the design of the compiler.  The example rendering system implements the core algorithms used by the game Starcraft 2 [Fil08]_.  Rendering systems typically use many different algorithms to produce a final image.  These algorithms are divided into "shader programs" that that are executed on batches of data sent to the GPU.  The example rendering system contains 8 different shader programs.  Shader programs are  subdivided into individual "shaders" that process different kinds of data, such as vertices in a 3D model or pixels being written into an image.  The code for one of the shader programs is included below.

.. code-block:: python

   class AmbientPass(ShaderProgram):

     def shadeVertex(self, context, pos, texCoord):
       context.position = pos
       return texCoord,

     def shadeFragment(self, context, texCoord):
       # Sample the underlying geometry
       g = self.gbuffer.sample(texCoord)
       # Sample the ambient occlusion
       ao = self.ao.texture(texCoord).xyz
       # Calculate the lighting
       ambientLight = self.env.ambientColor(g.normal)*ao
       # Modulate the output
       color=vec4(g.diffuse*ambientLight, 1.0)
       context.colors = (color,)

This shader program implements a special kind of lighting calculation for the example rendering system.  The class contains two shader methods: a vertex shader and a fragment shader.  The first two arguments for each shader are special.  The ``self`` argument holds data that is constant during the execution of the shader.  The ``context`` argument holds an object with shader-specific fields.  For example, colors written to the ``context.colors`` field inside of a fragment shader will be written into the image(s) being rendered once the shader exits.  All subsequent arguments correspond to streams of data being fed into the shader.  Return values correspond to streams of data produced by the shader.


Compiling Python
-----------------

PyStream takes a novel approach to compiling Python that is simpler and more flexible than previous approaches [Sal04], [Pyp11].  The key to this approach is keeping the representation of the program being compiled as simple as possible.  Compiling Python can be quite complicated because the language is filled with numerous special cases.  In fact, most operations can result in arbitrary Python code being executed.  Instead of trying to embed a complete knowledge of Python's semantics into its algorithms, the PyStream compiler treats the interpreter as if it were a library being called by the Python program. This allows PyStream to easily analyze Python's complex semantics without complicating the compiler.  This new intermediate representation is called the Simplified Python Intermediate Language (SPIL) [Bra11]_.  Because SPIL treats the interpreter as part of the program,  standard optimizations such as dead code elimination and function inlining are extremely effective at eliminating Python's runtime overhead.  Several Python-specific transformations are also performed, such as optimizing method calls to eliminate the creation of bound method objects.


Mapping Python onto the GPU
---------------------------

One of the biggest challenges in mapping a Python shader program onto the GPU is the presence of memory operations.  GLSL does not support pointers in any form: the address of an objects cannot be taken, and function arguments are passed by value.  PyStream is designed to aggressively eliminate as many memory operations as possible to simplify the translation to GLSL.  The optimizations it performs are a mixture of load/store elimination and PyStream-specific transformations such as flattening the input and output data structures for each shader into a list of local variables.  Some memory operations may still remain after these optimizations, however.  Any remaining memory operations and are implemented in GLSL using indirections through arrays.

The biggest constraint on PyStream's ability to map Python shaders onto the GPU is that it must be able to bound and statically allocate all of the memory a shader may use.  This means recursive data structures cannot be translated.  It is also often impossible to bound the size of mutable container objects, which means lists and dictionaries can only be used in specific ways.


Performance
-------------

Evaluating the performance of the PyStream compiler is difficult, because there is no other compiler like it.  A manual inspection of the generated GLSL code reveals that it is close to what would be written by hand.  Quantitatively, PyStream provides a massive speedup for the compiled shaders.  The following table shows the time taken to draw one million pixels with a Python shader program when it is compiled onto the GPU versus when it is interpreted on the CPU.  Measurements were taken on a AMD Athlon 64 X2 3800+ CPU with a NVidia 9800 GT GPU running Windows XP and Python 2.6.4.

.. For some reason Miktex doesn't like this table, so a raw version (below)is used instead.
   +-----------+-------------------+----------+
   |  Shader   |   GPU   |   CPU   | Speedup  |
   +===========+=========+=========+==========+
   | material  | 5.62 ms | 220.5 s | 39,211x  |
   | skybox    | 0.81 ms | 35.5 s  | 43,568x  |
   | ssao      | 1.44 ms | 444.9 s | 308,958x |
   | bilateral | 1.49 ms | 429.1 s | 288,956x |
   | ambient   | 0.84 ms | 64.1 s  | 76,310x  |
   | light     | 0.95 ms | 127.1 s | 133,789x |
   | blur      | 0.54 ms | 74.2 s  | 138,692x |
   | post      | 9.57 ms | 442.6 s | 46,272x  |
   | average   | 1.23 ms | 180.8 s | 146,712x |
   +-----------+---------+---------+----------+


.. raw:: latex

   \begin{tabular}{|c|c|c|c|}
   \hline 
   \textbf{Shader} & \textbf{GPU} & \textbf{CPU} & \textbf{Speedup} \tabularnewline
   \hline 
   material & 5.62 ms & 220.5 s & 39,211x \tabularnewline
   \hline 
   skybox & 0.81 ms & 35.5 s & 43,568x \tabularnewline
   \hline 
   ssao & 1.44 ms & 444.9 s & 308,958x \tabularnewline
   \hline 
   bilateral & 1.49 ms & 429.1 s & 288,956x \tabularnewline
   \hline 
   ambient & 0.84 ms & 64.1 s & 76,310x \tabularnewline
   \hline 
   light & 0.95 ms & 127.1 s & 133,789x \tabularnewline
   \hline 
   blur & 0.54 ms & 74.2 s & 138,692x \tabularnewline
   \hline 
   post & 9.57 ms & 442.6 s & 46,272x \tabularnewline
   \hline 
   average & 1.23 ms & 180.8 s & 146,712x \tabularnewline
   \hline 
   \end{tabular}

On average, the shaders in the example rendering system run 146,712x faster when compiled onto the GPU than when interpreted on the CPU.  The CPU timings only benchmark the execution time of the shader code and neglect the time required to sample textures and othter render system functionality.  The GPU timings take all costs into account, so the speedup is understated.  Five orders of magnitude speedup is reasonable, however.  Compiling an optimized Python program into C can provide two orders of magnitude speedup [Sal04]_.  For unoptimized programs taking full advantage of Python's abstraction mechanisms, an additional order of magnitude of speedup can be achieved because a static compiler can inline functions and globally optimize the program.  Switching from a CPU to a GPU can easily provide another two orders of magnitude speedup for real-time rendering, a task the GPU was designed for.  Taken together, this easily explains the net speedup.

.. figure:: object-scaling.png

   Performance of the example rendering system as the number of objects drawn is increased. :label:`object-scaling`

Figure :ref:`object-scaling` shows the performance of the example rendering system, in frames per second (FPS), as the number of objects drawn increases.  Drawing more objects requires more computation, and will naturally reduce the rate at which images are produced.  Rendering systems may be bottlenecked by factors other than computation; they can also be limited by the rate that glue code can transfer data to the GPU.  PyStream can generate glue code for both OpenGL 2 and OpenGL 3.    OpenGL 3 has features that let it transfer data more efficiently to the GPU.  As seen in the figure, these features can offer a ~20% speed improvement when the rendering system is bottlenecked by its glue code.  This demonstrates an interesting benefit of PyStream: future proofing.  PyStream can take advantage of new features offered by GPUs and GPU APIs without requiring modifications to the rendering system.


Conclusion
----------

PyStream demonstrates a unique approach to high-performance high-level programming.  It can map a significant portion of a general-purpose language onto a GPU, and allow a GPU-accelerated application to be written as a single code base.


References
----------
.. [Bra10] N. C. Bray.  *PyStream: Python Shaders Running on the GPU*,
           PhD thesis, Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign.

.. [Bra11] N. C. Bray.  *Using Partial Evaluation to Simplify a Python Compiler*, White Paper.

.. [Fil08] D. Filion and R. McNaughton, *Effects & techniques*, SIGGRAPH '08: ACM SIGGRAPH 2008 Classes, pp. 133-164.

.. [Pyp11] Online: http://codespeak.net/pypy/dist/pypy/doc/

.. [Sal04] M. Salib, *Starkiller: A static type inferencer and compiler for Python*, M.S. thesis, Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology.
