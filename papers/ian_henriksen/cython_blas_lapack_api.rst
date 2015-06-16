:author: Ian Henriksen
:email: iandh@byu.edu
:institution: Brigham Young University Math Department

---------------------------------------------------------------------
Circumventing The Linker: Using SciPy's BLAS and LAPACK Within Cython
---------------------------------------------------------------------

.. class:: abstract

   In Python, it is often said that you can code your scientific algorithms in a high level language and then optimize any performance-intensive parts with minimal modification in Cython or some other code accelerator.
   For simple algorithms that center entirely on looping and item-by-item access, this is true, but this simple idea starts to break down when optimizing code that relies heavily on calls to other libraries.
   When code like this isn't fast enough, users may be forced to rewrite larger portions of their existing codebase in another language, change the libraries they are using, and write new interfaces between Python and their low-level language of choice.
   To help alleviate these problems, I introduced a Cython API for BLAS and LAPACK.

   BLAS, LAPACK, and other libraries like them have formed the underpinnings of much of the scientific stack in Python.
   Up until now the standard practice in many packages for using BLAS and LAPACK has been to link each Python extension directly against the libraries needed.
   Each module that calls these low-level libraries directly has had to link against them independently.

   Cython has existing machinery that allows C-level declarations to be shared between Cython-compiled extension modules without linking against the original libraries.
   If used carefully, this functionality can be used to export Cython-level APIs exactly mirroring existing C and Fortran interfaces that can be used without any additional linking.
   The Cython BLAS and LAPACK API in SciPy makes it so that the same BLAS and LAPACK libraries that were used to compile SciPy can be used in Python extension modules via Cython.

   In this talk I will demonstrate how to create and use these APIs for both Fortran and C libraries in a platform-independent manner.
   I will also discuss how using these techniques mitigate the effects of dependency creep both within and between Python packages.

.. class:: keywords

   Cython, BLAS, LAPACK, SciPy

Introduction
------------

Many of the primary underpinnings of the scientific Python stack rely on interfacing with lower-level languages, rather than working with code that is exclusively written in Python.
There is a longstanding tradition of libraries and API's that mix Python, C, C++, and Fortran code in order to provide well-optimized algorithms together with simple APIs.
SciPy, for example, is a collection of algorithms and libraries implemented in a variety of languages that have all been wrapped to provide convenient and usable APIs within Python.
Because the need to create such extension modules is so prevalent, a variety of wrapping mechanisms have been introduced that aid in the creation of Python bindings for a given package.
F2PY, Fwrap, Cython, SWIG, CFFI, ctypes, and several other packages have all seen extensive use in recent years.

It is often said that, if Python isn't fast enough for a particular piece of code, getting more speed is simply a matter of writing a little extra code in a different language, exposing it to Python, and adapting the existing code to use the new external piece.
In practice, however, moving even small parts of a larger algorithm from Python to C can prove far more challenging that we often admit.
The fundamental problem is that, in order to use code within a different language, a programmer must either remove calls to existing libraries entirely, or call an existing package within the lower-level language.
Changing the language of a piece of code may involve changing the stack of libraries used.
Writing a few lines of code in a different language already adds a significant dependency and presents a significant burden in terms of packaging and reproducibility.
On the other hand, when using additional libraries from the other language, properly packaging the module can suddenly become very difficult.

A possible solution to this unusually painful problem is to make it so that existing Python packages that already export algorithms from a library written in in a lower-level language can provide ways to interface with the lower-level library directly without having to rebuild or search for the low-level library.
NumPy has provided some of this sort of functionality for BLAS and LAPACK by making it so that the locations of the system's BLAS and LAPACK libraries can be found using NumPy's distutils module.
Cython has also provided a similar sort of functionality by allowing C-level APIs to be exported between Cython modules.


The Cython API for BLAS and LAPACK
----------------------------------

Over the last year, a significant amount of work has been devoted to exposing the BLAS and LAPACK libraries within SciPy at the Cython level.
The primary goals of providing such an interface are twofold:
* Making the low-level routines in BLAS and LAPACK more readily available to users.
* Reducing the dependency burden on third party packages.

Using the new Cython API, users can now dynamically load the BLAS and LAPACK libraries used to compile SciPy without having to actually link against the original BLAS and LAPACK libraries or include the corresponding headers.
Modules that use the new API also no longer need to worry about which BLAS or LAPACK library is used.
If the correct versions of BLAS and LAPACK were used to compile SciPy, the correct versions will be used by the extension module.

BLAS and LAPACK proved to be particularly good candidates for a Cython API, resulting in several additional benefits:
* Python modules that use the Cython BLAS/LAPACK API no longer need to link statically to provide binary installers.
* The custom ABI wrappers and patches used in SciPy to provide a more stable and uniform interface across different BLAS/LAPACK libraries and  Fortran compilers are no longer needed for third party extensions.
* The naming schemes used within BLAS and LAPACK make it easy to write type-dispatching versions of BLAS and LAPACK routines using Cython's fused types.

In providing these low-level wrappers, it was simplest to follow the calling conventions of BLAS and LAPACK as closely as possible, so all arguments are passed as pointers.

Here's a minimal example

.. code-block:: python

   # cython: boundscheck = False
   # cython: wraparound = False
   import numpy as np
   from scipy.linalg.cython_blas cimport dgemm
   def test_dgemm():
       cdef:
           double[:,:] a = np.array([[2, 3], [2, 1]], 'd', order='F')
           double[:,:] b = np.array([[1, 4], [6, 4]], 'd', order='F')
           double[:,:] c = np.empty((2,2), order='F')
           int n = 2
           char trans = 'n'
           double al = 1., be = 0.
       # Axes lengths are all equal, so n can be reused.
       # Note that all arguments are passed as pointers.
       dgemm(&trans, &trans, &n, &n, &n, &al, &a[0,0],
             &n, &b[0,0], &n, &be, &c[0,0], &n)
       # Print the results to show they match np.dot.
       print np.array(c)
       print np.dot(a, b)

If these wrappers are needed in an extension module written in C, C++, or another low-level language, a small Cython shim can be used to export the needed functions.
Since Cython uses Python's capsule objects internally for the cimport mechanism, it is also possible to extract function pointers directly from the module's `__pyx_capi__` dictionary and cast them to the needed type without writing the extra shim.

Exporting Cython APIs for Existing C Libraries
----------------------------------------------

The process of exposing a Cython binding for a function or variable in an existing library is relatively simple.
First, as an example, consider the simple C file

.. code-block:: c

   // myfunc.c
   double f(double x, double y){
   return x * x - x * y + 3 * y;
   }

with the corresponding header file

.. code-block:: c

   // myfunc.h
   double f(double x, double y);
   \end{lstlisting}

This library can be compiled by running `clang -c myfunc.c -o myfunc.o`.

This can be exposed at the Cython level and exported as a part of the resulting Python module by including the header in the pyx file, using the function from the C file to create either a Cython shim or a function pointer with the proper signature, and then declaring the function or function pointer in the corresponding pxd file without including the header file.
Here's a minimal example of how to do that:

.. code-block:: python

   # cy_myfunc.pyx
   # Use a file-level directive to link against the compiled object.
   # distutils: extra_link_args = ['myfunc.o']
   cdef extern from 'myfunc.h':
       double f(double x, double y) nogil
   # Declare both the external function and the Cython function as nogil so they can be
   # without any Python operations (other than loading the module).
   cdef double cy_f(double x, double y) nogil:
       return f(x, y)

.. code-block:: python

   # cy_myfunc.pxd
   # Don't include the header here.
   # Only give the signature for the Cython-exposed version of the function.
   cdef double cy_f(double x, double y) nogil

.. code-block:: python

   # cy_myfunc_setup.py
   from distutils.core import setup
   from Cython.Build import cythonize
   setup(ext_modules=cythonize('cy_myfunc.pyx'))

From here, once the module is built, the Cython wrapper for the C-level function can be used in other modules without linking against the original library.

Exporting a Cython API for an existing Fortran library
------------------------------------------------------

When working with a Fortran library, the name mangling scheme used by the compiler must be taken in to account.
The simplest way to work around this would be to use Fortran 2003's ISO C binding module.
Since, for the sake of platform/compiler independence, such a recent version of Fortran cannot be used in SciPy, an existing header with a small macro was used to imitate the name mangling scheme used by the various Fortran compilers.
In addition, for this approach to work properly, all the Fortran functions in BLAS and LAPACK were first wrapped as subroutines (functions without return values) at the Fortran level.

.. code-block:: Fortran

   c     myffunc.f
   c     The function to be exported.
         double precision function f(x, y)
           double precision x, y
           f = x * x - x * y + 3 * y
         end function f

.. code-block:: Fortran

   c     myffuncwrap.f
   c     A subroutine wrapper for the function.
         subroutine fwrp(out, x, y)
           external f
           double precision f
           double precision out, x, y
           out = f(x, y)
         end

.. code-block:: c

   // fortran_defs.h
   // Define a macro to handle different Fortran naming conventions.
   // Copied verbatim from SciPy.
   #if defined(NO_APPEND_FORTRAN)
   #if defined(UPPERCASE_FORTRAN)
   #define F_FUNC(f,F) F
   #else
   #define F_FUNC(f,F) f
   #endif
   #else
   #if defined(UPPERCASE_FORTRAN)
   #define F_FUNC(f,F) F##_
   #else
   #define F_FUNC(f,F) f##_
   #endif
   #endif

.. code-block:: c

   // myffuncwrap.h
   #include "fortran_defs.h"
   void F_FUNC(fwrp, FWRP)(double *out, double *x, double *y);

.. code-block:: python

   # cyffunc.pyx
   cdef extern from 'myffuncwrap.h':
       void fort_f "F_FUNC(fwrp, FWRP)"(double *out, double *x, double *y) nogil
   cdef double f(double *x, double *y) nogil:
       cdef double out
       fort_f(&out, x, y)
       return out

.. code-block:: python

   # cyffunc.pxd
   cdef double f(double *x, double *y) nogil

Numpy's distutils package can be used to build the Fortran libraries and compile the final extension module.
The interoperability between NumPy's distutils package and Cython is limited, but the C file resulting from the Cython compilation can still be used to create the final extension module.

.. code-block:: python

   # cyffunc_setup.py
   from numpy.distutils.core import setup
   from numpy.distutils.misc_util import Configuration
   from Cython.Build import cythonize
   def configuration():
       config = Configuration()
       config.add_library('myffunc', sources=['myffunc.f'], libraries=[])
       config.add_library('myffuncwrap', sources=['myffuncwrap.f'],
                          libraries=['myffunc'])
       config.add_extension('cyffunc', sources=['cyffunc.c'],
                         libraries=['myffuncwrap'])
       return config
   # Run Cython to get the needed C files.
   # Doing this separately from the setup process
   # causes any Cython file-specific distutils directives
   # to be ignored.
   cythonize('cyffunc.pyx')
   setup(configuration=configuration)

Since there are many routines in BLAS and LAPACK and creating these wrappers currently still requires a large amount of boiler plate code, it was easiest to write Python scripts that used f2py's existing functionality for parsing Fortran files to generate a set of function signatures that could, in turn, be used to generate the needed wrapper files.

Since SciPy supports several versions of LAPACK, it was also necessary to determine which routines should be included as a part of the new Cython API.
In order to support all currently used versions of LAPACK, we limited the functions in the Cython API to include only those that had a uniform interface from version 3.1 through version 3.5.

Conclusion
----------

The new Cython API for BLAS and LAPACK in SciPy helps to alleviate the substantial packaging burden imposed on Python packages that use BLAS and LAPACK and provides a model for including access to lower-level libraries used within a Python package.
It makes BLAS and LAPACK much easier to use for new and expert users alike and makes it much easier for smaller modules to write platform and compiler independent code.
It also provides a model that can be extended to other packages to help fight dependency creep and reduce the burden of package maintenance.
Though it is certainly not trivial, it is still fairly easy to add new Cython bindings to an existing library.
Doing so makes the lower-level libraries vastly easier to use.

Possible future directions for this work include using Cython's fused types to expose a more type-generic interface to BLAS and LAPACK, writing better automated tools for generating wrappers that expose C, C++, and Fortran functions automatically, making similar interfaces available in ctypes and CFFI, and providing similar APIs for a wider variety of libraries.



References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


