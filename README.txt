SciPy Proceedings
=================

Paper Format
------------

General Guidelines
``````````````````
- All figures should have captions.
- License conditions on images and figures must be respected
  (Creative Commons, etc.).
- Code snippets should be formatted to fit inside a single column
  without overflow.
- Try to use as little custom LaTeX markup as possible.

Authors and affiliations
````````````````````````
Define the fields in the beginning of the paper::

  :author: My Name
  :email: myname@myplace.com
  :institution: Some University

  :author: Author Two
  :email: two@myplace.com
  :institution: Some University

Figures
```````
Use the following markup::

 .. figure:: filename.png

    Caption of figure goes here.

Mathematics
```````````
Use the inline ``math``-role::

 This is some inline :math:`$f(x)`.

Or insert an equation on a separate line::

 Let us examine the following equation:

 .. raw:: latex

    \begin{equation*}\int_0^\infty f(x) dx\end{equation*}

Build Process
-------------
::

  python publisher/build_paper.py directory_to_my_paper

Requirements
------------
 - IEEETran class
 - Latest docutils

