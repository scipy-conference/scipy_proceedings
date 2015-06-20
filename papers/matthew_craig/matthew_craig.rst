:author: Matthew Craig
:email: mcraig@mnstate.edu
:institution: Department of Physics and Astronomy, Minnesota State University Moorhead

--------------------------------------------------------------------------
Widgets and Astropy: Accomplishing Productive Research with Undergraduates
--------------------------------------------------------------------------

.. class:: abstract

    A set of IPython notebooks with a widget interface that are based on
    Astropy, a community-developed package of fundamental tools for astronomy,
    and Astropy affiliated packages. The widget interface makes astropy a much
    more useful tool to undergraduates or other non-experts doing research in
    astronomy, filling a niche for software that connects beginners to
    research-grade code.

.. class:: keywords

   astronomy

Introduction
------------

The Astropy project [Astropy2013]_ is a community-driven effort to develop
high-quality, open source tools for Python in astronomy. It, and project
like it, in principle make research-grade software available for non-experts.
In practice, despite extensive documentation, the barrier to entry is high.

The specific target population for the tools described here is beginning
undergraduates at a non-selective state university. The students have little
or no programming experience in any language and are only beginning to learn
astronomy. This population is often drawn to astronomy by the prospect of
doing their own observational astronomy research but the available tools for
analyzing observational data are either commercial, closed-sourced with
limited documentation of what algorithms are being used, and/or require
substantial programming knowledge. About half of the students use Windows; the
remainder use Mac OSX.

IPython [Pérez2007]_ notebooks introduced widgets in version 2, providing an
easy, Python-only way to program user interface elements in the IPython
notebook. A browser-based graphical interface is familiar to all students, and
because the widgets can be written in Python, it is easy to couple them to
software like astropy.

This paper describes a framework for coupling action-based widgets to Python
code, in the context of ``reducer``, a notebook for calibrating astronomical
images. In addition to coupling research-grade, open source software with a
straightforward graphical interface, the notebook itself is a record of the
choices the student made in calibrating the data.

.. [#] http://www.astropy.org/affiliated/

Aside: Bootstrapping a computing environment for students
---------------------------------------------------------

While the goal of this work is to minimize the amount of programming new users
need to do, there are a few things that cannot be avoided: installing Python
and the SciPy [scipy2001]_ stack, and learning a little about how to use a
terminal.

Students find the Anaconda Python distribution [#]_ easy to install and it is
available for all platforms. From a developer point of view, it also provides
a platform for distributing binary packages, particularly useful to the
students on Windows.

Students also need minimal familiarity with the terminal to install the
reducer package, generate a notebook for analyzing their data and launching
the notebook. The *Command Line Crash Course* from *Learn Code the Hard Way*
[#]_ is an excellent introduction, has tracks for each major platform, and is
very modular.

.. [#] https://store.continuum.io/cshop/anaconda/
.. [#] http://cli.learncodethehardway.org/book/


The ``reducer`` widget structure
--------------------------------

At the base of the reducer widget structure is an extension of a container
widget from IPython. This class, ``ToggleContainerWidget``, adds a toggle to
control display the contents of the container, and a list of child widgets
displayed in the container. [#]_  Since a ``ToggleContainerWidget`` can have
another ``ToggleContainerWidget`` as a child, this immediately provides an
interface for presenting a user with a nested list of options.

In IPython 2 it is not possible to preserve the state of widgets between
sessions, and in IPython 3 it remains difficult, so the
``ToggleContainerWidget`` class defines a ``__str__`` method to facilitate
printing the contents of the widget. The purpose of this is not to provide a
way to progammatically rebuild the widget; it is to provide a human reader of
the notebook a history of what was done in the notebook.

The widget also has an ``action`` method. This method must be overridden by
subclasses to do anything useful. It is used in some cases to set up an
environment for acting on data files and to invoke the action of each child
widget on each data file, in the order the children are listed in the widget.
In other cases, the action simply invokes a function that acts on the data
file.

An ``is_sane`` method that can be overridden by subclasses to indicate that
the settings in the widget are sensible. This can provide some minimal
validation of user input.

One subclass of ``ToggleContainerWidget``, a ``ToggleGoWidget``, styles the
toggle as a button instead of a checkbox, and adds a "Start" button that is
displayed only when the settings of the widget and all of its children is
"sane" as defined by the ``is_sane`` method. What the "Start" button is pushed
it invokes the ``action`` method of the ``ToggleGoWidget`` and displays a
progress bar while working.

**TODO:** Add screenshots of sample widgets,

.. [#] Classes in the current version of ``reducer`` use IPython 2-style class
       names ending in "WidgetW. Part of upgrading the package to IPython 3
       widgets will be removing that ending.

Background: calibration of astronomical images
----------------------------------------------

An image from a CCD camera on a telescope is simply an array of pixel values.
Several sources contribute to the brightness of an individual pixel in a raw
image:

+ Light from stars and other astronomical objects (this is the bit in which
  an astronomer is interested).
+ Light from the nighttime sky; even a “dark” sky is not perfectly black.
+ Noise that is related to the temperature of the camera and to the
  electronics that transfer the image from the detector chip in the camera
  to a computer.

The first stage of calibration is to remove the noise from each image. The
second stage is to correct for imperfections in the optical system that affect
how much light gets to each pixel in the camera. An example of this sort of
imperfection is dust on telescope elements.

After this calibration has been performed, the brightness of a pixel in the
image is directly proportional to the amount of light that arrived at that
pixel through the telescope. It is at this point that measurements can be made
from the image.

The ``reducer`` package and notebook
------------------------------------

``reducer`` is a pure Python package available on PyPI and as a conda  package
[#]_. The user-facing part of the package is a single script, also called
``reducer``. When invoked, it creates an IPython notebook,
called ``reduction.ipynb``, in the directory in which it is invoked.

Screen shots of the reduction notebook, showing a sample of the widgets, is
below (**TODO:** screenshot).

All of the image operations in reducer are performed by ``ccdproc``, an
Astropy-affiliated package for astronomical image reduction [ccdproc]_.

Image browser
-------------

Reducer also contains a basic image browser, which organizes the images based
on a table of metadata, and displays, when an image is selected, both the
image and all of the metadata in that image.

Use with students
-----------------

This package has been used with 8 undergraduate physics majors ranging from
first-semester freshman to seniors; it was also used in an astronomical
imaging course that included two non-physics majors. It typically took one
1-hour session to train the students to use the notebook. The other graphical
tool used in the course took considerably longer for the students to set up
and left no record the steps and settings the students followed in calibrating
the data.

Conclusion and future directions
--------------------------------

**TODO**

.. [#] Use channel ``mwcraig`` to get the conda package.

References
----------
.. [Astropy2013] Astropy Collaboration, Robitaille, T.~P., Tollerud, E.~J., et al.,
             *Astropy: A community Python package for astronomy*,
             Astronomy \& Astrophysics, 558: A33, October 2013.

.. [scipy2001] Jones, E., Oliphant, T., Peterson, P. *et al*,
               *SciPy: Open source scientific tools for Python*,
               http://scipy.org/ 2001

.. [Pérez2007] Pérez, F. and  Granger, B.E.
               *IPython: A System for Interactive Scientific Computing*,
               Computing in Science and Engineering, 9(3):21-29, May/June 2007

.. [ccdproc] Crawford, S and Craig, M., https://github.com/ccdproc
