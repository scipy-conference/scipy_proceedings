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

Incoming students interested in majoring in Physics at Minnesota State
University Moorhead are often interested in doing astronomical research. The
department encourages students to become involved in research as early as
possible to foster their interest in science and because early research
experiences are correlated with successful completion of a degree
[ref_needed]_.

The students typically have no programming experience, but even the smallest
project requires calibrating and taking measurements from a couple of hundred
images. To the extent possible, analysis needs to be automated. Roughly half
of the students use Windows, the rest Mac OSX.

The problem, described in more detail below, is that the GUI-based software
most accessible to these students is expensive, often available only on
Windows, and not clearly documented. The free options largely require
programming. The software must also leave a record of the choices made in
calibrating the images so that future researchers can use the images with
confidence.

The proposed solution is a widget-based IPython notebook [Pérez2007]_ for
calibrating astronomical images, called ``reducer``. A widget-based interface
was chosen because students at this level are more comfortable with a GUI than
with programming. An IPython notebook was chosen because of its rich display
format, the ability to save both code and text, and the persistence of output
in the notebook, which provides a record of the work done.

The back end of ``reducer`` is built on the Astropy project [Astropy2013]_, a
community-driven effort to develop high-quality, open source tools for Python
in astronomy, and on Astropy affiliated projects. [#]_ Astropy was chosen because it has a large developer community of professional astronomers.

Section W provides background on the science of image calibration. In section
X the problem is discussed more completely, including a review of some of the
available options for astronomical image processing. The section Y discusses
the use of ``reducer``, while section Z presents its implementation. The
widget classes in ``reducer`` are potentially useful in other applications.

.. [#] http://www.astropy.org/affiliated/

Background: Image analysis in optical stellar astronomy
-------------------------------------------------------

While a detailed description of astronomical data analysis is beyond the scope
of this talk, some appreciation of the steps involved is useful for
understanding its motivation.

An image from a CCD camera on a telescope is simply an array of pixel values.
Several sources contribute to the brightness of an individual pixel in a raw
image:

+ Light from stars and other astronomical objects.
+ Light from the nighttime sky; even a “dark” sky is not perfectly black.
+ Noise that is related to the temperature of the camera and to the
  electronics that transfer the image from the detector chip in the camera
  to a computer.
+ A DC offset to prevent negative pixel values.

The first stage of calibration is to remove the noise and offset from each
image. The second stage is to correct for imperfections in the optical system
that affect how much light gets to each pixel in the camera. An example of
this sort of imperfection is dust on the camera itself.

After calibration, the brightness of a pixel in the image is directly
proportional to the amount of light that arrived at that pixel through the
telescope. Note that light includes both starlight and light from the
atmosphere.

Extraction of the brightness of individual stars is called photometry. There
are several techniques for performing photometry, all of which estimate and
eliminate the sky background.

The problem
-----------

Several software packages can calibrate astronomical images and perform
photometry, which begs the question "Why write another one?"

Ideally, such software would:

1. Be easily usable by an undergraduate with limited or no programming
   experience.
2. Work on Windows and Mac.
3. Have its operation well tested in published articles and/or be open
   source so that the details of its implementation can be examined.
4. Leave behind a record of the settings used by the software in
   calibrating the images and measuring star brightness.
5. Be maintained by a large, thriving community of developers.

Commercial software, like *MaxIm DL* [#]_, typically meets the first criteria.
Past MSUM students were able to learn the software quickly. However, it leaves
behind almost no record of how calibration was done: a fully calibrated image
has one keyword added to its metadata: ``CALSTAT='BDF'``. While this does
indicate which corrections have been made [#]_, it omits important information
like whether cosmic rays were removed from the calibration images and how the
individual calibration images were combined.

The most extensively-tested and widely-used professional-grade package for
calibration and photometry is IRAF [IRAF1993]_. IRAF is both a scripting
language and a set of pre-defined scripts for carrying out common operations.
It is certainly widely used, with approximately 450 citations of the paper,
and, because IRAF scripts store settings in text files, there is a record of
what was done.

However, there are several challenges to using IRAF. It is easiest to install
in Linux, though distributions exist for Mac and it is possible to use on
Windows with Cygwin [#]_. The IRAF command language (CL) is difficult to
learn; undergraduates who have worked with it in summer REU programs report
spending 3-4 weeks learning IRAF. That makes it infeasible to use as part of a
one-semester research project. It is also no longer maintained [#]_.

One option that comes close to meeting all of the criteria is AstroImageJ
[#]_, a set of astronomy plug-ins for the Java-based ImageJ [ImageJ2012]_. It
has a nice graphical interface that students in both an introductory astronomy
course for non-majors and an upper-level course for majors found easy to use,
is open source,  free, and available on all platforms. Its two weaknesses are
that it leaves an incomplete record of the settings used in calibrating data
and measuring brightness and that it is maintained almost exclusively by one
person.

.. [#] http://www.cyanogen.com/
.. [#] The bias offset and dark current were subtracted and the result
       divided by a flat frame to correct for non-uniform illumination.
.. [#] http://www.cygwin.com/
.. [#] The last update was in 2012 according to the IRAF web site,
       http://iraf.noao.edu
.. [#] http://www.astro.louisville.edu/software/astroimagej/

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

.. [#] Use channel ``mwcraig`` to get the conda package.

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

Appendix: Bootstrapping a computing environment for students
------------------------------------------------------------

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

.. [ref_needed] *FILL ME IN*

.. [IRAF1993] Tody, D., *IRAF in the Nineties*, Astronomical Data Analysis
              Software and Systems II, A.S.P. Conference Series, Vol. 52, 1993

.. [ImageJ2012] Schneider, C.A., Rasband, W.S., Eliceiri, K.W.
                *NIH Image to ImageJ: 25 years of image analysis*,
                Nature Methods 9, 671-675, 2012.
