:author: Scott Determan
:email: scott.determan@gmail.com
:institution: Vision Spreadsheet

------------------------------------------------------
Vision Spreadsheet: An Environment for Computer Vision
------------------------------------------------------

.. class:: abstract

   Vision Spreadsheet is an environment for computer vision. It combines a
   spreadsheet with computer vision and scientific python. The cells in the
   spreadsheet are images, computations on images, measurements, and plots. There
   are many built in image processing and machine learning algorithms and it
   extensible by writing python functions and importing them into the
   spreadsheet.
   
.. class:: keywords

   computer vision,spreadsheet,OpenCV

Introduction
------------

Vision Spreadsheet is an environment for computer vision. Its purpose is to lower
the barrier to computer vision by:

- Having a shallow learning curve by using spreadsheets, an interface many people
  are familiar with.
- Providing immediate feedback to algorithm changes. The spreadsheet interface
  along with binding GUI controls to algorithm parameters lets you explore your
  problems.
- Letting you experiment with lots of different algorithms. All of the image
  processing and machine learning algorithms from OpenCV are available.
- Allowing you to easily add your own functions to the spreadsheet. You can use
  python to extend the spreadsheet.

It is difficult to get a feel for an interactive environment by reading a
paper. I would encourage you to visit http://visionspreadsheet.com to download the
program and watch some videos of vision spreadsheet in action.

The Spreadsheet
---------------

Figure :ref:`bloodcells` shows a screen shot of Vision Spreadsheet. There are
three main area to the GUI: the grid of cells, the repl below the cells, and the
GUI controls to the left of the cells.

.. figure:: bloodcells.png
   :scale: 25%
   :figclass: bht

   Vision Spreadsheet cells contain images, measurements, and plots. :label:`bloodcells`

Financial spreadsheets contain a grid of numbers and labels. Vision Spreadsheet's
grid of cells contains images, computations on images, measurements, and
plots. In a financial spreadsheet, if you sum a column of numbers and change a
number in the column the total automatically updates itself. Similarly, in vision
spreadsheet if you change a cell (for instance by loading a new image or changing
an algorithm parameter) then all of the cells that depend on the changed cell
will update themselves.

The repl (read-eval-print-loop) is a modified ipython shell used to specify what
a cells contains. The repl is also used to write new spreadsheet functions in
python.

The GUI controls area contains display parameters, overlays, and controls bound
to algorithm parameters for the current cell.


Vision Development Process
--------------------------

Let's look at the development process of one type of vision problem: classifying
white blood cells. The problem: given an image with one of five types of white
blood cells, build an algorithm that correctly identifies which type of blood
cell is in the image. The types of cells we will distinguish between are:
lyphosites, monocytes, eosinophils, neutrophils, and basophils (See figure :ref:
`bloodcell_types`).

.. figure:: bloodcell_types.png
   :scale: 25%
   :figclass: bht

   Bloodcell types. :label:`bloodcell_types`

The typical work flow for a classification problem like this is:

#. Collect sample images. There's a relationship between the number of images we
   collect and the number of features we'll be able to use and the type of
   classifier we'll be able to use without over training.  So get as many sample
   images as you can.
#. Ground truth the samples. Have an expert go through the samples and label what
 type of cell is in the image.
#. Randomly assign some samples to a training set and the rest of the samples to
 a scoring set.
#. Find features that distinguish the cells from one another. Looking at the
 cells, we can see some distinct parts: a dark blob surrounded by a lighter
 blob. The dark blob is the nucleus, the surrounding blob is the cytoplasm. The
 first step to our algorithm will be to segment the nucleus and cytoplasm and
 extract simple measurements from them (area, perimeter, average color, etc).
#. Pick a classifier type and train it using the features from the previous
 step. If you don't know what type of classifier to pick, don't sweat it. The choice
 of classifier is much less important than finding good features in step 4. A random forest
 algorithm is a good default choice.
#. Test how well the classifier does by running it against the scoring set. If
   the classifier doesn't work well enough, look at the misclassified images. Can
   we see things that might improve the classification. Maybe some images are
   brighter than others (the user turned up the light on the microscope). Or some
   images have a dark left side and a bright right side. Or sometimes the
   cytoplasm segmentation bleeds into the background. We'll see if we can address
   these problems. We may also think of new features that help us distinguish
   between the cells, like measuring the texture of the cytoplasm. We go back to
   step 4 and repeat until we're happy with the algorithm.

Of these steps, steps 1 and steps 4 are the critical steps. Vision Spreadsheet
can't collect more images for you, but it can help you explore what features
distinguish between your objects. The rest of this paper shows how Vision
Spreadsheet supports exploring vision problems like this.

The Cells Language
------------------

The cells language is a very simple language used to specify what to display in a
cell. A typical call looks like:

.. code-block:: python

   some_cell = some_function(parameter1,paramter2)

For example, to erode an image in cell a1, and put the result in cell b1:

.. code-block:: python

   b1 = erode(a1)

Now if you load a new image in cell a1, cell b1 will automatically update
itself. If you want to inhibit the automatic update behavior, you can preface a
variable with a '$'. For example, if you say:

.. code-block:: perl 

   b1 = erode($a1)

The '$' in front of the a1 variable prevents the statement from rerunning if a1
changes.

Functions may be nested, so one way to run a morphological open would be:

.. code-block:: python

   b1 = dilate(erode(a1))

Of course, morphological open is already built in. In fact, all of the image
processing and machine learning functions from OpenCV are available.

The arithmetic operators are available and follow the usual syntax and precedence
rules. There is an if function and select function.

Vision spreadsheet supports namespaces. This is most useful for referencing cells
in other sheets of the spreadsheet. The syntax is:

.. code-block:: python

   namespace_name::variable_name
   ::variable_name # global namespace

So if you have sheets g1 and g2, to refer to sheet g1 cell a1, you would say:

.. code-block:: python

   g1::a1

The cells language is meant to write single lines of code to specify what a cell
contains. It is not meant to write complex programs. For that, use python (and
python mode within vision spreadsheet).

Binding Parameters to GUI Controls
----------------------------------

One of my favorite features in vision spreadsheet is binding GUI controls to
algorithm parameters. The best way to explain this feature is to look at an example.
Let's say we want to threshold an image. There are a couple threshold operators, but
let's use the simplest: the '>.' operator. First load an image in cell a1. Next,
threshold it by typing:

.. code-block:: python

   b1 = a1 >. 128

This creates an image where values greater than 128 are set to 255 and values
less than or equal to 128 are set to zero. However, often we want to set
thresholds interactively. We could keep typing in numbers until we get the result
we get. A better way is to bind the parameter to a GUI control, say a
slider. The following command does this:

.. code-block:: python

   b1 = a1 >. slider(128,0,255)

This creates a slider with a default value of 128, a min value of 0 and a max value of 255. If you look
in the cell controls pane on the left of the GUI, you will see a slider (see figure :ref:`slider`). You can use this
slider to interactively change the parameter to the threshold function.

.. figure:: slider.png
   :scale: 25%
   :figclass: bht

   GUI controls may be bound to algorithm parameters. Here a slider is bound to a threshold. :label:`slider`

There are many other types of GUI controls that may be bound to parameters, such
as: radio buttons, sliders, spin controls, combo boxes and movie controls (radio
button are particularly useful to bind to file names so different images may be
easily loaded into a cell).

You may also use multiple GUI controls to control a single function
parameter. You do this by nesting the calls to the GUI controls. For example, to
have a spin control and a slider control the threshold:

.. code-block:: python

   b1 = a1 >. spin(slider(128,0,255))


Data Structures
---------------

There are three main data structures in vision spreadsheet: images, data frames,
and statistical models (classifiers, clusters, and regression algorithms).

Images are the data structure you will use the most. Taking an image and running
a filter, or an edge finder, or (most) segmentation algorithms transform images
to images. It is a two dimensional array of vectors. All the elements are of the
same numeric type (uchar through double are supported). Lots of image types are
supported: depth images (from the Kinect camera, for example), grayscale, color
(rgb, brg, hsi, cie lab, etc.). When an image is passed to a user defined python
function it is automatically converted into a numpy array.

Data frames are modeled after R's data frame structure. You will use data frames
to store measurements on images and to overlay images with shapes and regions of
interest (among other uses). It is a table where each column in the table may
have a different type. So a single data frame may have a column of numbers and a
column of strings. Supported column types are: numeric (uchar through double),
boolean, string, and region of interest. Like R's data frames, rows may contain
missing data. Data frames also support R's notion of factor columns. Factor
columns are usually used to specify responses when training classifiers. Unlike
R, vision spreadsheet supports grouping columns into a hierarchy. This is useful
for storing higher-level objects in a data frame.  For example, rectangles are
stored in a data frame by grouping together four numeric columns. These
rectangles may then be overlaid and edited on an image.

The last major data structure is a statistical model. You will use statistical
models to classify objects in images (among other uses). There are two main
functions to a statistical model: train and predict. Train takes a data frame of
features (labeled for supervised learning, unlabeled for supervised). Predict
takes a data frame and returns a prediction for each row in the data frame (the
predictions are classifications or regressions, depending on the type of
statistical model).

There are other data types in vision spreadsheet, but using only these three you
can solve many problems in computer vision.

Python Mode
-----------

The ipython shell at the bottom of the GUI supports two modes, cells mode and
python mode. To toggle between the two modes, type '##' and hit return. Cells
mode is the default mode. Python mode is just a regular ipython shell with two
differences: you can type '##' to toggle to cells mode and there is a module
called 'vis_sheet' that can be used to interact with the spreadsheet.

There are two interesting activities you typically do in python mode:

#. Extend the spreadsheet with new functions.
#. Get values from the spreadsheet, muck around with them interactively in
   python, and set the values back into the spreadsheet.

Let's look at adding a new function to the spreadsheet. Change to python mode by
typing '##'. The shell should now have a black background. Now, define a
subtraction function as follows:

.. code-block:: python

   def my_subtract(a,b): return a-b
   import vis_sheet
   vis_sheet.add_python_op(my_subtract)

Change back to cells mode by typing '##' (the shell should now have a white
background). Load an image in cell a1, erode it an put it in b1, and subtract b1
from a1 using our new function:

.. code-block:: python

   c1 = my_subtract(a1,b1)

You should see the edges from the image in cell a1. Note that the images in the
spreadsheet are automatically converted to numpy arrays before they are passed to
user defined functions. So the parameters a and b will be numpy arrays. If the
result is a numpy array, it will automatically be converted to an image.

You can get or set values in the spreadsheet from python mode with the following functions:

.. code-block:: python

   import vis_sheet
   vis_sheet.get_var_data('a1')
   vis_sheet.set_var_data('b1')


Kinect Camera
-------------

Although it isn't a major part of vision spreadsheet, it's too much fun not to
mention. There is an interface to the Microsoft Kinect camera. The function
grab_kinect_rgb will stream values from the rgb camera and the grab_kinect_depth
will stream values from the depth camera.

Status
------

Vision Spreadsheet isn't released yet. But I'm very, very close. All of the
following are done:the cells language, OpenCV is wrapped, binding GUI controls to
function parameters, multiple spreadsheets, data frames, adding python functions
to the spreadsheet work, the kinects camera interface, saving and loading, and
plots. It is quite useful as it is. But if I do just a little more, it will be
fantastically more useful.

I still have to implement the tools for classifiers (like ground truthing
images), tools for data frames (like overlaying the rectangles in a data frame on
an image and editing them, and editing factor columns).

Finally, I need some time to shake out the really bad bugs before I let anyone
else use it.

I had planned on releasing Vision Spreadsheet shortly before the conference. I
didn't make it. I'm sorry. When it is released, you can go to
http://visionspreadsheet.com to download it (free, of course).

Conclusion
----------

Vision spreadsheet provides an environment for interactively working with
computer vision. 

Thank You
---------

I used many great open source projects. I expecially want to thank the following
projects (alphabetical order): antlr, boost, cmake, ipython, libfreenect, numpy,
opencv, python, scipy, vigra, wxpython, and wxwidgets.

