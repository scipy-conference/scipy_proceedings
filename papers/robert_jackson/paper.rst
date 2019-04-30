:author: Robert Jackson
:email: rjackson@anl.gov
:institution: Argonne National Laboratory, Argonne, IL, USA

:author: Scott Collis
:email: scollis@anl.gov
:institution: Argonne National Laboratory, Argonne, IL, USA

:author: Timothy Lang
:email: timothy.j.lang@nasa.gov
:institution: NASA Marshall Space Flight Center, Huntsville, AL, USA

:author: Corey Potvin
:email: corey.potvin@noaa.gov
:institution: NOAA/OAR National Severe Storms Laboratory, Norman, OK, USA
:institution: School of Meteorology, University of Oklahoma, Norman, OK, USA

:author: Todd Munson
:email: tmunson@anl.gov
:institution: Argonne National Laboratory, Argonne, IL, USA
:bibliography: mybib


:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------
A Numerical Perspective to Terraforming a Desert
------------------------------------------------

.. class:: abstract

   PyDDA is a new community framework aimed at wind retrievals that depends
   only upon utilities in the SciPy ecosystem such as scipy, numpy, and dask.
   It can support retrievals of winds using information from weather radar
   networks constrained by high resolution forecast models over grids that
   cover thousands of kilometers at kilometer-scale resolution.
   Unlike past wind retrieval packages, this package can be installed using
   anaconda for easy installation and, with a focus on ease of use can retrieve
   winds from gridded radar and model data with just a few lines of code.


.. class:: keywords

   wind, retrieval, hurricane,

Introduction
------------

The retrieval of three dimensional winds from weather radars is a nontrivial
task. Given that the radar measures the speed of scatterers in the direction
of the radar beam rather than the full wind velocity, retrieving these
winds requires more information than the doppler velocities measured by a
single weather radar.

(Paragraph about methods used for wind retrievals)
Currently existing software for wind retrievals includes software based
off of the strong constraint technique such as CEDRIC (citation) as well
as software based off of the weak variational technique such as MultiDop
:cite:`Langetal2017`. Since CEDRIC uses a strong constraint
from mass continuity equation to retrieve winds, the addition of constraints
from other data sources is not possible with CEDRIC. Also, while CEDRIC was
revolutionary for its time, it is difficult to use as it uses a scripting
language as inputs for the retrieval. While MultiDop is based off of the
more customizable 3D variational technique, it is fixed to 2 or 3 radars and
is not scalable. In addition, the addition of constraints from other 3D
fields such as models is not supported. Finally, Multidop is a wrapper
around a program written in C which introduces issues related to packaging
in frameworks such as anaconda and scalability due to the non-threadsafe
nature of the wrapper.

The limitations in current wind retrieval software motivated development
an open source, entirely Pythonic wind retrieval package called PyDDA.
PyDDA is based entirely on the scientific Python ecosystem. This therefore
permits the easy installation of PyDDA using pip or anaconda. As will
be shown later, PyDDA can retrieve winds from multiple radars combined with
data from model reanalyses with just a few lines of code. In addition, the
open source nature of PyDDA encourages contributions by users in order
to

Three dimensional variational technique
---------------------------------------

The technique that PyDDA uses to create a wind retrieval is based off of
finding the wind vector field :math:`\vec{\textbf{V}}` of a cost function
that minimizes :math:`J(\textbf{V})`. This cost function is the weighted
sum of many different cost functions related to various constraints.

+--------------------------------+-------------------------------+
| Cost function                  | Basis of constraint           |
+================================+===============================+
| :math:`J_{o}(\vec{\textbf{V}})`| Radar observations            |
+--------------------------------+-------------------------------+
| :math:`J_{c}(\vec{\textbf{V}})`| Mass continuity equation      |
+--------------------------------+-------------------------------+
| :math:`J_{v}(\vec{\textbf{V}})`| Vertical vorticity equation   |
+--------------------------------+-------------------------------+
| :math:`J_{m}(\vec{\textbf{V}})`| Model field constraint        |
+--------------------------------+-------------------------------+
| :math:`J_{b}(\vec{\textbf{V}})`| Background constraint         |
|                                | (rawinsonde data)             |
+--------------------------------+-------------------------------+
| :math:`J_{s}(\vec{\textbf{V}})`| Smoothness constraint         |
+--------------------------------+-------------------------------+

The detailed formulas behind these cost functions can be found in
:cite:`Shapiroetal2009` and :cite:`Potvinetal2012`. The cost function
:math:`\vec{\textbf{V}}` is then typically expressed as:

.. math::

     J(\vec{\textbf{V}}) = J_{o}(\vec{\textbf{V}}) + J_{c}(\vec{\textbf{V}}) +
                           J_{v}(\vec{\textbf{V}}) + J_{m}(\vec{\textbf{V}}) +
                           J_{b}(\vec{\textbf{V}}) + J_{s}(\vec{\textbf{V}})

In order to find the :math:`\vec{\textbf{V}}` that minimizes
:math:`\vec{J(\textbf{V})}` an iterative optimization technique must be used.
A commonly used technique to minimize :math:`J(\textbf{V})` by iterating
:math:`\vec{\textbf{V_{n}}} = \vec{\textbf{V_{n-1}}} - \alpha\nabla{\vec{\textbf{V}}}`
for an :math:`\alpha > 0` until there is convergence to a solution. This first
requires the user to provide an initial guess :math:`\vec{\textbf{V_{0}}}`.
This is called the gradient descent method that finds the minimum by
decrementing :math:`\vec{\textbf{V}}` in the direction of steepest descent.
Multidop used the gradient descent method to minimize the cost function
:math:`\vec{J(\textbf{V})}`.

However, convergence can be slow with gradient descent. Therefore, in
order to ensure faster convergence, PyDDA uses the limited memory
Broyden–Fletcher–Goldfarb–Shanno (L-BGFS-B) technique that optimizes the gradient
descent method by using the inverse Hessian of the cost function to find an
optimal search direction and :math:`\alpha` for each retrieval. Since there
are physically realistic constraints to :math:`\vec{\textbf{V}}`, the L-BFGS
box (L-BFGS-B) variant of this technique can take advantage of this by only
using L-BFGS on what the algorithm identifies as free variables, optimizing
the retrieval further. The L-BFGS-B algorithm is implemented in SciPy. After
the initial wind field is set up, PyDDA calls the L-BFGS-B algorithm with this
line of code

.. code-block:: python

    from scipy.optimize import fmin_l_bfgs_b

    winds = fmin_l_bfgs_b(
            J_function, winds, args=(vrs, azs, els, wts, u_back, v_back,
            u_model, v_model, w_model, Co, Cm, Cx, Cy, Cz, Cb,
            Cv, Cmod, Ut, Vt, grid_shape,
            dx, dy, dz, z, rmsVr, weights, bg_weights, mod_weights,
            upper_bc, False), maxiter=10, pgtol=1e-3, bounds=bounds,
            fprime=grad_J, disp=0, iprint=-1)

This line of code is rather complex for the end user. Therefore, in order
to simplify this retrieval, PyDDA includes a wrapper function in its
retrieval module called get_dd_wind_field. With this line of code, if one
has grids that they have loaded using the Python ARM-Radar Toolkit into
list_of_grids and initial states of the wind field into arrays called
u_init, v_init, and w_init, retrieval of winds is as easy as

.. code-block:: python

    winds = pydda.retrieval.get_dd_wind_field(
        list_of_grids, u_init, v_init, w_init)

PyDDA even includes an intialization module that will generate
u_init, v_init, w_init for the user. For example, in order to generate an
initial wind field of :math:`\vec{\textbf{V}} = \vec{\textbf{0}}` in the
shape of any one of the grids in list_of_grids, simply do

.. code-block:: python

    u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
        list_of_grids[0], wind=(0.0, 0.0, 0.0))

Available features in PyDDA
---------------------------

In addition, PyDDA also supports 3 types of basic visualizations: wind barb
plots, quiver plots, and streamline plots. These plots are created using
matplotlib and return a matplotlib axis handle so that the user can use
matplotlib to make further customizations to the plots that they desire.

(Go over custom constraints and initalizations)

Hurricane Florence winds using NEXRAD and HRRR
----------------------------------------------

.. figure:: Figure1.png
   :align: center

   A streamline plot of the wind field retrieved by PyDDA from 2 NEXRAD
   radars and the HRRR in Hurricane Florence. The blue contour represents the
   region containing gale force winds, while the red contour represents the
   regions where hurricane force winds are present. :label:`small_hurricane`

.. code-block:: python

    import pyart
    import pydda

    from distributed import Client

    def load_file_and_grid(file_name):
        """
        Processes a radar file by filtering and dealiasing
        velocities using Py-ART

        Parameters
        ----------
        file_name: str
            The name of the file to process.

        Returns
        -------
        my_grid: Py-ART Grid
            The Py-ART Grid for the corresponding radar
        """

        my_radar = pyart.io.read(file_name)

        # Filter out noise
        gf = pyart.filters.GateFilter(mhx_radar)
        gf.exclude_below('cross_correlation_ratio', 0.5)
        gf.exclude_below('reflectivity', -20)

        # Dealias velocities
        dealiased_vel = pyart.correct.dealias_region_based(
            my_radar, gatefilter=gf)

        # Convert to Cartesian coordinates (z, y, x in m)
        grid_spec = (31, 1101, 1101)
        grid_z = (0., 15000.)
        grid_y = (-650000., 650000.)
        grid_x = (-650000., 650000.)
        my_grid = pyart.map.grid_from_radars(
           my_radar, grid_spec, (grid_z, grid_y, grid_x),
           fields=['reflectivity','corrected_velocity'],
           refl_field='reflectivity',roi_func='dist_beam',
           h_factor=0.,nb=0.6,bsp=1.,min_radius=200.,
           grid_origin=(mhx_radar.latitude['data'], mhx_radar.longitude['data']))

        return my_grid

    # Initialize dask client for your cluster
    client = Client(json_file='my_cluster_json.json')

    file_list = ['radar1.nc', 'radar2.nc']
    # Load radar grids using Py-ART
    pyart_grid1 = pyart.io.read_grid('first_radar.nc')
    pyart_grid2 = pyart.io.read_grid('second_radar.nc')
    my_grids = [pyart_grid1, pyart_grid2]

    # Add HRRR GRIB file
    hrrr_path = 'my_hrrr_file.grib'
    my_grids[0] = pydda.constraints.add_hrrr_constraint_to_grid(my_grids[0],
            hrrr_path)

    # Download and add ERA Interim data
    my_grids[0] = pydda.constraints.make_constraint_from_era_interim(
        my_grids[0])

    # Make the output grids
    u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
        grid_mhx, (0.0, 0.0, 0.0))
    out_grids = pydda.retrieval.get_dd_wind_field_nested(
        my_grids, u_init, v_init, w_init, Co=1.0, Cm=100.0,
        Cmod=1e-5, model_fields=["hrrr", "erainterim"],
        client=client)

Another example of the power of PyDDA is its ability to retrieve winds from
networks of radars over areas spanning thousands of kilometers with ease.
:ref:`big_hurricane` shows an example of a retrieval from PyDDA using 6
NEXRAD radars combined with the HRRR and ERA-Interim. Using a multigrid method
that first retrieves the wind field on a coarse grid and then splits the
fine grid retrieval into chunks, this technique can use dask to retrieve
the wind field in Figure :ref:`big_hurricane` about 30 minutes. The code to
retrieve the wind field from many radars and both models is as simple as

.. figure:: Figure2.png
   :align: center

   A wind barb plot showing the winds retrieved by PyDDA from 6 NEXRAD
   radars, the HRRR and the ERA-Interim. Contours are as in Figure
   :ref:`small_hurricane`. :label:`big_hurricane`

Tornado in Sydney, Australia using 4 radars
-------------------------------------------

Combining single weather radars with ERA-Interim
------------------------------------------------

Contributor Information
-----------------------

Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension. 

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work. 

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references. 

So what would previously have registered as a in text reference ``[Atr03]_`` for 

:: 

     [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

what you actually see will be an empty reference rendered as **[?]**.

E.g., [Atr03]_.


If you wish to have a block quote, you can just indent the text, as in 

    When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication. :cite:`hume48`

Dois in bibliographies
++++++++++++++++++++++

In order to include a doi in your bibliography, add the doi to your bibliography
entry as a string. For example:

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = "10.1017/CBO9780511808432",
   }


If there are errors when adding it due to non-alphanumeric characters, see if
wrapping the doi in ``\detokenize`` works to solve the issue.

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = \detokenize{10.1017/CBO9780511808432},
   }

Source code examples
--------------------

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

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
 
Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

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
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


