:author: Orhan Eroglu
:email: oero@ucar.edu
:institution: National Center for Atmospheric Research
:corresponding:

:author: GeoCAT member 1
:email: email
:institution: National Center for Atmospheric Research

:author: GeoCAT member 2
:email: email
:institution: National Center for Atmospheric Research

:author: GeoCAT member 3
:email: email
:institution: National Center for Atmospheric Research

:author: John Clyne
:email: clyne@ucar.edu
:institution: National Center for Atmospheric Research


:bibliography: references

:video: https://www.youtube.com/watch?v=34zFGkDwJPc

---------------------------------------------------------------------------------------------------------------------------
The Geoscience Community Analysis Toolkit: An Open Development, Community Driven Toolkit in the Scientific Python Ecosystem
---------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

The Geoscience Community Analysis Toolkit (GeoCAT) team develops and maintains
data analysis and visualization tools on structured and unstructured grids for
the geosciences community in the scientific Python ecosystem. In response to
dealing with increasing geoscientific data sizes, GeoCAT prioritizes scalability,
ensuring its implementations to be scalable from personal laptops to HPC clusters.
Another major goal of the GeoCAT team is to ensure community involvement throughout
the whole project lifecycle, which is realized through an open development mindset
by encouraging the users/contributors to get involved in any decision-making.

.. class:: keywords

   data analysis, geocat, geoscience, open development, open source, scalability,
   visualization

Introduction
------------

The Geoscience Community Analysis Toolkit (GeoCAT) team, established in 2019,
leads the software engineering efforts of the National Center for Atmospheric
Research (NCAR)’s “Pivot to Python” initiative :cite:`pivot19`. GeoCAT essentially
aims at creating scalable data analysis and visualization tools on structured and
unstructured grids for the geosciences community in the scientific Python
ecosystem. The GeoCAT team is committed to open development, which helps the
team prioritize community involvement at any level of the project lifecycle
alongside having the whole software stack open-sourced.

GeoCAT created several, now-established Python tools that are hosted and
managed publicly on Github to develop computation and visualization functions,
which  are built on cornerstone Pangeo :cite:`pangeo18` (i.e. a community platform
for big data geoscience) packages such as Xarray :cite:`xarray17` and Dask
:cite:`dask15`. Namely, GeoCAT-comp houses computational operators for
applications ranging from regridding and interpolation, to climatology and
meteorology. GeoCAT-examples provides over 140 publication-quality plotting
scripts in Python for Earth sciences. It also houses Jupyter notebooks with
high-performance, interactive plots that enable features such as pan and zoom
on fine-resolution geoscience data (e.g. ~3 km data rendered within a few
tens of seconds to few minutes on personal laptops). GeoCAT-viz enables
higher-level implementation of Matplotlib and Cartopy plotting capabilities
through it's variety of easy to use visualization convenience functions for
GeoCAT-examples. GeoCAT also maintains WRF-Python (Weather Research and
Forecasting), which works with WRF-ARW model output and provides diagnostic
and interpolation routines.

GeoCAT was recently awarded Project Raijin, which is an NSF EarthCube-funded
effort :cite:`raijinaward21`. Its goal is to enhance the open-source analysis and visualization tool
landscape by developing community-owned, sustainable, scalable tools that
facilitate operating on unstructured climate and global weather data in the
scientific Python ecosystem. To realize this, GeoCAT created the Xarray-based
Uxarray package to recognize unstructured grid models through partnership with
the geoscience community groups. Throughout this three-year project, GeoCAT
will work on the development of data analysis and visualization functions that
operate directly on the native grid as well as establish an active community
of user-contributors.

This paper will provide insights about the GeoCAT's open development methodology,
software stack and current status, team scope and near-term plans, as well as
ways of community involvement.

GeoCAT Software
---------------



Open Development
----------------

To ensure community involvement at any level in the development lifecycle, GeoCAT
is committed to an open development model. With this model, we not only
have our code-base open-sourced but also ensure most of the project assets that are
directly related to the software development lifecycle are publicly accessible.
In order to implement this model, GeoCAT provides all of its software tools as
Github repositories with publicly accessible Github project boards and roadmaps,
issue tracking and development reviewing, comprehensive documentation for
users/contributors such as Contributor’s Guide and toolkit-specific
documentation, along with community announcements on the GeoCAT blog.
Furthermore, GeoCAT encourages community feedback and contribution at any level
with inclusive and welcoming language.

GeoCAT employs a continuous delivery model, which has been achieved through the use
of a monthly package release cycle on package management systems and package indexes
such as Conda and PyPI. To assist this process, the team utilizes advanced continuous
integration/deployment (CI/CD) technologies throughout Github assets to ensure
automation, unit testing and code coverage, licensing and
reproducibility. Additionally, to further promote user engagement with the
geoscience community, GeoCAT has contributed multiple Python tutorials to
the web-based, community-owned, educational resources created through Project
Pythia. The GeoCAT team has also encouraged undergraduate and graduate student
engagement in the python ecosystem through participation in NCAR's Summer
Internships in Parallel Computational Science (SIParCS).


