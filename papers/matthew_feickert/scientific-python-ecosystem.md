## Employing the Scientific Python ecosystem

The multiple stages of physics data processing and analysis map onto different parts of the Scientific Python ecosystem.
This begins with the highly-structured but jagged nature of the event data in HEP.
The data structure of each event consists of variable length lists of physics objects (e.g. electrons, collections of tracks from charged objects).
To study the properties of the physics objects in a statistical manner, a fixed event analysis procedure is repeated over billions of events.
This has traditionally motivated the use of "event loops" that implicitly construct event-level quantities of interest and leveraged the `C++` compiler to produce efficient iterative code.
This precedent made it difficult to take advantage of array programming paradigms that are common in Scientific Python given NumPy [@numpy] vector operations.
The Scikit-HEP library Awkward Array [@Awkward_Array_zenodo] provides a path forward by providing NumPy-like idioms for nested, variable-sized (JSON-like) data and also brings analysts into an array programming paradigm [@Hartmann:2021qzp].

With the ability to operate on HEP data structures in an array programming &mdash; or "columnar" &mdash; approach, the next step is to be able to read and write with the HEP domain specific ROOT [@Brun:1997pa] file format &mdash; which has given the particle physics community columnar data structures with efficient compression since 1997 [@Pivarski:2020qcb].
This is accomplished with use of the `uproot` library [@Uproot_zenodo], which allows for efficient transformation of ROOT data to NumPy or Awkward arrays.
The data is then filtered through kinematic and physics signature motivated selections using Awkward manipulations and queries to create array collections that contain the passing events.
Through intense detector characterization and calibration efforts, the ATLAS collaboration has developed robust methods and tooling to apply corrections to the data and evaluate systematic uncertainties.
For instance, corrections to the signal collected by a specific calorimeter subsystem along with systematic uncertainties due to the imperfect knowledge of the subsystem.
Given the custom nature of the detector and correction implementations, these corrections are implemented in custom `C++` libraries in the ATLAS software framework, Athena [@ATL-SOFT-PUB-2021-001;@Athena_zenodo].
To expose these `C++` libraries to the Pythonic tooling layer, custom Python bindings are written using `nanobind` for high efficiency, as seen in @fig:access_layer_diagram.

:::{figure} figures/access_layer_diagram.png
:label: fig:access_layer_diagram

The data access abstract interface from the high level user facing Python API to the ATLAS Event Data Model (EDM) access library that exposes the shared ATLAS combined performance (CP) tools for reconstruction, identification, and measurement of physics objects. [@Kourlitis:2890478]
:::

To contend with the extreme data volume, efficient distributed computing is an essential requirement.
Given the success of Dask [@Dask] in the Scientific Python ecosystem, and its ability to be deployed across both traditional batch systems and cloud based infrastructure with Kubernetes, the Scikit-HEP ecosystem has built extensions to Dask that allow for native Dask collections of Awkward arrays [@Dask-awkward] and computing multidimensional `boost-histogram` objects [@Boost-histogram_zenodo] with Dask collections [@Dask-histogram].
Using Dask and these extensions, the data selection and systematic correction workflow is able to be horizontally scaled out across ATLAS collaboration compute resources to provide the data throughput necessary to make analysis feasible.
This is often achieved through use of the high level `coffea` columnar analysis framework [@coffea_zenodo] which was designed to integrate with Dask and these HEP specific Dask extensions.

The resulting data objects that are returned to analysts are histograms of physics quantity distributions &mdash; such as the reconstructed invariant-mass of a collection of particles or particle momentum.
Using the `hist` library [@hist_zenodo] for higher level data exploration and manipulation, physicists are then able to efficiently further manipulate the data distributions using tooling from the broader Scientific Python ecosystem and create domain-centric visualizations using the `mplhep` [@mplhep_zenodo] extension of Matplotlib [@matplotlib].
From these high level data representations of the relevant physics, physicists are then able to serialize the distributions and use them for the final stages of data analysis and statistical modeling and inference.
