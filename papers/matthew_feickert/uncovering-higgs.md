## Uncovering the Higgs boson

The most famous and revolutionary discovery in particle physics this century is the discovery of the Higgs boson &mdash; the particle corresponding to the quantum field that gives mass to fundamental particles through the Brout-Englert-Higgs mechanism &mdash; by the ATLAS and CMS experimental collaborations in 2012. [@HIGG-2012-27;@CMS-HIG-12-028]
This discovery work was done using large amounts of customized `C++` software, but in the following decade the state of the PyHEP community has advanced enough that the workflow can now be done using community Python tooling.
To provide an overview of the Pythonic tooling and functionality, a high level summary of a simplified analysis workflow [@IRIS-HEP-AGC] of a Higgs "decay" to two intermediate $Z$ bosons that decay to charged leptons $(\ell)$ (i.e. electrons ($e$) and muons ($\mu$)), $H \to Z Z^{*} \to 4 \ell$, on ATLAS open data [@ATLAS-open-data] is summarized in this section.

### Loading data

Given the size of the data, the files used in a real analysis will usually be cached at a national level "analysis facility" where the analysis code will run.
Using `coffea`, `uproot`, and Dask, these files can then be efficiently read and the tree structure of the data can populate Awkward arrays.

<!-- https://mystmd.org/guide/directives#directive-include -->
```{include} code/read.py
:lang: python
:caption: Using `coffea`, tree structured ROOT files are read with `uproot` from an efficient file cache, and the relevant branches for physics are filtered out into Awkward arrays. The operation is scaled out on a Dask cluster for read performance.
```

### Cleaning and selecting data

Once the data is in Awkward arrays, additional selections need to be applied before it can be analyzed.
Only physics objects of adequate quality are kept for further analysis and those should reconstruct the topology of interest.
In this particular case, due to the decay of the Higgs boson to two leptons, the data selected contain four charged leptons grouped in two opposite flavor lepton pairs (so that the total charge is zero, as the Higgs and the $Z$-bosons are electrically neutral).
Additionally, in order to compare various kinds of simulated data, the events need to be normalized/weighted given their relative appearance in reality and the amount of actual data collected by the experiment.

These selection and weighting can then be implemented in an analysis specific `coffea` processor, and then the processor can be executed used a Dask executor to horizontally scale out the analysis selection across the available compute.

```{include} code/coffea.py
:lang: python
:caption: A `coffea` processor designed to make physics motivated event selections to create accumulators of the 4-lepton invariant mass.
```

### Feature engineering: The invariant mass

In order to discriminate the events of interest, i.e. candidates of the Higgs boson decay, from the vast background which has the same experimental signature, a discriminating feature is constructed.
The example shown uses a simple, physics-inspired discriminant the "invariant mass" but the methods used can use complex feature engineering that involve machine learning methods to calculate more efficient discriminants.
The invariant mass is the mass of a system that remains constant regardless of the system's motion or the reference frame in which it is measured. Invariant mass is derived from the energy and momentum of a system of particles and is a fundamental property of the system:
```{math}
m = {\frac{\sqrt{E^2 - p(c)^2}}{c^2}}
```
where $E$ and $p$ is the total energy and momentum of the particles, respectively.

By detecting and measuring the energies and momenta of the detected particles at the experiment, we can reconstruct the invariant mass of the decay system. Particle systems originating from the decay of the Higgs boson will have a characteristic value of the invariant mass, which after the discovery in 2012 we know it is about 125$GeV/c^2$.
This is the quantity that will allow us to discriminate from particle systems that originate from background processes.

### Measurement uncertainties

One of the most expensive operations that happens during the event selections is the computation of systematic variations of the events to accommodate for imperfect knowledge of the detector systems.
This in practice requires applying complex, experiment specific corrections to each event, using algorithms implemented in `C++`.
Historically these tools were implemented for an event loop processing paradigm, but with recent tooling additions, as shown in @fig:access_layer_diagram, efficient on-the-fly systematic corrections can be computed for array programming paradigms.

The following provides an example of high level Python APIs that provide handles to these tools to use in the workflows described so far.
These tools are efficient enough to be able to apply multiple systematic variations in analysis workflows, as seen in @fig:Zee_mc_systematics.

```{include} code/corrections.py
:lang: python
:caption: Simplified example of what the Python API for a systematic correction tool with a columnar implementation looks like.
```

:::{figure} figures/Zee_mc_systematics.png
:label: fig:Zee_mc_systematics

Example of the reconstructed dilepton invariant mass distribution in simulation with the electron reconstruction and identification efficiency scale factor (SF) and corrections to the energy resolution (res) energy scale (scale) computed on-the-fly using the `nanobind` Python bindings to the ATLAS `C++` correction tools.
The total variation in the systematic corrections is plotted as a hashed band. [@Kourlitis:2890478]
:::

### The "discovery" plot

After running the `coffea` processors, the resulting data from the selections is accumulated into `boost-histogram` objects, as seen visualized in @fig:prefit_plot.


```{include} code/prefit_plot.py
:lang: python
:caption: Using `mplhep`, `hist`, and `matplotlib` the post-processed histograms of the simulation and the data are visualized in advance of any statistical inference of best-fit model parameters.
```

:::{figure} figures/prefit_plot.png
:label: fig:prefit_plot

Using `mplhep`, `hist`, and `matplotlib` the post-processed histograms of the simulation and the data are visualized in advance of any statistical inference of best-fit model parameters.
:::

These histograms are then serialized into files with `uproot` and used by the statistical modeling and inference libraries `pyhf` [@pyhf_zenodo;@pyhf_joss] and `cabinetry` [@cabinetry_zenodo] to build binned statistical models and efficiently fit the models to the observed data using vectorized computations and the optimization library `iminuit` [@iminuit_zenodo] for full uncertainties on all model parameters.
The resulting best-fit model parameters &mdash; such as the scale factor on the signal component of the model corresponding to the normalization on the Higgs contributions &mdash; are visualized in @fig:postfit_plot, where good agreement between the model predictions and the data is observed.
The signal component, clearly visible above the additional "background" components of the model, are Higgs boson events, with an observed count in agreement with theoretical expectations.

```{include} code/postfit_plot.py
:lang: python
:caption: Using `cabinetry`, `pyhf`, and `matplotlib` the data and the post-fit model prediction are visualized.
```

:::{figure} figures/postfit_plot.png
:label: fig:postfit_plot

Using `cabinetry`, `pyhf`, and `matplotlib` the data and the post-fit model prediction are visualized.
:::
