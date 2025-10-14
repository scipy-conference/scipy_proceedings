---
# Ensure that this title is the same as the one in `myst.yml`
title: Extension of the OpenMC depletion module for transport-independent depletion
abstract: |
  We have added functionality for running depletion simulations independently
  of neutron transport in OpenMC, an open source Monte Carlo particle
  transport code with an internal depletion module. Transport-independent
  depletion uses pre-computed static multigroup cross sections and fluxes to
  calculate reaction rates for OpenMC's depletion matrix solver. This
  accelerates the depletion calculation, but removes the spatial coupling
  between depletion and neutron transport.

  We used a simple PWR pincell to validate the method against the existing
  transport-coupled depletion method.  Nuclide concentration errors roughly
  scale with depletion time step size and are inversely proportional to the
  amount of the nuclide present in a depletable material. The magnitude of
  concentration error depends on the nuclide of interest. Concentration errors
  for low-abundance nuclides at longer (30-day) time steps exhibit large
  negative initial concentration the becomes more positive with time due to
  overestimation of nuclide production stemming from the lack of spatial
  coupling to neutron transport. For ten 3-day time steps, fission product
  concentration errors are all under 3\%.  Actinide concentration errors range
  from 10-15\% for Am and Cm, 5-7\% for Pu and Np, and 2\% and less for U.
  Surprisingly, the numbers are similar for 30-day time steps. These results
  demonstrate the potential of this new method with moderate accuracy and
  extraordinary time savings for low and medium fidelity simulations.
  Concentration error characterization on larger models remains an open area of
  interest.
acknowledgment: |
  This work was by the Exascale Computing Project (17-SC-20-SC), a collaborative
  effort of the U.S. Department of Energy Office of Science and the National
  Nuclear Security Administration. This work was additionally supported through
  the NRC Integrated University Grant Program Fellowship.

  Refactoring of the initial implementation of the `MicroXS` and
  `IndependentOperator` classes to support multi-group cross sections was
  supported by the U.S. Department of Energy Office of Fusion Energy Sciences
  under Award Number DE-SC0022033.

  We gratefully acknowledge the computing resources provided on Bebop, a
  high-performance computing cluster operated by the Laboratory Computing
  Resource Center at Argonne National Laboratory.  This research made use of the
  resources of the High Performance Computing Center at Idaho National
  Laboratory, which is supported by the Office of Nuclear Energy of the U.S.
  Department of Energy and the Nuclear Science User Facilities under Contract
  No. DE-AC07-05ID14517
data_availability: |
  The data that support the findings of this study are openly available at the
  following DOI: 10.5281/zenodo.15103379
abbreviations:
  PWR: Pressurized Water Reactor
  CRAM: Chebyshev Rational Approximation Method
---

(introduction)= 
## Introduction

Accurate simulations of advanced reactors on exascale computing hardware
requires state-of-the-art software tools in Monte Carlo transport and
heat-coupled computational fluid dynamics
[@romano2021nse; @merzari2023sc].  These methods are computationally
expensive, but deliver high confidence in their results. If any component in
this tool chain could be simplified or sped up, it would reduce the
computational cost on and economic cost from these exascale machines. For
Monte Carlo neutron transport, depletion simulations are one of the more
expensive simulations. _Depletion_ is the process by which the nuclide
composition changes due to nuclear transmutation and decay reactions
occurring in a material subject to irradiation. Depletion capabilities were
recently added [@romano_depletion_2021] to OpenMC
[@romano_openmc_2015], an open source, community developed Monte Carlo
particle transport code. The depletion solver utilizes OpenMC's Python API
to iteratively run a transport simulation, calculate transmutation reaction
rates, solve the depletion matrix ([Eq. %s](#eq:depletion-matrix-t)), and
update the material composition. This makes transport-coupled depletion
calculations computationally expensive, especially for large reactor models.

In the present work, we describe a new method for running depletion
calculations without the need to iteratively run transport calculations by
using pre-calculated microscopic cross sections and fluxes for each depletion
step. At most, only one transport calculation needs to be run to generate
these cross sections and fluxes. Neutron transport and depletion are tightly
coupled because the change in material composition can result in changes to
the spatial distribution of reaction rates. If few materials are used in a
transport-independent depletion calculation, this effectively removes
spatial effects of the changing material composition due to the static
fluxes and cross sections.

This limits the practical application of transport-independent depletion to
domains where capturing spatial effects are a secondary concern. One
such application is in global fuel-cycle analysis where we are more
concerned with the overall change in material composition. Fuel cycle
simulations typically model many different reactors simultaneously over
long time periods. The capabilities described in this paper
have been utilized in Cyclus [@huff_fundamental_2016] -- an open source
fuel cycle simulator -- for coupling depletion to the fuel cycle
simulaton [@bachmann_os_2024]. Another area of application is in
facilities where the coupling between neutron transport and depletion is
effectively one-way.  This is often the case for experimental facilities
with a neutron source where the source rate is high enough to cause
activation of materials but not high enough to produce meaningful burnup of
materials. Many proposed fusion energy facilities would fall into this
category. For such applications, it is unnecessary to run a fully coupled
simulation between neutron transport and depletion; instead, a neutron
transport simulation corresponding to the beginning of irradiation can
determine reaction rates relative to the neutron source rate, and then the
depletion calculation can be performed using the reaction rates scaled by
any changes in the neutron source rate over time. For fusion systems, these
calculations are often carried out with FISPACT-II [@sublet2017nds];
See [@eade2020nf] for a recent example. The capabilities described in
this paper have also been utilized for a recent study of shutdown dose rate
in fusion systems [@peterson2024nf].

This paper is organized as follows: In [](#methods), we describe
the methods and algorithms used to calculate reaction rates using the new
depletion capabilities. In [](#results), we describe our
simulation and analyze the results. In [](#conclusion), we
summarize our results, and discuss some gaps in the current implementation
of the new feature.

(methods)=
## Methods
Depletion is typically done in sub-regions that are small enough to have
little spatial variation in the flux (and hence in the reaction rates). For
example, in a model of a full reactor core, depletion regions may correspond to
individual fuel pins, or even smaller sub-pin regions of radial rings and
azimuthal segments. The general form of the equation governing depletion is
% The notation here matches what is in the OpenMC depletion paper from 2021
\begin{equation}
  \label{eq:depletion}
  \begin{split}
    \frac{dN_i}{dt} = &\sum\limits_{j=1}^n \left[ \int_0^\infty 
    f_{j \rightarrow i}(E) \sigma_j (E) \phi(E,t) dE \;+ \lambda_{j\rightarrow i}
    \right] N_j(t) \\ &- \left [\int_0^\infty \sigma_i (E) \phi(E,t) dE \; +
    \sum\limits_{j=1}^n \lambda_{i\rightarrow j} \right ] N_i(t),
  \end{split}
\end{equation}
where
\begin{equation*}
  \begin{split}
    N_i(t) &\equiv \text{density of nuclide $i$ at time $t$} \\
    \sigma_i(E) &\equiv \text{transmutation cross section for nuclide $i$ at energy $E$} \\
    \phi(E,t) &\equiv \text{neutron flux at energy $E$ and time $t$} \\
    f_{j \rightarrow i}(E) &\equiv \text{fraction of transmutation reactions in nuclide $j$ that produce nuclide $i$} \\
    \lambda_{j \rightarrow i} &\equiv \text{decay constant for decay modes in nuclide $j$ that produce nuclide $i$} \\
    n &\equiv \text{total number of nuclides.}
  \end{split}
\end{equation*}
The summations in Equation [](#eq:depletion) are effectively over all
nuclides except $i$, as $f_{i \rightarrow i}(E)$ and $\lambda_{i \rightarrow i}$ must
be zero by definition. Equation [](#eq:depletion) can be condensed into a
matrix-vector equation:
\begin{equation}
  \label{eq:depletion-matrix-t}
  \frac{d\mathbf{n}}{dt} = \mathbf{A}(\mathbf{n},t)\mathbf{n}
\end{equation}
where
\begin{equation}
  \mathbf{n} = \begin{pmatrix} N_1 \\ N_2 \\ \vdots \\ N_n \end{pmatrix}, \quad
\end{equation}
and $\mathbf{A}(\mathbf{n},t) \in \mathbb{R}^{n\times n}$ is the depletion
matrix containing the decay and transmutation coefficients. The coefficients
of $\mathbf{A}$ change with time and depend on the nuclide
composition vector $\mathbf{n}$, making this a nonlinear system of equations.
OpenMC utilized the CRAM method to solve this system of equations
[@romano_depletion_2021]. 

We had two design goals for
transport-independent depletion functionality:

1. Enable use of OpenMC depletion capabilities using only the Python API
2. Enable depletion of materials directly given a multi-group flux
        for each material.

We created the `IndependentOperator` and `MicroXS` classes to
achieve these design goals.

(microxs)=
  ### MicroXS
  The `MicroXS` class contains cross section data indexed by nuclide,
  reaction name, and energy group. It also contains methods to import and
  export `MicroXS` objects from `.csv` files. In principle, a
  user could create multi-group microscopic cross sections with a
  transport solver code, and then use the `MicroXS`  class to read in
  the cross section data for use in depletion. However, this is
  unnecessary as the `get_microxs_and_flux()` function runs an OpenMC
  $k$-eigenvalue calculation to create a `MicroXS` object and
  multi-group flux profile for each user-specified domain.

  In the initial release of this feature in OpenMC v0.13.1, `MicroXS`
  sub-classed the `pandas.DataFrame`. class to store data and assumed a
  one-group structure. The v0.14.0 release removed the `pandas` dependency
  and refactored `MicroXS` class to store multi-group cross section
  data.

(independentoperator)=
  ### IndependentOperator
  The `Operator` class contains data and methods for running
  transport-coupled depletion calculations and runtime processing of
  OpenMC statepoint files to extract the data needed to solve Equation
  [](#eq:depletion-matrix-t). In this work, we refactored the
  `Operator` class into a `CoupledOperator`  class maintaining
  the existing depletion capability. We implemented the
  transport-independent depletion machinery in a new class,
  `IndependentOperator` which uses an instance of the `MicroXS`
  class and corresponding fluxes for each depletion domain to calculate
  reaction rates for that domain using multi-group microscopic cross
  sections.

  Calculating reactions rates using multi-group cross sections is
  mathematically equivalent to tallying the reaction rates directly. In
  transport-coupled depletion calculation, the transmutation reaction
  rates $r_{ij}(t)$ are tallied directly as:
  \begin{equation}
      \label{eq:cont-rxn-rate}
      r_{ij}(t) = \left[\int_0^\infty \sigma_{ij}(E) \phi(E,t) dE \; \right]
      N_{i}(t)
  \end{equation}
  where $i$ and $j$ indicate the nuclide and reaction, respectively.

  To perform depletion using multi-group cross sections, the multi-group
  cross sections and fluxes are multiplied to get a per-source neutron
  reaction rate:
  \begin{equation}
      \label{eq:mg-rxn-rate}
      r_{ij}(t) = \left[\sum_{g} \sigma_{ij,g} \phi_{g}(t) \right]
      N_{i}(t) 
  \end{equation}
  In practice, this introduces a discretization error, but in the limit of
  infinitely small energy groups, Equation [](#eq:mg-rxn-rate) is
  equivalent to Equation [](#eq:cont-rxn-rate). To get a time-dependent
  microscopic cross section or flux, we must perform transport
  calculations (and in the case of time-varying microscopic cross
  sections, incorporate thermal feedbacks). Herein lies the primary
  assumption of transport-independent depletion: _microscopic cross
  sections and reactor fluxes are static_.

  %% THIS DOESN'T MATCH WITH OPENMCs COMMENTS
  Both the tallied and multi-group-calculated reaction rates have units of
  $\text{reactions}/\text{src particle}$. A normalization factor
  $f$ in $\text{src}/\text{s}$ is applied to obtain the more
  conventional unit of $\text{reactions}/\text{s}$:
  \begin{equation}
      R_{ij} = r_{ij} f
  \end{equation}
  This normalization factor is typically related to the power of the
  reactor and energy released from fission (`fission-q`
  normalization) or the external source strength (`source-rate`
  normalization). `IndependentOperator` may use either of these methods to normalize
  the reaction rates. While `CoupledOperator` may use several methods to
  calculate fission yields, including using the Monte Carlo simulation to
  directly tally fission yields, `IndependentOperator` relies on
  constant yield data included in the depletion chain file.

(results)=
## Results
To validate transport-independent depletion, we ran depletion simulations on
a PWR with a single depletion zone for three cases:
1. Transport-coupled depletion,
2. Transport-independent depletion,
3. Transport-independent depletion with microscopic cross sections
   updated after each depletion step.

The first case serves as a reference solution we use to estimate the error
in the solution of the new method. We used 25 inactive and 125 active
batches, with $10^6$ particles per batch. For the second case, we used three
different group structures to test the effect on solution accuracy: one
group, CASMO-8, and CASMO-40. The third case is much slower than the first
case as the cross sections need to be reloaded after each depletion step.
The third case is a sanity check that the cross sections are being computed
correctly, as it should have a smaller error than the second case.
Due to the cross section discretization that occurs before performing
depletion, we do not expect the third case results to converge to the first
case results in limit of infinite particles, but we would expect it to
converge in the limit of infinite particles _and_ energy groups. The
third case was only run for one-group transport-independent depletion. 

[](#tab:mat-params) and [](#tab:mat-comps) contain the material parameters and
compositions of our model, and [](#tab:geo-params) contains the geometric
parameters.  We used the ENDF/B-VII.1 nuclear data library available at
[openmc.org/data](https://openmc.org/data/#endf-b-vii-1). We used the ENDF/B-VII.1 depletion
chain in the PWR spectrum also available at [openmc.org/data](https://openmc.org/data/#endf-b-vii-1-chain-thermal-spectrum).

```{raw} latex
\begin{longtable*}{|c|c|c|c|}
    \caption{Material Parameters}
    \label{tab:mat-params}
    \toprule
    {\bf Item} & {\bf Fuel} & {\bf Cladding} & {\bf Water} \\ % Table header row
    \midrule
      Density [g cm$^{-3}$] & 10.4 & 6 & 1.0\\
      Volume [cm$^{3}$] & 0.1764$\pi$ & -- & -- \\
      S($\alpha$,$\beta$) & --  & -- & `c_H_in_H2O`\\
\end{longtable*}
```

:::{table} Material Compositions
:label: tab:mat-comps

<table>
<tr>
<th colspan="2">Fuel composition</th>
<th colspan="2">Cladding composition</th>
<th colspan="2">Water composition</th>
</tr>
<tr>
<th>Nuclide</th>
<th>Composition [atom %]</th>
<th>Nuclide</th>
<th>Composition [atom %]</th>
<th>Nuclide</th>
<th>Composition [atom %]</th>
</tr>
<tr>
<td><sup>15</sup>O</td><td>0.000758</td>
<td><sup>90</sup>Zr</td><td>0.5145</td>
<td><sup>1</sup>H</td><td>1.999689</td>
</tr>
<tr>
<td><sup>16</sup>O</td><td>1.999242</td>
<td><sup>91</sup>Zr</td><td>0.1122</td>
<td><sup>2</sup>H</td><td>0.000311</td>
</tr>
<tr>
<td><sup>234</sup>U</td><td>0.000385</td>
<td><sup>92</sup>Zr</td><td>0.1715</td>
<td><sup>15</sup>O</td><td>0.999621</td>
</tr>
<tr>
<td><sup>235</sup>U</td><td>0.043020</td>
<td><sup>94</sup>Zr</td><td>0.1738</td>
<td><sup>16</sup>O</td><td>0.000379</td>
</tr>
<tr>
<td><sup>236</sup>U</td><td>0.000197</td>
<td><sup>96</sup>Zr</td><td>0.028</td>
<td></td><td></td>
</tr>
<tr>
<td><sup>238</sup>U</td><td>0.956398</td>
<td></td><td></td>
<td></td><td></td>
</tr>
</table>
:::

```{raw} latex
\begin{longtable*}{|c|c|c|}
    \caption{Geometric Parameters}
    \label{tab:geo-params}
    \toprule
    {\bf Fuel Radius} [cm] & {\bf Clad Radius} [cm] & {\bf Water Bounding Box dimensions}
    [cm  $\times$ cm]\\
    \midrule
    0.42 & 0.45 &  1.24 $\times$ 1.24\\
\end{longtable*}
```

Depletion is a slow process whose effects on neutronics only start to apply
on longer timescales (days and months). For validation purposes we ran each
of the three cases with four different magnitudes of timestep size: 360
seconds, 4 hours, 3 days, and 30 days. All simulations ran for 10 depletion
steps. We used the `PredictorIntegrator` time stepper, which is
analogous to the Euler method. Predictor-corrector methods are not useful
for transport independent depletion. The correction step will utilize the
same reaction rates as the initial prediction due to the static fluxes and
cross sections.  This means any predictor-corrector method will produce the
same numerical results as a predictor method but with higher floating-point
errors due to the increased number of operations [^footnote-1]. We used constant
reaction rate Q values multiplied by the reaction rates to obtain the
normalization factor $f$ (`fission-q` normalization) for all cases with
a linear power density of 174 W/cm.

[^footnote-1]: Preliminary testing with predictor-corrector methods confirmed
    the higher floating point errors, however their magnitudes are marginal. 

The general trend in our results is that concentration errors are smaller
for shorter depletion time steps, and larger for longer depletion time
steps, however some nuclides exhibit more complicated behavior. 

::::{figure}
:label: actinide-error-days

```{figure} figs/actinides_constant_xs_predictor_fission_q_days.*
:label: fig:actinides-error-constant-xs-days
:width: 50%
```

```{figure} figs/actinides_updating_xs_predictor_fission_q_days.*
:label: fig:actinides-error-updating-xs-days
:width: 50%
```

Relative actinide concentration error using 3-day time steps at 3, 12, 21, and
30 days of depletion for [({subEnumerator})](#fig:actinides-error-constant-xs-days) constant
cross sections; [({subEnumerator})](#fig:actinides-error-updating-xs-days) updating cross
sections.
::::

[](#fig:actinides-error-constant-xs-days) and
[](#fig:actinides-error-updating-xs-days) show the relative error in predicted
actinide concentration for constant cross sections and updating cross sections,
respectively, using 3-day time steps. [](#fig:actinides-error-constant-xs-months)
and [](#fig:actinides-error-updating-xs-months) show the same respective
quantities for 30-day time steps. As expected, updating the cross sections at
each depletion step results in low predicted nuclide concentration errors, on
the order of a fraction of a percent. The error trend for constant cross
sections depends both on the cross section and time step size. For example,
constant cross sections using a 3-day time step
underpredicts the $^{241}$Pu concentration, but at longer time steps
overpredicts the concentration. [](#fig:pu240-n-gamma-months) shows
the overprediction is due to overcalculating the rate of ($n,\gamma$) reactions
that occur on $^{240}$Pu. 


:::{figure} figs/pu240-n-gamma-months.*
:label: fig:pu240-n-gamma-months
Relative $^{240}$Pu ($n,\gamma$) reaction rate error using constant cross
sections and 30-day time steps.
:::

The error in $^{241}$Pu concentration propagates to daughter nuclides
that are related to the amount of $^{241}$Pu, such as isotopes of Cm
and Am. We observe a general trend that the less abundant
nuclides have higher concentration error relative to transport-coupled
depletion. The less abundant actinides have increased sensitivity to
variations in production. The more abundant nuclides (U,
$^{239}$Np, $^{239}$Pu, $^{240}$Pu) tend to have low concentration
errors, (5\% or less), whereas the less abundant actinides (isotopes of Am,
isotopes of Cm, $^{241}$Pu, $^{242}$Pu) tend to have high (more
than 10\%) concentration errors depending on the nuclide of interest.


::::{figure}
:label: actinide-error-months

```{figure} figs/actinides_constant_xs_predictor_fission_q_months.*
:label: fig:actinides-error-constant-xs-months
:width: 50%
```

```{figure} figs/actinides_updating_xs_predictor_fission_q_months.*
:label: fig:actinides-error-updating-xs-months
:width: 50%
```

Relative actinide concentration error using 30-day time steps at 30, 120, 210, and
300 days of depletion for [({subEnumerator})](#fig:actinides-error-constant-xs-months) constant
cross sections; [({subEnumerator})](#fig:actinides-error-updating-xs-months) updating cross
sections.
::::

[](#fig:fp-error-constant-xs-days) and [](#fig:fp-error-updating-xs-days) show the
relative error in predicted fission product concentration using constant cross
sections and updating cross sections, respectively, using 3-day time steps.
[](#fig:fp-error-constant-xs-months) and
[](#fig:fp-error-updating-xs-months) show the same respective quantities for
30-day time steps. Similar to the actinides, updating the cross sections at each
depletion step results in low predicted nuclide concentration errors. There is a
lower concentration error across the board for many of these fission products
compared to the actinides. This is due to the low concentration error of
$^{235}$U, which propagates to the fission products. Other actinides with higher
concentration errors are orders of magnitude less abundant in the fuel than
$^{235}$U, so their effect on fission product concentration error is
proportionally small.

Concentration errors for low-abundance nuclides using 30-day time steps follow
an interesting trend where there is an initial large negative concentration
error that becomes more positive over time. This behavior is due to
the static cross sections and fluxes producing the same amount of these
nuclides every timestep, whereas in the transport-coupled case,
net production of the low-abundance nuclides decreases over time.

::::{figure}
:label: fp-error-days

```{figure} figs/fission_products_constant_xs_predictor_fission_q_days.*
:label: fig:fp-error-constant-xs-days
:width: 50%
```

```{figure} figs/fission_products_updating_xs_predictor_fission_q_days.*
:label: fig:fp-error-updating-xs-days
:width: 50%
```

Relative fission product concentration error using 3-day time steps at 3, 12, 21, and
30 days of depletion for [({subEnumerator})](#fig:fp-error-constant-xs-days) constant
cross sections; [({subEnumerator})](#fig:fp-error-updating-xs-days) updating cross
sections.
::::

::::{figure}
:label: fp-error-months

```{figure} figs/fission_products_constant_xs_predictor_fission_q_months.*
:label: fig:fp-error-constant-xs-months
:width: 50%
```

```{figure} figs/fission_products_updating_xs_predictor_fission_q_months.*
:label: fig:fp-error-updating-xs-months
:width: 50%
```

Relative fission product concentration error using 30-day time steps at 30, 120, 210, and
300 days of depletion for [({subEnumerator})](#fig:fp-error-constant-xs-months) constant
cross sections; [({subEnumerator})](#fig:fp-error-updating-xs-months) updating cross
sections.
::::

Repeating this analysis for both the CASMO-8 and CASMO-40 multi-group
structures did not yield noticeable decreases in nuclide concentration
errors for transport-independent depletion over the one-group case.
[](#fig:actinides-error-casmo8-constant-xs-days) and
[](#fig:actinides-error-casmo40-constant-xs-days) show the relative error in
predicted actinide concentration using the CASMO-8 and CASMO-40 group
structures, respectively, using 3-day time steps.
[](#fig:actinides-error-casmo8-constant-xs-months) and
[](#fig:actinides-error-casmo40-constant-xs-months) show the same respective
quantities for 30-day time steps.  It is possible in more complex models,
like full reactor depletion, that the multi-group structure could become more
important.

::::{figure}
:label: actinide-error-casmo-days

```{figure} figs/actinides_casmo8_constant_xs_predictor_fission_q_days.*
:label: fig:actinides-error-casmo8-constant-xs-days
:width: 50%
```

```{figure} figs/actinides_casmo40_constant_xs_predictor_fission_q_days.*
:label: fig:actinides-error-casmo40-constant-xs-days
:width: 50%
```

Relative actinide concentration error using constant cross sections and 3-day
time steps at 3, 12, 21, and 30 days of depletion for the
[({subEnumerator})](#fig:actinides-error-casmo8-constant-xs-days) CASMO-8 group
structure; [({subEnumerator})](#fig:actinides-error-casmo40-constant-xs-days)
CASMO-40 group structure.
::::

::::{figure}
:label: actinide-error-casmo-months

```{figure} figs/actinides_casmo8_constant_xs_predictor_fission_q_months.*
:label: fig:actinides-error-casmo8-constant-xs-months
:width: 50%
```

```{figure} figs/actinides_casmo40_constant_xs_predictor_fission_q_months.*
:label: fig:actinides-error-casmo40-constant-xs-months
:width: 50%
```

Relative actinide concentration error using constant cross sections and 3-day
time steps at 30, 120, 210, and 300 days of depletion for the
[({subEnumerator})](#fig:actinides-error-casmo8-constant-xs-months) CASMO-8 group
structure; [({subEnumerator})](#fig:actinides-error-casmo40-constant-xs-months)
CASMO-40 group structure.
::::

The transport-coupled simulations each took several hours to complete,
whereas the transport-independent simulations using
`IndependentOperator` took seconds to minutes to complete.


(conclusion)=
## Conclusion
In this paper, we introduced transport-independent depletion in OpenMC. This
method utilized pre-computed multi-group cross sections and fluxes to
calculate reaction rates for a depletion calculation. The new method is much
faster than running a transport-coupled depletion simulation, albeit with a
penalty to accuracy. Better accuracy will be obtained for models where the
neutron flux spectrum will be constant, i.e. in fusion systems and low power
fission reactors, so this new method should be used judiciously on problems
where they are applicable.

This study focused exclusively on nuclide compositions. Future work may include
building machinery to monitor criticality as the fuel depletes. A good
starting point may be in implementing a method similar to the method to estimate
$k_{\infty}$ used in [@LOVECKY2014333], wherein
\begin{equation}
    k_{\infty} = \frac{\sum_{i} \sum_{j} N_{i} \nu_{j}
    \sigma_{i}^{j}}{\sum_{i}\sum_{j} N_{i} \sigma_{i}^{j}}  
\end{equation}
where $j$ indicates the type of reaction, and $\nu_{j}$ is the neutron
multiplication of reaction $j$, defined as
\begin{equation}
    \nu_j = \begin{cases}
        0 & j=\text{ absorption reaction}\\
        x & j=(n,xn)\\
        \bar{\nu} & j=\text{fission}
    \end{cases}
\end{equation}
In OpenMC, $(n,xn)$ reactions are not used to calculate $k_\text{eff}$ (and
by extension $k_{\infty}$). In this case, $\nu_{x} = 0$ in the numerator for
all $x$, and we would use an 'effective' absorption cross section
$\overline{\sigma}_{a} = \sigma_{a} - \sigma_{(n,2n)} - 2\sigma_{(n,3n)} -
3\sigma_{(n,4n)}$. 

Future work could focus on characterizing the effect of sub-regions on the
concentration errors. The geometry used in this study is a simple PWR
pincell. It is possible that the error due to static fluxes could be reduced
by dividing the pincell into sub-regions to better capture radial and axial
variations as is done in high-fidelity depletion of large reactor models.
Another area for future work could be on studying the importance of energy
group discretization for more complex reactor models. We did not find a
difference in concentration errors between one-group, CASMO-8 group, and
CASMO-40 group transport-independent depletion for the pincell model. It is
possible that energy group discretization becomes more important for complex
models with distinct depletion zones.
