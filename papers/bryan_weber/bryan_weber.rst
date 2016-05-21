:author: Bryan W. Weber
:email: bryan.w.weber@gmail.com
:institution: Mechanical Engineering Department, University of Connecticut, Storrs, CT 06269

:author: Chih-Jen Sung
:email: chih-jen.sung@uconn.edu
:institution: Mechanical Engineering Department, University of Connecticut, Storrs, CT 06269

:bibliography: SciPy-2016

---------------------------------------------------------------------
UConnRCMPy: Python-based data analysis for Rapid Compression Machines
---------------------------------------------------------------------

.. class:: abstract

    The ignition delay of a fuel/air mixture is an important quantity in designing combustion
    devices, and these data are also used to validate computational kinetic models for combustion.
    One of the typical experimental devices used to measure the ignition delay is called a Rapid
    Compression Machine (RCM). This work presents UConnRCMPy, an open-source Python package to
    process experimental data from the RCM at the University of Connecticut. Given an experimental
    measurement, UConnRCMPy computes the thermodynamic conditions in the reaction chamber of the RCM
    during an experiment along with the ignition delay. UConnRCMPy relies on several packages from
    the SciPy stack and the broader scientific Python community. UConnRCMPy implements an extensible
    framework, so that alternative experimental data formats can be incorporated easily. In this
    way, UConnRCMPy improves the consistency of RCM data processing and enables reproducible
    analysis of the data.

.. class:: keywords

    rapid compression machine, engineering, kinetic models

Introduction
------------

The world relies heavily on combustion to provide energy in useful forms for human consumption; in
particular, the transportation sector accounts for nearly 30% of the energy use in the United States
and of that, more than 90% is supplied by combustion of fossil fuels :cite:`MER2016`. Unfortunately,
emissions from the combustion of traditional fossil fuels have been implicated in a host of
deleterious effects on human health and the environment :cite:`Avakian2002` and fluctuations in the
price of fossil fuels can have a negative impact on the economy :cite:`Owen2010`.

Two methods are being considered to reduce the impact of fossil fuel combustion in transportation on
the environment and the economy, namely: 1) development of new fuel sources and 2) development of
new engine technologies.

Unfortunately, it is not straightforward to combine new fuels with newly designed engines. For
instance, the design process of a new engine may become circular: the "best" alternative fuel should
be tested in the "best" engine, but the "best" engine depends on which is selected as the "best"
alternative fuel. One way to short circuit this process is by employing computer-aided design and
modeling of new engines with new fuels to design engines to be able to utilize several fuels. The
key to this process is the development of accurate and predictive combustion models.

These models of combustion are typically descriptions of the chemical kinetic pathways the fuel and
oxidizer undergo as they break down into carbon dioxide and water. There may be as many as several
tens of thousands of pathways in the model for combustion of a particular fuel, with each pathway
requiring several parameters to describe its rate. Therefore, it is important to thoroughly validate
the operation of the model by comparison to experimental data collected over a wide range of
conditions.

One type of data that is particularly relevant for transportation applications is the ignition
delay. The ignition delay is a global combustion property depending on the interaction of many of
the pathways present in the model. There are several methods to measure the ignition delay at
engine-relevant conditions, including shock tubes and rapid compression machines (RCMs).

An RCM is typically designed with one or two pistons that rapidly compress a homogeneous fuel and
oxidizer mixture inside a reaction chamber. After the end of compression (EOC), the piston(s) is
(are) locked in place, creating a constant volume reaction chamber. The primary diagnostic in most
RCM experiments is the pressure measured as a function of time in the reaction chamber. This
pressure trace is then processed to extract the ignition delay.

In this work, the design and operation of a software package to process the pressure data collected
from RCMs is described. This package, called UConnRCMPy :cite:`uconnrcmpy`, is designed to enable
reproducible analysis of the data acquired from the RCM at the University of Connecticut. Despite
the initial focus on data from the UConn RCM, the package is designed to be extensible so that it
can be used for data in different formats while providing a consistent interface to the user.

First, this paper will introduce the fundamentals of RCM operation and data processing. Then, the
implementation of UConnRCMPy will be described, including the use of many packages from the
scientific Python software ecosystem. Finally, a brief demonstration of the use of UConnRCMPy and
its extensibility will be given.

Background
----------

The RCMs at the University of Connecticut have been described extensively elsewhere
:cite:`Das2012,Mittal2007a`, and will be summarized here for reference. The RCMs use a single piston
that is pneumatically accelerated and hydraulically decelerated. In a typical experiment, the
reaction chamber is first evacuated to an absolute pressure near 1 Torr, measured by a high-accuracy
static pressure transducer. Next, the reactants are filled in to the desired initial pressure
(|P0|), and a low dead volume valve on the reaction chamber is closed. Compression is triggered by a
digital control circuit. After compression, the piston is held in place to create a constant volume
chamber in which reactions proceed. For appropriate combinations of pressure, temperature, and
mixture composition, ignition will occur after some delay period. A single
compression-delay-ignition sequence is referred to as an experiment or a run. Each experiment is
repeated approximately 5 times at the same nominal initial conditions to ensure repeatability of the
data, and this set of experiments is referred to in the following as a condition.

During and after the compression, the pressure is monitored using a dynamic pressure transducer.
The pressure trace is processed to determine the quantities of interest, including
the pressure and temperature at the EOC, |PC| and |TC| respectively, and the ignition delay. The
ignition delay (|tau|) is typically measured at several values of |TC| for a given value of |PC| and
mixture composition; this is referred to in the following as a data set.

RCM Signal Processing Procedure
-------------------------------
Signal measurement
==================

As mentioned previously, the primary diagnostic on the RCM is the reaction chamber pressure,
measured by a dynamic pressure transducer (separate from the static transducer used to measure
|P0|). The dynamic transducer outputs a charge signal that is converted to a voltage signal by a
charge amplifier. This system measures changes in pressure in the reaction chamber (as opposed to
the absolute pressure measured by the static transducer) and as such, has a nominal output of 0 V
prior to the start of compression. In addition, the output range of 0 V to 10 V is set by the
operator to correspond to a particular pressure range by setting a "scale factor". Typical values
for the scale factor range between 10 bar/V and 100 bar/V.

The voltage output from the charge amplifier is digitized by a hardware DAQ and recorded into a
plain text file by a LabView Virtual Instrument. The voltage is sampled at rate chosen by the
operator, typically between 50 kHz and 100 kHz. This provides sufficient resolution for events on
the order of milliseconds; the typical ignition delay measured in this system approximately ranges
from 5 ms to 100 ms.

.. figure:: figure1.png

    Raw voltage trace and the voltage trace after filtering and smoothing from a typical RCM
    experiment. :label:`raw-voltage`

Figure :ref:`raw-voltage` shows a typical voltage trace measured from the RCM at UConn. Several
features are apparent from this figure. First, the compression stroke takes approximately 30 ms to
40 ms, with the EOC used to set the reference time of :math:`t = 0` (the determination of the time
of the EOC will be discussed in due course). Approximately 50% of the pressure rise occurs in the
last 5 ms of compression. Second, there is a slow pressure decrease after the EOC due to heat
transfer from the reactants to the relatively colder chamber walls. Third, after some delay period
there is a spike in the pressure corresponding to rapid heat release due to combustion. Finally, the
signal is somewhat noisy, and the measured initial voltage may offset from the nominal 0 V by a few
millivolts.

Filtering and Smoothing
=======================

To produce a useful pressure trace, the voltage signal must be filtered and/or smoothed. Several
algorithms have been considered to smooth the voltage trace, including a simple moving average, a
low-pass filter, and some combination of these two methods. In the current version of UConnRCMPy
:cite:`uconnrcmpy`, the voltage is first filtered using a low-pass filter with a cutoff frequency of
10 kHz. The filter is constructed using the ``firwin`` function from the ``signals`` module of SciPy
:cite:`Jones2001` with the Blackman window :cite:`Blackman1958,Oppenheim1999` and a filter order of
:math:`2^{14}-1`. The cutoff frequency, window type, and filter order were determined empirically.
Methods to select a cutoff frequency that optimizes the signal-to-noise ratio are currently being
investigated.

After filtering, the signal is smoothed by a moving average filter with a width of 21 points. It is
desired that the signal remain the same length through this operation, but the convolution operation
used to apply the moving average zero-pads the first and last 10 points. To avoid a bias in the
initial voltage, the first 10 points are set equal to the value of the 11th point; the final 10
points are not important in the rest of the analysis and are ignored. The result of the filtering
and smoothing operations is shown on Fig. :ref:`raw-voltage`.

Offset Correction and Pressure Calculation
==========================================

In general, the voltage trace can be converted to a pressure trace by

.. math::
    :label: pressure-trace

    P(t) = \overline{V}(t) + F \cdot P_0

where :math:`\overline{V}(t)` is the filtered and smoothed voltage trace and :math:`F` is the scale
factor from the charge amplifier. However, as can be seen in Fig. :ref:`raw-voltage` there is a
small offset in the initial voltage relative to the nominal value of 0 V. To correct for this
offset, it can be subtracted from the voltage trace

.. math::
    :label: corrected-pressure-trace

    P(t) = \left[\overline{V}(t) - \overline{V}(0)\right] + F \cdot P_0

where :math:`\overline{V}(0)` is the initial voltage. The result is a vector of pressure values that
must be further processed to determine the time of the EOC and the ignition delay.

Finding the EOC
===============

There are several methods to determine the EOC of a particular experiment. Since the piston is held
in place at the end of its stroke, the pressure will be a maximum (in the absence of ignition) at
the EOC. Therefore, the EOC can be found either by searching for this maximum value or by
calculating the first derivative of the pressure with respect to time and finding the zero crossing.
As the signal is noisy, even after smoothing, the derivative will tend to increase the noise in the
signal :cite:`Chapra2010` leading to difficulty in specifying the correct zero crossing. On the
other hand, finding the maximum of the pressure in the time prior to ignition is not straightforward
either. In general, the pressure after ignition has occured will be higher than the pressure at the
EOC and the width of the ignition peak is unknown. However, we can take advantage of the fact
that there is some pressure drop after the EOC to eliminate the ignition from consideration.

In the current version of UConnRCMPy :cite:`uconnrcmpy`, this is done by searching backwards in time
from the maximum pressure in the pressure trace (typically, the global maximum pressure is after
ignition has occured) until a minimum in the pressure is reached. Since the precise time of the
minimum is not important for this method, the search is done by comparing the pressure at a given
index :math:`i` to the pressure at point :math:`i-50`, starting with the index of the global maximum
pressure. This offset is used to avoid the influence of noise. If :math:`P(i) \geq P(i-50)`, the
index is decremented and the process is repeated until :math:`P(i) < P(i-50)`. This value of
:math:`i` is approximately the minimum of pressure prior to ignition, so the maximum of the pressure
in points to the left of the minimum will be the EOC.

This method is generally robust, but it fails when there is no minimum in the pressure between the
EOC and ignition, or the minimum pressure is very close to the EOC pressure. This may be the case
for short ignition delays, on the order of 5 ms or less. In these cases, the comparison offset can
be reduced to improve the granularity of the search; if that method fails, manual intervention is
necessary to determine the EOC. In either case, the value of the pressure at the EOC, |PC|, is
recorded and the time at the EOC is taken to be :math:`t=0`.

Calculating Ignition Delay
==========================

The ignition delay is determined as the time difference between the EOC and the point of ignition.
There are several definitions of the point of ignition; the most commonly used in RCM experiments is
the inflection point in the pressure trace due to ignition. As before, finding zero crossings of the
second time derivative of the pressure to define the inflection point is difficult due to noise;
however, finding the maximum of the first derivative is trivial, particularly since the time before
and shortly after the EOC can be excluded to avoid the peak in the derivative around the EOC.

In the current version of UConnRCMPy :cite:`uconnrcmpy`, the first derivative of the experimental
pressure trace is computed by a second-order forward differencing method. The derivative is then
smoothed by the moving average algorithm with a width of 151 points. This value for the moving
average window was chosen empirically.

For some conditions, the reactants may undergo two distinct stages of ignition. These cases can be
distinguished by a pair of peaks in the first time derivative of the pressure. For some two-stage
ignition cases, the pressure rise (and consequently the peak in the derivative) are relatively weak,
making it hard to distinguish the peak due to ignition from the background noise. This is currently
the the area requiring the most manual intervention, and one area where significant improvements
can be made by improving the differentiation and filtering/smoothing algorithms. An experiment that
shows two clear peaks in the derivative is shown in Fig. :ref:`ign-delay-def` to demonstrate the
definition of the ignition delays.

.. figure:: figure1.png

    Illustration of the definition of the ignition delay in a two-stage ignition case.
    :label:`ign-delay-def`

Calculating the EOC Temperature
===============================

The final parameter of interest presently is the EOC temperature, |TC|. This temperature is often
used as the reference temperature when reporting ignition delays. In general, it is difficult to
measure the temperature as a function of time in the reaction chamber of the RCM, so methods to
estimate the temperature from the pressure trace are generally used.

The law of conservation of energy written for the ideal gases in the reaction chamber is:

.. math::
    :label: first-law

    c_v \frac{dT}{dt} = -P \frac{dv}{dt} - \sum_k u_k \frac{d Y_k}{dt}

where :math:`c_v` is the specific heat at constant volume of the mixture, :math:`v` is the specific
volume, :math:`u_k` and :math:`Y_k` are the specific internal energy and mass fraction of the
species :math:`k`, and :math:`t` is time. For a constant-area piston, the rate of change of the
volume is equal to the piston velocity. In UConnRCMPy, Eq. :ref:`first-law` is integrated by the
software Cantera :cite:`cantera`.

In Cantera, intensive thermodynamic information about the system is stored in an instance of the
``Solution`` class. The ``Solution`` classes used in this study model simple, compressible systems
and require two independent intensive properties, plus the composition, to fix the state. In
addition to evaluating thermodynamic data, Cantera :cite:`cantera` contains several objects used to
model homogeneous reacting systems; the two used in this paper are a ``Reservoir`` and an
``IdealGasReactor``, which are subclasses of the generic ``Reactor`` class. The specific
``IdealGasReactor`` class is preferred over the generic ``Reactor`` class in this study because the
energy equation is directly solved in terms of the temperature (i.e., Eq. :ref:`first-law`) in an
``IdealGasReactor``. A ``Solution`` object is installed in each ``Reactor`` subclass to manage the
state information and evaluate thermodynamic properties. The difference between the ``Reservoir``
and the ``IdealGasReactor`` is simply that the state (i.e., the pressure, temperature, and chemical
composition) of the ``Solution`` in a ``Reservoir`` is fixed.

Integrating Eq. :ref:`first-law` requires knowledge of the volume of the reaction chamber as a
function of time. To calculate the volume as a function of time, it is assumed that there is a core
of gas in the reaction chamber that undergoes an isentropic compression and the effects of the
boundary layer can be ignored :cite:`Lee1998`. Furthermore, it is assumed that there is negligible
reactant consumption during the compression stroke. Then, a Cantera ``Solution`` object is
initialized at the initial temperature, pressure, and composition of the reaction chamber.

After initialization the initial mass-specific entropy (|s0|) and density (|rho0|) are recorded. The
initial volume is arbitrarily taken to be :math:`V_0=1.0\,\text{m}^3`. The initial volume used in
constructing the volume trace is arbitrary provided that the same value for the volume is used for
the initial volume in the ``Reactor``-based simulations described below. However, extensive
quantities such as the total heat release during ignition cannot be compared to experimental values.

The measured pressure at each point in the pressure trace (:math:`P_i`) is used with the previously
recorded initial entropy (|s0|) to set the state of the ``Solution`` object sequentially. At each
point, the volume is computed by applying the ideal gas law:

.. math::
    :label: ideal-gas-law

    V_i = V_0 \frac{\rho_0}{\rho_i}

where :math:`\rho_i` is the density at each point computed by the Cantera ``Solution``. This
procedure effects a constant composition isentropic compression process.

Once the volume trace has been generated, it can be utilized in the ``IdealGasReactor`` and the
solution of Eq. :ref:`first-law` by installing an instance of the ``Wall`` class. In Cantera
:cite:`cantera`, ``Wall``\ s have several uses, including allowing heat transfer into or out of the
``Reactor``, allowing heterogeneous reactions on the surface of the ``Wall``, or causing the volume
of the ``Reactor`` to vary. In this study, only the last function is used (i.e., the reaction
chamber is adiabatic and homogeneous). ``Wall``\ s must be installed between instances of
``Reactor``\ s, so in UConnRCMPy a ``Wall`` is installed between the ``IdealGasReactor`` that
represents the reaction chamber and an instance of the ``Reservoir`` class. By specifying the
velocity of the ``Wall`` using the volume trace computed previously, the ``IdealGasReactor`` will
proceed through the same states as the reaction chamber in the experiment.

The velocity of the ``Wall`` is specified by using an instance of the ``VolumeProfile`` class from
the CanSen software :cite:`cansen`. Then, the ``IdealGasReactor`` is installed into an instance of
``ReactorNet`` from Cantera :cite:`cantera`. The ``ReactorNet`` implements the interface to the
solver CVODES. CVODES is an adaptive-time-stepping solver, distributed as part of the SUNDIALS suite
:cite:`Hindmarsh2005`. As the solver steps towards the end time of the simulation, the state of the
system is stored on each integrator time step, producing simulated pressure, volume, and temperature
traces. Finally, the EOC temperature is recorded as the simulated temperature at the EOC.

Two simulations are conducted using this procedure. In the first, the multiplier for all the
reaction rates is set to zero, to simulate a constant composition (non-reactive) process. In the
second, the reactions are allowed to proceed as normal. The temperature traces from the two
simulations are compared to each other and the inclusion of the reactions does not change |TC|,
validating the assumption of adiabatic, constant composition compression. Nonetheless, when
conducting simulations to compare a kinetic model to experimental results, it is important to
include the species equations in the solution of Eq. :ref:`first-law` due to the buildup of a pool
of radicals that affects the processes after the EOC :cite:`Mittal2008`, although it does not affect
the computation of |TC|.

Acknowledgements
----------------

This material is based on work supported by the National Science Foundation under Grant No.
CBET-1402231.

.. |TC| replace:: :math:`T_C`
.. |PC| replace:: :math:`P_C`
.. |O2| replace:: O\ :sub:`2`
.. |N2| replace:: N\ :sub:`2`
.. |P0| replace:: :math:`P_0`
.. |T0| replace:: :math:`T_0`
.. |gamma| replace:: :math:`\gamma`
.. |tau| replace:: :math:`\tau`
.. |s0| replace:: :math:`s_0`
.. |rho0| replace:: :math:`\rho_0`
