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
    measurement, UConnRCMPy computes the thermodynamic conditions in the reactor of the RCM during
    an experiment along with the ignition delay. UConnRCMPy relies on several packages from the
    SciPy stack and the broader scientific Python community. UConnRCMPy implements an extensible
    framework, so that alternative experimental data formats can be incorporated easily. In this
    way, UConnRCMPy improves the consistency of RCM data processing and enables reproducible
    analysis of the data.

.. class:: keywords

    rapid compression machine, engineering, kinetic models

Introduction
============

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
oxidizer mixture inside a reactor. After the end of compression (EOC), the piston(s) is (are) locked
in place, creating a constant volume reactor. The primary diagnostic in most RCM experiments is the
pressure measured as a function of time in the reaction chamber. This pressure trace is then
processed to extract the ignition delay.

In this work, the design and operation of a software package to process the pressure data collected
from RCMs is described. This package, called UConnRCMPy :cite:`Weber2016`, is designed to enable
reproducible analysis of the data acquired from the RCM at the University of Connecticut. Despite
the initial focus on data from the UConn RCM, the package is designed to be extensible so that it
can be used for data in different formats while providing a consistent interface to the user.

First, this paper will introduce the fundamentals of RCM operation and data processing. Then, the
implementation of UConnRCMPy will be described, including the use of many packages from the
scientific Python software ecosystem. Finally, a brief demonstration of the use of UConnRCMPy and
its extensibility will be given.

Background
==========

The RCMs at the University of Connecticut have been described extensively elsewhere
:cite:`Das2012,Mittal2007a`, and will be summarized here for reference. The RCMs use a single piston
that is pneumatically accelerated and hydraulically decelerated. The total time for compression is
near 30 ms. The EOC conditions in the reaction chamber are typically the ones of interest, and are
represented by |PC| and |TC| for the EOC pressure and temperature respectively. |PC| and |TC| can be
varied independently by varying the geometric compression ratio, initial pressure, and initial
temperature. The piston in the reaction chamber is machined with crevices that contain the roll-up
vortex that would be created by the piston motion and promote homogeneous conditions in the reactor
after the EOC :cite:`Mittal2006`.

As mentioned previously, the primary diagnostic on the RCM is the reaction chamber pressure,
measured on the UConn RCMs by a Kistler 6125C dynamic transducer coupled with a Kistler 5010B charge
amplifier. The voltage output from the charge amplifier is digitized by a National Instruments DAQ
(two are used, depending on the RCM), and recorded into a plain text file by a LabView Virtual
Instrument. The voltage is sampled from the DAQ at rate chosen by the machine operator, typically
between 50 kHz and 100 kHz.

The compression stroke of the RCM brings the homogeneous fuel/oxidizer mixture to the EOC
conditions, and for suitable values of |TC| and |PC|, the mixture will ignite. For
some conditions, the mixture undergoes two stages of ignition. In general, the ignition delay is
defined as the time from the EOC until a peak in the time derivative of the pressure occurs; for two
stage ignition, two peaks will occur, while for a single stage only a single peak is present.

In addition to reactive experiments, non-reactive experiments are carried out to determine the
influence of machine specific operating parameters on the experiment. In these experiments, |O2| in
the oxidizer is replaced with |N2| to maintain a similar specific heat ratio but suppress the
oxidation reactions that lead to ignition. If the pressure at the EOC of the non-reactive
experiments matches that at the EOC of the reactive experiments, it is assumed that no substantial
heat release has occurred during the compression stroke, and the temperature at the EOC can be
estimated by applying the adiabatic core hypothesis :cite:`Lee1998` and the isentropic relations
between pressure and temperature during the compression stroke:

.. math::

    \ln{\left(\frac{P_C}{P_0}\right)} = \int_{T_0}^{T_C} \frac{\gamma}{\gamma - 1}\frac{dT}{T}

where |P0| is the initial pressure, |T0| is the initial temperature, and |gamma| is the
temperature-dependent ratio of specific heats. Since |gamma| is temperature-dependent, the value
reached for |TC| for a given |P0|, |T0|, |PC| set depends on the path taken during the compression.
Under the adiabatic core hypothesis, it is assumed that the core gases in the reaction chamber (away
from the boundary layer near the wall) undergo an adiabatic compression process, so that the
equation can be integrated to give |TC|. In reality, the gases in the reaction chamber do not
undergo an adiabatic process; nonetheless, experimental measurements of the temperature during and
after compression have shown that the adiabatic core hypothesis is adequate to determine the
temperature evolution of the reactants :cite:`Das2012a,Uddi2011`.

.. figure:: figure1.png

    Graph of a typical pressure trace from an RCM experiment showing the definition of the ignition
    delay. :label:`ign-delay-def`

Figure :ref:`ign-delay-def` shows the definition of the ignition delay typically used in RCM studies
along with a typical pressure trace acquired from the experiment. The time of EOC is chosen as the
maximum of the pressure prior to ignition, and this is arbitrarily chosen to be :math:`t = 0`.

Acknowledgements
================

This material is based on work supported by the National Science Foundation under Grant No.
CBET-1402231.

.. |TC| replace:: :math:`T_C`
.. |PC| replace:: :math:`P_C`
.. |O2| replace:: O\ :sub:`2`
.. |N2| replace:: N\ :sub:`2`
.. |P0| replace:: :math:`P_0`
.. |T0| replace:: :math:`T_0`
.. |gamma| replace:: :math:`\gamma`
