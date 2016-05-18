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
alternative fuel. One way to shortcircuit this process is by employing computer-aided design and
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
can be used for data in different formats while providing a consitent interface to the user.

First, this paper will introduce the fundamentals of RCM operation and data processing. Then, the
implementation of UConnRCMPy will be described, including the use of many packages from the
scientific Python software ecosystem. Finally, a brief demonstration of the use of UConnRCMPy and
its extensibility will be given.

.. First, using new sources of fuel for transportation can reduce the economic
.. impact of swings in the price of current fuels and potentially reduce emissions. Second, using new
.. engine technologies can simultaneously reduce emissions and increase fuel efficiency.
..
.. Many new sources of fuels have been investigated recently. The most promising of these in the long
.. term are renewable biological sources, which are used to produce fuels known as biofuels. In the
.. ideal case, biofuels could be used as drop-in replacements for traditional fuels, requiring few
.. changes in engine design. However, the combustion properties of biofuels may be substantially
.. different from the fuels they are intended to replace, and thus may require extensive modifications
.. to engine designs.
..
.. In addition to new fuel sources, new advanced engines, known as Low-Temperature Combustion (LTC)
.. engines, are being developed. These engines operate at conditions that optimize efficiency avoid the
.. generation of emissions. However, the combustion timing in these devices is largely controlled by
.. the chemical kinetics of the ignition of the fuel being used.
..
.. Neither of these approaches---new fuels and new engine technologies---is able to mitigate all of the
.. negative impacts of combustion by itself. By switching to biofuels but retaining the same engines,
.. the efficiency and emissions targets may not be met; by only developing new engines, our sources of
.. fuel will continue to cause economic distress, turmoil, and negative effects on the environment. It
.. will take a concerted effort to bring these two pathways of innovation together.

Background
==========
