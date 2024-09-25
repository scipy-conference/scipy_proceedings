## Introduction

<!-- Here we can talk about the ATLAS experiment, the detector, its purpose, the amount of data collected, etc. -->

The field of high energy physics (HEP) is devoted to the study of the fundamental forces of Nature and their interactions with matter.
To study the structure of the Universe on the smallest scales requires the highest energy density environments possible &mdash; similar to those of the early Universe.
These extreme energy density environments are created at the CERN international laboratory, in Geneva, Switzerland, using the Large Hadron Collider (LHC) to collide "bunches" of billions of protons at a center-of-mass energy of $\sqrt{s} = 13~\mathrm{TeV}$.
The resulting collisions are recorded with building-sized particle detectors positioned around the LHC's $27~\mathrm{km}$ ring that are designed to measure subatomic particle properties.
Given the rarity of the subatomic phenomena of interest, the rate of the beam crossings is a tremendous $40~\mathrm{MHz}$ to maximize the number of high quality collisions that can be captured and read out by the detectors.
Even with real-time onboard processing ("triggering") of the experiment detector readout to save only the most interesting collisions, detectors like the ATLAS experiment [@PERF-2007-01] still produce multiple petabytes of data per year.
These data are then further filtered through selection criteria on the topology and kinematic quantities of the particle collision "events" recorded into specialized datasets for different kinds of physics analysis.
The final datasets that physicists use in their physics analyses in ATLAS is still on the order of hundreds of terabytes, which poses challenges of compute scale and analyst time to efficiently use while maximizing physics value.

Traditionally, the ATLAS and the other LHC experiment have created experiment-specific custom `C++` frameworks to handle all stages of the data processing pipeline, from the initial construction of high-level physics objects from the raw data to the final statistical analyses.
Motivated by the broad success of the Scientific Python ecosystem across many domains of science, and the rise of the Scikit-HEP ecosystem of Pythonic tooling for particle physics [@Rodrigues:2020syo;@henry_schreiner-proc-scipy-2022] and community tools produced by the Institute for Research and Innovation in Software for High Energy Physics (IRIS-HEP) [@S2I2HEPSP;@CWPDOC], there has been a broad community-driven shift in HEP towards use of the Scientific Python ecosystem for analysis of physics data &mdash; a PyHEP ecosystem [@PyHEP].
The use of dataframes and array programming for data analysis has enhanced the user experience while providing efficient computations without the need of coding optimized low-level routines.
The ATLAS collaboration is further extending this ecosystem of tooling to include high-level custom Python bindings to the low level `C++` frameworks using `nanobind` [@nanobind].
Collectively, these tools are modernizing the methods which researchers are engaging data analysis at large scale and providing a novel end-to-end analysis ecosystem for the ATLAS collaboration.
