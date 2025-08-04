---
# Ensure that this title is the same as the one in `myst.yml`
title: Empowering Learners - Teaching Reproducible Research with Open-Source Tools
abstract: |
  Reproducibility and open science are increasingly recognized as essential components of rigorous geoscientific research. However, gaps in training often hinder early-career scientists from fully adopting best practices in computational workflows and FAIR (Findable, Accessible, Interoperable, Reusable) publishing. The Facilitating Reproducible Open Geoscience (FROGS) initiative addresses this need by offering an ongoing series of integrated courses and self-paced online modules designed to build capacity in Python and R programming, time series analysis, and open science publishing within the geosciences. To date, three sequential courses have been completed, and this paper reports on these initial offerings. During the courses, participants engaged in hands-on learning activities combining synchronous instruction with asynchronous exercises, supported by the LeapFROGS platform. Survey results demonstrate high participant satisfaction, increased confidence in applying reproducible research methods, and active contributions to open source geoscience projects. This paper details the course design, curriculum content, participant outcomes, and the broader impact of FROGS in fostering a sustainable community of practice dedicated to open and reproducible geoscience. Our findings underscore the critical role of integrated training programs in advancing open science and highlight strategies for scaling such initiatives across scientific domains.
acknowledgements: | 
  Deborah Khider and Julien Emile-Geay are supported by the U.S. National Science Foundation (NSF) under award number RISE#2324732. David Edge and Nicholas McKay are supported by the U.S. NSF under award number RISE#2324733.
---

## Introduction and Motivation

Sharing research data, software, and workflows enhances reproducibility, collaboration, and the directions of future research [@doi:10.1002/2015EA000136] and is fundamental to building a FAIR open science ecosystem. The desire to share and reuse scientific data and software has grown in recent years as funders [e.g. @doi:10.1108/TG-03-2014-0008; @vandereynden2016; @Zuiderwijk2014] and publishers [e.g. @AGUSoftwareGuidelines; @AGUWorkflowGuidelines; @doi:10.1038/s41592-019-0350-x] have introduced open science policies emphasizing reproducibility, and as scientists increasingly recognize the benefits of open science [@doi:10.1038/s41559-017-0160; @doi:10.7554/elife.16800]. Consequently, the last decade has seen a proliferation of frameworks to promote open source resources, collaboration among scientists, and sharing of scientific research products. One critical element to the success and long-term sustainability of these resources is the training of scientists, especially in their early-career, in their use. This training needs to take place across scientific practice and publishing. 

Facilitating Reproducible Open Geoscience (FROGS) is a U.S. National Science Foundation (NSF) funded initiative designed to speed up the integration of open source tools into the research and publication workflows of geoscientists. FROGS has offered three seven-week long courses on geoscience analysis in Python and R and FAIR publishing principles. The courses emphasized science publishing as an integral part of a research product. Each course included a synchronous workshop that introduced overarching concepts, followed by asynchronous, detailed online materials combined with biweekly office hours designed to help participants integrate the resources into their practice. The asynchronous component was facilitated by the [LeapFROGS](https://linked.earth/LeapFROGS/) learning platform [@LeapFROGS], which provided summaries of the key concepts, curated additional resources from the open science community, and offered challenges for learners to assess their comprehension.

This paper provides an overview of the training activities in [](#method-section), detailing the goals and design of the LeapFROGS platform, before discussing educational outcomes and contributions to open science in [](#result-section). 

(method-section)=
## Overview of training activities

### LeapFROGS

The LeapFROGS platform combines lecture content with self-graded exercises to deliver self-paced modules covering diverse aspects of (geo)scientific research. Built using Gatsby and supported by a myBinder backend [@doi:10.25080/Majora-4af1f417-011], it offers a Python sandbox environment for hands-on learning, bypassing environment setup difficulties that are often a limiting step for first-timers. The platform features seven modules that explore different facets of scientific practice and publishing. These modules leverage open source tools from the scientific Python community and are specifically tailored to geoscience applications. The platform’s primary aim is to serve as a gateway into the extensive Python educational ecosystem. 

The first module, *Introduction to Python*, covers fundamental concepts including numbers, variables, logic, strings, lists, tuples, sets, dictionaries, conditionals, loops, functions, and classes. The lessons reference the Trinket Python course [@trinket], while additional coding exercises on LeapFROGS offer geoscience-relevant challenges to reinforce learning. The coding exercises are designed as fill-in-the-blank tasks (@fig:python). A key advantage of the platform is its ability to provide hints and reveal solutions to learners who encounter difficulties.

:::{figure} FROGS - Python.png
:label: fig:python
Example of training model on the LeapFROGS platform. (a) An exercise on numbers in Python with links to the Trinket Python course, and (b) an tutorial on Pandas using materials developed by Project Pythia. LeapFROGS tests understanding of these concepts by asking to fill in the blank in a code cell than can be executed on myBinder so learners can run their code to check against the solution. Learners can also ask for helpful hints or see the solution. 
:::

The second module, *The Scientific Python Stack*, introduces libraries common to scientific Python such as Jupyter [@jupyter], NumPy [@numpy], pandas [@pandas1; @pandas2], Matplotlib [@matplotlib], Cartopy [@Cartopy], Seaborn [@Seaborn], statsmodels [@statsmodels], scikit-learn [@sklearn1; @sklearn2], PyTorch [@pytorch], and Xarray [@xarray]. This module adopts the same approach as the *Introduction to Python* module, featuring lessons from Project Pythia Foundations [@PythiaFoundations], alongside tutorials from library developers and coding exercises tailored to reinforce learning within a geoscience context (@fig:python). 

The third module, *Timeseries Analysis*, introduces key concepts and practicums about timeseries analysis, a common task in geoscience studies. The lecture materials were adapted from Emile-Geay's class at the University of Southern California (USC) on data analysis in the Earth and Environmental Sciences and its accompanying e-book [@JEG_dataAnalysis]. The module covers common data processing techniques, measures of association between timeseries, the use of surrogates for significance testing, and spectral and wavelet analyses. The module is supported by a [Jupyter Book](http://linked.earth/PyRATES_practicums_py/intro.html) [@Emile-Geay_Time_Series_Analysis_2025] that makes use of the Pyleoclim software package [@doi:10.1029/2022PA004509].

The fourth module, *The Scientific Paper of The Future*, introduces concepts in FAIR scientific publishing. As its name indicates, this module presents an updated version [@KhiderGPF] of the Geoscience Paper of the Future (GPF) [@doi:10.1002/2015EA000136]. The GPF represented one of the earliest community efforts to fully document, share, and cite all research products, including data, software, and computational provenance (i.e., describing the workflow used in a study, which links together data and software with their associated parameter values for reproducibility). Although the GPF was published before the FAIR principles [@doi:10.1038/sdata.2016.18; @doi:10.3233/DS-190026; @doi:10.1162/dint_a_00033], the guidelines outlined in this paper allow for compliance with FAIR. The updated materials reflect this compliance and provide additional guidance on the technologies such as Binder [@binder] that support FAIR, open and reproducible science. Learners on the platform are able to test their understanding of these concepts using a multiple choice quiz (@fig:quiz).  

The fifth module, *Using GitHub for Your Research*, focuses on GitHub as a platform to collaborate, share scientific workflow and software, and support project management. The module covers the basics of repositories, forks, pull requests, understanding and setting up actions for automation, and linking a GitHub repository to Zenodo to obtain a persistent identifier. This module provides a high-level overview of GitHub Actions, including their purpose and the basic structure of a workflow file, while more detailed applications—such as containerization and continuous integration—are addressed in the final two modules. Lessons from this chapter were summarized from various sources including the GitHub documentation, blogs from Medium and other sources, and Project Pythia Foundations [@PythiaFoundations].

:::{figure} FROGS - Publishing.png
:label: fig:quiz
:width: 500px
Example of training model on the LeapFROGS platform. Learners can test their understanding using a self-paced quiz that provides feedback in case of an incorrect answer. 
:::

The sixth module, *Sharing Reproducible Workflows*, provides information about using [Docker](https://www.docker.com) and [myBinder](https://mybinder.org) to share reproducible science results. 

Finally, the last module, *Packaging Your Software for Sharing* walks through instructions and tutorials on creating Python packages, publishing it on software registries (e.g., The [Python Package Index](https://pypi.org), PyPI), on creating documentation using Sphinx [@brandl2021sphinx] and publishing it on [readthedocs.org](https://readthedocs.org), on unit testing, on setting up GitHub actions for continuous integration, and on publishing on PyPI. Most of these tutorials were based on the materials developed by the [PyOpenSci](https://www.pyopensci.org) community. 

### Training activities

We facilitated three complementary training workshops designed to build geoscientists’ skills in computational research, open science publishing, and software development. Participants in each workshop were selected to maximize diversity across geoscientific disciplines and backgrounds. A breakdown of participants by career stage and geoscientific discipline is given in @fig:participants.

:::{figure} participants.png
:label: fig:participants
Distribution of participants by geoscience disciplines and career stage across all three training workshops. 
:::

#### PyRATES: Python and R Analysis of Time Series

Held in-person in June 2024 at the USC Information Sciences Institute (ISI), PyRATES offered foundational training in Python and R tailored to geoscientists, with an emphasis on time series analysis — a core expertise of the principal investigators that spans multiple geoscience domains. This workshop targeted early-career researchers with little prior programming experience and aimed to introduce the basics of scientific Python and R, statistical concepts in time series analysis, and FAIR science publishing principles. The workshop gathered 18 participants and covered modules 1 through 4 on the LeapFROGS platform. At the end of the asynchronous period, participants were asked to submit a reproducible notebook applying FAIR principles.

#### FAIRLeap: FAIR Publishing in the Geosciences

FAIRLeap was designed for researchers actively engaged in geoscience projects, including those preparing manuscripts or reproducibility studies. The workshop introduced FAIR publishing concepts, GitHub for project and software management, and tools such as Docker, Binder, and myBinder for sharing reproducible workflows (modules 4-6 on the LeapFROGS platform). Held virtually in February 2025, FAIRLeap gathered 21 participants, attending mostly asynchronously. 

#### Open Geoscience Hackathon

The Open Geoscience Hackathon targeted researchers interested in open-source software development and contributions. The workshop focused on practical skills including opening pull requests, software packaging, unit testing, and continuous integration (module 7 on LeapFROGS). Held in person at USC ISI in May 2025, the hackathon gathered 17 participants. Morning sessions were reserved for high-level lectures and guided homework on the PyCatSim [@PyCatSim] toy package. This Python package, which simulates cat ownership, was specifically designed for this hackathon. The functions were straightforward to implement and served to illustrate object-oriented programming principles, methods for integrating data into packages, and error handling, all while demonstrating the process of contributing to an open-source project. In the afternoons, participants worked on their own package. By the end of the workshop, participants were able to share the foundation of their package on GitHub.

(result-section)=
## Contributions to open science

FROGS has contributed to open science in two main ways. First, it has trained the next generation of scientists in Python programming, data science libraries, time series analysis, geoscience data science, and open science sharing and publishing practices. In turn, these scientists have produced exemplars of reproducible studies within their respective communities. 

### Training the next generation of scientists

Exit surveys conducted following the three workshops demonstrated a highly favorable reception among participants. Specifically, 95% of respondents indicated they would recommend the workshops to colleagues, 90% reported that the activities enhanced their understanding of key concepts in geoscience practice and publishing, and 94% expressed increased confidence in applying these principles within their own research endeavors. Participants consistently highlighted the effectiveness of the workshops’ hands-on, project-oriented approach, emphasizing the value of integrating lectures with practical coding exercises, such as contributing to exemplar packages like PyCatSim. Instructional modules on automation, software testing, documentation, and version control via GitHub were particularly well received.

Nevertheless, participant feedback also identified areas for refinement. A notable challenge was accommodating the wide range of prior experience among attendees; some less-experienced participants encountered difficulties with advanced topics, including object-oriented programming, underscoring the need for preparatory materials or more structured introductory sessions. Suggestions were also made to extend the duration of workshops or incorporate additional applied exercises and case studies to foster deeper engagement. Furthermore, enhanced guidance on software environment setup and package publication workflows was recommended to mitigate technical barriers.

Taken together, these findings indicate that the workshops effectively impart practical skills and foster confidence in open science methodologies, while providing actionable insights to improve accessibility and pedagogical design for diverse participant cohorts.

### Contributed workflows and software

In addition to providing comprehensive training, a significant outcome of the workshops was the active contribution of participants to open science projects. These contributions not only reinforced the practical skills acquired during the sessions but also fostered engagement within the open source geoscience community. The repositories developed or enhanced by participants span a diverse range of geoscience domains and are publicly available on GitHub. @tbl:contributions summarizes these contributions, listing the associated scientific domains alongside the corresponding GitHub repositories.

```{list-table} Open science projects contributed by participants, with repository names shown as user/repository_name on GitHub.
:label: tbl:contributions
:header-rows: 1
* - Research Area
  - Repository
* - Paleoceanography and Paleoclimatology
  - frozenarchives/pyRATES
* - Volcanology, Geochemistry, and Petrology
  - ruixiabai/XYplot
* - Ocean Sciences
  - ksc005/pyrates/
* - Earth and Planetary Surface Processes
  - Dewan-cpu/Decoding-Landslide-Hazard-Assessment
* - Atmospheric Sciences
  - sputhiyamadam/PYRATES_workshop
* - Geodesy
  - chongjh11/pyrates2024
* - Geoinformatics
  - IGCCP/mindat-locality
* - Seismology
  - vikkybass/PyRates-reproduc
* - Paleoceanography and Paleoclimatology
  - kurtlindberg/leafwaxtools
* - Cryosphere Sciences
  - Duyi-Li/multipol
* - GeoHealth
  - jeanmico/structuralnoisebarriers
* - Hydrology
  - preetika11
* - Paleoceanography and Paleoclimatology
  - tanaya-g/sedproxy_python
* - Hydrology
  - surabhiupadhyay/pyclimproj
* - Hydrology
  - jsacerot/Rpftools
* - Science and Society
  - nqulizada835/geocleaner
* - Earth and Planetary Surface Processes
  - lefitzpatrick/nbspredictor
* - Atmospheric Sciences
  - bosup/toy_package
* - Hydrology
  - zuhlmann/dataFrameGis
* - GeoInformatics
  - PratyushTripathy/pyrsgisHydrology
```
(conclusion-section)=
## Conclusions

In total, the workshops engaged 56 participants spanning over a dozen geoscience subfields and multiple career stages. Participants produced 8 reproducible notebooks and contributed to 9 open-source repositories for software development, underscoring tangible skill acquisition and active engagement with open science practices. These contributions not only exemplify the immediate outputs of the training but also serve as building blocks for a sustainable community of practice that extends beyond the duration of the workshops. By fostering collaborative development and dissemination of geoscience software and workflows, FROGS has catalyzed lasting shifts in research norms toward greater openness and reproducibility.

FROGS has demonstrated that targeted training in computational methods and FAIR publishing principles can effectively embed open science practices into routine geoscience research workflows. Beyond individual capacity building, the initiative has nurtured a vibrant community of practice that continues to support collaborative development and knowledge sharing. Future efforts will focus on scaling these training models, integrating them with institutional curricula, and expanding partnerships with related open science initiatives to amplify impact. Continued evaluation of long-term participant engagement and contributions will guide ongoing refinement and ensure that FROGS remains responsive to the evolving needs of the geoscience community.
