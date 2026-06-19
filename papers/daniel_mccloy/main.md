---
# Ensure that this title is the same as the one in `myst.yml`
title: On-boarding and retaining maintainer talent for MNE-Python
abstract: |
  MNE-Python is open-source software for analyzing electrophysiological data in neuroscience.
  Like many projects, we struggle to attract and retain maintainers:
  most neuroscientists lack software maintenance expertise,
  and most software engineers lack neuroscience domain knowledge.
  To address this, we organized progressive training sprints with open applications and a participation stipend.
  Currently, we are onboarding four alumni of those sprints as new maintainers.
  We are seeing positive outcomes from this approach, but at a high cost.
  We are now developing a curriculum to streamline future onboarding efforts.
---

## Background

MNE-Python [@mne_python] is open-source software for analyzing electrophysiological data in neuroscience.
It was created in 2010 by [Alex Gramfort](https://alexandre.gramfort.net/) as a port of [MNE](https://mne.tools/stable/install/mne_c.html) software (originally written in C by [Matti Hämäläinen](https://research.aalto.fi/en/persons/matti-h%C3%A4m%C3%A4l%C3%A4inen/)).
The package is pure Python with no compiled code, and wraps many foundational Scientific Python libraries (NumPy, SciPy, Scikit-Learn, Pandas, Statsmodels, and others) with 2D visualizations via Matplotlib and PyQtGraph, and 3d visualizations via PyVista.

MNE-Python provides functionality for electrophysiological signal pre-processing (including filtering, downsampling, artifact detection and suppression), signal analysis (time-domain, spectral, spectrotemporal, clustering, decoding, and more), inverse imaging (estimation of cortical sources based on external sensor signals) and visualization.
Its functionality is further expanded by more than 50 compatible satellite packages; examples include extensions for specialized analyses (brain connectivity [@mne_connectivity], microstate analysis [@mne_microstates;@pycrostates], representational similarity analysis [@mne_rsa], phase-amplitude coupling [@pactools]), for specific data types (intracranial electrodes [@Rockhill2022], near-infrared spectroscopy [@mne_nirs]), for specific data resources (OpenNeuro [@openneuropy], the Human Connectome Project [@mne_hcp]), for organizing data to conform to the Brain Imaging Data Structure standard [@mne_bids], for handling real-time data streams [@mne_lsl], and for managing data processing pipelines for large datasets [@mne_bids_pipeline;@autoreject;@pyprep].
Many of these related packages are hosted within the MNE-Tools organization on GitHub, and looked after by members of the MNE-Python Maintainer Team.

### Ecosystem and Governance

In 2024, the MNE-Python governance changed from a BDFL-plus-maintainers model[^bdfl] to a more decentralized model comprising an advisory board, a steering council, and a maintainer team.
<!-- TODO insert ecosystem diagram -->
Our user base spans neuroscience research, clinical neurology, and applied neurotechnology, and is estimated to be in the range of 5000-10000 users.
The community convenes mostly online through a [Q&A forum](https://mne.discourse.group), biweekly live office hours on Discord, our GitHub repositories' issue trackers, and occasional in-person small-group sprints.
MNE-Python has no community manager; to the extent that community management happens at all, it is handled mostly by the steering council chair.

[^bdfl]: The founding BDFL was Alexandre Gramfort, replaced by Daniel McCloy in 2022.

## Problem Statement
Like many open-source software projects, MNE-Python struggles retain maintainers and reach a comfortable Truck Factor [@AvelinoEtAl2016].
Part of the problem is that the pool is small: contributors and maintainers need relevant neuroscience knowledge (in order to implement sensible default behavior, stay on top of new methodological developments, accurately communicate trade-offs for different analysis choices, *etc.*) and also need software development expertise (not just "writing good code", but also testing, continuous integration and deployment, dependency and security management, *etc.*).
Since most MNE-Python users are not formally trained in software engineering, this makes finding suitable maintainers difficult; the problem is aggravated by academic incentive systems which devalue open source work compared to scientific publications [@WestnerEtAl2025].
Therefor our task becomes finding interested neuroscientists and teaching them how to develop and maintain scientific software.[^domaindev]
Our efforts to achieve this are the topic of the rest of this paper.
<!-- TODO BOLSTER WITH IDEAS FROM OUR "CYCLING" PAPER? -->

[^domaindev]: Theoretically, another option would be engaging a competent developer and teaching them how to think like a neuroscientist, but anecdotally the consensus among leaders of other scientific software projects seems to be that it's much easier to train scientists to develop software than *vice-versa*.

## Interventions
<!-- 3 parts: new dev spr, intermed spr, POSE onboarding. for each: example of program, work they did, etc -->

To increase our contributor pool, we organized two New Developer Sprints and one Intermediate Developer Sprint.
These one-week courses were open to applications from the community, and participants received a stipend.
Holding the courses online (and with flexible work times to accommodate different time zones) allowed for the participation of developers from all over the world.
We used the Discord platform because it supports fast, high-resolution screen-sharing and the creation of an arbitrary number of voice/video "rooms" for participants to work in.
Each sprint had an associated kanban-style project board on GitHub to facilitate finding and tracking the pieces of work to be done.
Both types of sprint involved participants pair-programming with each other and with seasoned maintainers.
Sprint mentors (MNE-Python maintainers) staffed the Discord server in shifts, waiting in an "available mentors" room for participants to ask questions, request reviews, or seek their next piece of work.
Short plenary lectures and tutorials augmented the time spent coding.

### New Developer Sprints

In 2021 and 2022 we held two New Developer Sprints, which aimed at equipping users of our software with the knowledge necessary to become open-source contributors.
Most of our users are from the fields of neuroscience and cognitive science, and many do not have formal training in computer science or programming.
Thus most lack the knowledge of how open-source contribution happens.
Setting up the development environment, understanding git and GitHub, writing documentation and tests, and navigating reviews and CI logs: all are knowledge gaps that prospective contributors must bridge.
For the New Developers Sprints, our application process included questions designed to guarantee that applicants had, at minimum, a functioning development environment and an up-to-date clone of their own fork of the upstream MNE-Python repository.
We provided support to help applicants get this part right, and proactively reached out to applicants before the deadline if their answers indicated a wrong or incomplete setup.
By doing this, we avoided spending any sprint time on environment setup, git configuration, or other preliminaries.

The content of the New Developers' Sprints was also curated in advance.
Dozens of issues were tagged as reserved for the sprint, including several nearly-identical small documentation tasks spread across the codebase.
Participants were given brief instructions in pair programming practices, assigned a partner, and each pair was given several small documentation issues to work through.
<!-- TODO ADD EVIDENCE/CITATION ON PAIR PROGRAMMING BEING USEFUL? -->
The purpose of these small issues was to repeatedly expose participants to the full contribution life cycle: fetch upstream, branch, code, test, commit, push, open PR, check CIs, and respond to reviews.
After a day or so spent on the small documentation issues, nearly all participants took on more complex tasks involving editing module code, writing tests, or updating tutorials.

Each day, a short lecture or tutorial was provided by senior community members within and outside of academia.
Lectures focused on career trajectory, experiences in the MNE-Python community, and how the speakers benefitted from being contributors.
Tutorials focused on knowledge relevant to the contributing workflow: writing and running tests, checking code coverage, reading tracebacks, and similar topics.

<!-- TODO ADD INFO ON RETENTION/CONTINUED CONTRIBUTIONS -->

### Intermediate Sprint

In 2023, we held an Intermediate Sprint, that was aimed at people who were already contributing to MNE-Python but who wanted to advance their contributing skills.
This sprint was again held online, covering several time zones.
Participants chose larger contributing projects in advance, pairing up with each other or mentors for pair-programming sessions.
This was complemented by daily short lectures on pertinent topics such as running and writing tests,
building documentation, deprecations, using Continuous Integration and understanding the output of the services used by MNE-Python, and more.

### Maintainer Onboarding

Currently, we are onboarding four alumni of those sprints as maintainers.
For the first time, we are providing a structured onboarding process that spans two years.
During this period, the prospective maintainers obtain a stipend.
<!-- 
- What are they learning / learning objectives
- What are their responsibilities
- How is this structured (recurring meetings etc)
- Do we want to say something about the funding of this?
 -->

### Curriculum

During the sprints and onboarding activities, we are collecting notes and materials to transform into a written MNE-Python onboarding curriculum.
This documentation is meant to mainly support future onboarding efforts, but will also be worked into our Contributing Guide where applicable.
The domain-general parts will be extracted and published separately from the MNE-Python-specific curriculum.


## Discussion

### Onboarding Results

Past contributors and maintainers mostly came from labs where the lab director had a vested interest in MNE-Python, or were recruited at conferences to contribute their methodological developments.
<!-- ↑ This could be duplicated above as "problem statement" -->
In contrast, our current approach has been bottom-up: first training users how to contribute, then upskilling contributors to facilitate repeat contributions, and finally providing intensive training in maintainer-specific skills.
In our experience, providing education on how to contribute to open source, especially information specific to our project,
greatly lowers the threshold for our users to be willing to attempt a contribution.
This approach also allowed us to prioritize inclusivity in our recruitment,
leading to a slight increase in the diversity of our regular contributors and maintainers.

### Onboarding Costs

The onboarding efforts, however, came with considerable monetary and personnel costs.
The sprints and maintainer onboarding were funded by three separate grants over a six-year period.
Beyond the monetary requirements (compensation for the sprint attendees and key personnel),
they were a huge investment of existing maintainers' time. 
<!-- Do we want to say something about investing into future maintainers vs fixing current things? -->

### Onboarding Obstacles and Outlook

The incentive structure of academia still works against retaining our contributors and maintainers long-term.
We hope that by publicizing our onboarding curriculum and creating other “contributor ladder” resources,
we will empower more users to self-educate about the open-source contribution process.
This will hopefully increase the “input stream” of contributors, and may also increase retention:
by making contribution easier through upskilling, hopefully each single contribution becomes less effortful and thus more likely to be attempted.
At the same time, we hope to spark a discussion among open source software maintainers about the communities’ efforts toward educating and retaining talented maintainers.


## Funding acknowledgment
This project has been made possible in part by NSF POSE award 2449064, and by grant numbers 2020-219006 and 2021-237679 from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation.
