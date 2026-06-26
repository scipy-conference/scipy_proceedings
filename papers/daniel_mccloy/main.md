---
# Ensure that this title is the same as the one in `myst.yml`
title: On-boarding and retaining maintainer talent for MNE-Python
abstract: |
  MNE-Python is open-source software for analyzing electrophysiological data in neuroscience.
  Like many projects, we struggle to attract and retain maintainers.
  Besides the usual reasons (overworked researchers facing career instability
  and a publication-focused incentive structure),
  our challenge is especially acute:
  most neuroscientists lack software maintenance expertise,
  and most software engineers lack neuroscience domain knowledge.
  To address this, we organized progressive training sprints over several years.
  Currently, four alumni of those sprints are being onboarded as maintainers.
  We are seeing positive outcomes from this approach, but at a high cost.
  We are now developing a curriculum to streamline future onboarding efforts.
---

## Background

MNE-Python [@mne_python] is open-source software for analyzing electrophysiological data in human neuroscience.
It was created in 2010 by [Alexandre Gramfort](https://alexandre.gramfort.net/) as a port of the original [MNE](https://mne.tools/stable/install/mne_c.html) software (written in C by [Matti Hämäläinen](https://research.aalto.fi/en/persons/matti-h%C3%A4m%C3%A4l%C3%A4inen/)).
The package is pure Python with no compiled code, and wraps many foundational Scientific Python libraries (NumPy [@numpy], SciPy [@scipy], Scikit-Learn [@sklearn1;@sklearn2], Pandas [@pandas1;@pandas2], Statsmodels [@statsmodels], and others) with 2D visualizations via Matplotlib [@matplotlib] and PyQtGraph [@pyqtgraph], and 3D visualizations via PyVista [@pyvista].

MNE-Python's initial emphasis was MEG and EEG signals, but has since expanded to encompass ECoG, fNIRS, and eyetracking data as well.
It provides functionality for pre-processing (including filtering, downsampling, artifact detection and suppression), signal analysis (time-domain, spectral, spectrotemporal, clustering, decoding, and more), inverse imaging (estimation of cortical sources based on external sensor signals), and visualization.

### Ecosystem

MNE-Python's functionality is further expanded by more than 50 compatible satellite packages; examples include extensions for specialized analyses (brain connectivity [@mne_connectivity], microstate analysis [@mne_microstates;@pycrostates], representational similarity analysis [@mne_rsa], phase-amplitude coupling [@pactools]), for specific data types (intracranial electrodes [@Rockhill2022], near-infrared spectroscopy [@LukeEtAl2021]), for specific data resources (OpenNeuro [@openneuropy], the Human Connectome Project [@mne_hcp]), for organizing data to conform to the Brain Imaging Data Structure standard [@mne_bids;@bids], for handling real-time data streams [@mne_lsl], and for managing data processing pipelines for large datasets [@mne_bids_pipeline;@autoreject;@pyprep].
Many of these related packages are hosted within the MNE-Tools organization on GitHub, and looked after by members of the MNE-Python Maintainer Team.

### Community and Governance

<!-- TODO insert ecosystem diagram -->
In 2024, the MNE-Python governance changed from a BDFL-plus-maintainers model[^bdfl] to a more decentralized power-sharing model comprising an advisory board, a steering council, and a maintainer team.
Our user base spans neuroscience research, clinical neurology, and applied neurotechnology, and is estimated to be in the range of 5000-10000 users.
The community convenes mostly online through a [Q&A forum](https://mne.discourse.group), biweekly live online office hours, our repositories' issue trackers, and occasional in-person small-group sprints.
MNE-Python has no community manager; to the extent that community management happens at all, it is handled mostly by the steering council chair.

[^bdfl]: The founding BDFL was Alexandre Gramfort, replaced by Daniel McCloy in 2022.

## Problem Statement
Like many open-source software projects, MNE-Python struggles retain maintainers and reach a comfortable [Truck Factor](https://en.wikipedia.org/wiki/Bus_factor) [@AvelinoEtAl2016].
Part of the problem is that the pool is small: contributors and maintainers need relevant neuroscience knowledge (in order to implement sensible default behavior, stay on top of new methodological developments, and accurately communicate trade-offs for different analysis choices) and also need software development expertise (not just "writing good code", but also knowledge of testing, continuous integration and deployment, dependency and security management, and related topics).
Since most MNE-Python users are not formally trained in software engineering, finding suitable candidates is difficult.
The problem of attracting and retaining contributors is further aggravated by academic incentive systems which devalue open-source work compared to scientific publications [@WestnerEtAl2025];
those barriers to open-source contributions in academia also reinforce existing representational biases in developer demographics, since some groups (such as women, Non-White persons, more senior academics) face greater obstacles to contributing [@Nafus2012;@WestnerEtAl2025].

In the past, regular contributors often came from research groups that already had strong ties to the MNE-Python development team, or groups where the principal investigator had a vested interest in contributing.
While those contributors usually had a better-than-average set of relevant skills, this network-based approach almost surely missed many promising contributors.
This approach is also reliant on social ties that may atrophy when key network members experience career or life changes, making the approach somewhat brittle.
Accordingly, we sought to mitigate the risk posed by low maintainer numbers by striving to remove some of the barriers to MNE-Python contribution — especially those related to (lack of) open-source contribution skills.
Therefore, our task became finding interested neuroscientists and teaching them how to develop and maintain scientific software.[^domaindev]
Our efforts to achieve this are the topic of the rest of this paper.

[^domaindev]: Theoretically, another option would be engaging a competent developer and teaching them how to think like a neuroscientist, but anecdotally the consensus among leaders of other scientific software projects seems to be that it is much easier to train scientists to develop software than *vice-versa*.

## Interventions
<!-- 3 parts: new dev spr, intermed spr, POSE onboarding. for each: example of program, work they did, etc -->

To increase our contributor pool, we organized two New Developer Sprints and one Intermediate Sprint.
Because we provided stipends for participants, admission was competitive and hence a non-trivial application process was needed.
The goal of these one-week courses was to teach software development and open-source contribution skills to interested community members.
Holding the courses online (and with flexible work times to accommodate different time zones) allowed for the participation of scientists from all over the world.
In total, 29 participants (12 of them female) took part in one of the two New Developer Sprints, coming from eleven countries in Europe, North and South America, and West Asia.
Eleven of these participants would return for the Intermediate Sprint.
We used a video conferencing platform that supports fast, high-resolution screen-sharing and the creation of an arbitrary number of voice/video/screen "rooms" for participants to work in.
Each sprint had an associated kanban-style project board to facilitate finding and tracking the pieces of work to be done.
Both types of sprint involved participants pair-programming with each other and with mentors.
Sprint mentors (MNE-Python maintainers) were available in shifts, waiting in an "available mentors" room for participants to ask questions, request reviews or pair-programming sessions, or seek their next piece of work.
Short plenary lectures and tutorials augmented the time spent coding.

### New Developer Sprints

In 2021 and 2022 we held New Developer Sprints, aimed at equipping users of our software with the knowledge necessary to become contributors.
Since most of our users are from the fields of neuroscience or cognitive science, and many do not have formal training in computer science or programming,
tasks like setting up a development environment, using git and GitHub, writing documentation and tests, and navigating reviews and CI logs are all knowledge gaps that prospective contributors must bridge.
For the New Developer Sprints, our application process included questions designed to guarantee that applicants had, at minimum, a functioning development environment and an up-to-date clone of their own fork of the upstream MNE-Python repository.[^example_app_questions]
We provided support to help applicants get this part right, and proactively reached out before the deadline if their submissions indicated a wrong or incomplete setup.
By doing this, we avoided spending any sprint time on environment setup, git configuration, or other preliminaries.

[^example_app_questions]: Example application questions: (1) change directory to the root of your repository clone, and paste the output of the command `git remote -v`. (2) start an IPython session and run the commands `import mne; mne.sys_info()` and paste the output here.

The content of the New Developer Sprints was also curated in advance.
Dozens of issues were tagged as reserved for the sprint, including several nearly-identical small documentation tasks spread across the codebase.
Participants were given brief instructions in pair programming practices (for review of benefits, see @HanksEtAl2011), assigned a partner, and each pair was given several small documentation issues to work through.
The purpose of these small issues was to repeatedly expose participants to the full contribution life cycle: fetch upstream → make branch → write code → build docs/run tests → commit changes → push → open PR → check CIs → respond to reviews.
After a day or so spent on the small documentation issues, nearly all participants took on more complex tasks involving editing module code, writing tests, or updating tutorials.

Each day, a short lecture or tutorial was provided by senior community members within and outside of academia.
Lectures focused on career trajectory, experiences in the MNE-Python community, and how the speakers benefitted from being contributors.
Tutorials focused on knowledge relevant to the contributing workflow: writing and running tests, checking code coverage, reading tracebacks, and similar topics.

:::{figure} figure1.png :label: fig:reten Contributor retention of the two New Developer Sprints. The lines show how many years the contributors stayed active after the sprint. The teal-colored line represents the 2021 cohort, the purple line the 2022 cohort. :::

Figure @fig:reten shows the retention of New Developer Sprint attendees as contributors to MNE-Python.
For both sprints, around half of the attendees continued to contribute beyond the first year. Taking both sprints together, ten attendees contributed for two years or longer.
In 2026, five and four years after the sprints, 4 attendees are still regularly contributing.
Our contributor statistics show furthermore, that seven sprint attendees are currently ranking in the top 10% of all-time contributors to the project.
This includes some contributors who have been active for a smaller number of years but have been extraordinarily productive in this time.

### Intermediate Sprint

In 2023, we held an Intermediate Sprint, open to alumni of the New Developer Sprints who wanted to advance their contributing skills.
Applications were again competitive, stipends were provided, and the sprint was again held online with staggered work times to accommodate participants' time zones.
Participants chose larger contributing projects in advance, and could propose their own project (in their application) or choose from a range of pre-selected projects.
The projects were more complex and took participants several days if not the whole week to complete.
During the sprint, they joined with each other or with mentors for pair-programming sessions;
hands-on contributing was complemented by daily short lectures on pertinent topics such as debugging, building documentation, deprecations, understanding output of the CIs we use, and more.
While the New Developer Sprints had focused mostly on contribution skills, the projects at the Intermediate Sprint relied on neuroscience knowledge as well, making the experience more akin to the work done by seasoned contributors.
Examples of sprint projects include how to handle bad channels when working with multiple recordings, adding functionality to equalize observation numbers ("epochs") for time-frequency-resolved data, fixing a bug in our permutation statistics code, or annotating eyetracking data.

### Maintainer Onboarding

After completion of the Intermediate Sprint we sought funding to support the onboarding new MNE-Python maintainers.
We received funding in 2025, and began training three alumni of our Intermediate Sprint and one exceptionally good GSoC contributor.
The funding again allowed us to provide substantial stipends to the trainees.
For the first 6 months, trainees met weekly (online) with an experienced maintainer to discuss topics such as optimization, configuring CIs, conducting code reviews, advanced testing concepts (fixtures, mocking, parametrization), environment management, packaging, dependencies, refactoring, release processes, and similar topics.

After 6 months, onboarding activities transitioned from pre-prepared lessons to a combination of three activities:
triage and PR review sessions, planning and execution of non-trivial documentation improvements,
and phased improvements to our CI configuration.
Plans for the second year of onboarding include deeper engagement with our user forum
(including development of semi-automated approach to triaging user support queries),
assessment and adoption of triage management tooling,
involving the trainee maintainers in long-term planning (roadmap and funding) discussions,
and integrating the lessons learned in year 1 into our in-process onboarding curriculum (see [](#curriculum) below).

In addition to working directly with experienced MNE-Python maintainers, the trainee maintainers have also attended a variety of workshops and short courses, including training in Code-of-conduct response and enforcement, community management, and project management.
Our funding for this onboarding project also supports two in-person sprints to integrate the trainee maintainers with the existing maintainer team.

(curriculum)=
### Curriculum

Throughout the onboarding activities, we are collecting notes and creating materials with the aim of publishing an MNE-Python maintainer onboarding curriculum.
This documentation is meant to streamline future onboarding efforts,
but experience so far shows that it will also precipitate changes to our Contributing Guide.
The domain-general parts of this onboarding curriculum will be extracted and published separately,
for use by the wider Scientific Python Ecosystem.
This will complement existing resources such as [The Turing Way](https://book.the-turing-way.org/) (focused on reproducibility and collaboration), [PyOpenSci](https://www.pyopensci.org/) (focused on DIY packaging of research code), and the [Scientific Python Library Development Guide](https://learn.scientific-python.org/development/) (focused on best-practice recommendations for maintainers, but misses some topics where differences between projects make such general recommendations impractical).[^projectculture]


[^projectculture]: For example, recommendations about triage practices strongly depend on the current size of a project's maintainer team, and the heterogeneity of the domain knowledge needed across different submodules of a package.

## Discussion

MNE-Python has been struggling to attract and retain contributor and maintainer talent.
This is due to a mixture of factors: a small pool of qualified people, the unfavorable-to-open-source academic incentive structure, and our past recruitment practices.

### Onboarding Results

In contrast to our past network-based recruitment practices, our current approach has been bottom-up: first training users how to contribute, then upskilling contributors to facilitate repeat contributions, and finally providing intensive training in maintainer-specific skills.
In our experience, providing education on how to contribute to open-source software, especially information specific to our project,
greatly lowers the threshold for our users to be willing to attempt a contribution.
<!-- TODO sprint attendee retention numbers could go here too/instead -->
This approach also allowed us to prioritize inclusivity in our recruitment and to address barriers that disproportionately impact underrepresented groups in our training.
As of now, this has lead to a slight increase in the diversity of our regular contributors and maintainers.

### Onboarding Costs

The onboarding efforts, however, came with considerable monetary and personnel costs.
The sprints and maintainer onboarding were funded by three separate grants over a six-year period.
Beyond the monetary requirements (compensation for the trainees and key personnel),
they were a huge investment of existing maintainers' time.
To spell it out: the time spent training new maintainers could not be used to fix bugs or make progress on our roadmap, thereby putting further strain on the maintenance of the software (at least in the short term).
Nonetheless, looking ahead we still consider this a valuable investment that hopefully will make future maintenance easier by distributing it across a larger team, and by creating resources that make it easier to keep the size of that team stable going forward.

### Onboarding Obstacles and Outlook

The incentive structure of academia still works against retaining our contributors and maintainers long-term,
though the rise of academic OSPOs and [recent efforts](https://archive.rd-alliance.org/groups/policies-research-organisations-research-software-pro4rs/) by organizations like [ReSA](https://www.researchsoft.org/), [RDA](https://www.rd-alliance.org/), and [CURIOSS](https://curioss.org/) seem to be gradually improving the situation.
The funding landscape for scientific software also seems to be slowly improving,
but in general still prioritizes new development over maintenance or technical debt reduction.
We hope that by publicizing our onboarding curriculum and creating other "contributor ladder" resources,
we will empower more users to self-educate about the open-source contribution process and the path to maintainership.
This will hopefully increase the input stream of contributors to open-source scientific software projects, and may also increase retention:
by making contribution easier through upskilling, hopefully each single contribution becomes less effortful and thus more likely to be attempted.
At the same time, we hope to spark a discussion among open-source software maintainers about the communities’ efforts toward educating and retaining talented maintainers.


## Funding acknowledgment
This project has been made possible by NSF POSE award [2449064](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2449064), and by grant numbers [2020-219006](https://doi.org/10.37921/890928yoegkg) and [2021-237679](https://chanzuckerberg.com/eoss/proposals/building-pediatric-and-clinical-data-pipelines-for-mne-python/) from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation (funder DOI [10.13039/100014989](https://doi.org/10.13039/100014989)).
