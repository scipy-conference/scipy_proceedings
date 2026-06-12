---
# Ensure that this title is the same as the one in `myst.yml`
title: On-boarding and retaining maintainer talent for MNE-Python
abstract: |
  MNE-Python is open-source software for analyzing electrophysiological data in neuroscience.
  Like many projects, we struggle to retain maintainers.
  Finding maintainers in our user community is hard; most have little formal training in programming.
  To address this, we organized progressive training sprints with open applications and a participation stipend.
  Currently, we are onboarding four alumni of those sprints as new maintainers.
  We've seen positive outcomes from this approach, but at a high cost.
  We are now developing a curriculum for future onboarding efforts.
  We hope to spark discussions with other project leaders about their efforts toward educating and retaining talented maintainers.
---

## Introduction

MNE-Python [@mne_python] is open-source software for analyzing electrophysiological data in neuroscience.
We have a broad user base spanning neuroscience research, clinical neurology, and applied neurotechnology.
<!-- TODO INSERT HISTORY/USES HERE -->

Like many open-source software projects, MNE-Python is struggling to retain maintainers and reach a comfortable Truck Factor [@AvelinoEtAl2016].
This is aggravated by academic incentive systems which devalue open source work compared to scientific publications [@WestnerEtAl2025],
and the fact that many MNE-Python users are not formally trained in programming.
Moreover, MNE-Python's status as domain software makes it difficult for capable programmers lacking neuroscience backgrounds to fill the maintenance gap:
there are too many domain-specific details that one must know to effectively maintain the codebase.
<!-- TODO BOLSTER WITH IDEAS FROM OUR PAPER? -->
<!-- TODO POSE GRANT APPLICATION HAS SOME GOOD TEXT WE CAN REUSE -->

## Interventions
<!-- 3 parts: new dev spr, intermed spr, POSE onboarding. for each: example of program, work they did, etc -->

To increase our contributor pool, we organized two New Developer Sprints and one Intermediate Developer Sprint.
These one-week courses were open to applications from the community, and participants received a stipend.
Holding the courses online allowed for the participation of developers from all over the world.
To further account for this, we ran the sprints in different time zones,
with maintainers from the community leading different time zones.
Both types of sprint involved participants pair-programming with each other or with seasoned maintainers and participants choosing projects to tackle during the week.
The programs were complemented with plenary lectures.

### New Developer Sprints

In 2021 and 2022 we held two New Developer Sprints, which aimed at equipping users of our software with the knowledge necessary to become open-source contributors.
Most of our users are from the wider fields of neuroscience and cognitive science, and usually do not have a formal education in computer science.
This means that they can be lacking the necessary knowledge on how to contribute to open source,
and the (perceived) barriers to contribution can thus be high.
With the New Developers Sprints, we aimed at providing an educational space to acquire the open-source-general and MNE-Python-specific skills for successful contributions to our project.
The curriculum of these sprints consisted of short presentations and pair-programming.
<!-- TODO ADD EVIDENCE/CITATION ON PAIR PROGRAMMING BEING USEFUL? -->
The presentations were delivered by senior community members within and outside of academia, who shared their experiences in the MNE-Python community and how they benefitted from being contributors.
The focus on the sprints, however, was on acquiring hands-on contributing experience through online pair-programming.
Before the sprints, we had curated a list of suitable “First Issues” that the participants could choose from.
The participants paired up with each other and tackled these issues in pairs, connected on a platform that enables screen-sharing and video calls.
Meanwhile, mentors from the MNE-Python maintainer team were available to answer questions or jump into the pair-programming sessions. 
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
