---
# Ensure that this title is the same as the one in `myst.yml`
title: How the Scientific Python ecosystem helps answer fundamental questions of the Universe
abstract: |
  The ATLAS experiment at CERN explores vast amounts of physics data to answer the most fundamental questions of the Universe.
  The prevalence of Python in scientific computing motivated ATLAS to adopt it for its data analysis workflows while enhancing users' experience.
  This talk will describe to a broad audience how a large scientific collaboration leverages the power of the Scientific Python ecosystem to tackle domain-specific challenges and advance our understanding of the Cosmos.
  Through a simplified example of the renowned Higgs boson discovery, attendees will gain insights into the utilization of Python libraries to discriminate a signal in immersive noise, through tasks such as data cleaning, feature engineering, statistical interpretation and visualization at scale.
exports:
  - format: pdf
    template: eartharxiv
    output: exports/scipy_proceedings_draft.pdf
---
:::{include} introduction.md
:::

:::{include} scientific-python-ecosystem.md
:::

:::{include} uncovering-higgs.md
:::

:::{include} conclusions.md
:::

:::{include} acknowledgements.md
:::
