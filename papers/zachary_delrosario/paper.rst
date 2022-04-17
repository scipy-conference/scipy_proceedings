:author: Zachary del Rosario
:email: zdelrosario@olin.edu
:institution: Olin College of Engineering

:bibliography: references

-------------------------------------------------------------------------------------
Enabling Active Learning Pedagogy and Insight Mining with a Grammar of Model Analysis
-------------------------------------------------------------------------------------

.. class:: abstract

   Modern engineering models are complex, with dozens of inputs, uncertainties arising from simplifying assumptions, and dense output data. While major strides have been made in the computational scalability of complex models, relatively less attention has been paid to user-friendly, reusable tools to explore and make sense of these models. Such tools have the potential to help engineers mine their models for valuable insights, and to enable more effective engineering pedagogy through hands-on active learning approaches. Grama is a python package aimed at supporting these activities.

   Grama is a grammar of model analysis: An ontology that specifies data (in tidy form), models (with quantified uncertainties), and the verbs that connect these objects. The design of grama is inspired by the Tidyverse, which uses a unifying concept of "tidy data" to organize the design of interoperable data manipulation tools. Grama similarly defines a "grama model" that combines input metadata with an input-to-output mapping. This definition enables a reusable set of "evaluation" verbs that provide a consistent analysis toolkit across different grama models.

   This paper presents three case studies that illustrate the benefits of a model grammar: 1. Invariants in the "grama model" design provide teachable moments that encourage more sound analysis. 2. Reusable tools encourage users to self-initiate healthy modeling behaviors. 3. Reusable tools enable *exploratory model analysis* (EMA), an analog of exploratory data analysis augmented with a data generation loop.

.. class:: keywords

   engineering, engineering education, exploratory model analysis, software design, uncertainty quantification

Introduction
------------

 Test citations with :cite:`wickham2019welcome`.
