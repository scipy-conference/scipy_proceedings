:author: Hongsup Shin
:email: hongsup.shin@arm.com
:institution: Arm Research
:corresponding:

------------------------------------------------
Case study: Real-world machine learning application for hardware failure detection
------------------------------------------------

.. class:: abstract

   When designing microprocessors, engineers must verify whether the proposed design, defined in hardware description language, do what is intended. Engineers run simulation tests and can fix bugs if tests have failed. Due to the complexity of design, the baseline approach is to provide random stimuli to verify random parts. 

   However, this method is time-consuming and redundant. To increase efficiency, I trained an ensemble of supervised and unsupervised (classification) models to predict failures so that engineers can run fail-prone tests only. In this talk, I will also mention evaluating retraining scenarios, often not discussed in data science presentations.

   The conference SciPy celebrates a variety of topics where machine learning and data science is applied by using Python. This ranges from geological studies to healthcare applications. Machine learning application in computer hardware is still relatively new and thus introducing an actual project with a full scope, starting from prototyping to model evaluation and retraining, will provide another good example of data-driven approach to the SciPy community.

   Since the majority of data science and machine learning presentations mostly focus on algorithms without realistic and practical context, I would like to shed more light on how to work around and optimize within practical constraints and customize solutions in a machine learning project. Furthermore, I would like to discuss various ways of retraining deployed models, which are rarely discussed in data science talks.

.. class:: keywords

   hardware verification, machine learning, deployment, retraining, model selection

Introduction
------------


Methods
-------

Problem statement and constraints
#################################

Proposed solution
#################

Data
####

Models
######

Results
-------

Model performance
#################

Retraining
##########

Conclusions
-----------

