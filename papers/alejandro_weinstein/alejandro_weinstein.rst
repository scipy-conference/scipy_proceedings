:author: Alejandro Weinstein
:email: alejandro.weinstein@uv.cl
:institution: Universidad de Valparaiso, Chile
:institution: Advanced Center for Electrical and Electronic Engineering
:corresponding:

:author: Wael El-Deredy
:email: wael.el-deredy@uv.cl
:institution: Universidad de Valparaiso, Chile
:institution: Advanced Center for Electrical and Electronic Engineering

:author: Stéren Chabert
:email: steren.chabert@uv.cl
:institution: Universidad de Valparaiso, Chile

:author: Myriam Fuentes
:email: mfuentes018@gmail.com
:institution: Universidad de Valparaiso, Chile

--------------------------------------------------
Fitting Human Decision Making Models using Python
--------------------------------------------------

.. class:: abstract

A topic of interest in experimental psychology and cognitive neuroscience is to understand how humans make decisions. A common approach involves using computational models to represent the decision making process, and use the model parameters to analyze brain imaging data. These computational models are based on the Reinforcement Learning (RL) paradigm, where an agent learns to make decisions based on the difference between what it expects and what it gets each time it interacts with the environment. In the typical experimental setup, subjects are presented with a set of options, each one associated to different numerical rewards. The task for each subject is to learn, by taking a series of sequential actions, which option maximizes their total reward. The sequence of actions made by the subject and the obtained rewards are used to fit a parametric RL model. The model is fit by maximizing the likelihood of the parameters given the experiment data. In this work we present a Python implementation of this model fitting procedure. We extend the implementation to fit a model of the experimental setup known as the "contextual bandit", where the probabilities of the experiment change from trial to trial. We also developed an artificial agent that can simulate the behavior of a human taking decision under the RL paradigm. We use this artificial agent to validate the model fitting by comparing the parameters estimated from the data with the known agent parameters. We also present the results of a model fitted with experimental data. We use the standard scientific Python stack (NumPy/SciPy) to compute the likelihood function and to find its maximum. The code organization allows to easily change the RL model. We also use the Seaborn library to create a visualization with the behavior of all the subjects. The simulation results validate the correctness of the implementation. The experimental results shows the usefulness and simplicity of the program when working with experimental data. The source code of the program is available at https://github.com/aweinstein/bandit.


.. class:: keywords

   decision making, reinforcement learning

Introduction
------------

As stated by the classic work of Rescorla and Wagner [Res72]_

  *"... organisms only learn when events violate their expectations. Certain
  expectations are built up about the events following a stimulus complex;
  expectations initiated by that complex and its component stimuli are then
  only modified when consequent events disagree with the composite
  expectation."*

This paradigm allows to use the framework of Reinforcement Learning (RL) to model the process of human decision making. In the fields of experimental psycology and cognitive neuroscience these models are used to fit experimental data. Once such a model is fitted, one can use the model parameters to draw conclusions about the subjects that participated in the experiments, and in more general terms, to improve the understanding of how certain areas of the brain work. For an in-depth discussion about the connections between cognitive neuroscience and RL see chapter 16 of [Wie12]_. 

In this work we present a Python program able to fit experimental data to a RL model. The fitting is based on a maximum likelihood approach [Cas02]_. We present simulation and experimental data results. 

A Decision Making Model
-----------------------

In this section we present the model used in this work to describe how an agent (either an artificial one or a human) learn to interact with an environment. The setup assumes that at the discrete time :math:`t` the agent selects action :math:`a_t` from the set :math:`\mathcal{A}=\{1, \ldots, n\}`. After executing that action the agent gets a reward :math:`r_t \in \mathbb{R}`, according to the properties of the environment. Typically these properties are stochastic and are defined in terms of probabilities conditioned by the action. This sequence repeats :math:`T` times. The objective of the agent is to take actions to maximize the total reward

.. math::

   R = \sum_{t=1}^{T} r_t.

In the RL literature, this setup is known as the "n-armed bandit problem" [Sut98]_.

According to the Q-learning paradigm [Sut98]_, the agent keeps track of its perceived value for each action through the so called action-value function :math:`Q(a)`. When the agent selects action :math:`a_t` at time :math:`t`, it updates the action-value function according to

.. math::

   Q_{t+1}(a_t) = Q_t(a_t) + \alpha (r_t - Q_t(a_t)),

where :math:`0 \leq \alpha \leq 1` is a parameter of the agent known as *learning rate*. To make a decision, the agent selects an action at random from the set :math:`\mathcal{A}` with probabilities for each action given by the softmax rule

.. math::

   P(a_t = a) = \frac{e^{\beta Q_t(a)}}{\sum_{i=1}^n e^{ \beta Q_t(a_i)}},

where :math:`\beta > 0` is a parameter of the agent known as *inverse temperature*.

In this work we consider the case were the probabilities associated to the reward, in addition to being conditioned by the action, are also conditioned by a context of the environment. This context change at each time step and is observed by the agent. This means that the action-value function, the softmax rule, :math:`\alpha`, and :math:`\beta` also depend on the current context of the environment. In this scenario, the update action-value and softmax rules become

.. math::
   :label: EqUpdate
	   
   Q_{t+1}(a_t, c_t) = Q_t(a_t, c_t) + \alpha_{c_t} (r_t - Q_t(a_t, c_t))

.. math::
   :label: EqSoftmax

   P(a_t = a, c_t) = \frac{e^{\beta_{c_t} Q_t(a, c_t)}}{\sum_{i=1}^n e^{ \beta_{c_t} Q_t(a_i, c_t)}},

where :math:`c_t` is the cue observed at time :math:`t`. In the literature, this setup is known as *associative search* [Sut98]_ or *contextual bandit* [Lan08]_.

In summary, each interaction, or trial, between the agent and the environment starts by the agent observing the environment context, or cue. Based on that observed cue and on what the agent have learned so far from previous interactions, the agent makes a decision about what action to execute next. It then gets a reward, and based on the value of that reward it updates the action-value function accordingly.

Fitting the Model Using Maximum Likelihood
------------------------------------------

In cognitive neuroscience and experimental psychology one is interested in fitting a decision making model, as the one described in the previous section, to experimental data [Daw11]_.

In our case, this means to find, given the sequences of cues, actions and rewards

.. math::

   (c_1, a_1, r_1), (c_2, a_2, r_2) \ldots, (c_T, a_T, r_T)
   
the corresponding :math:`\alpha_c` and :math:`\beta_c`. The model is fit by maximizing the likelihood of the parameters :math:`\alpha_c` and :math:`\beta_c` given the experiment data. The likelihood function of the parameters is given by

.. math::

   \mathcal{L}(\alpha_t, \beta_t) = \prod_{t=1}^T P(a_t, c_t),

where the probability :math:`P(a_t, c_t)` is calculated using equations :ref:`EqUpdate` and :ref:`EqSoftmax`. 

Once one have access to the likelihood function, the parameters are found by finding the :math:`\alpha_c` and :math:`\beta_c` that maximize the function. In practice, this is done by minimizing the negative of the logarithm of the likelihood function [Daw11]_. In other words, the estimate of the model parameters are given by

.. math::

    \widehat{\alpha}_c, \widehat{\beta}_c =\underset{0\leq\alpha \leq 1, \beta \geq 0}{\operatorname{argmin}} \mathcal{L}(\alpha_c, \beta_c).

Details about the calculation of the likelihood function and its optimization  are given in the *Implementation and Results* section.


Experimental Data
-----------------

The data used in this work consists on the record of a computarised card game played by the participants of the experiment. The game consists of 360 trials. Each trial begins with the presentation of a cue during one second. This cue can be a circle, a square or a triangle. The cue indicates the probability of winning on that trial. These probabilities are 20%, 50% and 80%, and are unknown to the participants. The trial continues with the presentation of four cards with values 23, 14, 8 and 3. The participant select one of these cards and wins or lose the amount of points selected in the card, according to the probabilities defined by the clue. The outcome of the trial is indicated by a stimulus that last one second. The trial finalize with a blank inter-trial stimulus that also last one second. Figure :ref:`FigStimulus` shows a schematic of the stimulus presentation. Participants were instructed to maximize their winnings. See [Mas12]_ for more details about the experimental design.

The study was approved by the University of Manchester research ethics committee. Informed written consent was obtained from all participants.

.. figure:: stimulus.pdf
   :align: center

   Schematic of the stimulus presentation. A trial begins with the presentation
   of a cue. This cue can be a circle, a square or a triangle and is associated
   with the probability of winning in that trial. These probabilities are 20%,
   50% and 80%, and are unknown to the participants. The trial continues with
   the presentation of four cards with values 23, 14, 8 and 3. After selecting
   a card, the participant wins or lose the amount of points indicated in the
   card, according to the probabilities associated with the cue. The outcome of
   the trial is indicated by a stimulus and finalize with a blank inter-trial
   stimulus. :label:`FigStimulus`

Implementation and results
--------------------------



.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b


or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

.. figure:: actions_0.pdf
   :align: center
   :figclass: w
   :scale: 50%

   This is the caption. :label:`egfig`


Discussion
----------


Acknowledgments
---------------

We thanks Liam Mason for sharing the experimental data used in this work. This research was partially supported by the Advanced Center for Electrical and
Electronic Engineering, Basal Project FB0008, Conicyt.


References
----------


.. [Cas02] G. Casella and R. L. Berger, Statistical Inference. Thomson
           Learning, 2002.

.. [Daw11] N. D. Daw, *Trial-by-trial data analysis using computational
           models*, Decision making, affect, and learning: Attention and
           performance XXIII, vol. 23, p. 1, 2011.

.. [Lan08] J. Langford, and T. Zhang, *The epoch-greedy algorithm for
           multi-armed bandits with side information*, Advances in neural
           information processing systems (2008).

.. [Mas12] L. Mason, N. O’Sullivan, R. P. Bentall, and W. El-Deredy, *Better
           Than I Thought: Positive Evaluation Bias in Hypomania*, PLoS ONE,
           vol. 7, no. 10, p. e47754, Oct. 2012.
	   
.. [Res72] R. A. Rescorla and A. R. Wagner, *A theory of Pavlovian
           conditioning: Variations in the effectiveness of reinforcement and
           nonreinforcement*, Classical conditioning II: Current research and
           theory, vol. 2, pp. 64–99, 1972.

.. [Sut98] Sutton, R. S., & Barto, A. G. (1998). Reinforcement
           Learning. Cambridge, Massachusetts: The MIT press.

.. [Wie12] M. Wiering and M. van Otterlo, Eds., Reinforcement Learning,
           vol. 12. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012.



..  LocalWords:  neuroscience
