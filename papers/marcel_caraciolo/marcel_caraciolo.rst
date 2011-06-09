:author: Marcel Caraciolo
:email: marcel@muricoca.com
:institution: Muricoca Labs

:author: Bruno Melo
:email: bruno@muricoca.com
:institution: Muricoca Labs

:author: Ricardo Caspirro
:email: ricardo@muricoca.com
:institution: Muricoca Labs

--------------------------------------------------
Crab: A Recommendation Engine Framework for Python
--------------------------------------------------


.. class:: abstract

	Crab is a flexible, fast recommender engine for Python that integrates classic information filtering
	recommendation algorithms in the world of scientific Python packages (NumPy,SciPy, Matplotlib).The engine 
	aims to provide a rich set of components from which you can construct a customized recommender system from 
	a set of algorithms. It is designed for scability, flexibility and performance making use of scientific 
	optimized python packages in order to provide simple and efficient solutions that are acessible to everybody
	and reusable in various contexts: science and engineering.
	The engine takes users' preferences for items and returns estimated preferences for other items. For instance,
	a web site that sells movies could easily use Crab to figure out, from past purchase data, which movies a
	customer might be interested in watching to. This work presents our inniative in developing this framework
	in Python following the standards of the well-known machine learning toolkit Scikit-Learn to be an alternative
	solution for Mahout Taste collaborative framework for Java. Finally, we will present its main features,
	real scenarios where this framework is already applied and future extensions.

.. class:: keywords

   data mining, machine learning, recommendation systems, information filtering, framework


Introduction
------------
With the great advancements of machine learning in the past few years, many new learning algorithms have been
proposed and one of the most recognizable techniques in use today are the recommender engines. There are several
services or sites that attempt to recommend books or movies or articles based on users past actions. By trying to
infer tastes and preferences, those systems focus to identify unknown items that are are of interest given an user.
Although people's tastes vary, they do follow patterns. People tend to like things that are similar to other items
they like. For instance, because a person loves bacon-lettuce-and-tomato sandwiches, the recommender system could
guess that he would enjoy a club sandwich, which is mostly the same sandwich, with turkey.  Likewise, people tend
to like things that similar people like.  When a friend entered design school, he saw that just about every other
design student owned a Macintoshi computer - which was no surprise, as she already a lifetime Mac User. Recommendation
is all about predicting these patterns of taste, and using them to discover new and desirable things a person
didn't know about.

Recommendation engines have been implemented in programming languages such as C/C++, Java and made 
publicly available. One of the most popular implementations is the open-source recommendation library Taste, which was
included in the Mahout framework project in 2008 (reference). Mahout is a well-known machine learning toolkit written in Java for
building scalable apache licensed machine libraries (reference). It is specially a great resource for developers who are willing to 
take a step into recommendation and machine learning technologies. Taste has enabled systematic comparisons between
standard developed recommendation methods, leading to an increased visibility, and supporting their broad adoption
in the community of machine learning, information filtering and industry as well as in many others (reference here.) There
are also several another publicly available implementations of recommender engines toolkits in the web. Each one comes with
its own interface, sometimes even not updated anymore by the project owners, a small set of recommendation techniques implemented, and
unique benefits and drawbacks. For Python developers, which is considered a remarkable ammount of machine learning developers and 
researchers there is no single unified way of interfacing and testing these recommendation algorithms, even though there are many
approaches but found incomplete or missing the required set for creating and testing new methods. This restraints
developers and researchers from fully taking advantage of the recent developments in recommendation engines algorithms as also an
obstacle for machine learning researchers that will not want to learn complex programming languages for writing their recommender approaches.
Python has been considered for many years a excellent choice for programmers since it is easy to learn with simple syntax, portable and 
extensive. In scientific computing field, high-quality extensive libraries such as Scipy, MatPlotlib and Numpy have given Python an 
attractive alternative for researchers in academy and industry to write machine learning implementations and toolkits such as Brain,
Shogun, Scikit-learn, Milk and many others (reference).




 



Recommender Engines 
-------------------

Representing Data
-----------------

Making Recommendations
----------------------

Taking Recommenders to Production
---------------------------------

Distributing Recommendation Computations
----------------------------------------

Conclusion and Future Works
---------------------------


References
----------