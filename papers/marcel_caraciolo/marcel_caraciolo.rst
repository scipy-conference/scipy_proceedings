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
proposed and one of the most recognizable techniques in use today are the recommender engines [Adoma2005]_. There are several
services or sites that attempt to recommend books or movies or articles based on users past actions [Linden2003]_ , [Abhinandan2007]_ .
By trying to infer tastes and preferences, those systems focus to identify unknown items that are are of interest given an user.
Although people's tastes vary, they do follow patterns. People tend to like things that are similar to other items
they like. For instance, because a person loves bacon-lettuce-and-tomato sandwiches, the recommender system could
guess that he would enjoy a club sandwich, which is mostly the same sandwich, with turkey.  Likewise, people tend
to like things that similar people like.  When a friend entered design school, he saw that just about every other
design student owned a Macintoshi computer - which was no surprise, as she already a lifetime Mac User. Recommendation
is all about predicting these patterns of taste, and using them to discover new and desirable things a person
didn't know about.

Recommendation engines have been implemented in programming languages such as C/C++, Java among others and made 
publicly available. One of the most popular implementations is the open-source recommendation library Taste, which was
included in the Mahout framework project in 2008 [Taste2008]_ . Mahout is a well-known machine learning toolkit written in Java for
building scalable apache licensed machine libraries [Mahout2011]_ . It is specially a great resource for developers who are willing to 
take a step into recommendation and machine learning technologies. Taste has enabled systematic comparisons between
standard developed recommendation methods, leading to an increased visibility, and supporting their broad adoption
in the community of machine learning, information filtering and industry. There
are also several another publicly available implementations of recommender engines toolkits in the web [EasyRec2011]_ , [MyMediaLite2011]_.
Each one comes with its own interface, sometimes even not updated anymore by the project owners, a small set of recommendation techniques implemented, and
unique benefits and drawbacks.

For Python developers, which there is a considerable ammount of machine learning developers and 
researchers, there is no single unified way of interfacing and testing these recommendation algorithms, even though there are some
approaches but found incomplete or missing the required set for creating and evaluating new methods [PySuggest2005]_, [djangorecommender2008]_.
This restraints developers and researchers from fully taking advantage of the recent developments in recommendation engines algorithms as also an
obstacle for machine learning researchers that will not want to learn complex programming languages for writing their recommender approaches.
Python has been considered for many years a excellent choice for programming beginners since it is easy to learn with simple syntax, portable and 
extensive. In scientific computing field, high-quality extensive libraries such as Scipy, MatPlotlib and Numpy have given Python an 
attractive alternative for researchers in academy and industry to write machine learning implementations and toolkits such as Brain,
Shogun, Scikit-learn, Milk and many others.

The reason of not having an alternative for python machine learning developers by providing an unified and easy-to-use recommendation framework 
motivated us to develop a recommendation engine toolbox that provides a rich set of features from which the developer can build a customized recommender system
from a set of algorithms. The result is a framework, called CRAB, with focus on large-scale recommendations making use of scientific python packages
such as Scipy, Numpy and Matplotlib to provide simple and efficient solutions for constructing recommender systems
that are acessible and reusable in various contexts. The framework Crab provides a generic interface for recommender systems implementations,
among them the collaborative filtering approaches such as Slope-One, User-Based and Item-Based filtering, which are already available for use.
The recommender interfaces can be easily combined with more than 10 different pairwise metrics already implemented, like the cosine, tanimoto,
pearson, euclidean already using Scipy and Numpy basic optimized functions [Breese1998]_. Moreover, it offers support
for using similarities functions such as user-to-user or item-to-item and allows easy integration with different input domains like databases,
text files or python dictionaries.

Currently, the collaborative filtering algorithms are widely supported. In addition to the User-Based and Item-Based filtering 
techniques, Crab implements several pairwise metrics and provides the basic interfaces for developers to build their own 
customized recommender algorithms. Finally, several commonly used performance measures, such as accuracy, precision, recall are
implemented in Crab.

An important aspect in the design of Crab was to enable very large-scale recommendations. Crab is currently being rewritten
to support optimized scientific computations by using Scipy and Numpy routines. Another feature concerned by the current mantainers
is to make Crab support sparse and large datasets in a way that there is a little as possible overhead for storing the data
and intermediate results. Moreover, Crab also aims to support scaling in recommender systems in order to build high-scale, 
dynamic and fast recommendations over simple calls. Futhermore, Crab is also planned to support distributed 
recommendation computation by interfacing with the distributed computation library MrJob written in Python currently
developed by Yelp [MrJob2010]_. What sets Crab apart from many other recommender systems toolboxes, is that it provides 
interactive interfaces to build, deploy and evaluate customized recommender algorithms written in Python running on several
platforms such as Linux, BSD, Mac OS and Windows.

The outline of this paper is as follows. We first discuss the Crab's main features by explaining the architecture of the framework.
Next, we provide our current approach for representing the data in our system and current challenges. Then, we also presents
how Crab can be used in production by showing a real scenario where it is already deployed. Finally, we discuss about our plans
to handle with distributed recommendation computations. Also, our conclusions and future works are also presented at the end of
this paper.


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
.. [Adoma2005] Adomavicius, G.; Tuzhilin, A. *Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions*,
      IEEE Transactions on Knowledge and Data Engineering; 17(6):734–749, June 2005.

.. [Linden2003] Greg Linden, Brent Smith, and Jeremy York. *Amazon.com Recommendations: Item-to-Item Collaborative Filtering.*,
      IEEE Internet Computing 7, 1, 76-80,  January 2003.

.. [Abhinandan2007] Abhinandan S. Das, Mayur Datar, Ashutosh Garg, and Shyam Rajaram, *Google news personalization: scalable online collaborative filtering.*,
	 In Proceedings of the 16th international conference on World Wide Web (WWW '07). ACM, New York, NY, USA, 271-280, 2007.

.. [Taste2008]  *Taste - Collaborative Filtering For Java* , accessible at: http://taste.sourceforge.net/.

.. [Mahout2011] *Mahout - Apache Machine Learning Toolkit* ,accessible at: http://mahout.apache.org/

.. [EasyRec2011] *EasyRec* ,accessible at: http://www.easyrec.org/

.. [MyMediaLite2011] *MyMediaLite Recommender System Library*, accessible at: http://www.ismll.uni-hildesheim.de/mymedialite/

.. [PySuggest2005] *PySuggest*, accessible at: http://code.google.com/p/pysuggest/

.. [djangorecommender2008] *Django-recommender* accessible at: http://code.google.com/p/django-recommender/

.. [Breese1998] J. S. Breese, D. Heckerman, and C. Kadie. *Empirical analysis of predictive algorithms for collaborative filtering.*,
                UAI, Madison, WI, USA, pp. 43-52, 1998.
.. [MrJob2010] *mrjob*  , accessible at:  https://github.com/Yelp/mrjob
