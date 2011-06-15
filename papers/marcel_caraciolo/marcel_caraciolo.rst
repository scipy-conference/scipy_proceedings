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
Crab contains a recommender engine, in fact, several types beginning with conventional in the literature
user-based and item-based recommenders. Crab provides an assortment of components that may be plugged together
and customized to create an ideal recommender for a particular domain. The toolkit is implemented using Python
and the scientific enviroments for numerical applications such as Scipy and NumPy. The decision of choosing those 
libraries is because they are widely used in scientific computations specially in python programs. Another reason
is because the framework uses the Scikit-learn toolkit as dependant, which provides basic components from our recommender
interfaces derive. The Figure 01 presents the relationship between these basic components. Not all Crab-based recommenders
will look like this -- some will employ different components with different relationships, but this gives a sense 
of the role of each component. 

FIGURE 01


The Data Model implementation stores and provides access to all the preference, user and item data needed in the recommendation. The Similarity
interface provides the notion of how similar two users or items are; where this could be based on one of many possible metrics or calculations.
Finally, a Recommender interface which inherits the BaseEstimator from scikit-learn pull all these components together to recommend items
to users, and related functionality. 

It is easy to explore recommendations with Crab. Let's go through a trivial example. First, we need input to the recommender, data on which
to base recommendations. Generally, this data takes the form of preferences which are associations from users to items, where these users and items
could be anything. A preference consist of a user ID and an item ID, and usually a number expressing the strength of the user's preference
for the item. IDs in Crab can be represented by any type indexable such as string, integers, etc. The preference value could be anything,
as long as larger values mean strong positive preferences. For instance, these values can be considered as ratings on a scale of 1 to 5, where
the user has assigned "1" to items he can't stand, and "5" to his favorites.

Crab is able to work with text files containing information about users and their preferences. The current state of the framework allows
developers to connect with databases via Django's ORM or text files containing the user IDs, product IDs and preferences. For instance, 
we will consider a simple dataset including data about users, cleverly named "1" to "5" and their preferences for four movies, which we call
"101" through "104". By loading this dataset and passing as parameter to the dataset loader, all the inputs will be loaded in memory by creating
a Data Model object.  

Analyzing the data set, it is possible to notice that Users 1 and 5 seem to have similar tastes. Users 1 and 3 don't overlap much since the only
movie they both express a preference for is 101. On other hand, users 1 and 2 tastes are opposite- 1 likes 101 while 2 doesn't, and 1 likes 103
while 2 is just the opposite. By using one of recommender algorithms available in Crab such as the User-Based-Filtering with the given data set 
loaded in a Data Model as input, just run this script using your favorite IDE as you can see the snippet code in the Figure 02 below.


The output of running program should be:    . We asked for one top recommendation, and got one. The recommender engine recommended the
book 104 to user 1. This happens because it estimated user 1's preference for book 104 to be about 4.3 and that was the highest among
all the items eligible for recommendations. It is important to notice that all recommenders are estimators, so they estimate how much
users may like certain items. The recommender worked well considering a small data set. Analyzing the data you can see that the recommender
picked the movie 104 over all items, since 104 is a bit more highly rated overall. This can be refforced since user 1 has similar preferences
to the users 4 and 5, where both have highly rated.

For small data sets, producing recommendations appears trivial as showed above. However, for 
data sets that are huge and noisy, it is a different situation. For instance, consider a popular news
site recommending new articles to readers. Preferences are inferred from article clicks. But,
many of these "preferences" may be noisy - maybe a reader clicked an article but did not like it,
or, had clicked the wrong story. Imagine also the size of the data set - perhaps billions of clicks in a 
month. It is necessary for recommender engines to handle with real-life data sets, and Crab as Mahout
is focusing on how to deal with large and sparse data sets, one of the main issues faced by Crab developers.


Therefore, before deploying recommender engines in Crab into production, it is necessary to present 
another main concept in Crab at the next section: representation of data.



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

In this paper we have presented our efforts in building this toolkit in Python, which we believe that may be useful and make an increasing impact
beyond the recommendation systems community by benefiting diverse applications. 


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
