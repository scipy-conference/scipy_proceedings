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

FIGURE 02

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
another main concept in our framework at the next section: representation of data.

Representing Data
-----------------
Recommender systems are data-intensive and runtime performance is greatly affected by quantiy of data and its representation. In Crab
the recommender-related data is encapsulated in the implementations of DataModel. DataModel provide efficient acces to data required
by several recommender algorithms. For instance, a DataModel can provide a count or an array of all user IDs in the input data, or 
provide access to all preferences associated to an item. 

One of the implementations available in Crab  is the in-memory implementation DictDataModels. This model is appropriate if the developer
wants to construct his data representation in memory by passing a dictionary of user IDs and their preferences for item IDs. One of benefits
of this model is that it can easily work with JSON files, which is common as output at web services and REST APIs, since Python converts
the json input into a bult-in dictionary. 

FIGURE 3 Simple Data Model

Typically the model that developers will use is the FileDataModel - which reads data from a file and stores the resulting preference data in memory,
in a DictDataModel. Comma-separated-value or Tab-separated files which each line contains one datum: user ID, item ID and preference value are
acceptable as input to the model. Zipped and gzipped files will also work, since they are commonly used for store huge data in a compressed format.

For data sets which ignore the preference values, that is, ignore the strength of preference, Crab also has an appropriate DataModel twin of 
DictDataModel called BooleanDictDataModel. This is likewise as in-memory DictDataModel implementation, but one which internally does not 
store the preference values. These preferences also called "boolean preferences" have two states: exists, or does not exist and happens when
preferences values aren't available to begin with. For instance, imagine a news site recommending articles to user based on previously viewed
article. It is not typical for users to rate articles. So the recommender recommend articles based on previously viewed articles, whic establishes
some association between user and item, an interesting scenario for using the BooleanDictModel.

FIGURE 4 Simple Data Model with Boolean Data.

Crab also support store and access preference data from a relational database. The developer can easily implement their recommender by using
customized DataModels integrated with serveral databases. One example is the MongoDB, a NoN-SQL database commonly used for non-structured
data. By using the Django's ORM, a popular web framework in Python and MongoEngine, a ORM adapter for integrating MongoDB with Django, we could
easily set up a customized Data Model to access and retrieve data from MongoDB databases easily. In fact, it is already in production at 
a recommender engine using Crab for a brazilian social network called Atepassar. We will explore more about it in the next sections.

One of the current challenges that we are facing is how to handle with all this data in-memory. Specially for recommender algorithms, which
are data intensive. We ar currently investigating how to store data in memory and work with databases directly
without using in-memory data representations. We are concerned that it is necessary for Crab to handle with huge data sets and keep all
this data in memory can affects the performance of the recommender engines implemented with our framework. By now we are using Numpy arrays
for storing the matrices and in the organization of this paper at the time we were discussing about using scipy.sparse packages, a Scipy 2-D
sparse matrix package implemented for handling with sparse a matrices in a efficiently way.  

Now we have discussed about how Crab represents the data input to recommender, in the next section it will examine the recommenders implemented
in detail as also how to evaluate recommenders using Crab tools.

Making Recommendations
----------------------

Crab already supports the collaborative recommender user-based and item-based approaches. They are considered in some of the earliest
research in the field. The user-based recommender algorithm can be described as a process of recommending items to some user, denoted by u,
as follows:

Colocar o algoritmo.

The outer loop suggests we should consider every known item that the user hasn't already expressed a preference for as a candidate
for recommendation. The inner loop suggests that we should look to any other user who has expressed a preference for this candidate
item and see what his or her preference value for it was. In the end, those values are averaged to come up with an estimate -- a 
weighted average.  Each preference value is weigthed in the average by how similar that user is to the target user. The more similar
a user, the more heavily that we weight his or her preference value. In the standard user-based recommendation algorithm, in the step
of searching for every known item in the data set, instead, a "neighborhood" of most similar users is computed first, and only items
known to those users are considered.

In the first section we have already presented a user-based recommender in action. Let's go back to it in order to explore the 
components the approach uses.

SNIPPET CODE.

UserSimilarity encapsulates the concept of similarity amongst users. The UserNeighborhood encapsulates the notion of a group
of most-similar users. The UserNeighborhood uses a UserSimilarity, which extends the basic interface BaseSimilarity. However,
the developers are encouraged to to plug in new ideas of similarity - just creating new BaseSimilarity implementations - 
and get quite different results. As you will see, Crab is not one recommender engine at all, but a set of components that may be
plugged together in order to create customized recommender systems for a particular domain. Here we sum up the components used in 
the user-based approach:

* Data model implemented via DataModel
* User-to-User similarity metric implemented via UserSimilarity
* User neighborhood definition implementd via UserNeighborhood
* Recommender engine implemented via Recommender, in this case, UserBasedRecommender

The same approach can be used at UserNeighborhood where developers also can create their customized neighborhood approaches 
for defining the set of most similar users. Another important part of recommenders to examine is the pairwise metrics implementation.
In the case of the User-based recommender, it relies most of all in this component. Crab implements several similarity metrics
using the Numpy and Scipy scientific libraries such as Pearson Correlation, Euclidean distance, Cosine measure and metric implementations
that ignore preferences entirely like as Tanimoto coefficient and Log-likehood.

Another approach to recommendation implemented in Crab is the item-based recommender. Item-based recommendation is derived from how similar
items are to items, instead of users to users. The algorithm implemented is familiar to the user-based recommender:

ALGORITHM


In this algorithm it is evaluated the item-item similarity, not user-user similarities as shown at the user-based approach. Although they
look similar, there are different properties. For instance, the running time of an item-based recommender scales up as the number of 
items increases, whereas a user-based recommender's running time goes up as the number of users increases. The perfomance advantage
in item-based approach is significant compared to the user-based one.


Let's see how to use item-based recommender in Crab with the following code. Here it employs ItemBasedRecommender rather than
UserBasedRecommender, and it requires a simpler set of dependencies. It also implements the ItemSimilarity interface,
which is similar to the UserSimilarity interface that we've already seen. The ItemSimilarity also works with the pairwise metrics
used in the UserSimilarity. There is no itemneighborhood, since it compares series of preferences expressed by many users for one item
instead of by one user for many items.

* Evaluation
* Future extensions with Slope one , SVD, Boltzman and Fatorization and content-based.


Taking Recommenders to Production
---------------------------------

So far we have presented the recommender algorithms and variants that Crab provides. we also presented how Crab handles with
performance and accuracy evaluation of a recommender. But another important step for a recommender lifecycle is to turn it into a
deployable production-ready web service.




Distributing Recommendation Computations
----------------------------------------
For large data sets with millions of preferences, the current approaches for single machines would have trouble processing recommendations
in the way we have seen in the last sections. It is necessary to deploy a new type of recommender algorithms using a distributed 
computing approach. One of the most popular paradigms is the MapReduce and Hadoop.

Crab didn't support at the time of writting this paper distributed computing, but we are planning to develop variations on the item-based
recommender approach in order to run it in the distributed world. One of our plans is to use the Yelp framework mrJob which supports
Hadoop and it is written in Python, so we may easily integrate with our framework. One of the main concerns in this topic
is to give Crab a scalable and efficient recommender implementation without having high memory and resources consumption as the number of items grows.

Another concern is to investigate and develop other distributed implementations such as Slope One, Matrix Factorization, giving the developer 
alternatives for choosing the best solution for its need specially when handling with large data sets using the power of Hadoop's MapReduce
computations.


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
