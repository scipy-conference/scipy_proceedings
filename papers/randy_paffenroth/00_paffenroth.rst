:author: Randy Paffenroth
:email: rcpaffenroth@wpi.edu
:institution: Worecester Polytechnic Institute, Mathematical Sciences Department and Data Science Program

:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------
Python in Data Science Research and Education
------------------------------------------------

.. class:: abstract

  Here we demonstrate how Python can be used throughout an entire
  lifecycle of a graduate program in Data Science.  In
  interdisciplinary fields, such as Data Science, the students often
  come from a variety of different backgrounds where, for example,
  some students may have strong mathematical training but less
  experience in programming.  Python’s ease of use, open source
  license, and access to a vast array of libraries make it
  particularly suited for such students.  In particular, we will
  discuss how Python, IPython notebooks, scikit-learn, NumPy, SciPy,
  and pandas can be used in several phases of graduate Data Science
  education, starting from introductory classes (covering topics such
  as data gathering, data cleaning, statistics, regression,
  classification, machine learning, etc.) and culminating in degree
  capstone research projects using more advanced ideas such as convex
  optimization, non-linear dimension reduction, and compressed
  sensing.  One particular item of note is the scikit-learn library,
  which provides numerous routines for machine learning.  Having
  access to such a library allows interesting problems to be addressed
  early in the educational process and the experience gained with such
  “black box” routines provides a firm foundation for the students own
  software development, analysis, and research later in their academic
  experience.  

.. class:: keywords

   data science, education, machine learning

Introduction
------------

Data Science is a burgeoning field of study that lays at the
intersection of statistics, computer science, and numerous applied
scientific domains.  As is common within such *interdisciplinary*
domains of study, Data Science education, mentoring, and research
draws ideas from, and is inspired by, several other more classic
areas.  Perhaps just as importantly, student who wish to pursue
education and careers in Data Science common from similarly diverse
backgrounds.  Accordingly, the challenges and opportunities of being
an educator in such a domain requires one to reflect on appropriate
tools and approaches that promote educational success.  In is the
author's view, and experience, that the Python scripting language can
be an effective part of the Data Science curriculum for several
reasons including its ease of use, its open source license, and its
access to a vast array of libraries covering many topics of interest
to Data Science.

Worcester Polytechnic Institute (WPI) has recently (Fall 2014) begun
admitting students into its new Data Science Master's degree program.
Already the program appears to be experiencing some preliminary
success, with far more applications each semester than we can
reasonably admit, and many students already having success finding
internships and having bright prospects for future jobs.  Of course,
it is much to early to make any comments on the sustained of the
program, however, perhaps the gentle reader will find some value in
the author's experiences, even at this early date.

Of course, we are not the first to suggest Python's effectiveness in
an education and research environment.  In fact, the Python
programming language is quite popular in numerous problem domains and
Python has seen wide used in education, see e.g., [Mye07]_ and
[Sta00]_.  Accordingly, it is not our purpose here to focus on Python
in the large, but rather to focus on its use in *Data Science*
education and research.  With that in mind, herein we will focus on a
small number of case studies that provide insights into how Python can 
leveraged.   

In particular, herein we will discuss the use of Python at three
different levels of Data Science education and research.  First, one
of the courses that is offered as part of the Data Science curriculum,
and in which the author and others have leveraged Python, is DS501
|--| "Introduction to Data Science".  The idea of DS501 is to provide
an introductory overview of the many fields that comprise Data Science
and it is intended that DS501 be one of the first classes a new
student takes when entering the program.  Second, the author has also
used Python to support MA542 |--| "Regression Analysis".  MA542 is a
somewhat more advanced class that is a (core) elective in the Data
Science program as well as being a class taken by many students who
are seeking degrees in the Mathematical Sciences department.  Finally,
the author mentors a number of student's research projects both within
the Data Science Program and the Mathematical Sciences Department.
These research projects all leverage Python, in various ways,
and having access to a common code base allows the various student
projects to build off of one another.

Two key themes will permeate our discussion in the following sections.
First, Python provides easy access to a vast array of libraries.  Even
though Data Science education and research draws from many other
domains, Python was always there with a library ready to support our
work.  Second, and perhaps more subtly, having access to a language
which is easy to use, but provides access to many advanced libraries,
allows one to carefully craft the difficulty and scope of homework
assignments, class projects, and research problems.  In particular,
Python allows students to tackle particular aspects of real world
problems, without being overly burdened with details that are
extraneous to their particular learning objectives.  Both properties make
Python particularly advantageous.

TODO Get rid of after references are done.
TODO Put in references.

Python [Lut13]_, IPython notebooks [Per07]_, scikit-learn [Ped11],
  NumPy [Wal11]_, SciPy [Oli01]_, and pandas [McK10]_

matplotlib [Hun07]_
Cython [Beh11]_
Mayavi [Ram11]_

TODO Compare to R?
TODO More mention of learning objectives earlier in the introduction.


DS501 Introduction to Data Science
----------------------------------

DS501 |--| "Introduction to Data Science" is intended to be one of the
first classes a new student takes when entering the Data Science
program at WPI, and the goal is to provide a high level overview of a
wide swath of the material that a burgeoning Data Scientist should know.
In particular, the course is described as:

.. raw:: latex  
 
  \begin{quotation} 

    This course provides an overview of Data Science, covering a broad
    selection of key challenges in and methodologies for working with
    big data. Topics to be covered include data collection,
    integration, management, modeling, analysis, visualization,
    prediction and informed decision making, as well as data security
    and data privacy. This introductory course is integrative across
    the core disciplines of Data Science, including databases, data
    warehousing, statistics, data mining, data visualization, high
    performance computing, cloud computing, and business
    intelligence. Professional skills, such as communication,
    presentation, and storytelling with data, will be
    fostered. Students will acquire a working knowledge of data
    science through hands-on projects and case studies in a variety of
    business, engineering, social sciences, or life sciences
    domains. Issues of ethics, leadership, and teamwork are
    highlighted. --
    {\footnotesize http://www.wpi.edu/academics/catalogs/grad/dscourses.html}

  \end{quotation}

As one might imagine for such a course, finding the right level of
detail for the course can be quite challenging especially considering
the fact that many of the students have quite varied backgrounds, with
some being experts in mathematics, but with perhaps less training in
computer science or software development, while others find themselves
in the opposite situation.  

In the author's view, an important feature of such a class is that the
students should be able to *get "their hands dirty" playing with real
data*.  Students can often find inspiration by seeing the ideas discussed
in class put to use on problems of practical interest (TODO citation).

With all of the above in mind, the author and the others involved in
the design of DS501 decided to have as a focus of the class be an
interconnected set of four *case studies*.  Each case study is
intended to build upon the previous one until the students are able to
solve some interesting and pertinent problems in Data Science.  And it
is precisely here were Python had a substantial role to play.

Case Study 1
~~~~~~~~~~~~

The idea of the first case study in DS501 is to perform basic data
gathering, cleaning, and collection of statistics.  For this case
study we choose our data source to be the Twitter Data Streaming API
TODO cite.  Already, Python begins to demonstrate its usefulness,
since it allows ready access to the Twitter API through python-twitter
TODO CITE https://code.google.com/p/python-twitter/.

Another key feature of the case studies in DS501 is that we chose to
use IPython notebooks [Per07]_ both to provide the assignments to the
students and to have the students submit their results.  Using IPython
notebooks for both of these tasks provided a number of advantages.
First and foremost, it let the instructors to provide the students
with skeleton implementations of their assignments and allowed the 
students to focus on their learning objectives.  

For example, in the IPython notebooks we provided code examples
similar to the following:

.. code-block:: python

   import twitter
   #---------------------------------------------
   # Define a Function to Login Twitter API
   def oauth_login():
       # Go to http://twitter.com/apps/new to create an 
       # app and get values for these credentials that you'll 
       # need to provide in place of these empty string values 
       # that are defined as placeholders.  
       # See https://dev.twitter.com/docs/auth/oauth for 
       # more information on Twitter's OAuth implementation.
    
       CONSUMER_KEY = '<Insert your key>'
       CONSUMER_SECRET ='<Insert your key>'
       OAUTH_TOKEN = '<Insert your token>'
       OAUTH_TOKEN_SECRET = '<Insert your token>'
    
       auth = twitter.oauth.OAuth(OAUTH_TOKEN, 
		                  OAUTH_TOKEN_SECRET,
                                  CONSUMER_KEY, 
                                  CONSUMER_SECRET)
    
       twitter_api = twitter.Twitter(auth=auth)
       return twitter_api

   #----------------------------------------------
   # Your code starts here
   #   Please add comments or text cells in between to 
   #   explain the general idea of each block of the code.
   #   Please feel free to add more cells below this cell 
   #   if necessary

In this example we provide a skeleton that allows the students to not
have to struggle with the details of Twitter authentication, but
rather focus on the objective of analyzing tweets and hashtags with
frequency analysis.  Using Python, and the skeleton code provided by
the instructors, the student where able to gather and analyze many
thousands of tweets and learn important lessons about data gathering,
data APIs, data storage, and basic analytics.

Case Study 2
~~~~~~~~~~~~

The next case built on the previous one by asking the students to analyze
the MovieLens 1M Data Set (CITE)....  

STOPPED HERE.

Desired outcome of the case study.

* In this case study we will look at the MovieLens 1M Data Set.
* It contains data about users and how the rate movies.
* The idea is to analyze the data set, make conjectures, support
  or refute those conjectures with data, and tell a story about
  the data!
    
Required Readings:

* Chapter 2 (only the section on the MovieLens 1M Data Set), Chapter
  5, Chapter 6 Pg 171-172, and Chapter 8 of the book [Python for Data
  Analysis](http://shop.oreilly.com/product/0636920023784.do)
  (available from the WPI library as an e-book).
* If you want your code to be really compact then you might want to
  also look into the pivot_table method of Panda's DataFrame, though
  there are many other ways to complete the case study!

Required Python libraries:

* Pandas (pandas.pydata.org)
* Matplotlib (matplotlib.org)
* If you need help installing them then you can refer to Chapter 1 of Python for Data Analysis book above.

Problem 1: Importing the MovieLens data set and merging it into a
single Pandas DataFrame

* Download the 1 million ratings data set from
  http://grouplens.org/datasets/movielens/ (though if you are feeling
  adventerous you can download the 10 million ratings file instead)
* Merge all of the data into a single Pandas DataFrame
* Store the data into an HDF5 file.

Report some basic details of the data you collected.  For example:

* How many movies have an average rating over 4.5 overall?
* How many movies have an average rating over 4.5 among men?  How
  about women?
* How many movies have an *median* rating over 4.5 among men over age
  30?  How about women over age 30?
* What are the ten most popular movies?
* Choose what you consider to be a reasonable defintion of
  "popular".
* Be perpared to defend this choice.
* Make some conjectures about how easy various groups are to please?
  Support your answers with data!
* For example, one might conjecture that people between the ages
  of 1 and 10 are the easiest to please since they are all young
  children.  This conjecture may or may not be true, but how
  would you support or disprove either conclusion with with data?
* Be sure to come up with your own conjectures and support them
  with data!

Problem 2: Expand our investigation to histograms
An obvious issue with any inferences drawn from Problem 1 is that we
did not consider how many times a movie was rated.

* Plot a histogram of the ratings of all movies.
* Plot a histogram of the *number* of ratings each movie recieved.
* Plot a histogram of the *average rating* for each movie.
* Plot a histogram of the *average rating* for movies which are rated
  more than 100 times.
* What do you observe about the tails of the histogram where you
  use all the movies versus the one where you only use movies
  rated more than 100 times?
* Which highly rated movies would you trust are actually good?
  Those rated more than 100 times or those rated less than 100
  times?
* Make some conjectures about the distribution of ratings? Support
  your answers with data!
* For example, what age range do you think has more extreme
  ratings?  Do you think children are more or less likely to rate
  a movie 1 or 5?
* Be sure to come up with your own conjectures and support them
  with data!

Problem 3: Correlation:  Men versus women

Let look more closely at the relationship between the pieces of data
we have.

* Make a scatter plot of men versus women and their mean rating for
  every movie.
* Make a scatter plot of men versus women and their mean rating for
  movies rated more than 200 times.
* Compute the *correlation coefficent* between the ratings of men and
  women.
* Are the ratings similiar or not? Support your answer with data!
* Conjecture under what circumstances the rating given by one gender
  can be used to predict the rating given by the other gender.
* For example, are men and women more similar when they are
  younger or older?
* Be sure to come up with your own conjectures and support them
  with data!

Problem 4: Open Ended Question:  Business Intelligence

* Do any of your conjectures in Problems 1, 2, and 3 provide insights
  that a movie company might be interested in?
* Propose a business question that you think this data can answer.
* Suppose you are a Data Sciencetist at a movie company.  Convince
  your boss that your conjecture is correct!

Case Study 3
~~~~~~~~~~~~

Text processing to generate predictors.  Much harder, but made much
easier with sklearn.

Textual analysis of movie reviews

Desired outcome of the case study.

* In this case study we will look at movie reviews from the v2.0
  polarity dataset comes from the
  http://www.cs.cornell.edu/people/pabo/movie-review-data.
* It contains written reviews of movies divided into positive and
  negative reviews.
* As in Case Study 2 idea is to *analyze* the data set, make
  *conjectures*, support or refute those conjectures with *data*, and
  *tell a story* about the data!
    
Required Readings:

* This case study will be based upon the scikit-learn Python library
* We will build upon the turtorial "Working With Text Data" which can
  be found at
  http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

Required Python libraries:

* Numpy (www.numpy.org) (should already be installed from Case
  Study 2)
* Matplotlib (matplotlib.org) (should already be installed from Case
  Study 2)
* Scikit-learn (scikit-learn.org) (installation instructions can be
  found on the web page)

Problem 1: Complete Exercise 2: Sentiment Analysis on movie reviews
from
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

* Assuming that you have downloaded the scikit-learn source code:
* The data cane be downloaded using
  doc/tutorial/text_analytics/data/movie_reviews/fetch_data.py
* A skeleton for the solution can be found in
  doc/tutorial/text_analytics/skeletons/exercise_02_sentiment.py
* A completed solution can be found in
  doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py
* It is ok to use the solution provided in the scikit-learn
  distribution as a starting place for your work.

Modify the solution to Exercise 2 so that it can run in this iPython
notebook

* This will likely involved moving around data files and/or small
  modifications to the script.

Problem 2: Explore the scikit-learn TfidVectorizer class

Read the documentation for the TfidVectorizer class at
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.

* Define the term frequency–inverse document frequency (TF-IDF)
  statistic (http://en.wikipedia.org/wiki/Tf\%E2\%80\%93idf will likely
  help).
* Run the TfidVectorizer class on the training data above
  (docs_train).
* Explore the min_df and max_df parameters of TfidVectorizer.  What do
  they mean? How do they change the features you get?
* Explore the ngram_range parameter of TfidVectorizer.  What does it
  mean? How does it change the features you get? (Note, large values
  of ngram_range may take a long time to run!)

Problem 3: Machine learning algorithms

* Based upon Problem 2 pick some parameters for TfidfVectorizer
* "fit" your TfidfVectorizer using docs_train
* Compute "Xtrain", a Tf-idf-weighted document-term matrix using
  the transform function on docs_train
* Compute "Xtest", a Tf-idf-weighted document-term matrix using
  the transform function on docs_test
* Note, be sure to use the same Tf-idf-weighted class ("fit"
  using docs_train) to transform both docs_test and
  docs_train
* Examine two classifiers provided by scikit-learn
* LinearSVC
* KNeighborsClassifier
* Try a number of different parameter settings for each and judge
  your performance using a confusion matrix (see Problem 1 for an
  example).
* Does one classifier, or one set of parameters work better?
* Why do you think it might be working better?
* For a particular choice of parameters and classifier, look at 2
  examples where the prediction was incorrect.
* Can you conjecture on why the classifier made a mistake for this
  prediction?

Problem 4: Open Ended Question:  Finding the right plot

* Can you find a two dimensional plot in which the positive and
  negative reviews are separated?
* This problem is hard since you will likely have thousands of
  features for review, and you will need to transform these
  thousands of features into just two numbers (so that you can
  make a 2D plot).
* Note, I was not able to find such a plot myself!
* So, this problem is about trying but perhaps not
  necessarily succeeding!
* I tried two things, neither of which worked very well.
* I first plotted the length of the review versus the number of
  features we compute that are in that review
* Second I used Principle Component Analysis on a subset of the
  features.
* Can you do better than I did!?

Case Study 4
~~~~~~~~~~~~

Yelp Dataset Challenge

Hadoop for large scale processing.

Problem 1: Data Collection
Download the Yelp Dataset from the
http://www.yelp.com/dataset_challenge (Click "Get the Data", and
register to get data) This data set contains information about 42,153
business, 252,898 users, and 1,125,458 reviews in Phoenix, Las Vegas,
Madison, Waterloo and Edinburgh.

* Preprocessing can be time consuming but is critical for analyzing
  big data.
* Let's first take a look at the data by loading the json files into
  IPython notebook
* NOTE: the whole dataset can be too big for your computer. If the
  following codes don't work in your computer, try changing the code
  by loading only a few lines of each file.

Question 1: Choose a business category

* Select one business *category* that you are interested in, for
  example, "Restaurants", "Bars", "Shopping", "Hotels" or "Auto
  Repair". (Hint: check the categories field of each business
  object.)
* Collect all the business entities of the category you selected from
  the whole dataset. (It would be recommended that the number of
  business of the category should be larger than 500.) For example,
  collect all the restaurants, i.e., all the business objects that
  have "Restaurants" in their "categories" lists.
* Store the business data you collected into a local file (txt file or
  json file)

Report some statistics about the business you collected 

* The category of interest: 
* The total number of business collected: 
* Plot a histogram of different number of stars for the above business set.

Question 2: Collect all the reviews for the business set in Question 1

* Collect all the reviews for the business entities you collected in
  Question 1. (Hint: Use *business_id* to filter the review data file;
  the data file can be too large to be loaded into the main memory,
  try to avoid using up all the memory in your computer.)
* Store the review data you collected into a local file (txt file or
  json file)

Report some statistics about the reviews you collected 

* The total number of reviews collected: 
* Plot a histogram of different number of stars for the review set.
* Collect all the text data in these reviews (Hint: check the *text* field of the review objects)
* Use the TfidVectorizer class on the text data above, and convert each review into a feature vector.
* Plot a table of the most frequent words (as least top 10, please
  remove the stop words by setting max_df to some reasonable value)
  for all the reviews with FIVE stars.
* Plot a similar table for all reviews with FOUR stars.
* Then three stars, two stars, and one stars.

Question 3: Collect all the users for the review set in Question 2

* Collect all the users for the reviews you collected in
  Qustion 2. (Hint: Use *user_id* to filter the user data file)
* Store the user data you collected into a local file (txt file or
  json file)

Report some statistics about the users you collected 

* The total number of users collected: 
* Plot a histogram of different review_count for the user set.
* Plot a histogram of different average_stars for the user set.
* Plot a histogram of different number of friends for the user set.

Problem 2: MapRudce Data Analysis

* Store the text data of all the collected reviews into a local txt
  file, where each line of the file contains the text of one review.
* Convert the above txt file into TFIDF format using MapReduce. (Hint:
  use *mrjob* package) Please write your answers into the python file:
  "mr_tfidf.py"
* NOTE: you may need to work on this question by runing the python
  file ("mr_tfidf.py") in the terminal (not in IPython notebook)
* Hint: you could test the correctness of your code by comparing with
  the result from TfidVectorizer in the Problem 1 with the result from
  MapReduce. The two results should be the same.  Note that if the
  stop words were removed in TfidVectorizer by setting some max_df,
  the MapReduce version should also remove these stop words. Otherwise
  the results of TfidVectorizer and MapReduce will be different.)

Problem 3: Finding Important Users through PageRank Algorithm

* Convert the friendship information among the users into a graph,
  where each node is a user, each edge represents the friendship
  relationship between the two users.

* Compute the PageRank scores of each user using PageRank algorithm
  (Note: please implement the PageRank algorithm by yourself, but you
  could read and learn from the following code:
  https://code.google.com/p/python-graph/source/browse/trunk/core/pygraph/algorithms/pagerank.py?r=702)

Problem 4: Reweighting the review scores based upon user's PageRank scores

* In the current setting of Yelp, all users are considered as equally
  important. So the average_star of a business is computed by giving
  all reviews with equal importance. However, in many cases, the
  reviews of important users are more influential than those of
  unimportant users. In this problem, please re-compute the avearage
  stars of each business through re-weighting each review by users's
  PageRank scores.


MA542 Regression Analysis
-------------------------

A nice side effect is that you can carefully control the difficulty
and focus be saying which parts they do and which parts are ok to be a
black box.  Nice segue into DS501, where we wanted to focus on specific
ideas, but have the problem be interesting.

More advanced class, but perhaps with a greater concentration of
students who are mathematically focused.  Also, may students were
first time Python users, with the majority of the exceptions having 
taken DS501.

So many libraries that any homework question is probably trivially
answerable if they look hard enough.  Need to be careful that the
ground rules are set correctly.  For example, need to say that they
need to solve the regression problem using the *normal equations*.  It
is ok to debug their code using the black box routine, but they still
need to write their own code.  For example, I insist that they hand in
code.  *Not for grading* but to see how they did it.

Numpy, matplotlib, and Pandas provided almost all of the functionality
they needed for the bulk of the class.   Even though book was more focused
on things like SAS and SPSS (double check book to make sure).

Were able to focus on the mathematics and not have the language, get
in the way.
Look at comments from students!


Student research projects
-------------------------

Python us extensively used in scientific computing [Mil11]_ and [Oli07]_

Convex optimization, deep learning, large scale robust PCA (be careful to 
describe just the right amount), graphical models, communitie analysis,
supervised learning in BGP data.
Yes, they are all related at a deep mathematical level, but I won't bore you 
with the details.

Libraries available for them all!

Also discuss Turing with pycuda and mpi4py.

Finally, discuss manifold learning, and show 3D visualization using mayavi
of the WPI logo embedded in a non-linear manifold.  Make it colorful.
Brings all the pieces together.  Just looking for good Ph.D. student to
work on.

.. figure:: WPI3D.png
   :align: center
   :figclass: w

   An example of a 3D visualization of a manifold using Mayavi .


Conclusion
----------
Python rocks!
It can be used at all levels, and each level builds on the previous one.
There is such a broad array of libraries available in Data Science (or 
whatever you want to call it) that students can focus on what is important
to them.

Sample Stuff
------------
Twelve hundred years ago  |---| in a galaxy just across the hill...

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }
 
Sample Stuff 2
--------------
Test some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

In tellus metus, elementum vitae tincidunt ac, volutpat sit amet
mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. cython
.. [Beh11] Stefan Behnel, Robert Bradshaw, Craig Citro, Lisandro
           Dalcin, Dag Sverre Seljebotn and Kurt Smith. Cython: The
           Best of Both Worlds, Computing in Science and Engineering,
           13, 31-39 (2011), DOI:10.1109/MCSE.2010.118 (publisher
           link)

.. matplotlib
.. [Hun07] John D. Hunter. Matplotlib: A 2D Graphics Environment,
           Computing in Science & Engineering, 9, 90-95 (2007),
           DOI:10.1109/MCSE.2007.55 (publisher link)

.. python
.. [Lut13] Lutz, Mark. *Programming python*. 5th edition, O'Reilly
           Media, Inc., 2010.

.. pandas
.. [McK10] Wes McKinney. Data Structures for Statistical Computing in
           Python, Proceedings of the 9th Python in Science
           Conference, 51-56 (2010) (publisher link)

.. scientific computing in python
.. [Mil11] K. Jarrod Millman and Michael Aivazis. Python for
           Scientists and Engineers, Computing in Science &
           Engineering, 13, 9-12 (2011), DOI:10.1109/MCSE.2011.36

.. python for education
.. [Mye07] Myers, Christopher R., and James P. Sethna. *Python for
	   education: Computational methods for nonlinear systems.*
	   Computing in Science & Engineering 9.3 (2007): 75-79.

.. scipy
.. [Oli01] Jones E, Oliphant E, Peterson P, et al. *SciPy: Open Source
           Scientific Tools for Python*, 2001-, http://www.scipy.org/
           [Online; accessed 2015-05-31].

.. scientific computing in python
.. [Oli07] Travis E. Oliphant. *Python for Scientific Computing*,
           Computing in Science & Engineering, 9, 10-20 (2007),
           DOI:10.1109/MCSE.2007.58

.. sklearn
.. [Ped11] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort,
           Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu
           Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg,
           Jake Vanderplas, Alexandre Passos, David Cournapeau,
           Matthieu Brucher, Matthieu Perrot, Édouard
           Duchesnay. Scikit-learn: Machine Learning in Python,
           Journal of Machine Learning Research, 12, 2825-2830 (2011)
           (publisher link)

.. ipython
.. [Per07] Fernando Pérez and Brian E. Granger. IPython: A System for
           Interactive Scientific Computing, Computing in Science &
           Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53
           (publisher link)

.. mayavi
.. [Ram11] Ramachandran, P. and Varoquaux, G., `Mayavi: 3D
           Visualization of Scientific Data` IEEE Computing in Science
           & Engineering, 13 (2), pp. 40-51 (2011)

.. education
.. [Sta00] Stajano, Frank. *Python in education: Raising a generation
	   of native speakers.* Proceedings of 8th International
	   Python Conference. 2000.

.. numpy and scipy
.. [Wal11] Stéfan van der Walt, S. Chris Colbert and Gaël
           Varoquaux. The NumPy Array: A Structure for Efficient
           Numerical Computation, Computing in Science & Engineering,
           13, 22-30 (2011), DOI:10.1109/MCSE.2011.37 (publisher link)

