:author: Edward Raff
:email: raff_edward@bah.com
:institution: Booz Allen Hamilton
:institution: University of Maryland, Baltimore County
:corresponding:

:author: Joe Aurelio
:email: jaurelio@umbc.edu
:institution: University of Maryland, Baltimore County
:institution: Booz Allen Hamilton

:author: Charles Nicholas
:email: nicholas@umbc.edu
:institution: University of Maryland, Baltimore County

:bibliography: Bib


------------------------------------------------
PyLZJD: An Easy to Use Tool for Machine Learning
------------------------------------------------

.. class:: abstract

    As Machine Learning (ML) becomes more widely known and popular, so too does the desire for new users from other backgrounds to apply ML techniques to their own domains. A difficult prerequisite that often confounds new users is the feature creation and engineering process. This is especially true when users attempt to apply ML to domains that have not historically received attention from the ML community (e.g., outside of text, images, and audio). The Lempel Ziv Jaccard Distance (LZJD) is a compression based technique that can be used for many machine learning tasks. Because of its compression background, users do not need to specify any feature extraction, making it easy to apply to new domains. We introduce pyLZJD, a library that implements LZJD in a manner meant to be easy to use and and apply for novice practitioners. We will discuss the intuition and high-level mechanics behind LZJD, followed by examples of how to use it on problems of disparate data types. 

.. class:: keywords

   compression, complex data, machine learning

Introduction
------------

Machine Learning (ML) has become an increasingly popular tool, with libraries like Scikit-Learn :cite:`scikit-learn` and others :cite:`xgboost,JMLR:v18:16-131,JMLR:v17:15-237,Hall2009` making ML algorithms available to a wide audience of potential users. However, ML can be daunting for news and amateur users to pick up and use. Before even considering what algorithm should be used for a given problem, feature creation and engineering is a prerequisite step that is not easy to perform, nor is it easy to automate. 

In normal use, we as ML practitioners would describe our data as a matrix :math:`\boldsymbol{X}` that has :math:`n` rows and :math:`d` columns. Each of the :math:`n` rows corresponds to one of our data points (i.e., an example), and each of the :math:`d` columns corresponds to one of our features. Using cars as an analogy problem, we may want to know what color a car is, how old it is, or its odometer mileage, as features. We want to have these features in every row :math:`n` of our matrix so that we have the information for every car.  Once done, we might train a model :math:`m(\cdot)` to perform a classification problem (e.g., is the car an SUV or sedan?), or use some distance measure :math:`d(\cdot, \cdot)` to help us find similar or related examples (e.g., which used car that has been sold is most like my own?). 

The question becomes, how do we determine what to use as our features? One could begin enumerating every property a car might have, but that would be time consuming, and not all of the features would be relevant to all tasks. If we had an image of a car, we might use a Neural Network to help us extract information or find similar looking images. But if one does not have prior experience with machine learning, these tasks can be daunting. For some types of complex data, feature engineering can be challenging even for experts.

To help new users avoid this difficult task, we have developed the PyLZJD library. PyLZJD makes it easy to get started with ML algorithms and retrieval tasks without needing any kind of feature specification, selection, or engineering, to be done by the user. Instead, all a user needs to do is represent their data as a file (i.e., one file for every data point, for :math:`n` total files). PyLZJD will automatically process the file and can be used with Scikit-Learn to tackle many common tasks. While PyLZJD will likely not be the best method to use for most problems, it provides an avenue for new users to begin using machine learning with minimal effort and time. 

The Lempel Ziv Jaccard Distance
-------------------------------

LZJD stands for "Lempel Ziv Jaccard Distance" :cite:`raff_lzjd_2017`, and is the algorithm implemented in PyLZJD. The inspiration and high-level understanding of LZJD comes from compression algorithms. Let :math:`C(\cdot)`  represent your favorite compression algorithm (e.g., zip, bz2, etc.), which will take an input :math:`x` and produce a compressed version :math:`C(x)`. Using a decompressor, you can recover the original object or file :math:`x` from :math:`C(x)`. The purpose of this compression is to reduce the size of the file stored on disk. So if :math:`|x|` represents how many bytes it takes to represent the file :math:`x`, the goal is that :math:`|C(x)| < |x|`. 

What if we wanted to compare the similarity of two files, :math:`x` and :math:`y`? We can use compression to help us do that. Consider two files :math:`x` and :math:`y`, with absolutely no shared content. Then we would expect that if we concatenated :math:`x` and :math:`y` together to make one larger file, :math:`x \Vert y`, then compressing the concatenated version of the files should be about the same size as the files compressed separately, :math:`|C(x \Vert y)| = |C(x)| + |C(y)|`. But what if :math:`|C(x \Vert y)| < |C(x)| + |C(y)|`? For that to be true, there must be some overlapping content between :math:`x` and :math:`y` that our compressor :math:`C(\cdot)` was able to reuse in order to achieve a smaller output. The more similarity between :math:`x` and :math:`y`, the greater difference in file size we should see. In which case, we could use the ratio of compressed file lengths to tell us how similar the files are. We could call this a "Compression Distance Metric" :cite:`Keogh:2004:TPD:1014052.1014077` as shown in Equation :ref:`cdm`, where CDM(:math:`x,y`) returns a smaller value the more similar :math:`x` and :math:`y` are, and a larger value if they are different. 

.. math::
    :label: cdm

    \text{CDM}(x,y) = \frac{C(x \Vert y)}{|C(x)| + |C(y)|}


The CDM distance we just described gives the intuition behind LZJD, that is, we can use compression to measure the similarity between arbitrary files. CDM has been used to perform time series clustering and classification :cite:`Keogh:2004:TPD:1014052.1014077`. A large number of compression based distance measures have been proposed :cite:`Sculley:2006:CML:1126009.1126054` and used for tasks such as DNA clustering :cite:`Li2004`, image retrieval :cite:`doi:10.1117/12.704334`, and malware classification :cite:`Borbely2015`. 

.. raw:: latex

	\subsection{Mechanics of LZJD} 

While the above strategy has seen much success, it also suffers from drawbacks. Using a compression algorithm for every similarity comparison %makes prior methods
is slow, and the mechanics of standard compression algorithms are not optimized for machine learning tasks. LZJD rectifies these issues by converting a specific compression algorithm, LZMA, into a dedicated distance metric :cite:`raff_lzjd_2017`. By doing so, LZJD is fast enough to use for larger datasets, and maintains the properties of a true distance metric\footnote{symmetry, indiscernibility, and the triangle inequality
}. LZJD works by first creating the compression dictionary of the Lempel Ziv algorithm :cite:`Lempel1976`.


.. code-block:: python

    def lzset(b): #b should be a list
        s = set()
        start = 0
        end = 1
        while end <= len(b):
            b_s = b[start:end]
            if b_s not in s:
                s.add(b_s)
                start = end
            end += 1
        return s
    
    def sim(A, B): # A & B should be set objects
        return len(A & B) / len(A | B)


The :code:`lzset` method shows the Lempel compression dictionary creation process. Since LZJD cares about similarity as a direct goal, we do not put in the extra work or code normally required to make an effective compressor. Instead, we simply create a Python set of many different sub-strings of the input sequence :code:`b`. Because the :code:`lzset` method gives us a set of objects, we use the well-known Jaccard similarity to measure how close the two sets are. This is defined in the :code:`sim` method above, and mathematically in Equation :ref:`jaccard`. 

.. math:: 
    :label: jaccard

    J(A, B)=\frac{|A \cap B|}{|A \cup B|}=\frac{|A \cap B|}{|A|+|B|-|A \cap B|}


Defining a distance as :math:`d(A,B) = 1-J(A,B)` is a valid metric, and thus provides all the tools necessary to measure the similarity between arbitrary sequences or files. If :math:`a` and :math:`b` represent different sequences, their LZJD would be computed as:

.. code-block:: python

    dist = 1.0-sim(lzset(a),lzset(b))


While the procedure above will implement the LZJD algorithm, it does not include the speedups that have been incorporated into PyLZJD. Following :cite:`raff_lzjd_2017` we use Min-Hashing :cite:`Broder:1998:MIP:276698.276781` to convert a set :math:`A` into a more compact representation :math:`A'`, which is of a fixed size :math:`k` (i.e., :math:`|A'|=k`) but guarantees that :math:`J(A, B) \approx J(A', B')`. :cite:`raff_lzjd_digest` reduced computational time and memory use further by mapping every sub-sequence to a hash, and performing :code:`lzset` construction using a rolling hash function to ensure every byte of input was only processed once. To handle class imbalance scenarios, a stochastic variant of LZJD allows over-sampling to improve accuracy :cite:`raff_shwel`. All of these optimizations were implemented with Cython :cite:`behnel2010cython` in order to make PyLZJD as fast as possible. 

.. raw:: latex

	\subsection{Vectorizing Inputs}

The LZJD algorithm as discussed so far provides only a distance metric. This is valuable for search and information retrieval problems, many clustering algorithms, and :math:`k`-nearest-neighbor style classification, but does not avail ourselves to all the algorithms that would be available in Scikit-Learn. Prior work proposed one method of vectorizing LZSets :cite:`raff_shwel` based on feature hashing :cite:`Weinberger2009a`, where every item in the set is mapped to a random position in a large and high dimensional input (they used :math:`d=2^{20}`). For new users, we want to avoid such high dimensional spaces to avoid the "curse of dimensionality" :cite:`Bellman1957`, a phenomena that makes obtaining meaningful results in higher dimensions difficult. 

Working in such high dimensional spaces often requires greater consideration and expertise. To make PyLZJD easier for novices to use, we have developed a different vectorization strategy. To make this possible, we use a new version of Min-Hashing called "SuperMinHash", :cite:`Ertl2017`. The new SuperMinHash adds a fixed overhead compared to the prior method, but enables us to use what is known as :math:`b`-bit minwise hashing to convert sets to a more compact vectorized representation :cite:`Li:2011:TAB:1978542.1978566`. Since :math:`k \leq 1024` in most cases, and :math:`b \leq 8`, we arrive at a more modest :math:`d=k\cdot b \leq 8,192`. By keeping the dimension smaller, we make PyLZJD easier to use and a wider selection of algorithms from Scikit-Learn should produce reasonable results. 

.. raw:: latex

	\subsection{Over-Sampling Data }

Another feature introduced in :cite:`raff_shwel` is the ability to stochastically over-sample data to create artificially larger datasets. This is particularly useful when working with imbalanced datasets. Given a value :code:`false_seen_prob`, their approach modifies the inner if statement of :code:`lzset` to falsely "see" a sub-string that it has not seen before. This is a one line change that looks like the following:

.. code-block:: python

    if b_s not in s 
      or random.uniform() < false_seen_prob:


By doing so, the set of sub-strings returned will be altered. However, the altered set is still true to the data in that every string in the set is a real and valid sub-string from the corpus. This works because the Lempel Ziv dictionary creation is sensitive to small changes in the input, so a few small alterations can propagate forward and cause a number of differences in the entire process. By making the condition random, we can repeat the process several times and get  different results each time, providing additional diversity that can help train a model. When :code:`false_seen_prob` = 0, we get the standard LZJD output. To perform oversampling, we recommend using small values like :code:`false_seen_prob` :math:`\leq 0.05`. 


Using PyLZJD
-------------

Now that we have given the intuition and described how LZJD works, we show by example how the implementation PyLZJD can be used to do machine learning on a number of different problems, without having to specify a feature processing pipeline. PyLZJD, along with complete versions of these examples, can be found at https://github.com/EdwardRaff/pyLZJD. 

To use PyLZJD, at most three functions need to be imported, as shown below. 

.. code-block:: python

    from pyLZJD import digest, sim, vectorize


These three functions work as follows:


- :code:`digest`: takes in a string as data to convert to a digest, or a path to a file, and converts the file's content to an LZJD digest. If a list is given as input, each element of the list will be processed to return a list of digests. 
- :code:`vectorize`: works the same as digest, but instead of returning a list, returns a numpy array representing a feature vector. 
- :code:`sim`: takes two LZJD digests, and returns the similarity score between two files. 1.0 indicating they are exactly similar, and 0.0 indicating no similarity. 


The above is all that is needed for practitioners to use pyLZJD in their code. Below we will go through three examples of how to use these functions in conjunction with Scikit-Learn to get decent results on these problems. For new users, we recommend considering LZJD as a first-pass easy-to-use algorithm so long as the length of the input data is 200 bytes/characters or more. This recommendation comes from the fact that LZJD is compression based, and it is difficult to compress very short sequences. Another way to consider if LZJD may work for your data is to try manually compressing data points with your favorite compression algorithm. If the files compress well, LZJD may work. If the files do not compress well, LZJD is less likely to work. 

.. raw:: latex

	\subsection{T5 Corpus Example}

The first example we will use is a dataset called "T5", which has historically been used for computer forensics :cite:`Roussev2011`. It contains 4,457 files that are of one of 8 different file types: html, pdf, text, doc, ppt, jpg, xls, or gif. As a simple first step to using pyLZJD, we will attempt to classify a file as one of these 8 file types. Our code starts by collecting the paths to each file into a list :code:`X_paths`. Creating a LZJD digest for each of these files is simple, and calls the :code:`digest` function, as shown below:

.. code-block:: python

    X_hashes = digest(X_paths, processes=-1)


The processes argument is optional. By setting it to -1, as many processor cores as are available will be used. If set to any positive value :math:`n`, then :math:`n` cores will be used. A list of digests will be returned with the same corresponding order as the input. The :code:`digest` function will automatically load every file path from disk, and perform the LZJD process outlined above. 

To use LZJD with Scikit-Learn, we need to massage the files into a form that it expects. Scikit-Learn needs a distance function between data stored as a list of vectors (i.e., a matrix :math:`X`). However, our digests are not vectors in the way that Scikit-Learn understands them, and needs to know how to properly measure distances. An easy way to do this, which is compatible with other specialized distance a user may want to leverage, is to create a 1-D list of vectors. Each vector will store the index of its digest in the created :code:`X_hashes` list.  Then we can can create a distance function which uses the index, and returns the correct value. While wordy to explain, it takes only a few lines of code:

.. code-block:: python

    X = [ [i] for i in range(len(X_hashes))]
    
    def lzjd_dist(a, b):
    	a_i = X_hashes[int(a[0])]
    	b_i = X_hashes[int(b[0])]
    	return 1.0-sim(a_i, b_i)


This is all we need to use the tools already built into Scikit-learn. For example, we can perform :math:`k`-nearest-neighbor classification with cross validation to get an idea about how accurately we are able to predict a file's type. 

.. code-block:: python

    knn_model = KNeighborsClassifier(n_neighbors=5,
        algorithm='brute', metric=lzjd_dist)
    
    scores = cross_val_score(knn_model, X, Y)
    print("Accuracy: %0.2f (+/- %0.2f)" 
        % (scores.mean(), scores.std() * 2))


The above code returns a value of 91\% accuracy. This was all done without us having to specify anything about the associated file formats, how to parse them, or any feature engineering work. We can also leverage other distance metric based tools that Scikit-Learn provides. For example, we can use the t-SNE :cite:`Maaten2008` algorithm to create a 2D embedding of our data that we can then visualize with matplotlib. Using Scikit-Learn, this is only one line of code:

.. code-block:: python

    X_embedded = TSNE(n_components=2, perplexity=5, 
        metric=lzjd_dist).fit_transform(X)

.. figure:: t5_perp5.pdf
    :align: center
    :figclass: h
    
    Example of t-SNE visualization created using LZJD. Best viewed digitally and in color.


A plot of the result is shown in Figure 1, where we see that the groups are mostly clustered into separate regions, but that there is a significant collection of points that were difficult to organize with their respective groups. While a tutorial on effective use t-SNE is beyond the scope of this paper, LZJD allows us to leverage this popular tool for immediate visual feedback and exploration. 

.. raw:: latex

	\subsection{Spam Image Classification}

The prior example used files of varying types, which is similar to the problem domain that LZJD was developed for. In this example, we will change the type of data and how we approach the problem. Here, our goal will be to predict if an email image attachment is a "spam" image (i.e., undesirable) or a "ham" image (i.e., desirable - or at least, more desirable than spam). This dataset was collected in 2007 :cite:`imageSpam2007`, with 3298 spam and 2021 ham images. 

In this example, we will use the :code:`vectorize` function to create feature vectors for each data point. This may be desirable in order to build models that avoid the nearest neighbor search, which can be slow and cumbersome to deploy. The trade off is we spend more time during the training phase of the algorithm. Doing this with pyLZJD is simple, and the below code snippet handles the work of creating the labels, loading the files, and creating feature vectors, again, without us having to specify anything about the input. 

.. code-block:: python

    spam_paths = glob.glob("personal_image_spam/*")
    ham_paths = glob.glob("personal_image_ham/*")
    
    all_paths = spam_paths + ham_paths
    yBad = [1 for i in range(len(spam_paths))]
    yGood = [0 for i in range(len(ham_paths))]
    y = yBad + yGood
    X = vectorize(all_paths)


Now that we have feature vectors, we can train a Logistic Regression model to predict if a new image is a spam or not. The code to do that and evaluate it (by several metrics) is shown below. 

.. code-block:: python

    X_train, X_test, y_train, y_test = 
      train_test_split(X, y, test_size=0.2, 
        random_state=42) 
    
    lgs = LogisticRegression(class_weight='balanced')
    lgs.fit(X_train, y_train) #training our model
    
    predicted = lgs.predict(X_test)
    
    fpr, tpr, _ = metrics.roc_curve(y_test, 
      (lgs.predict_proba(X_test)[:, 1]))
    auc = metrics.auc(fpr, tpr)
    print("Accuracy: %f" % 
      lgs.score(X_test, y_test)) 
    print("Precision: %f" %
      metrics.precision_score(y_test, predicted))
    print("Recall: %f" % 
      metrics.recall_score(y_test, predicted))
    print("F1-Score: %f" % 
      metrics.f1_score(y_test, predicted))
    print("AUC: %f" % auc)


This should produce an accuracy of about 94.6\%, and an AUC of 98.7\%. In the above code snippet, we included the :code:`class_weight` parameter in an effort to aid the model with the class imbalance that is present in the data. There are more examples of spam images, which can bias a model toward calling most inputs "spam" by default. Using a 'balanced' class weight re-weights the data as if there was an equal number of examples of each class. With pyLZJD, you can perform a special type of over-sampling to help further reduce this impact and improve accuracy. In the below code segment, we show a simple version of using this ability. 

.. code-block:: python

    paths_train, paths_test, y_train, y_test = 
      train_test_split(all_paths, y, 
        test_size=0.2, random_state=42)
    
    X_train = vectorize(paths_train*10, 
      false_seen_prob=0.05)
    X_test = vectorize(paths_test)


In this code, we have over-sampled both the spam and ham training data 10 times. Normally, this would create 10 copies of the same vectors, and have no impact on the solution learned. But, we added the :code:`false_seen_prob` flag, which alters how the :code:`lzset` is constructed: this flag turns on the stochastic component described above that makes you get a different result every time you call the function, so that we can get a variety of different (but realistic) examples for each datapoint. If we train a new logistic regression model on this data, we get improved results, which are shown in Table :ref:`spamImgResults`. 

.. raw:: latex

    \begin{table}[!h]
    \centering
    \caption{Results on training a Logistic Regression model for spam image detection. Over-sampled scores show results when 'false\_seen\_prob' is used.   }
    \label{spamImgResults}
	\begin{tabular}{lcc}
	\hline
	\multicolumn{1}{c}{Metric} & Score    & Over-sampled Score \\ \hline
	Accuracy                   & 0.946429 & 0.956767           \\
	Precision                  & 0.950437 & 0.953824           \\
	Recall                     & 0.965926 & 0.979259           \\
	F1-Score                   & 0.958119 & 0.966374           \\
	AUC                        & 0.987108 & 0.991602           \\ \hline
	\end{tabular}
    \end{table}

LZJD won't always be effective for images, and convolutional neural networks (CNNs) are a better approach if you need the best possible accuracy. However, this example demonstrates that LZJD can still be useful, and has been used successfully to find slightly altered images :cite:`Faria-joao`. This example also shows how to build a more deployable classifier with pyLZJD and tackle class-imbalance situations. 

.. raw:: latex

	\subsection{Text Classification}

As our last example, we will use a text-classification problem. While other methods will work better, the purpose it to show that LZJD can be used in a wide array of potential applications. For this, we will use the well-known 20 Newsgroups dataset, which is available in Scikit-Learn. We use this dataset because LZJD works best with longer input sequences. For simplicity we will stick with distinguishing between the newsgroup categories of 'alt.atheism' and 'comp.graphics'. An example of a email from the later group is shown below. 


	By '8 grey level images' you mean 8 items of 1bit images?
	It does work(!), but it doesn't work if you have more than 1bit
	in your screen and if the screen intensity is non-linear.
	
	With 2 bit per pixel; there could be 1*c_1 + 4*c_2 timing,
	this gives 16 levels, but they are linear if screen intensity is
	linear.
	With 1*c_1 + 2*c_2 it works, but we have to find the best
	compinations -- there's 10 levels, but 16 choises; best 10 must be
	chosen. Different compinations for the same level, varies a bit, but
	the levels keeps their order.

	Readers should verify what I wrote... :-)


When a string is not a valid path to a file, pyLZJD will processes the string itself to create a digest. This makes working with data stored as strings simple, and getting results is as easy as the code snippet below: 

.. code-block:: python

    X_train = vectorize(newsgroups_train.data)
    X_test = vectorize(newsgroups_test.data)
    
    clf = LogisticRegression()
    clf.fit(X_train, newsgroups_train.target)
    
    pred = clf.predict(X_test)
    metrics.f1_score(newsgroups_test.target, 
        pred, average='macro')


With the above code, we get an :math:`F_1` score of 83\%. It is important to note that in this case, using Scikit-Learn's TfidfVectorizer one can get 89\% :math:`F_1`. The point here is that with pyLZJD we can get decent results without having to think about what kind of vectorization is being performed, and that any string encoded data can be feed directly into the :code:`vectorize` or :code:`digest` functions to get immediate results. 

Conclusion
----------

We have shown, by example, how to use pyLZJD on a number of different datasets composed of raw binary files, images, and regular ASCII text. In all cases, we did not have to do any feature specifications to use pyLZJD, making application simpler and easier. This shortcut is particularly useful when feature specification is hard, such as raw file types, but can also make it easier for people to get into applying Machine Learning. 