:author: Jacob Montiel
:email: jacob.montiel@waikato.ac.nz
:institution: Department of Computer Science, University of Waikato

:bibliography: bibliography

-----------------------------------
Learning from evolving data streams
-----------------------------------

.. class:: abstract

   Ubiquitous data poses challenges on current machine learning systems to store, handle and analyze data at scale. Traditionally, this task is tackled by dividing the data into (large) batches. Models are trained on a data batch and then used to obtain predictions.  As new data becomes available, new models are created which may contain previous data or not. This training-testing cycle is repeated continuously. Stream learning is an active field where the goal is to learn from infinite data streams. This gives rise to additional challenges to those found in the traditional batch setting: First, data is not stored (it is infinite), thus models are exposed only once to single samples of the data, and once processed those samples are not seen again. Models shall be ready to provide predictions at any time. Resources such are memory and time are limited, consequently, they shall be carefully managed. The data can change (evolve) over time and models shall be able to adapt accordingly. This is a key difference with respect to batch learning, where data is assumed static and models will fail in the presence of change. Model degradation is a side-effect of batch learning in many real-world applications requiring additional efforts to address it. This papers provides a brief overview of the core concepts of machine learning for data streams and describes ``scikit-multiflow``, an open-source Python library specifically created for machine learning on data streams. ``scikit-multiflow`` is built to serve two main purposes: easy to design and run experiments, easy to extend and modify existing methods.

.. class:: keywords

   machine learning, data streams, concept drift, scikit, open-source

Introduction
------------

The minimum pipeline in machine learning is composed of: (1) data collection and processing, (2) model training and (3) model deployment. Traditionally, data is collected and processed in batches. Although this approach is the state-of-the-art in multiple applications, it is not suitable in the context of evolving data streams as it implies important compromises. In batch learning, data is assumed to be accessible (usually stored on a physical device), the limitations on this approach have been pushed as storage and processing power became cheaper. Similarly, advances have been done towards data management and  advances in efficient   has enable to handle large volumes of data. However, there exist instances where storing large volumes of data is not practical. For example:

- Financial markets generate huge volumes of data on a daily basis. Depending on the state of such markets and multiple external factors data can become obsolete quickly rendering it useless for creating accurate models. Predictive models must be able to adapt fast in order to be useful in this dynamic environment.
- Predictive maintenance. IoT sensors are a continuous source of data that reflects the health of multiple systems, from complex systems such as airplanes to simpler ones such as house appliances. Predictive systems are required to react fast as to prevent disruptions from malfunctioning elements.
- Online fraud detection. The speed of reaction of an automatic system is also an important factor in multiple applications. Fraud detection in online banking presents similar challenges in terms of data collections and processing. However, it also involves additional challenges, fraud detection systems must adapt quickly to changes such as consumer behaviour (for example during holidays), stability of the financial markets, as well as the fact that attackers constantly change their behaviour in order to beat these systems.
- Supply chain. Several sectors use automatic systems in their supply chain in order to cope with the demand of products in an efficient way. However, the covid-19 pandemic brought to attention the fragility of these systems to sudden changes, e.g., in less than 1 week, products related to  the pandemic such as face masks filled the top 10 searched terms in Amazon [#]_. Many automatic systems failed resulting in a disruption in the supply chain.
- Climate change. Environmental data is a quintessential example of the five *v*'s of big data: volume, velocity, variety, veracity and value. Understanding this data has many implications in our daily lives, e.g., food production can be severally impacted by climate change, disruption of the water cycle has resulted on a rise of heavy rains with the associated risk of floodings. IoT sensors are now making environmental data available at a faster rate and machine learning systems must adapt to this new norm.

.. [#] https://www.technologyreview.com/2020/05/11/1001563/covid-pandemic-broken-ai-machine-learning-amazon-retail-fraud-humans-in-the-loop/

.. figure:: stream_opportunity.png
   :align: center
   :scale: 70%
   :figclass: w

   Batch learning systems are bounded to a investment in resources like memory and training time as the data volume increases. Once a reasonable investment threshold is reached, data becomes unusable turning into a missed opportunity. On the other hand, efficient management of resources makes stream learning an interesting alternative for big data applications. :label:`fig:investment`

As shown in the previous examples, dynamic environments pose an additional set of challenges to batch learning systems. Model degradation is an predominant problem in multiple real-world applications. As enough data has been generated and collected, proactive users might decide to train their models to make sure that they agree with the current data. This is complicated for two reasons: First, batch models (in general) are not able to use new data into account, so the machine learning pipeline must be run multiples times as data is collected over time. Second, the decision for such an action is not trivial, and involves multiple aspects. For example, should a new model be trained only on new data? This depends on the amount of variation in the data. Small variations might not be enough to justify retraining and redeploying a model. This is why a reactive approach is predominantly employed in the industry. Model degradation is monitored and corrective measures are enforced if an user defined threshold is exceeded (accuracy, type I and type II errors, etc.). Fig. :ref:`fig:investment` depicts another important aspect to consider, the tradeoff between the investment in resources such as memory and time (and associated cost) and the pay-off in predictive performance. In stream learning, resource-wise efficiency is fundamental, predictive models not only must be accurate but also must be able to handle theoretically infinite data streams. Models must fit in memory no mather the amount of data seen (constant memory). On the other hand, training time is expected to grow sub-linearly with respect to the volume of data processed. New samples must be processed as soon as they become available so it is vital to process them as fast as possible in order to be ready for the next iteration.

Machine learning for streaming data
-----------------------------------

Formally, the task of supervised learning from evolving data streams is defined as follows. Consider a stream of data :math:`S=\{(\vec{x}_t,y_t)\} | t = 1,\ldots,T` where :math:`T \rightarrow \infty`. Input :math:`\vec{x}_t` is a feature vector and :math:`y_t` the corresponding target where :math:`y` is continuous in the case of regression and discrete for classification. The objective is to predict the target :math:`\hat{y}` for an unknown sample :math:`\vec{x}`. For illustrative purposes, this paper focuses on the classification task. However, it is important to mention that the field of machine learning for streaming data covers other tasks such as regression, clustering, anomaly detections, to name a few.

In stream learning, models are trained incrementally, one sample at a time, as new samples :math:`(\vec{x}_t,y_t)` become available. Since streams are theoretically infinite, the training phase is non-stop and predictive models are continuously updating their internal state in agreement with incoming data. This is fundamentally different to the batch learning approach, where models have access to all (available) data during training. As previously mentioned, in the stream learning paradigm, predictive models must be resource-wise efficient. For this purpose, a set of requirements :cite:`Bifet2011DataStreamMining` must be fulfilled by streaming methods:

- **Process one example at a time, and inspect it only once.** The assumption is that there is not enough time nor space to store multiple samples, failing to meet this requirement implies the risk of missing incoming data.
- **Use a limited amount of memory.** Data streams are assumed infinite, thus storing data for further processing is impractical.
- **Work in a limited amount of time.** In other words, avoid bottlenecks generated by time consuming tasks which in the long run could make the algorithm fail.
- **Be ready to predict at any point.** Stream models are continuously *updated* and must be able to provide predictions at any point in time.

Concept drift
+++++++++++++
An challenging element of dynamic environments is the chances that the underlying relationship between features :math:`X` and target(s) :math:`\vec{y}` can evolve (change) over time. This phenomenon is known as **Concept Drift**. Real concept drift is defined as changes in the posterior distribution of the data :math:`p(\vec{y}|X)`. Real concept drift means that the unlabeled data distribution does not change, whereas data evolution refers to the unconditional data distribution :math:`p(X)`. In batch learning, the joint distribution of data :math:`p(X,\vec{y})` is, in general, assumed to remain stationary. In the context of evolving data streams, concept drift is defined between two points in time :math:`t_o, t_1` as

.. figure:: drift_patterns.png
   :align: center
   :scale: 100%
   :figclass: wt

   Drift patterns depicted as the change of mean data values over time. Note that an outlier is not a change but *noise* in the data. This figure is based on :cite:`Gama2014Survey`. :label:`fig:driftpatterns` 

.. math::

   p_{t_0}(X,\vec{y}) \neq p_{t_1}(X,\vec{y}) 

Learning is known to be affected by the presence of concept drift :cite:`Gama2014Survey`. The following patterns, shown in Fig. :ref:`fig:driftpatterns`, are usually considered: 

- **Abrupt.** When a new concept is immediately introduced. The transition between concepts is minimal. In this case, adaptation time is vital since the old concept becomes is no longer valid.
- **Incremental.** Can be interpreted as the transition from an old concept into a new concept where intermediate concepts appear during the transition.
- **Gradual.** When old and new concepts concur within the transition period. Can be challenging since both concepts are somewhat valid during the transition.
- **Recurring.** If an old concept is seen again as the stream progresses. For example, when the data corresponds to a seasonal phenomenon such as the circadian rhythm.

Although the incremental nature of stream methods provides some robustness to concept drift, specialized methods have been proposed to detect drift. Multiple methods have been proposed in the literature, the authors in :cite:`Gama2014Survey` provide a thorough survey of this topic. In general, the goal of drift detection methods is to accurately detect changes in the data distribution while showing robustness to noise and being resources-wise efficient. Drift-aware methods use drift detection mechanisms to react faster and efficiently to changes. For example, the *Hoeffding Tree* algorithm :cite:`Domingos2000HT`, a kind of decision tree for streams, does not handle concept drift explicitly. An drift-aware version is the *Hoeffding Adaptive Tree* :cite:`Bifet2009HAT`, which uses *ADaptive WINdowing* (*ADWIN*) :cite:`Bifet2007ADWIN` to detect drifts. If a drift is detected at a given branch, an alternate branch is created and eventually replaces the original branch if it shows better performance on new data.

*ADWIN*, a popular drift detection method with mathematical guarantees, keeps a variable-length window of recent items; such that it holds that there has no been change in the data distribution. Internally, two sub-windows :math:`(W_0, W_1)` are used to determine if a change has happened. With each new item observed, the average values of items in :math:`W_0` and :math:`W_1` are compared to confirm that they correspond to the same distribution. If the distribution equality no longer holds, then an alarm signal is raised indicating that a drift has occurred. Upon detecting a drift, :math:`W_0` is replaced by :math:`W_1` and a new :math:`W_1` is initialized.

Performance evaluation
++++++++++++++++++++++

Predictive performance :math:`P` of a given model :math:`h` is usually measured using some loss function :math:`\ell` that evaluates the difference between expected (true) class labels :math:`y` and the predicted class labels :math:`\hat{y}`.

.. math::

   P(h) = \ell(y,\hat{y})

A popular and straightforward loss function for classification is the *zero-one loss function* which corresponds to the notion of weather the model made a mistake or not when predicting. 

.. math::

   \ell(y,\hat{y}) = \begin{cases} 0, & y = \hat{y} \\
                                  1, & y \neq \hat{y}
                     \end{cases}

Due to the incremental nature of stream leaning methods, special considerations are used to evaluate their performance. Two prevalent methods in the literature are *hold-out* and *prequential* evaluation. The hold-out evaluation is a popular method in both batch and stream learning where testing is performed on an independent set of samples. On the other hand, prequential evaluation :cite:`dawid1984prequential`, is specific to the stream setting. In prequential evaluation, tests are performed on new data samples *before* they are used to train (update) the model. The benefit of this approach is that all samples are used for both test and training.

Previous discussion is just an brief overview of machine learning for streaming data, we direct the reader to :cite:`Gomes2017` for an extensive and deeper description of this field, state-of-the-art and the current challenges.

``scikit-multiflow``
--------------------

scikit-mutliflow :cite:`skmultiflow` is a machine learning library for multi-output/multi-label and stream data written in Python. Developed under the principles of free and open source software and distributed under the BSD 3-Clause License. Following the **SciKits** philosophy, scikit-multiflow extends the existing set of tools for scientific purposes. It features a collection of state-of-the-art methods for classification, regression, concept drift detection and anomaly detection, alongside a set of data generators and evaluators. scikit-multiflow is designed to seemingly interact with NumPy :cite:`NumPy` and SciPy :cite:`SciPy` and is compatible with Jupyter Notebooks. Additionally, it contributes to the democratization of machine learning for data streams by leveraging the popularity of the Python language. scikit-multiflow is mainly written in Python, and some core elements are written in Cython :cite:`Cython` for performance.

scikit-multiflow is intended for users with different levels of expertise. Its design is intended to make it friendly to new users and familiar to more experienced ones. Its conception and development follow two main objectives:

1. To be easy to design and run experiments. This follows the need for a platform that allows fast prototyping and experimentation. Complex experiments can be setup using evaluation classes. Different data streams and models can be analyzed under multiple conditions, and the amount of implementation required by the user is kept to the minimum.
2. Easy to extend existing methods. For advanced users, existing methods can be extended and modified to create or enable new capabilities.

scikit-multiflow is not intended as a stand alone solution for machine learning. It integrates with other Python libraries such as Matplotlib :cite:`Matplotlib` for plotting, scikit-learn :cite:`scikit-learn`  for incremental learning [#]_ compatible with the streaming setting, Pandas :cite:`Pandas` for data manipulation, Numpy and SciPy for numerical and scientific computations. However, it is important to note that scikit-multiflow does not extend scikit-learn, whose main focus in on batch learning. A key difference is that estimators in scikit-multiflow are incremental by design and training is performed by calling multiple times the :code:`partial_fit()` method. The majority of estimators implemented in scikit-multiflow are instance-incremental, meaning single instances are used to update their internal state. A small number of estimators are batch-incremental, where mini-batches of data are used. On the other hand, calling :code:`fit()` multiple times on a scikit-learn estimator will result on it overwriting its internal state on each call.

.. [#] Only a small number of methods in scikit-learn are incremental.

As of version 0.5.0, the following sub-packages are available:

- :code:`anomaly_detection`: anomaly detection methods.
- :code:`data`: data stream methods including methods for batch-to-stream conversion and generators.
- :code:`drift_detection`: methods for concept drift detection.
- :code:`evaluation`: evaluation methods for stream learning.
- :code:`lazy`: methods in which generalization of the training data is delayed until a query is received, e.g., neighbors-based methods such as kNN.
- :code:`meta`: meta learning (also known as ensemble) methods.
- :code:`neural_networks`: methods based on neural networks.
- :code:`prototype`: prototype-based learning methods.
- :code:`rules`: rule-based learning methods.
- :code:`transform`: perform data transformations.
- :code:`trees`: tree-based methods,

In a nutshell
-------------

In this section we provide a quick overview of different elements of scikit-multiflow and show how to easily setup and run experiments in scikit-multiflow. Specifically, we provide examples of classification and drift detection.

Architecture
++++++++++++

Here we describe the basic components of scikit-multiflow. The ``BaseSKMObject`` class is the base class. All estimators in scikit-multiflow are created by extending the base class and the corresponding task-specific mixin(s): ``ClassifierMixin``, ``RegressorMixin``, ``MetaEstimatorMixin`` and ``MultiOutputMixin``.

The ``ClassifierMixin`` defines the following methods:

* ``partial_fit`` -- Incrementally train the estimator with the provided labeled data.
* ``fit`` -- Interface used for passing training data as batches. Internally calls ``partial_fit``.
* ``predict`` -- Predict the class-value for the passed unlabeled data .
* ``predict_proba`` -- Calculates the probability of a sample pertaining to a given class.

During a learning task, three main tasks are performed: data is provided by the stream, the estimator is trained on incoming data, the estimator performance is evaluated. In scikit-multiflow, data is represented by the ``Stream`` class, where the ``next_sample()`` method is used to request new data. The ``StreamEvaluator`` class provides an easy way to set-up experiments. Implementations for the hold-out and prequential evaluation methods are available. A stream and one or more estimators can be passed to an evaluator.

Classification task
+++++++++++++++++++

In this example we will use the SEA generator. A stream generator does not store any data, but generates it on demand. The ``SEAGenerator`` class creates data corresponding to a binary classification problem. The data contains 3 numerical features, from which only 2 are relevant for learning [#]_. We will use the data from the generator to train a Naive Bayes classifier. For compactness, the following examples does not include import statements and external libraries are referenced by standard aliases.

As previously mentioned, a popular method to monitor the performance of stream learning methods is the prequential evaluation. When a new data sample ``(X, y)`` arrives: 1. Predictions are obtained for the new data sample (X) to evaluate how well the model performs. 2. Then the new data sample ``(X, y)`` is used to train the model so it updates its internal state. The prequential evaluation can be easily implemented as a loop:

.. [#] Some data generators and estimator use random numbers generators. When set, the ``random_state`` parameter enforces reproducible results.

.. code-block:: python
   
   stream = SEAGenerator(random_state=1)
   classifier = NaiveBayes()

   n_samples = 0
   correct_cnt = 0
   max_samples = 2000

   # Prequential evaluation loop
   while n_samples < max_samples and \
   stream.has_more_samples():
       X, y = stream.next_sample()
       # Predict class for new data
       y_pred = classifier.predict(X)
       if y[0] == y_pred[0]:
           correct_cnt += 1
       # Partially fit (train) model with new data
       classifier.partial_fit(X, y)
       n_samples += 1

   print('{} samples analyzed.'.format(n_samples))   
   print('Accuracy: {}'.format(correct_cnt / n_samples))
   
   >> 2000 samples analyzed.
   >> NaiveBayes classifier accuracy: 0.9395

The previous example shows that the Naive Bayes classifier achieves an accuracy of 93.95%. However, it is important to remember that learning from data streams is a continuous task, so it is desirable to observe performance at multiple points over the stream.

.. figure:: experiment_1.png
   :align: center
   :scale: 60%
   :figclass: wt

   Performance comparison between ``NaiveBayes`` and ``SGDClassifier`` using the ``EvaluatePrequential`` class. :label:`fig:prequential`

The evaluate prequential method is implemented in the ``EvaluatePrequential`` class. This class provides extra functionalities including:

- Easy setup of different evaluation configurations
- Selection of different performance metrics
- Visualization of performance over time
- Ability to benchmark multiple models concurrently
- Saving evaluation results to a csv file

Let's run the same experiment on the SEA data but this time we will compare two classifiers: ``NaiveBayes`` and ``SGDClassifier`` (linear SVM with SGD training). We use the ``SGDClassifier`` in order to demonstrate the compatibility with incremental methods from scikit-learn.

.. code-block:: python
   
   stream = SEAGenerator(random_state=1)
   nb = NaiveBayes()
   svm = SGDClassifier()
   # Setup the evaluation
   metrics = ['accuracy', 'kappa',
              'running_time', 'model_size']
   eval = EvaluatePrequential(show_plot=True,
                              max_samples=20000,
                              metrics=metrics)
   # Run the evaluation
   eval.evaluate(stream=stream, model=[nb, svm],
                        model_names=['NB', 'SVM']);

During the evaluation, a dynamic plot displays the performance of both estimators over the stream, Fig. :ref:`fig:prequential`. Once the evaluation is completed, a summary is displayed in the terminal. For this example and considering the evaluation configuration::

   Processed samples: 20000
   Mean performance:
   NB - Accuracy     : 0.9430
   NB - Kappa        : 0.8621
   NB - Training time (s)  : 0.56
   NB - Testing time  (s)  : 1.31
   NB - Total time    (s)  : 1.87
   NB - Size (kB)          : 6.8076
   SVM - Accuracy     : 0.9560
   SVM - Kappa        : 0.8984
   SVM - Training time (s)  : 4.70
   SVM - Testing time  (s)  : 1.73
   SVM - Total time    (s)  : 6.43
   SVM - Size (kB)          : 3.4531

In Fig. :ref:`fig:prequential`, we observe the evolution of both estimators as they are trined on data from the stream. Although ``NaiveBayes`` has better performance at the beginning of the stream, ``SGDClassifier`` eventually outperforms it. In the plot we show performance measured by a given metric (accuracy, kappa, etc.) in two ways: *Mean* corresponds to the performance over the entire stream, resulting in a smooth line. *Current* indicates the performance over a sliding windows with the latest data from the stream, The size of the sliding window can be defined by the user and is useful to analyze the 'current' performance of an estimator. In this experiment we also measure resources in terms of time (training + testing) and memory. ``NaiveBayes``is faster and uses slightly more memory. On the other hand, ``SGDClassifier`` is slower and has a smaller memory footprint.

Concept drift detection
+++++++++++++++++++++++

For this example, we will generate a synthetic data stream. The first half of the stream (500 samples) contains a sequence corresponding to a normal distribution with :math:`\mu=0.6`, :math:`\sigma=0.1` and the second half (500 samples) is a normal distribution with :math:`\mu=0.4`, :math:`\sigma=0.1`. We transform the values in the stream to binary values [0, 1] to simulate correct/incorrect predictions in a classification task. The distribution of data in the described synthetic stream is shown in Fig. :ref:`fig:drift`.

.. figure:: synthetic_drift.png
   :scale: 200%

   Synthetic data simulating a drift. The stream is composed by two distributions of 500 samples. :label:`fig:drift`

.. figure:: experiment_2.png
   :align: center
   :scale: 60%
   :figclass: w

   Benchmarking the Hoeffding Tree vs the Hoeffding Adaptive Tree on presence of drift. :label:`fig:trees`

.. code-block:: python

   dist_a = np.random.normal(0.6, 0.1, 500)
   dist_b = np.random.normal(0.4, 0.1, 500)
   stream = np.rint(np.concatenate((dist_a, dist_b)))

We will use the ADaptive WINdowing (ADWIN) drift detection method. The goal is to detect that a drift has occurred after sample 500 in the synthetic data stream.

.. code-block:: python

   drift_detector = ADWIN()

   for i, val in enumerate(stream_int):
      drift_detector.add_element(val)
      if drift_detector.detected_change():
         print('Change detected at index {}'.format(i))
         drift_detector.reset()
   
   >> Change detected at index 575

Impact of drift on learning
+++++++++++++++++++++++++++

Concept drift can have a significant impact on predictive performance if not handled properly. Most batch models will fail in presence of drift as they are essentially trained on different data. On the other hand, stream learning methods continuously update themselves and can adapt to new concepts. Furthermore, drift-aware methods use change detection methods to trigger mitigation mechanisms if a change in performance is detected.

In this example we compare two popular stream models: the ``HoeffdingTreeClassifier`` and its drift-aware version ``HoeffdingAdaptiveTreeClassifier``.

For this example we will load the data from a csv file using the ``FileStream`` class. The data corresponds to the output of the ``AGRAWALGenerator`` with 3 gradual drifts at the 5k, 10k and 15k marks. A gradual drift means that the old concept is gradually replaced by a new one, in other words, there exists a transition period in which the two concepts are present.

.. code-block:: python

   stream = FileStream("agr_a_20k.csv")
   ht = HoeffdingTreeClassifier(),
   hat = HoeffdingAdaptiveTreeClassifier()
   # Setup the evaluation
   metrics = ['accuracy', 'kappa', 'model_size']
   eval = EvaluatePrequential(show_plot=True,
                              metrics=metrics,
                              n_wait=100)
   # Run the evaluation
   eval.evaluate(stream=stream, model=[hy, hat],
                 model_names=['HT', 'HAT']);

The summary of the evaluation is::

   Processed samples: 20000
   Mean performance:
   HT - Accuracy     : 0.7279
   HT - Kappa        : 0.4530
   HT - Size (kB)          : 175.8711
   HAT - Accuracy     : 0.8070
   HAT - Kappa        : 0.6122
   HAT - Size (kB)          : 122.0986

The result of this experiment is shown in Fig. :ref:`fig:trees`. During the first 5K samples, we see that both methods behave in a very similar way, which is expected as the ``HoeffdingAdaptiveTreeClassifier`` essentially works as the ``HoeffdingTreeClassifier`` when there is no drift. At the 5K mark, the first drift is observable by the sudden drop in the performance of both estimators. However, notice that the ``HoeffdingAdaptiveTreeClassifier`` has the edge and recovers faster. The same behavior is observed after the drift in the 15K mark. Interestingly, after the drift at 10K, the ``HoeffdingTreeClassifier`` is better for a small period but is quickly overtaken. In this experiment we can also see that the *current* performance evaluation provides richer insights on the performance of each estimator. It is worth noting the difference in memory between these estimators. The ``HoeffdingAdaptiveTreeClassifier`` achieves better performance while requiering less space in memory. This indicates that the branch replacement mechanisms triggered by ADWIN has been applied, resulting in a less complex tree structure representing the data.

Note that the volume of data in the previous examples is for illustrative purposes only. Real streaming data applications usually are exposed to data in the magnitude of millions of samples.

Get ``scikit-multiflow``
------------------------

scikit-multiflow work with Python 3.5+ and can be used on Linux, macOS and Windows systems. The source code is publicly available in a GitHub. The stable release version is available via ``conda-forge`` (recommended) and ``pip``:

.. code-block:: console

   $ conda install -c conda-forge scikit-multiflow

   $ pip install -U scikit-multiflow

The latest development version is available in the project's repository: https://github.com/scikit-multiflow/scikit-multiflow. Stable and development version are also available as ``docker`` images.

Conclusions and final remarks
-----------------------------

In this paper, we provide a brief overview of machine learning for data streams. Stream learning is an alternative to standard batch learning in dynamic environments where data is continuously generated (potentially infinite) and data is non-stationary but evolves (concept drift).  We present examples of applications, and describe the challenges and requirements of machine learning techniques to be used on streaming data in an effective and efficient manner. 

We also describe ``scikit-multiflow``, an open source machine learning library for data streams in Python. The design of scikit-multiflow is based on two principles: to be easy to design and run experiments, and to be easy to extend and modify existing methods. We provide a quick overview of the core elements of scikit-multiflow and show how it can be used for the tasks of classification and drift detection.

Acknowledgements
----------------

The author is particularly grateful to Prof. Albert Bifet from the Department of Computer Science at the University of Waikato for his continuous support. We also thank Saulo Martiello Mastelini from the Institute of Mathematics and Computer Sciences at the University of São Paulo, for his ongoing collaboration on scikit-multiflow and his valuable work as one of the maintainers of the project. We thank interns who have contributed to scikit-multiflow and the open source community who helps and motivate us to improve this project. We gratefully acknowledge constructive comments of the reviewers. 