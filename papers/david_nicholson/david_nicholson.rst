:author: David Nicholson
:email: dnicho4@emory.edu
:institution: Emory University

---------------------------------------------------------------------------------
Comparison of machine learning methods applied to birdsong element classification
---------------------------------------------------------------------------------

.. class:: abstract

    Songbirds provide neuroscience with a model system for understanding how the brain learns and produces
    motor skills, such as speech. Similar to humans, songbirds learn their vocalizations from social 
    interactions during a critical period in development. Each bird’s song consists of repeated elements 
    referred to as “syllables”. To analyze song, scientists label syllables by hand, but songbirds can 
    produce hundreds of songs a day, many more than can be labeled. Several groups have developed machine 
    learning algorithms that automate labeling of syllables, but little work has been done comparing these 
    various methods. For example, there are methods using support vector machines (SVM), K-nearest 
    neighbors (KNN), and even a hybrid deep learning system to automate labeling of Bengalese finch song (a 
    species whose behavior has made it the subject of an increasing number of neuroscience studies). Here I 
    compare approaches to classifying Bengalese finch syllables (building on my previous work 
    [https://youtu.be/ghgniK4X_Js]). I propose that the best method is the one that yields the highest accuracy
    from the smallest amount of hand-labeled data. My results show that the previously published SVM method is 
    impaired by syllables in some individuals’ songs, but I can get it to outperform KNN by using a non-
    linear kernel. I then show domain-specific features can be used to further increase accuracy, and that 
    this increase is large enough to affect analysis for data sets of the size used in actual experiments. 
    Testing of machine learning algorithms was carried out using Scikit-learn and Numpy/Scipy via Anaconda. 
    This paper in Jupyter notebook form, as well as code and links to data, are here: 
    https://github.com/NickleDave/ML-comparison-birdsong
    

.. class:: keywords

    machine learning,birdsong,scikit-learn

Introduction
------------

Songbirds provide a model system for the study of learned vocalizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like humans, songbirds learn to vocalize during a critical period in development. During that critical period, they require social interactions, sensory feedback, and practice to learn their vocalizations--again, just like humans. Songbirds have a network of brain areas dedicated to learning and producing their song, known as the song system. These brain areas occur only in songbirds, not in birds that do not learn their song. At the same time, the bird brain contains most of the major regions found in the human brain, and studies of songbirds have changed our understanding of certain brain areas, e.g., the basal ganglia. Because of these shared behaviors and brain areas, songbirds provide an excellent model system through which we can understand how the brain learns and produces motor skills like speech that are learned during a critical period in development.

Machine-learning methods for labeling elements of song
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each bird’s song consists of repeated elements referred to as “syllables”:

.. figure:: spectrogram.png
    :align: center
    :figclass: w

    Figure 1. Spectrogram of Bengalese finch song. 
    Letters below the time axis, e..g, "i","a","b",..., are labels for syllables,
    discrete elements of song separated by brief silent intervals.
    Frequency (kHz) on the y axis and time on the x axis. :label: fig1

Each individual has a unique song which usually bears some similarity to the song of the bird that tutored it, but is not a direct copy. To analyze song, experimenters label syllables by hand. However, songbirds produce thousands of songs a day, more than can be labeled.

In order to deal with this mountain of data, some labs have developed automated analyses of birdsong. One popular approach scores songs based on similarity of spectrograms, without labeling syllables [TCHER2000]. Another method uses semi-automated clustering to label syllables, and then measures changes in acoustic and temporal structure of song over days using a distance metric [WU2008]. Other approaches make use of standard supervised learning algorithms to classify syllables, such as hidden Markov Models [KOGAN2008]. While code for some of these automated analyses is freely available, and there are some repositories of song on-line, to my knowledge almost no work has been done to compare the different methods.

I set out to compare methods using one species, the Bengalese finch. This species is of interest for several reasons. For example, Bengalese finches depend heavily on auditory feedback throughout life to maintain their vocalizations, much like humans ([SOBER2009] and references therein). In addition, their song tends to have relatively easy-to-quantify acoustic features (e.g., many of the syllables are "low entropy", having a pitchy, whistle-like timbre). Several previously-published studies or open-sourced libraries have applied various machine learning techniques to Bengalese finch song, including: support vector machines (SVMs) [TACH2014], and k-Nearest Neighbors (kNNs) (http://www.utsa.edu/troyerlab/software.html). Again, no study has compared these methods with open-source code and freely shared data.

Results
----------



Conclusion
----------

The results suggest SVM-RBF provides the highest accuracy across different birds' songs. For three of four birds, labeling as few as fifteen songs by hand provides enough training data to achieve greater than 99.2% average accuracy with SVM-RBF. This is approximately 40 seconds of song (assuming 50 milliseconds per syllable and 50 syllables per song), in comparison to previous papers that reported using a minute of song to achieve such accuracies. The success of SVM-RBF is likely because the kernel allows for non-linear decision boundaries that can separate more variable types of syllables, such as the low-amplitude, high entropy "intro" syllables. Further experiments should determine how well these algorithms deal with the presence of sounds that are not part of song, e.g., calls, movement of the bird on its perch, etc. Such experiments are necessary to further reduce the amount of work required on the part of the experimenter. I expect that the SVM-RBF will succeed in the face of these non-songbird elements as well. As the results above demonstrate, it is important to actually test how the algorithms compare with each other on varied data sets, and to present the results and code used to obtain those results in as accessible a manner as possible.


Methods
----------


References
----------
[TCHER2000] Tchernichovski, Ofer, et al. "A procedure for an automated measurement of song similarity." Animal Behaviour 59.6 (2000): 1167-1176.

[WU2008] Wu, Wei, et al. "A statistical method for quantifying songbird phonology and syntax." Journal of neuroscience methods 174.1 (2008): 147-154.

[KOGAN2008] Kogan, Joseph A., and Daniel Margoliash. "Automated recognition of bird song elements from continuous recordings using dynamic time warping and hidden Markov models: A comparative study." The Journal of the Acoustical Society of America 103.4 (1998): 2185-2196.

[SOBER2009] Sober, Samuel J., and Michael S. Brainard. "Adult birdsong is actively maintained by error correction." Nature neuroscience 12.7 (2009): 927-931.

[TACH2014] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-automatic classification of birdsong elements using a linear support vector machine." PloS one 9.3 (2014): e92584.
