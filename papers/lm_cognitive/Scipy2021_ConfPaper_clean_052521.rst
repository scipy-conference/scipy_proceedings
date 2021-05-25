:author: Amanda E. Kraft
:email: amanda.e.kraft@lmco.com
:institution: Lockheed Martin Advanced Technology Laboratories
:equal-contributor:

:author: Matthew Widjaja
:email: matthew.widjaja@lmco.com
:institution: Lockheed Martin Advanced Technology Laboratories
:corresponding:
:equal-contributor:

:author: Trevor M. Sands
:email: trevor.m.sands@lmco.com
:institution: Lockheed Martin Advanced Technology Laboratories

:author: Brad J. Galego
:email: brad.j.galego@lmco.com
:institution: Lockheed Martin Advanced Technology Laboratories

-----------------------------------------------------------------------------
Programmatically Identifying Cognitive Biases Present in Software Development
-----------------------------------------------------------------------------

.. class:: abstract

   Mitigating bias in AI-enabled systems is a topic of great concern within the
   research community. While efforts are underway to increase model
   interpretability and de-bias datasets, little attention has been given to
   identifying biases that are introduced by developers as part of the software
   engineering process. To address this, we began developing an approach to 
   identify a subset of cognitive biases that may be present in development 
   artifacts (e.g., version control commit messages): anchoring bias, 
   availability bias, confirmation bias, and hyperbolic discounting. We 
   developed multiple natural language processing (NLP) models to identify and 
   classify the presence of bias in text originating from software development 
   artifacts.

.. class:: keywords

   cognitive bias, software engineering, natural language processing

Introduction
============

Artificial Intelligence- (AI) and Machine Learning- (ML) based systems are
increasingly supporting decision-making, reasoning, and evaluation of rapidly
changing environments in objective manners. As AI-enabled systems are finding
increasing use across domains and industries, there is concern that the
objectivity of such systems may be negatively impacted by biases introduced by
the developers either in the design of the system or in the training data
itself. Though efforts are underway to make AI/ML systems more explainable and
debias datasets, little research is directed at human-centric cognitive biases
that developers unintentionally introduce as a part of the software engineering
(SE) process. As a result, ensuring unbiased and transparent algorithmic
decision-making is a complex challenge for AI creators and causes wide-ranging
implications for the future use of AI in society.

Cognitive biases are systematic deviations from rationality in judgment,
reasoning, evaluation, or other cognitive processes. There are hundreds of
biases that have been described across fields of research [1]_, approximately 40
of which have been investigated in the SE domain [2]_. We selected four of the
most commonly reported cognitive biases in software engineering to focus our
analysis on: 

* **Anchoring Bias**: Tendency to rely too heavily on pre-existing or first
  information found when making a quantitative judgment [2]_.
* **Availability Bias**: Tendency to overestimate the likelihood of events based
  on the ease of which examples come to mind [3]_.
* **Confirmation Bias**: Tendency to search for and focus on information that
  confirms one’s preconception(s) while ignoring or rejecting sources that
  challenge it [4]_.
* **Hyperbolic Discounting**: Tendency to prefer immediate payoffs over larger
  rewards at a later point [2]_.

Each of these biases may be influenced by self-generated factors, such as past
development experience, or externally generated factors, such as system
documentation [5]_. A tool to detect biases in software must be capable of
assessing multiple sources of information about the system, including commit
messages, comments, in-source docstrings, external technical documentation, and
diagrams. This study takes the first steps toward this objective by identifying
cognitive biases in software commit messages and comments from previously
completed projects.

Research Methods
================

Data Curation
-------------

To address the lack of research identifying cognitive biases in software
artifacts developed as part of a naturally occurring development process, we
collated data from two internally developed codebases. The first project
(“Project A”) was selected to represent capturing the whole software engineering
process for AI/ML-enabled systems (data management to feature extraction to
model training and evaluation); the second project (“Project B”) is similar in
structure to the first project, but the software artifacts gathered only include
the latter half of the development cycle (feature extraction to model training
and evaluation). The contents from both codebases were collated into datasets
based on the source of the development artifacts: commit messages, in-source
code comments, and documentation strings (docstrings). Given the time
limitations for the project, the software artifact types were prioritized where
commit messages were annotated for all three datasets, while comments and
docstrings were annotated for the second of the two internal projects. 

To validate models trained on the content from the internal projects, we
identified an open-source dataset: Code Smell [6]_; this dataset contains commit
messages extracted from refactoring activities across the software lifecycle for
various open-source software projects. 

Python scripts were developed to programmatically extract and format the text
content from the artifacts. Specifically, the following operations were
performed: commit message content had whitespace trimmed and artifact
identifiers removed; comments spanning multiple lines were combined into a
single entry; executable segments of code were removed; entries with non-ASCII
characters were removed; references to individual names, collaboration teams,
applications, and projects were redacted and replaced with an identifier string
(e.g., “NAME1”).

Bias Annotation: Prodigy Setup
------------------------------

The processed text data was then annotated in Prodigy to produce a structured
JSON document. Prodigy  is a commercially licensed software tool for data
annotation and allows for multiple individuals to collaborate on labeling text
entries. A custom annotation recipe was developed to define the four biases
described above as the possible labels; an additional label option,
“Subjective/Other” was included to provide reviewers a chance to flag entries
containing a form of bias other than the available options. Figure 1 provides an
example of what individual reviewers see when annotating a given dataset using
this custom recipe. For each entry, the reviewer must decide whether an entry is
valid, and if so, if the language indicates that the author may have introduced
bias into the system. When reviewers determined an entry contains bias, they
selected one or more labels and pressed “accept”; otherwise, the reviewer
pressed “reject” to indicate no language indicating bias was present. 

.. figure:: media/image1.png
   :scale: 60%
   
   Example view of a comment in reviewer mode; this particular comment 
   could be rejected as not containing bias.

Bias Annotation: Manual Annotation
----------------------------------

A total of six reviewers were engaged in this project, where each reviewer had
at least two years of programming experience. An annotation guide for
classifying open-ended text entries was developed for reviewers to remain
consistent. The guide provides examples of several biased commit messages such
as:

* Anchoring Bias

  * "Extended module to allow a more traditional approach to interface
    engineering"
  * "Applying back-changes from my original fix patch"
  * "Correct the temperature unit - assumes anything under 45 is C"

* Availability Heuristic

  * "Renamed method to more sensible wording”
  * "Tighter coupling of variable names with other modules"

* Confirmation Bias

  * "The use of [X] rather than [Y] allows each module to reuse the same
    functionality without having to extend a base class"
  * "We're now a bit smarter about the size of tables that we create by default,
    which was the root of the prior problems"

* Hyperbolic Discounting

  * "Throwing out the Key and Value classes for now to reduce the overall
    complexity"
  * "Modified function to account for type errors. Will likely have to recreate
    the db every time, unless other solutions come up"
  * "Module incorporated but fails"
  * "Quick and dirty method to add features"

* Subjective/Other

  * "I was too over-zealous with removing a module"
  * "Duplicate code is my nemesis..."


The guide reminds reviewers that they are to label if the language indicates the
author may have introduced bias into the system, not if the language indicates
the author may be addressing bias previously introduced. The guide further
advises the reviewer to flag entries as invalid if they should be excluded from
the training or testing datasets; the exclusion criteria include blank messages,
machine-generated messages (e.g., automated branch merging messages), messages
only containing an artifact or issue identifier, and “TODO” or “FIXME” comments
with no accompanying description. Reviewers were also encouraged to accept
samples that may be borderline cases, as a group consensus would decide final
classification labels. 

Bias Annotation: Finalizing Bias Labels
---------------------------------------

After all reviewers submitted their final annotations for a dataset, one
reviewer was selected to finalize the labels to be used for training and testing
models. For consistency, the same reviewer was selected to finalize labels on
all datasets. The review process itself was facilitated by Prodigy, which offers
a built-in review recipe, allowing a user to specify the annotation databases to
be compiled. With this recipe, Prodigy extracts all instances where an entry was
marked as “accepted” or “ignored” by at least one reviewer. These are compiled
and displayed similar to the initial review, noting which review session(s)
indicated which label(s).

In the final review, a “best fit” label was selected, rather than accepting
multiple labels for a single entry as allowed in the initial review stage. This
decision was implemented in order to provide non-overlapping classification
boundaries for model training and testing. The final reviewer followed a set of
guidelines for determining best fit labels, such as cross-referencing the
annotation guide or identifying the word or phrase that may have triggered the
response when multiple reviewers selected different biases for a single message.

If the final reviewer thought the best fit label was ambiguous or if the label
selected was only reported by themselves during the initial review process, the
message was logged for additional review. These flagged messages were compiled
in an Excel workbook along with the selected answer (First degree label), the
next best answer (Second degree label), and the labels marked by the initial
review sessions. The workbook was sent to at least two individuals to respond to
these entries, indicating their judgement of whether the first or second degree
label was the best fit or if another label option may have been overlooked.
Scoring of their responses was automated using the following rules: (1) if both
agreed with the first degree label, it was kept; (2) if both agreed with the
second degree label, the final label was switched; (3) if the first degree label
was not “reject” and one accepted while the other rejected, the first degree
label was kept. On the rare occasion when none of these conditions were met, the
final reviewer decided the label selection based on the feedback.

The results of the final review (i.e., entries labeled as biased) were merged
with the source dataset (i.e., non-biased entries) to comprise the training and
testing datasets for modeling. 

Models
------
To determine whether a tool can classify software artifacts as containing
indicators of bias, we developed text classification models using spaCy. Binary
and multi-class models were considered, where binary models were concerned with
identifying the presence or absence of biased language and multi-class models
concerned with identifying the type of bias present (if one is present at all).
Anticipating that the class distributions would be highly imbalanced towards not
containing bias, we implemented down sampling by taking the mean of the quantity
of data present across each label type to improve model training. This method
was randomized, with ten models trained on different data distributions.

Focusing on the ability of the trained models to perform on different codebases,
we prioritized evaluating the models independently trained on the two internal
commit datasets and applied each to the Code Smell dataset. As a secondary task,
we then combined the internal commits in a single training set and applied them
to Code Smell. Additionally, to determine if commit messages can predict bias in
comments, we trained a model on the internal commits and tested against comments
for the same project. Finally, we evaluate the combined internal dataset against
Code Smell.

For each model, we report the mean F1 score and standard deviation. During
training, three model hyperparameters are considered: the maximum number of
samples used to train, the percentage of data to drop, and the size of the
training batches and their compounding rate at each epoch. The maximum sample
capacity mitigates the impact of the imbalance in labels by limiting the total
number of positive and negative samples. Dropout denotes the percentage of
connections that are dropped from the neural network component of the ensemble
learners and is used as a mechanism to prevent over-training the model; typical
sweep values are 20%, 40%, and 60%. Finally, we used batching to determine how
much data is passed to the trainer at each iteration from a minimum (batch start
size) to a maximum (batch stop size) with a given rate of growth (compounding
rate). For all models, the compounding rate was left at the spaCy recommended
value of 1.001. The modeling process is illustrated in Figure 2.

.. figure:: media/image2.png
   :scale: 30%
   
   Overview of the modeling workflow.

Results and Discussion
======================

Annotated Datasets 
------------------
An overview of the four datasets in terms of total number of items, number of
duplicate entries, final number of items after accounting for duplicates, and
number of reviewers to annotate is provided in Table 1.    

.. raw:: latex

   \begin{table}[]
   \centering
   \footnotesize
   \begin{tabular}{|l|c|c|c|c|}
   \hline
   \textbf{Dataset}                                               & \multicolumn{1}{c|}{\textbf{\begin{tabular}[c]{@{}c@{}}Total\\ Items\end{tabular}}} & \multicolumn{1}{c|}{\textbf{\begin{tabular}[c]{@{}c@{}}Duplicate\\ Items\end{tabular}}} & \multicolumn{1}{c|}{\textbf{\begin{tabular}[c]{@{}c@{}}Final Item\\ Count\end{tabular}}} & \multicolumn{1}{c|}{\textbf{Reviewers}} \\ \hline
   \begin{tabular}[c]{@{}l@{}}Code Smell\\ Commits\end{tabular}   & 471                                                                                & 30                                                                                     & 441                                                                                     & 5                                      \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A\\ Commits\end{tabular}    & 1536                                                                               & 131                                                                                    & 1405                                                                                    & 6                                      \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Commits\end{tabular}    & 238                                                                                & 11                                                                                     & 227                                                                                     & 5                                      \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Comments\end{tabular}   & 469                                                                                & 0                                                                                      & 469                                                                                     & 5                                      \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Docstrings\end{tabular} & 181                                                                                & 0                                                                                      & 181                                                                                     & 5                                      \\ \hline                                      
   
   \end{tabular}
   \caption{Overview of dataset entries and reviewers.}
   \label{tab:table1}
   \end{table}

.. raw:: latex

   \begin{table}[]
   \centering
   \footnotesize
   \begin{tabular}{|l|c|c|c|c|}
   \hline
   \textbf{Dataset}                                               & \textbf{\begin{tabular}[c]{@{}c@{}}Answer\\ \end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Annotation\\ \end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Sub\\ Annotation\\ \end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Bias\\ \end{tabular}} \\ \hline
   \begin{tabular}[c]{@{}l@{}}Code Smell\\ Commits\end{tabular}   & 0.85 ± 0.23                                                           & 0.83 ± 0.28                                                               & 0.44 ± 0.19                                                                     & 0.22 ± 0.35                                                         \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A\\ Commits\end{tabular}    & 0.86 ± 0.21                                                           & 0.87 ± 0.24                                                               & 0.50 ± 0.20                                                                     & 0.39 ± 0.40                                                         \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Commits\end{tabular}    & 0.78 ± 0.24                                                           & 0.89 ± 0.24                                                               & 0.43 ± 0.21                                                                     & 0.35 ± 0.38                                                         \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Comments\end{tabular}   & 0.91 ± 0.19                                                           & 0.92 ± 0.20                                                               & 0.51 ± 0.17                                                                     & 0.43 ± 0.48                                                         \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Docstrings\end{tabular} & 0.95 ± 0.15                                                           & 0.94 ± 0.16                                                               & 0.51 ± 0.15                                                                     & 0.42 ± 0.49                                                         \\ \hline
   \end{tabular}
   \caption{Interrater reliability across datasets with standard deviation.}
   \label{tab:table2}
   \end{table}

.. raw:: latex

   \begin{table}[]
   \centering
   \footnotesize
   \begin{tabular}{|l|c|c|c|c|}
   \hline
   \textbf{Dataset}                                               & \textbf{\begin{tabular}[c]{@{}c@{}}Total\\ Items\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Rejected\\ (Not Biased)\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Accepted\\ (Biased)\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Ignored\\ (Excluded)\end{tabular}} \\ \hline
   \begin{tabular}[c]{@{}l@{}}Code Smell\\ Commits\end{tabular}   & 441                                                            & 389                                                                      & 51                                                                   & 1                                                                     \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A\\ Commits\end{tabular}    & 1,405                                                          & 1,154                                                                    & 162                                                                  & 89                                                                    \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Commits\end{tabular}    & 227                                                            & 140                                                                      & 26                                                                   & 61                                                                    \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Comments\end{tabular}   & 469                                                            & 430                                                                      & 27                                                                   & 12                                                                    \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Docstrings\end{tabular} & 181                                                            & 174                                                                      & 7                                                                    & 0                                                                     \\ \hline
   \end{tabular}
   \caption{Overview of Final Annotations.}
   \label{tab:table3}
   \end{table}

.. figure:: media/image3.png
   :scale: 50%
   
   Interrater reliability across datasets. Error bars show standard
   deviation in the reliability scores.

To quantify variance in interpretation of bias presentation in software commit
messages and comments, interrater reliability was computed based on percent
agreement across reviewers. Percent agreement is computed as the number of
matching pairs over the number of total possible pairs.

For answer reliability, the number of matching answer (i.e., “accept”, “reject”,
or “ignore”) pairs is divided by the total number of possible pairs. For label
reliability, we start with the high-level measure of all label options,
including the empty label string that results from selection of “reject” or
“ignore”. We refer to this measure as annotation reliability, as it accounts for
a combination of answer and label selection, though at the cost of instances of
“reject” and “ignore” being indistinguishable. Given the expected imbalance of
bias versus non-biased entries, we also provide an average of the reliability
scores for the subset in which at least one bias label is selected. We refer to
this measure as sub-annotation reliability. Lastly, we compute a bias
reliability measure in which we compare only the label options available when a
reviewer “accepts” an entry as biased.

There were six reviewers for the Project A Commits dataset and five reviewers
for all other datasets. Interrater reliability was computed across reviewer
annotations and are summarized in Figure 3 and Table 2. The distributions of
bias labels for each dataset are represented in Figure 4 and Figure 5. Overall,
reliability measures ranged from 0.78 to 0.91 for answers, 0.83 to 0.92 for
annotations, 0.43 to 0.51 for sub-annotations, and 0.22 to 0.43 for bias labels
across the four datasets. An overview of the final annotation labels is provided
in Table 3.


Given the nature of the data being annotated, we expected a significant amount
of variance in how reviewers interpret commit messages and in-source comments,
especially without additional context about the relevant code. This was
confirmed with the interrater reliability for top-level answers (i.e., accept,
reject, or ignore) averaging to 85% agreement, while reliability on bias type
averaged to 35%. 

.. figure:: media/stacked4.png
   :scale: 115%
   
   Distribution of bias labels selected by reviewers per dataset

.. figure:: media/image5.png
   :scale: 50%
   
   Final Bias Label Distribution.

Modeling
--------
Table 4 summarizes the results for each model, along with the best-preforming
hyperparameters as determined by a parameter sweep. Each model was trained using
10 different training set distributions. No models were trained for the dataset
composed of docstrings, due to only 7 (of 181) entries being labeled as biased. 

.. raw:: latex

   \begin{table}[]
   \centering
   \begin{tabular}{|l|c|c|c|c|c|c|}
   \hline
   \textbf{Dataset} & \textbf{\begin{tabular}[c]{@{}c@{}}Model\\ Type\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Max\\ Samples\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Drop\\ Rate\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Batch\\ Range\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Mean\\ F1\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Std.\\ Dev.\end{tabular}} \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A\\ Commits\end{tabular} & Binary & 220 & 40\% & 4-64 & 81.2\% & 2.6\% \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Commits\end{tabular} & Binary & 28 & 20\% & 8-64 & 65.9\% & 14.0\% \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A + \\ B Commits\end{tabular} & Binary & 247 & 20\% & 4-64 & 79.0\% & 5.1\% \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project A + \\ B Commits\end{tabular} & \begin{tabular}[c]{@{}c@{}}Multi\\ Label\end{tabular} & 188 & 20\% & 8-32 & 72.1\% & 5.8\% \\ \hline
   \begin{tabular}[c]{@{}l@{}}Project B\\ Commits +\\ Comments\end{tabular} & Binary & 104 & 40\% & 8-32 & 78.6\% & 6.8\% \\ \hline
   \begin{tabular}[c]{@{}l@{}}All Internal\\ Data\end{tabular} & Binary & 324 & 40\% & 8-64 & 82.3\% & 3.9\% \\ \hline
   \end{tabular}
   \caption{Hyperparameters selected and results for each model.}
   \label{tab:table4}
   \end{table}

All binary classification models performed in parity with one another, with mean
F1 scores ranging from 78.6% to 82.3%, with the exception of the model trained
on the Project B Commits data only, which performed at 65.9%, most likely due to
the significantly smaller size of the training dataset. The best performing
model (F1 = 82.3%) was trained using the largest dataset (i.e., the combined
commit messages and comments for both Projects A & B).

The multi-label model (F1 = 72.1%) consistently predicted that no bias was
present. This model was over-trained because the former category of biased
entries were now split among four bias types, which meant the data that was
not biased was significantly larger in size. This issue in data distribution,
plus interrater reliability, negatively impacted the multi-label model.


Conclusions & Implications
==========================

Through this project, two well-curated datasets were generated: one derived from
the commit messages of Projects A & B and the other created by labeling an
existing collection of code refactoring-related commit messages from various
free and open-source software projects [6]_. This data is valuable not only 
because it is the first of its kind, but also because it is representative of 
technical artifacts generated during the software development process.

The level of variability in bias annotations across reviewers emphasizes the
difficulty in discerning whether a statement is biased without insight of the
surrounding code and/or context.  This is further exacerbated when it comes to
type of bias. Perhaps the level of disagreement could be mitigated by a more
explicit guide or additional training. Further, limiting reviewers to a single
annotation per entry may alleviate the risk of reviewers selecting multiple
labels when uncertain rather than referring to the annotation guide to determine
the best fit. Our interrater reliability inherently resulted in lower scores for
multi-label annotations. For example, ['ANCHORING,HYPERBOLIC'] and
['HYPERBOLIC'] results in bias reliability of 0 even though both individuals
thought hyperbolic discounting was present. The level of variation may also
arise from individual differences in writing commit messages and comments –
messages that are longer or enumerate each change made are more likely to elicit
language suggestive of bias compared to bare minimum worded messages. Properly
identifying bias in software artifacts may require consideration for informing
software teams on message structuring for consistency and utility.

Future research efforts that can build on these results include the generation
of datasets and models that consider the impact of individual words or short
phrases on bias classification, application of such a tool in tracing the source
of a significant failure to the engineering process (e.g., as opposed to line of
code), and investigation of the impact of cognitive bias on code quality
metrics. Additionally, larger datasets, especially ones containing in-code
comments and document strings, are necessary to quantify the impact of cognitive
biases on the quality of finished software systems. In the future, larger
projects may require the development of post-mortem reports to identify which
aspects of the research, design, and development cycles are most impactful to
overall project success or failure. With such data available, researchers can
begin to answer the central question regarding the impact of individual biases
from a holistic perspective.

Acknowledgements
================

We thank Michael Krein, Lisa Baraniecki, and Owen Gift for their contributions
to annotating the datasets used in this effort.



.. [1] M. Delgado-Rodriguez and J. Llorca, *Bias*,
           Journal of Epidemiology & Community Health, 58(8):635-641, 2004.
.. [2] R. Mohanani, I. Salman, B. Turhan, P. Rodríguez and P. Ralph,
           *Cognitive biases in software engineering: a systematic mapping study*,
           IEEE Transactions on Software Engineering, 2018. 
.. [3] W. Stacy and J. MacMillan, *Cognitive bias in software engineering*,
           Communications of the ACM, 38(6):57-63, 1995. 
.. [4] G. Calikli and A. Bener,
           *Empirical analysis of factors affecting confirmation bias levels of software engineers*,
           Software Quality Journal, 23(4):695-722, 2015. 
.. [5] K. Mohan and R. Jain,
           *Using traceability to mitigate cognitive biases in software development*,
           Communications of the ACM, 51(9):110-114, 2008. 
.. [6] E. AlOmar, M. W. Mkaouer and A. Ouni,
           *Can refactoring be self-affirmed? an exploratory study on how developers document their refactoring activities in commit messages*,
           IEEE, no. 2019 IEEE/ACM 3rd International Workshop on Refactoring (IWoR),
           2019. 