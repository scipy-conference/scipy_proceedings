---
title: "Bridging the Technical Gap: A Student-Led RAG Pipeline for Community-Driven Document Analysis"
---

(abstract)=
## Abstract
This paper presents the impacts of using Large Language Models (LLMs) to extract and process information from complex PDF documents, assessing its operational efficiency compared to traditional human labeling workflows. Utilizing a student-led Retrieval-Augmented Generation (RAG) pipeline optimized for community-driven document analysis, we evaluate performance metrics across real-world datasets. We present comparative results highlighting fuzzy matches between human annotations and LLM-generated labels, outlining trade-offs in accuracy, cost, and scalability for community research initiatives.

## 1. Introduction
[cite_start]Police Misconduct Investigations [cite: 2]

[cite_start]Community organizations, such as Communities United Against Police Brutality (CUAPB) take on the important task of tackling police misconduct[cite: 3]. [cite_start]However, two main bottlenecks appear in our way[cite: 4]. [cite_start]First, data sources regarding police conduct are primarily unstructured and need to be parsed, often by a human reader due to the sensitive nature of the topic[cite: 4]. [cite_start]Second, events portrayed in news articles can be graphic and emotionally exhausting for human readers to parse on masse[cite: 5]. [cite_start]Consequently, work has been done to improve data extraction from unstructured sources[cite: 6].

### 1.1 Police Trends
[cite_start]One clear area of data extraction, among many, is location identification in articles[cite: 8]. [cite_start]Particular topic has been heavily explored in the literature and lends itself well to collaboration, due to the somewhat unambiguous nature of locations[cite: 9]. [cite_start]**[Describe some ARCGIS projects]**[cite: 10]. [cite_start]Though helpful for collaboration, police misconduct information can also greatly benefit local efforts for police accountability, helping auditors target their efforts at specific regions and allowing a high level view of policing trends[cite: 10]. [cite_start]**[CITATION]**[cite: 11]. [cite_start]Additionally, almost all articles regarding police misconduct will have location data, whereas categorization data and demographic data are sometimes omitted[cite: 11].

### 1.2 Data Scraping of Articles
[cite_start]Despite the unambiguity of a location, data scraping from News Articles can be difficult due to the varied nature of the data[cite: 13]. [cite_start]Not only can the addresses included be incomplete, such as 4th avenue, the types of address can be different, such as relative and absolute addresses[cite: 14]. [cite_start]But these questions arrive only after extracting the location[cite: 15]. [cite_start]For the moment, the question to tackle is extracting a location in the first place[cite: 15].

[cite_start]Common data scraping techniques can struggle, especially with articles that may include social media screenshots, those that resist OCR, or otherwise have an unusual table outline with many advertisements[cite: 16]. [cite_start]Even older physical news articles may appear in a tabular form that requires further pre-processing[cite: 17].

### 1.3 Named Entity Recognition
[cite_start]An initial technique, after the text has been extracted, has been named entity recognition[cite: 19]. [cite_start]NER has a benefit of extracting specific proper nouns, some of which can contain important location data[cite: 20]. [cite_start]However, a drawback of NER is that the data still requires parsing to manage[cite: 21].

#### Context Difficulties
[cite_start]In particular, contexts are not captured with NER, which aims to capture all entities simpliciter[cite: 23]. [cite_start]Some articles may display the article along with other relevant addresses, not all of which are related to the police incident at hand[cite: 24].

#### Can a simple LLM Model Match Human Performance?
[cite_start]We therefore turn to LLMs to capture this context[cite: 26]. [cite_start]If a large language model can be fed an entire article and is tasked to simply extract the location, it can be used to draw locations from the paper to map that article for further data analysis[cite: 26].

---

## 2. Methodology
[cite_start]To test the efficacy of LLMs in information extraction, we extract data from CUAPB’s article [cite: 28]

### 2.1 Data Extraction
[cite_start]CUAPB contains a repository of articles, all stored in PDF form[cite: 30]. [cite_start]To extract the data, we made use of PyMuPDF to extract pre-existing PCR data from the the repository[cite: 31]. [cite_start]To conserve context space, we extract only the first 7 pages of all articles[cite: 32]. [cite_start]Documents longer than 7 pages are highly likely to be court proceedings instead of the source news articles[cite: 33]. [cite_start]The extraction process for the articles, after parallelizing, took a total of 4 hours[cite: 34].

[cite_start]After a simple PyMuPDF extraction, we increased the quality of our extraction by flagging articles with the following features[cite: 35].

#### PUA
[cite_start]Private Use Access characters are characters not properly recognized by the usual UTF character sets[cite: 37]. [cite_start]We flagged these characters by notating characters of a certain hexadecimal range[cite: 38].

#### Images
[cite_start]Some articles were not text included either[cite: 40]. [cite_start]As the PDFs were simply collections of images, the library could not capture these[cite: 40]. [cite_start]We flagged these articles by notig the character length[cite: 41]. [cite_start]In particular, characters of under **[TJIS amount of characters]** were [cite: 41]

[cite_start]To handle these two cases, we implemented OCR (optical character recognitions)[cite: 42]. [cite_start]This workflow optimzies time, as an OCR scrape of all 2700 articles would be too time exhaustive[cite: 43]. [cite_start]The additional OCR procedure added around 100 news articles, increasing our test sample[cite: 44].

### 2.2 Data Labeling
[cite_start]We proceed to labeling the data using an LLM[cite: 45]. [cite_start]We made use of Lamma 3.1B instruct, which is a light language model[cite: 46]. [cite_start]While other models were available, we opted to use a model accessible to organizations with less compute power[cite: 47]. [cite_start]We then called it with the following prompt[cite: 48]:

[cite_start]“ [cite: 49]

[cite_start]“ [cite: 50]

[cite_start]We then fed it this article with the prompt[cite: 51]. [cite_start]We successfully scrapped[cite: 51].

### 2.3 Human labeling
[cite_start]We still needed a human dataset to serve as the ground truth[cite: 53]. [cite_start]While a similar process could be done with a single human, having crowdsourced evaluation helped speed up the process and account of individual human bias[cite: 54]. [cite_start]We procured the data byhosting a datathon across several campuses [Seattle University, Hamline University, and Carlton College][cite: 55]. [cite_start]We had succeeded in recruiting 48 different volunteers to label the articles[cite: 56].

[cite_start]To streamline the workflow, volunteers were asked to only perform data labeling on news articles, reading through the article to label relevant officers, civillians, and incident details[cite: 57]. [cite_start]We made use of two different environments for our datathon, exploring ways to more quickly crowdsource data[cite: 58].

#### Datathon 1
[cite_start]Baserow is a platform that allows users to build databases and share permissions, similar to an online spreadsheet tool[cite: 60]. [cite_start]The initial benefits to Baserow were first, its open-source structure, allowing for high degrees of accessibility to community organizations, and second, its ability to display various views[cite: 61].

[cite_start]While coordinating large, synchronous data entry, there is the risk that entries will overlap, such as when two users try to make an edit simultaneously on a Google cloud document[cite: 62]. [cite_start]While the risk remains for a tabular database such as Baserow to fail, Baserow allows for multiple frontend views, including card formatted data that allow subsets of data to be assigned to teams of volunteers[cite: 63]. [cite_start]This data segmentation helps mitigate simultaneous data entry[cite: 64].

[cite_start]The first datathon successfully labeled 460 articles and allowed double verification of 60 articles over the course of 2 hours[cite: 65]. [cite_start]Double verification is particularly important, as this will allow us to analyze the variance and noise in the responses, as human reponses are very unlikely to be deterministic due to the nature of responses[cite: 66]. [cite_start]For example, some volunteers may opt to offer more or less information, such a additional information about landmarks, in their data entry[cite: 67]. [cite_start]Additionally, certain volunteers may provide further information about the address such as the city or state[cite: 68].

[cite_start]Despite the enhanced structure that the advanced views gave, our datathon still faced strong limits, namely continued doubled verification issues[cite: 69]. [cite_start]The latency of the database prevented completely real time updates to be made[cite: 70]. [cite_start]Additionally, because there was not easy way to preview articles in Baserow[cite: 71]. [cite_start]Consequently, tasks could be rather taxing and navigating several pages, even with the more helpful views, was quite demanding for volunteers[cite: 72].

#### Zooniverse Integration
[cite_start]These issues led to use Zooniverse as an area of data collection[cite: 73]. [cite_start]Zooniverse is on online platformed designed specifically for data annotation, as opposed to being a more formal database as Baserow was[cite: 74]. [cite_start]Crucially, it is designed more as a survey than a typical tabular database[cite: 75]. [cite_start]Users are given items that they provide answers about[cite: 76]. [cite_start]In our case, we took the first 5 pages of each article, converted it into 5 images as a preview, providing users a way in-app to read through the articles and answer relevant questions[cite: 76].

[cite_start]We reaped several advantages to Zooniverse compared to Baserow,namely[cite: 77]:
* [cite_start]Streamlined work environment [cite: 78]
* [cite_start]Lack of latency issues [cite: 79]
* [cite_start]Data Integrity [cite: 80]

---

## 3. Evaluation Metric
[cite_start]Our evaluation metric is designed to capture similarity between responses[cite: 82]. [cite_start]While a semantic matching metric would be ideal, the short nature of the responses make close matches seperate[cite: 83]. [cite_start]Instead, we are interested in text similarity[cite: 84]. [cite_start]In particular, when the location is present, we want to ensure that both the human and LLM return the same chunk of text[cite: 84]. [cite_start]To account for natural variance in writing and potential mispelling, we use fuzzy matching as our measure of success[cite: 85].

[cite_start]Fuzzy matching assigns a score from 0 to 100 based on how closely the the characters in one chunk of text match the characters in the other chunk of text[cite: 86]. [cite_start]We additionally use subset fuzzy matching, which avoids assigning a penalty if one chunk has more text than another chunk, but both texts have overlapping text[cite: 87].

[cite_start]The choice to use a text based match is also meant to encourage tuning the model to replicate a human’s identification as much as possible, where the human is instructued to be as conservative as possible in their answers[cite: 88]. [cite_start]While there is no hard cut off on how high a fuzzy score an answer should be to be considered a match, scores about a 70 fuzzy match score tend to contain the same content, with 70 being a rather conservative response[cite: 89].

### 3.1 Fuzzy Comparison
#### [cite_start]Human-Human Comparisons [cite: 91]
[cite_start]However, a fuzzy score by itself cannot be the sole determiner of the success of the LLM[cite: 92]. [cite_start]Such a measurement would only be true if human responses to the labeling questions are deterministic[cite: 93]. [cite_start]However, as mentioned above, human responses may naturally differ as well[cite: 94]. [cite_start]What we do instead then is perform a fuzzy comparison between two human responses[cite: 95]. [cite_start]Then, we perform a fuzzy comparison between the LLM and one of the human responses[cite: 96]. [cite_start]We can then compare the distributions of those responses[cite: 97].

---

## 4. Results
[cite_start]After running both datathons, we compailed two datasets and managed to double-verify 120 articles[cite: 99].

[cite_start]**[Fuzzy Comparison Data/Plots Here]** [cite: 100]

```{figure} ./Dataset1.png
:name: human-human-comparison
:alt: Human-to-Human fuzzy comparison distribution plot
:align: center

Distribution of fuzzy match scores for double-verified human-to-human annotations.

```{figure} ./Table2.png
:name: human-human-comparison
:alt: Human-to-Human fuzzy comparison distribution plot
:align: center

Distribution of fuzzy match scores for double-verified human-to-human annotations.



---

## 5. Discussion

### [cite_start]5.1 Limitations [cite: 102]

### [cite_start]5.2 Future Work [cite: 103]

---

## [cite_start]6. Conclusion [cite: 104]
