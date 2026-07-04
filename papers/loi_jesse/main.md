---
title: "Bridging the Technical Gap: A Student-Led RAG Pipeline for Community-Driven Document Analysis (title to be altered)"
---

(abstract)=
## Abstract
This paper presents the impacts of using Large Language Models (LLMs) to extract and process information from complex PDF documents, assessing its operational efficiency compared to traditional human labeling workflows. Hosting a multi-campus datathon, we label over 400 documents to create a ground truth for our model evaluation. We present comparative results highlighting fuzzy matches between human annotations and LLM-generated labels. As a result, we demonstrate the viability of large language models in document analysis.

## 1. Introduction
Police Misconduct Investigations

Community organizations, such as Communities United Against Police Brutality (CUAPB) take on the important task of tackling police misconduct. However, two main bottlenecks appear in our way. First, data sources regarding police conduct are primarily unstructured and need to be parsed, often by a human reader to avoid false information. Second, events portrayed in news articles can be graphic and emotionally exhausting for human readers to parse on masse

Trauma Spillover Effects and the Need for a Cognitive Shield

Reading through massive quantities of documents can be exhausting for any researcher. LLMs have already proven their value on this end. However, another effect of researching police misconduct that goes unnoticed are the traumatic effects that spillover. While the civilians in misconduct incidents suffer trauma, the negative effects continue to spread outside of the incident itself. Bor et al (2018) describe spillover effects of how trauma affects Black Americans in the surrounding community. However, more pertinent to this research is Dubberley et al's (2020) work on on researchers who conduct research on war crimes. Their results show 44% of researchers report worse mental health as a result of their research. Thus, those working towards police accountability, such as CUAPB, are very likely to face similar psychological burden from their research. This trauma spillover from the initial event itself serves as a serious hindrance towards community efforts.

Consequently, we should consider ways to shield researchers from the direct details of the event, when appropriate. For example, volunteers and organizers chronicling the occurances or locations of the events likely do not need the full extend of the details in the articles. Selective censoring of portions already require someone to have read the article. Instead, we should look towards LLMs as a way to assist in reading through documents with sensitive information. Marengo et al (2026) have helpfully discussed ways to categorize or detect trauma inducing text. Consequently, they describe how LLMs can be used to screen for severe content and assist in datathons. Already, this will help filter out documents with especially graphic details. However, with articles specifically about police misconduct, very few articles will remain. We will use LLMs not to remove articles, but to assist in reading articles. The particular data we wish to read from the text will be location.

### 1.1 Police Trends
One clear area of data extraction, among many, is location identification in articles. Particular topic has been heavily explored in the literature and lends itself well to collaboration, due to the somewhat unambiguous nature of locations (see Lawton et al. [2001] for more). Etienne and Romo have already developed global information system (GIS) dashboards regarding police misconduct. Though helpful for collaboration, police misconduct information can also greatly benefit local efforts for police accountability, helping auditors target their efforts at specific regions and allowing a high level view of policing trends. For example, aforementioned CUAPB and the Atlanta Citizen Review Board (ACRB) are both examples of organizations. Additionally, almost all articles regarding police misconduct will have location data, demographic data are sometimes omitted. While the categorization of the misconduct incident itself is important, we set this aside due to its subjective nature. Address, on the other hand, is an objective piece of information for all incidents.

### 1.2 Data Scraping of Articles
Despite the unambiguity of a location, data scraping from news articles can be difficult due to the varied nature of the data. Not only can the addresses included be incomplete, such as 4th avenue, the types of address can be different, such as relative and absolute addresses. But these questions arrive only after extracting the location. For the moment, the question to tackle is extracting a location in the first place.

Common data scraping techniques can struggle, especially with articles that may include social media screenshots, those that resist OCR, or otherwise have an unusual table outline with many advertisements. Even older physical news articles may appear in a tabular form that requires further pre-processing.

### 1.3 Named Entity Recognition
An initial technique, after the text has been extracted, has been named entity recognition. Birks et al. (2020) use natural language processing to read narratives and identify particular trends of burglaries. More recently, Duca (2024) takes a different approach to extraction by using named entity recognition (NER) to extract repeated entities in a text. The use of NER can be quite conducive when searching texts for dates, officers, and incident locations. NER has a benefit of extracting specific proper nouns, some of which can contain important location data.However, a drawback of NER is that the data still requires parsing to manage. In particular, NER provides a series of nouns in the data, but cannot help us determine which of these nouns contains the location of the incident. For example, NER could identify the address of the indcident as well as some other location mentioned in the article that are completely unrelated to the police incident. Consequently, we needa more discriminate tool that can identify the location of the incident proper.


### LLMs and Hallucination

While we have discussed reasons to deploying LLMs for data extraction, we should be aware of major weaknesses, namely hallucination. Huang et al. (2023) and Xu et al. (2024) have shown the difficulty of removing hallucination. Given the nature of police accountability, we will be especially vigilant in suppressing hallucination. To help, we draw on recent RAG literature. Reuter et al. (2025) have recently discussed the use of RAG to combat hallucination in the context of long legal texts.

The benefit of RAG that Reuter describes is that the model has context to refer to in responding. For our purposes, because we already provide the article text as context, our model becomes less likely to hallucinate. Drawing from Pahoo et al.'s (2024) paper on prompt engineering, we can deploy certain techniques to instruct our model to be more conservative in its responses.

#### Can a simple LLM Model Match Human Performance?
With hallucinations intially addressed, we therefore turn to LLMs to capture this context. If a large language model can be fed an entire article and is tasked to simply extract the location, it can be used to draw locations from the paper to map that article for further data analysis.

---

## 2. Methodology
To test the efficacy of LLMs in information extraction, we extract data from CUAPB’s article repository at (https://complaints.cuapb.org/).

### 2.1 Data Extraction
CUAPB contains a repository of articles, all stored in PDF form. To extract the data, we made use of PyMuPDF to extract pre-existing PCR data from the the repository. To conserve context space, we extract only the first 7 pages of all articles. Documents longer than 7 pages are highly likely to be court proceedings instead of the source news articles. The extraction process for the articles, after parallelizing, took a total of 4 hours.

We made use of a light language model with the following prompt for each set of text:


incident_prompt = """
Analyze the provided news article and identify the LOCAL location of the incident in terms of the street address or physical landmarks. If the location cannot be found or is uncertain, return UNKNOWN instead. Only return a location you are confident with.
Output only the address and no comments. Just the address. No need to provide a state or a city. Just a local address that can be accurately located in the document. IF YOU CANNOT THEN ENTER UNKNOWN
DO NOT HALLUCINATE A LOCATION
An example output could be:
"A Kroger on 3rd Street"
"1923 E Pine St St Louis 42019"
"City Hall"
"Defendent's home"
"Unknown"
"Location not found"
"UNKNOWN"
These are all appropriate responses.
"""


After a simple PyMuPDF extraction, we increased the quality of our extraction by flagging articles with the following features.

#### PUA
Private Use Access characters are characters not properly recognized by the usual UTF character sets. In particular, many of these characters include stylized numeric characters such as the following. PyMuPDF fails to read these characters, resulting in  a street address with no building number. We flagged these characters by notating characters of a certain hexadecimal range between E000 to F8FF.

```{figure} ./PUA_Text.png
:name: Private Use Acess Text
:alt: Screenshot of article located at https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1627047812/Off-duty_McLeod_County_deputy__Norwood_Young_America_man_from_April_17_shooting_identified.pdf?1627047812
:align: center
```


#### Images
Some articles were scraped, but with extremely low character counts. These PDFs were simply collections of images and the library could not capture the embedded text. We flagged these articles by notig the character length. In particular, characters of under 876 were flagged as requiring OCR.

```{figure} ./Text_Embedded_Img.png
:name: Social media screenshot with incident location on news article.
:alt: Screenshot of article located at https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1626748591/Minneapolis_cop_says_he_threatened_Somali_over_flag__department_starts_internal_probe_–_Twin_Cities.pdf?1626748591
:align: center
```

To handle these two cases, we implemented OCR (optical character recognitions). This workflow optimzies time, as an OCR scrape of all 2700 articles would be too time exhaustive[cite: 43]. The additional OCR procedure added around 100 news articles, increasing our test sample.

### 2.2 Data Labeling
We proceed to labeling the data using an LLM[cite: 45]. [cite_start]We made use of Lamma 3.1B instruct, which is a light language model[cite: 46]. [cite_start]While other models were available, we opted to use a model accessible to organizations with less compute power[cite: 47]. [cite_start]We then called it with the following prompt[cite: 48]:

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


We have very promising data from our Baserow dataset. What we have so far is 123 LLM-human pairs and 85 human-human pairs, where human-human matches are those where we had two seperate volunteers identify a location for an article.


However, the data from the Zooniverse dataset is not as promising, with only 13 human-human pairs with 123 LLM-human pairs . This is likely due to the fact that Zooniverse has no native way to capture news articles only, leading to more entries for legal documents and other types of documents.

We compare the distribution for the baserow dataset here the first table displays the Baserow dataset comparison and the second table displays the Zooniverse dataset comparison.



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

Here we compare the two distributions. Upon visual inspection, the results look promising, despite the large differences. We see in both cases a tri-modal distribution, with answers on the high end indicating almost exact matches, answers on the low end indicating almost no match, and answers around the 60-70 point range indicating some match.

To elaborate on the significance of this comparison, we discuss examples of high, low, and mid ranking pairs from our dataset. These are represent some examples of what is included in the dataset, but are not strictly included in the final dataset.

| Response 1 | Response 2 | Fuzzy Score |
| :--- | :--- | :--- |
| "53rd and Emerson Avenues N." | intersection of 53rd and Emerson Avenues N | 93 |
| "1235 Reform St" | 520 Reform St., Norwood Young America MN | 63|
| N/A | “Minnesota City Court” | 0 |


Answers on the high end over 90 points indicate that the model and human captured the same text in their responses, with only minor phrase or wording changes. Scores around this range show promise about the model, though might represent sets of articles with a location clearly provided.

We also see here that while scores as high as 63 may seem to point to the human and model reading the same chunk of text, the human reader is able to capture some crucial details that the model cannot. We go into more details of this in the discussion, but these scores likely represent articles with a clear incident location mentioned, but cannot be accessed due to scraping problems.

Finally, answers on the low end, around 0-10 on the fuzzy scale, indicate almost no match between the terms. These likely represent articles where there is no clear incident location. In such cases, the human volunteer likely enters terms such as "N/A", "no location", or some other indicator that the address is unclear. In contrast, while the model is prompted to give a response without hallucinating, it will give an overly vague response or a mentioned location but that is not the incident location.



However, employing both the K-S test and the Wasserstrein difference gives us a different result.

K-S Statistic: 0.1027
p-value: 0.6158
Wasserstein Distance: 3.3351

We see that the the K-S statistic shows that there is not yet enough evidence that the distributions are different. More importantly, the Wassterstein distance shows that only a 3.3% shift in the data is necessary to match one distribution to the other.

For our much smaller dataset, the results are not as conclusive and show a larger difference, but still have a high p value, indicating that do not have sufficient evidence to claim that the distributions are different. Additionally, we see that the Wasserstein distance shows that only a 9% shift in the data is required to transform one distribution to the other.



K-S Statistic: 0.2245
p-value: 0.5192
Wasserstein Distance: 9.0125



## 5. Discussion

We consider our results. The initial view of the data is not promising, offering only modest success. We noticed that overall fuzzy comparisons between the model and a human to be unpromising, but such comparisons are not apt for a proper evaluation of a model. The literature discusses ways to evaluate an LLM [CITATION], but I will be making use of [CITATION] way of resolving this issue.

They handle the problem from the paradigm of "one-to-many" questions in the literature. In other words, they discuss questions for which the text generated answer is open-ended.

The question of location extraction might seem at first to be a closed-ended question. After all, a police incident can only occur at

However, what allows a more reasonable comparison was


### Impacts on Retrieval Augmented Generation

As mentioned previously, the goal of using these models is to provide additional context in retrieval augmented generation, especially to assist community organizations. The results here are promising in lowering the efforts in extracting information. A salient problem that remains is ensuring private use access characters are flagged and extracted using OCR, but this problem has solutions mentioned above.

### 5.1 Limitations

Private use access


Limited Articles


Model Limitations


### [cite_start]5.2 Future Work [cite: 103]

The results derived from this research are only cursory, and seek only to lay the groundwork for later investigation to how an LLM can mimic a human's ability to identify a location in text.

#### Additional data

While the trials done collected over 400 data points, only 123 of these points were used to do an analysis of the data because the news article data was not labeled beforehand. Future analyses can benefit from human labeling of the data. Out of the 1300, we only have 460 articles labeled. This is only a third of the articles from the CUAPB database. There is already infrastruture for this work in the form of the Zooniverse datathon page. Outside of the 1300 scraped articles, researchers in the field may be interested in exploring more data sources.

Even aside from data that remains unlabeled, such comparisons with the data can be done on for both new data and with new volunteers. Recall that we only had 26 total volunteers across both datathons. While this ratio remains high, with about 16 labels per volunteer, different biases among the volunteers and predilections towards certain answers may skew the results. In particular, because the datathons were organized as community events, we can be fairly confident that some labels were not made independently of each other.

### Stronger Models

Aside from further data exploration, comparison and exploration of different large language models may benefit our audience. Recall that our current model only made use of Ollama's llama 3.1:8B as the data extractor. The intention of this choice of model was to use as few parameters as possible to reduce computute resources and use a general-use model to make linking to a general RAG pipeline easier. Both of these considerations were with respect to accessibility. However, researchers interested in those areas may use more powerful models to derive more accurate locations, especially in cases where the location is unclear or implied.

## 6. Conclusion

We present a generalized outline for how to use LLMs in data extraction, especially for mass-quantity document analysis. We discuss data scraping and cleaning as well as the effectiveness of LLMs in searching through source documents. Our initial results are promising, showing that LLMs provide similar to that of a human, sparing precious manpower as well avoiding unneccessary psychological stress from having to read details about brutal events, serving as a cognitive shield.
