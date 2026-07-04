---
title: "Bridging the Technical Gap: A Student-Led RAG Pipeline for Community-Driven Document Analysis (title to be altered)"
---

(abstract)=
## Abstract
This paper presents the impacts of using Large Language Models (LLMs) to extract and process information from complex PDF documents, assessing its operational efficiency compared to traditional human labeling workflows. Hosting a multi-campus datathon, we label over 400 documents to create a ground truth for our model evaluation. We present comparative results highlighting fuzzy matches between human annotations and LLM-generated labels. As a result, we demonstrate the viability of large language models in document analysis.

## Introduction
Police Misconduct Investigations

Community organizations, such as Communities United Against Police Brutality (CUAPB) take on the important task of tackling police misconduct. However, two main bottlenecks appear in our way. First, data sources regarding police conduct are primarily unstructured and need to be parsed, often by a human reader to avoid false information. Second, events portrayed in news articles can be graphic and emotionally exhausting for human readers to parse on masse

Trauma Spillover Effects and the Need for a Cognitive Shield

Reading through massive quantities of documents can be exhausting for any researcher. LLMs have already proven their value on this end. However, another effect of researching police misconduct that goes unnoticed are the traumatic effects that spillover. While the civilians in misconduct incidents suffer trauma, the negative effects continue to spread outside of the incident itself. Bor et al (2018) describe spillover effects of how trauma affects Black Americans in the surrounding community. However, more pertinent to this research is Dubberley et al's (2020) work on on researchers who conduct research on war crimes. Their results show 44% of researchers report worse mental health as a result of their research. Thus, those working towards police accountability, such as CUAPB, are very likely to face similar psychological burden from their research. This trauma spillover from the initial event itself serves as a serious hindrance towards community efforts.

Consequently, we should consider ways to shield researchers from the direct details of the event, when appropriate. For example, volunteers and organizers chronicling the occurances or locations of the events likely do not need the full extend of the details in the articles. Selective censoring of portions already require someone to have read the article. Instead, we should look towards LLMs as a way to assist in reading through documents with sensitive information. Marengo et al (2026) have helpfully discussed ways to categorize or detect trauma inducing text. Consequently, they describe how LLMs can be used to screen for severe content and assist in datathons. Already, this will help filter out documents with especially graphic details. However, with articles specifically about police misconduct, very few articles will remain. We will use LLMs not to remove articles, but to assist in reading articles. The particular data we wish to read from the text will be location.

### Police Trends
One clear area of data extraction, among many, is location identification in articles. Particular topic has been heavily explored in the literature and lends itself well to collaboration, due to the somewhat unambiguous nature of locations (see Lawton et al. [2001] for more). Etienne and Romo have already developed global information system (GIS) dashboards regarding police misconduct. Though helpful for collaboration, police misconduct information can also greatly benefit local efforts for police accountability, helping auditors target their efforts at specific regions and allowing a high level view of policing trends. For example, aforementioned CUAPB and the Atlanta Citizen Review Board (ACRB) are both examples of organizations. Additionally, almost all articles regarding police misconduct will have location data, demographic data are sometimes omitted. While the categorization of the misconduct incident itself is important, we set this aside due to its subjective nature. Address, on the other hand, is an objective piece of information for all incidents.

### Data Scraping of Articles
Despite the unambiguity of a location, data scraping from news articles can be difficult due to the varied nature of the data. Not only can the addresses included be incomplete, such as 4th avenue, the types of address can be different, such as relative and absolute addresses. But these questions arrive only after extracting the location. For the moment, the question to tackle is extracting a location in the first place.

Common data scraping techniques can struggle, especially with articles that may include social media screenshots, those that resist OCR, or otherwise have an unusual table outline with many advertisements. Even older physical news articles may appear in a tabular form that requires further pre-processing.

### Named Entity Recognition
An initial technique, after the text has been extracted, has been named entity recognition. Birks et al. (2020) use natural language processing to read narratives and identify particular trends of burglaries. More recently, Duca (2024) takes a different approach to extraction by using named entity recognition (NER) to extract repeated entities in a text. The use of NER can be quite conducive when searching texts for dates, officers, and incident locations. NER has a benefit of extracting specific proper nouns, some of which can contain important location data.However, a drawback of NER is that the data still requires parsing to manage. In particular, NER provides a series of nouns in the data, but cannot help us determine which of these nouns contains the location of the incident. For example, NER could identify the address of the indcident as well as some other location mentioned in the article that are completely unrelated to the police incident. Consequently, we needa more discriminate tool that can identify the location of the incident proper.


### LLMs and Hallucination

While we have discussed reasons to deploying LLMs for data extraction, we should be aware of major weaknesses, namely hallucination. Huang et al. (2023) and Xu et al. (2024) have shown the difficulty of removing hallucination. Given the nature of police accountability, we will be especially vigilant in suppressing hallucination. To help, we draw on recent RAG literature. Reuter et al. (2025) have recently discussed the use of RAG to combat hallucination in the context of long legal texts.

The benefit of RAG that Reuter describes is that the model has context to refer to in responding. For our purposes, because we already provide the article text as context, our model becomes less likely to hallucinate. Drawing from Pahoo et al.'s (2024) paper on prompt engineering, we can deploy certain techniques to instruct our model to be more conservative in its responses.

#### Can a simple LLM Model Match Human Performance?
With hallucinations intially addressed, we therefore turn to LLMs to capture this context. If a large language model can be fed an entire article and is tasked to simply extract the location, it can be used to draw locations from the paper to map that article for further data analysis.

---

## Methodology
To test the efficacy of LLMs in information extraction, we extract data from CUAPB’s article repository at (https://complaints.cuapb.org/).

### Data Extraction
CUAPB contains a repository of articles, all stored in PDF form. To extract the data, we made use of PyMuPDF to extract pre-existing PCR data from the the repository. To conserve context space, we extract only the first 7 pages of all articles. Documents longer than 7 pages are highly likely to be court proceedings instead of the source news articles. The extraction process for the articles, after parallelizing, took a total of 4 hours.


After a simple PyMuPDF extraction, we increased the quality of our extraction by flagging articles with the following features.

#### PUA
Private Use Access characters are characters not properly recognized by the usual UTF character sets. In particular, many of these characters include stylized numeric characters such as the following. PyMuPDF fails to read these characters, resulting in  a street address with no building number. We flagged these characters by notating characters of a certain hexadecimal range between E000 to F8FF.

```{figure} ./PUA_Text.png
:name: PrivateUseAcessText
:alt: Screenshot of article
:align: center
```


#### Images
Some articles were scraped, but with extremely low character counts. These PDFs were simply collections of images and the library could not capture the embedded text. We flagged these articles by notig the character length. In particular, characters of under 876 were flagged as requiring OCR.

```{figure} ./Text_Embedded_Img.png
:name: SocialMediaScreenshot
:alt: Screenshot of article
:align: center
Screenshot of article located at https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1626748591/Minneapolis_cop_says_he_threatened_Somali_over_flag__department_starts_internal_probe_–_Twin_Cities.pdf?1626748591

```

To handle these two cases, we implemented OCR (optical character recognitions). This workflow optimzies time, as an OCR scrape of all 2700 articles would be too time exhaustive. The additional OCR procedure added around 100 news articles, increasing our test sample by a reasonable amount.

### LLM Labeling
We proceed to labeling the data using an LLM. We made use of Ollama's llamma 3.1:8B instruct, which is a light language model, with only 8 billion parameteres. While other models were available, we opted to use a model accessible to organizations with less compute power. Additionally, a general instruction trained model can be used for tasks other than simply just location extraction.


We then called the model with the following prompt:


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

### Human labeling
We still needed a human dataset to serve as the ground truth. While a similar process could be done with a single human, having crowdsourced evaluation helped speed up the process and account of individual human bias. We procured the data byhosting a datathon across several campuses [Seattle University, Hamline University, and Carlton College]. We had succeeded in recruiting 48 different volunteers to label the articles.

To streamline the workflow, volunteers, primarily students, were asked to only perform data labeling on news articles, reading through the article to label relevant officers, civillians, and incident details. We made use of two different environments for our datathon, exploring ways to more quickly crowdsource data.

#### Datathon 1
Baserow is a platform that allows users to build databases and share permissions, similar to an online spreadsheet tool. The initial benefits to Baserow were first, its open-source structure, allowing for high degrees of accessibility to community organizations, and second, its ability to display various views.

While coordinating large, synchronous data entry, there is the risk that entries will overlap, such as when two users try to make an edit simultaneously on a Google cloud document. While the risk remains for a tabular database such as Baserow to fail, Baserow allows for multiple frontend views, including card formatted data that allow subsets of data to be assigned to teams of volunteers. This data segmentation helps mitigate simultaneous data entry.

The first datathon successfully labeled 460 articles and allowed double verification of 60 articles over the course of 2 hours. Double verification is particularly important, as this will allow us to analyze the variance and noise in the responses, as human reponses are very unlikely to be deterministic due to the nature of responses. For example, some volunteers may opt to offer more or less information, such a additional information about landmarks, in their data entry. Additionally, certain volunteers may provide further information about the address such as the city or state.

Despite the enhanced structure that the advanced views gave, our datathon still faced strong limits, namely continued doubled verification issues. The latency of the database prevented completely real time updates to be made. Additionally, because there was not easy way to preview articles in Baserow. Consequently, tasks could be rather taxing and navigating several pages, even with the more helpful views, was quite demanding for volunteers.

```{figure} ./Baserow_Table.png
:name: BaserowTabularView
:alt: Baserow table containing volunteer entries
:align: center
Tabular view of the Baserow dataset
```


```{figure} ./Baserow_Cards.png
:name: BaserowVolunteerView
:alt: Baserow table containing censored volunteer entries.
:align: center
Card view of the Baserow dataset, allowing easier use by volunteers.
```


#### Datathon 2: Zooniverse Integration
These issues led to use Zooniverse as an area of data collection. Zooniverse is on online platformed designed specifically for data annotation, as opposed to being a more formal database as Baserow was. Crucially, it is designed more as a survey than a typical tabular database. Users are given items that they provide answers about. In our case, we took the first 5 pages of each article, converted it into 5 images as a preview, providing users a way in-app to read through the articles and answer relevant questions.


We reaped several advantages to Zooniverse compared to Baserow,namely:
* Streamlined work environment: Volunteers no longer need to manually open an URL and instead simply need to work from a single window without interuptions.
* Lack of latency issues: A major issue was that volunteers in Baserow continued to accidentally fill in the same data rows. This was because Baserow had a small bit of latency where there was a delay between user data entries. This resulted in frustration from the volunteers. Zooniverse does not face this challenge.
* Data Integrity: Each user could not modify the responses of another volunteer. Additionally, volunteers would not be influenced by the responses of another volunteer.

```{figure} ./Zooniverse_Interface.png
:name: ZooniverseVolunteerView
:alt: Article to be read with questions along the side.
:align: center
Zooniverse Volunteer View
```


---

## Evaluation Metric
Our evaluation metric is designed to capture similarity between responses. While a semantic matching metric would be ideal, the short nature of the responses make close matches seperate. Instead, we are interested in text similarity. In particular, when the location is present, we want to ensure that both the human and LLM return the same chunk of text. To account for natural variance in writing and potential mispelling, we use fuzzy matching as our measure of success.

Fuzzy matching assigns a score from 0 to 100 based on how closely the the characters in one chunk of text match the characters in the other chunk of text. We additionally use subset fuzzy matching, which avoids assigning a penalty if one chunk has more text than another chunk, but both texts have overlapping text.

The choice to use a text based match is also meant to encourage tuning the model to replicate a human’s identification as much as possible, where the human is instructued to be as conservative as possible in their answers. While there is no hard cut off on how high a fuzzy score an answer should be to be considered a match, scores about a 70 fuzzy match score tend to contain the same content, with 70 being a rather conservative response.

Dev et al. (2025) have previously used fuzzy matching as an evaluation metric in the past. The main competitor as an evaluation metric would be to use an LLM-as-judge. However, to produce as deterministic of a response as we can, we use fuzzy matching primariyl. Dev et al. remark that fuzzy matching remains strong so long as nuance does not need to be captured. The problem of location extraction at hand is simple enough to the point where an LLM-as-judge is likely unnecessary and not computationally worth it, especially given current problems with its use as an evaluation tool (Li et al. 2025).


###  Fuzzy Comparison
#### Human-Human Comparisons
However, a fuzzy score by itself cannot be the sole determiner of the success of the LLM. Such a measurement would only be true if human responses to the labeling questions are deterministic. However, as mentioned above, human responses may naturally differ as well. What we do instead then is perform a fuzzy comparison between two human responses. Then, we perform a fuzzy comparison between the LLM and one of the human responses. We can then compare the distributions of those responses. This is a methodology adopted by both Hans et al. (2025) and Wang et al. (2023). Hans et al. specifically call this evaluation by human agreement. The strategy involves, in some way, a type of Turing test where model success is not determined by closeness of model response, but rather the closeness of the distributions of the fuzzy pairs.

---

## Results
After running both datathons, we compailed two datasets and managed to double-verify 120 articles.


We have very promising data from our Baserow dataset. What we have so far is 123 LLM-human pairs and 85 human-human pairs, where human-human matches are those where we had two seperate volunteers identify a location for an article.


However, the data from the Zooniverse dataset is not as promising, with only 13 human-human pairs with 123 LLM-human pairs . This is likely due to the fact that Zooniverse has no native way to capture news articles only, leading to more entries for legal documents and other types of documents.

We compare the distribution for the baserow dataset here the first table displays the Baserow dataset comparison and the second table displays the Zooniverse dataset comparison.



```{figure} ./Dataset1.png
:name: human-human-comparison-baserow
:alt: Human-to-Human fuzzy comparison distribution plot
:align: center

Distribution of fuzzy match scores for double-verified human-to-human annotations for the Baserow dataset.
```

```{figure} ./Table2.png
:name: human-human-comparison-zooniverse
:alt: Human-to-Human fuzzy comparison distribution plot
:align: center

Distribution of fuzzy match scores for double-verified human-to-human annotations for the Zooniverse dataset.
```

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

| Dataset | News Articles Compared | Mean Fuzzy Score | Median Fuzzy Score Score |
| :--- | :---: | :---: | :---: |
| **Baserow R1** | 69 | 72 | 93 |
| **Baserow R2** | 42 | 69 | 91 |
| **Zooniverse** | 79 | 70.8 | 88 |


At an initial glance, the data looks promising. With a mean of at least 88, we can suspect at least half the data is a reasonable match. That being said, this still entails that half the data is below, with a mass of points clustering around a fuzzy score of 0. However, fuzzy score matches by themselves are not enough.


Employing both the Kolmogorov-Smirnov (K-S) test and the Wasserstrein difference gives us a different result. The K-S test takes two samples and provides a similarity score with p-value as to whether the difference between these distributions is statistically significant. Yu & Wang (2023) have already made use of this test for evaluating LLMs. A low statistic and a high p value for these tests would indicate that the distributions are not distinct enough to tell apart.

The Wasserstein Distance is a metric, usually between two probability distributions, that provides a measure of how much one dataset needs to be altered to convert one dataset to another. For example, in two classes of 100 students and 50 students, a Wasserstein distance of 5 would entail 5% of students, 5 in one class and 2 in the other, would need to change to match the distributions together. We will use this to show that little change is required between the human-human and human-LLM fuzzy pairs.

Baserow Complete Dataset

K-S Statistic: 0.1027
p-value: 0.6158
Wasserstein Distance: 3.3351

We see that the the K-S statistic shows that there is not yet enough evidence that the distributions are different. More importantly, the Wassterstein distance shows that only a 3.3% shift in the data is necessary to match one distribution to the other.

For our much smaller dataset, the results are not as conclusive and show a larger difference, but still have a high p value, indicating that do not have sufficient evidence to claim that the distributions are different. Additionally, we see that the Wasserstein distance shows that only a 9% shift in the data is required to transform one distribution to the other.

Zooniverse Dataset

K-S Statistic: 0.2245
p-value: 0.5192
Wasserstein Distance: 9.0125



## Discussion

We consider our results. The initial view of the data is not promising, offering only modest success. We noticed that overall fuzzy comparisons between the model and a human to be unpromising, but such comparisons are not apt for a proper evaluation of a model. We therefore opt for Hans et al.'s evaluation methodology.

They handle the problem from the paradigm of "one-to-many" questions in the literature. In other words, they discuss questions for which the text generated answer is open-ended, hence "many answers" to "one question".

The question of location extraction might seem at first to be a closed-ended question. After all, a police incident can only occur at a single location. However, recall that several terms can refer to a single address. Consequently, answers differing by a word or two should not be heavily penalized.


### Impacts on Retrieval Augmented Generation

NOTE TO Reviewer: I will likely remove this section. The "RAG" portion of the paper is less important than my initial submission suggestion. I hope to remove this and rename the paper.


As mentioned previously, the goal of using these models is to provide additional context in retrieval augmented generation, especially to assist community organizations. The results here are promising in lowering the efforts in extracting information. A salient problem that remains is ensuring private use access characters are flagged and extracted using OCR, but this problem has solutions mentioned above.

### Limitations

Encoding and reading errors:

While we had initially addressed Private Use Access Characters and text embedded in images, the solutions raised are not comprehensive. When analyzing larger and more diverse sets of articles, it is highly likely that many newer PUA characters are unreadable with the current infrastructure.

More salient of an issue is that longer news articles with small snippets of text within images are not read at all. This is because our method of detecting text-based images was due to character count. In an article with high character count but many images, our data extraction pipeline is significantly weakened.


Data Limitations:

While our datathons labeled over 800 documents, only about a third of these were news articles, and even fewer of these articles were double-reviewed. This greatly limited the comparisons made in the results section. If more data were collected, it could be that the distributions compared would be different in a statistically significant manner.





### Future Work

The results derived from this research are only cursory, and seek only to lay the groundwork for later investigation to how an LLM can mimic a human's ability to identify a location in text.

#### Additional data

While the trials done collected over 400 data points, only 123 of these points were used to do an analysis of the data because the news article data was not labeled beforehand. Future analyses can benefit from human labeling of the data. Out of the 1300, we only have 460 articles labeled. This is only a third of the articles from the CUAPB database. There is already infrastruture for this work in the form of the Zooniverse datathon page. Outside of the 1300 scraped articles, researchers in the field may be interested in exploring more data sources.

Even aside from data that remains unlabeled, such comparisons with the data can be done on for both new data and with new volunteers. Recall that we only had 48 total volunteers across both datathons. While this ratio remains high, with about 8 documents per volunteer, different biases among the volunteers and predilections towards certain answers may skew the results. In particular, because the datathons were organized as community events, we can be fairly confident that some labels were not made independently of each other.

### Stronger Models

Aside from further data exploration, comparison and exploration of different large language models may benefit our audience. Recall that our current model only made use of Ollama's llama 3.1:8B as the data extractor. The intention of this choice of model was to use as few parameters as possible to reduce computute resources and use a general-use model to make linking to a general RAG pipeline easier. Both of these considerations were with respect to accessibility. However, researchers interested in those areas may use more powerful models to derive more accurate locations, especially in cases where the location is unclear or implied.

## Conclusion

We present a generalized outline for how to use LLMs in data extraction, especially for mass-quantity document analysis. We discuss data scraping and cleaning as well as the effectiveness of LLMs in searching through source documents. Our initial results are promising, showing that LLMs provide similar to that of a human, sparing precious manpower as well avoiding unneccessary psychological stress from having to read details about brutal events, serving as a cognitive shield.


## References

Balasubramanian, J. B., Adams, D., Roxanis, I., Berrington de Gonzalez, A., Coulson, P., Almeida, J. S., & García-Closas, M. (2025). Leveraging large language models for structured information extraction from pathology reports. *Journal of Pathology Informatics*, *19*, Article 100521. https://doi.org/10.1016/j.jpi.2025.100521

Birks, D., Coleman, A., & Jackson, D. (2020). Unsupervised identification of crime problems from police free-text data. *Crime Science*, *9*, Article 18. https://doi.org/10.1186/s40163-020-00127-4

Bor, J., Venkataramani, A. S., Williams, D. R., & Tsai, A. C. (2018). Police killings and their spillover effects on the mental health of black Americans: A population-based, quasi-experimental study. *The Lancet*, *392*(10144), 302–310.

Dev, S., Paskov, P., Sloan, A., Wei, K., de Lima, P. N., Chowdhury, S., Johnson, J., & Marcellino, W. (2026). *Simpler is better for autograders: Toward cost-effective LLM evaluations for open-ended tasks* (Research Report No. RR-A4618-1). RAND Corporation.

Dixon, A., & Birks, D. (2021). Improving policing with natural language processing. In *Proceedings of the 1st Workshop on NLP for Positive Impact* (pp. 115–124). Association for Computational Linguistics.

Duca, A. L. (2024). An empirical study to use large language models to extract named entities from repetitive texts. In *Proceedings of the 20th International Conference on Web Information Systems and Technologies (WEBIST 2024)* (pp. 417–424). SCITEPRESS. https://doi.org/10.5220/0013066500003825

Dubberley, S., Griffin, F., & Bal, M. (2020). Safer viewing: A study of secondary trauma mitigation techniques in open source investigations. *Health and Human Rights Journal*, *22*(1), 281–294. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7348432/

Etienne, H., & Romo, F. (n.d.). *Mapping police violence 2015-2020* [ArcGIS Dashboard]. RomoGIS. Retrieved July 3, 2026, from https://www.arcgis.com/apps/dashboards/7964469c84514a3fb89d9d5b9ea9f958

Han, S., Junior, G. T., Balough, T., & Zhou, W. (2025). *Judge's verdict: A comprehensive analysis of LLM judge capability through human agreement* (arXiv:2510.09738). arXiv. https://doi.org/10.48550/arxiv.2510.09738

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., … Liu, T. (2025). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Transactions on Information Systems*, *43*(2), 1–55.

Hussain, T., Akram, M. U., & Salam, A. A. (2025). A novel data extraction framework using natural language processing (DEFNLP) techniques. *Natural Language Processing Journal*, *11*, Article 100149. https://doi.org/10.1016/j.nlp.2025.100149

Li, D., Jiang, B., Huang, L., Beigi, A., Zhao, C., Tan, Z., ... & Liu, H. (2025, November). From generation to judgment: Opportunities and challenges of LLM-as-a-judge. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 2757–2791).

Marengo, D., Hoeboer, C., & Olff, M. (2026). Investigating the use of large language models for the detection of trauma-related symptoms: An exploratory study using trauma narratives. *Journal of Anxiety Disorders*, *119*, Article 103151. https://doi.org/10.1016/j.janxdis.2026.103151

Ni, X., Li, P., & Li, H. (2023). *Unified text structuralization with instruction-tuned language models* (arXiv:2303.14956). arXiv. https://arxiv.org/abs/2303.14956

Premasiri, D., Ranasinghe, T., Mitkov, R., et al. (2025). Survey on legal information extraction: Current status and open challenges. *Knowledge and Information Systems*, *67*, 11287–11358. https://doi.org/10.1007/s10115-025-02600-5

Relins, S., Birks, D., & Lloyd, C. (2025). Using instruction-tuned large language models to identify indicators of vulnerability in police incident narratives. *Journal of Quantitative Criminology*, *41*, 647–684. https://doi.org/10.1007/s10940-025-09611-z

Reuter, M., Lingenberg, T., Liepina, R., Lagioia, F., Lippi, M., Sartor, G., ... & Sayin, B. (2025, November). Towards reliable retrieval in RAG systems for large legal datasets. In *Proceedings of the Natural Legal Language Processing Workshop 2025* (pp. 17–30).

Sahoo, P., Singh, A. K., Saha, S., Jain, V., Mondal, S., & Chadha, A. (2025). *A systematic survey of prompt engineering in large language models: Techniques and applications* (arXiv:2402.07927v2). arXiv. https://doi.org/10.48550/arXiv.2402.07927

Wang, D., Yang, K., Zhu, H., Yang, X., Cohen, A., Li, L., & Tian, Y. (2023). *Learning personalized alignment for evaluating open-ended text generation* (arXiv:2310.03304). arXiv. https://doi.org/10.48550/arxiv.2310.03304

Xu, Z., Jain, S., & Kankanhalli, M. (2024). *Hallucination is inevitable: An innate limitation of large language models* (arXiv:2401.11817). arXiv. https://arxiv.org/abs/2401.11817
