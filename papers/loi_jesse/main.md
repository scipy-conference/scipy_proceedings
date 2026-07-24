---
title: "Bridging the Technical Gap: A Student-Led RAG Pipeline for Community-Driven Document Analysis (title to be altered)"

---

(abstract)=
## Abstract
This paper presents the impacts of using Large Language Models (LLMs) to extract and process information from complex PDF documents, assessing its operational efficiency compared to traditional human labeling workflows. Hosting a multi-campus datathon, we label over 400 documents to create a ground truth for our model evaluation. We present comparative results highlighting fuzzy matches between human annotations and LLM-generated labels. As a result, we demonstrate the viability of large language models in document analysis.

## Introduction


Community organizations, such as Communities United Against Police Brutality (CUAPB) take on the important task of tackling police misconduct. However, two main bottlenecks appear in our way. First, data sources regarding police conduct are primarily unstructured and need to be parsed, often by a human reader to avoid false information. Second, events portrayed in news articles can be graphic and emotionally exhausting for human readers to parse en masse.

Reading through massive quantities of documents can be exhausting for any researcher and so LLMs have already proven their value on this end. However, another effect of researching police misconduct that goes unnoticed are the traumatic effects that spillover. While the civilians in misconduct incidents suffer trauma, the negative effects continue to spread outside of the incident itself. [@bor2018] describe spillover effects of how trauma affects Black Americans in the surrounding community. However, more pertinent to this research is the work of [@baker2020] on researchers who conduct research on war crimes. Their results show 44% of researchers report worse mental health as a result of their research. Thus, those working towards police accountability, such as CUAPB, are very likely to face a similar psychological burden from their research. This trauma spillover from the initial event itself serves as a serious hindrance towards community efforts.

We should consider ways to shield researchers from unnecessary details of the event, when appropriate. For example, volunteers and organizers chronicling the occurrences or locations of the events likely do not need the full extent of the details in the articles. However, selective censoring of portions often requires reading the article. Instead, we should look towards LLMs as a way to assist in reading through documents with sensitive information. [@marengo2026] have discussed ways to categorize or detect trauma inducing text. They describe how LLMs can be used to screen for severe content and assist in datathons. Already, this will help filter out documents with especially graphic details. However, with articles specifically about police misconduct, few will remain. We will use LLMs not to remove articles, but to assist in reading articles for a particular feature that does not require human judgment.

One promising area of data extraction, among many, is location. This topic has been heavily explored in the literature and lends itself well to collaboration, due to the somewhat unambiguous nature of locations. [@etienne2026] have already developed global information system (GIS) dashboards regarding police misconduct. This information can also greatly benefit individual local efforts, helping auditors target their efforts at specific regions and allowing a high level view of policing trends. The aforementioned CUAPB and the Atlanta Citizen Review Board (ACRB) groups are both examples of such organizations. Additionally, almost all articles regarding police misconduct will have location data; demographic data are sometimes omitted. While the categorization of the misconduct incident itself is important, we set this aside due to its subjective nature. Address, on the other hand, is an objective piece of information for all incidents. But scraping location is not without issues. Common data scraping techniques can struggle with articles that may include social media screenshots, those that resist OCR, or otherwise have an unusual table outline with many advertisements. Even older physical news articles may appear in a tabular form that requires further pre-processing. Work in natural language processing (NLP) aims to address extraction difficulties.

After scraping, analysts sometimes use named entity recognition (NER) to isolate candidates for data extraction. [@birks2020] use natural language processing to read narratives and identify particular trends of burglaries. More recently, [@duca2024] takes a different approach to extraction by using NER to extract repeated entities in a text. The use of NER can be quite conducive when searching texts for dates, officers, and incident locations, as it has a benefit of extracting specific proper nouns, such as addresses and landmarks. However, a drawback of NER is that the data still requires parsing to manage. In particular, NER may provide nouns but cannot help us determine which contains the location of the incident. It may, for example, provide some unrelated address. Consequently, we need a more discriminating tool that can identify the location of the incident properly, such as an LLM. However, these tools have their own faults.

While we have discussed reasons to deploy LLMs for data extraction, we should be aware of major weaknesses, namely hallucination. [@huang2025] and [@xu2024] have shown the difficulty of removing hallucination. Given the nature of police accountability, one must be especially vigilant in suppressing hallucination. To help, we draw on recent RAG literature. [@reuter2025] have recently discussed the use of RAG to combat hallucination in the context of long legal texts. The benefit of RAG that Reuter describes is that the model has context to refer to in responding. For our purposes, because we already provide the article text as context, our model becomes less likely to hallucinate. Additionally, drawing from [@sahoo2025]'s paper on prompt engineering, we can deploy certain techniques to instruct our model to be more conservative in its responses. We elaborate on this further in our methodology. With hallucinations initially addressed, we turn to LLMs to capture location context.


---

## Methodology
To test the efficacy of LLMs in information extraction, we extract data from CUAPB’s article repository at (https://complaints.cuapb.org/).


### Data Preprocessing

CUAPB contains a repository of articles, all stored in PDF form at (https://complaints.cuapb.org/). To extract the data, we made use of PyMuPDF to extract pre-existing PCR data from the the repository. To conserve context space, we extract only the first 7 pages of all articles. Documents longer than 7 pages are highly likely to be court proceedings instead of the source news articles. The extraction process for the articles, after parallelizing, took a total of 4 hours.Once we had the data in text form, we refine our source data flagging problematic features. Namely, we identified Private Use Access (PUA) characters and images-as-PDFs as features that we needed to address.

PUA characters are characters not properly recognized by the usual UTF character sets. In particular, many of these characters include stylized numeric characters such as the following. PyMuPDF fails to read these characters, resulting in  a street address with no building number. We flagged these characters by notating characters of a certain hexadecimal range between E000 to F8FF. This range captures a majority of the PUA characters in the dataset, but this range may be adapted for a wider set of characters.

```{figure} ./PUA_Text.png
:name: PrivateUseAcessText
:alt: Screenshot of article
:align: center
```

Some articles were scraped, but with extremely low character counts. These PDFs were simply collections of images and the library could not capture the embedded text. We flagged these articles by noting the character length. In particular, characters of under 876, representing the bottom 4% of documents in terms of character count, were flagged as potential screenshots of documents, as documents of such short length are very likely to contain more text.

```{figure} ./Text_Embedded_Img.png
:name: SocialMediaScreenshot
:alt: Screenshot of article
:align: center
Screenshot of article located at https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1626748591/Minneapolis_cop_says_he_threatened_Somali_over_flag__department_starts_internal_probe_–_Twin_Cities.pdf?1626748591

```

To handle these two cases, we implemented OCR (optical character recognitions). This workflow optimizes time, as an OCR scrape of all 2700 articles would be too time exhaustive. The additional OCR procedure added around 100 news articles, increasing our test sample by a reasonable amount. With the assistance of OCR, our set has been enhanced enough for an LLM to read its documents and label locations.

###  Data Labeling

We label our data with an LLM and with human volunteers to serve as a ground truth set for the models. We made use of Ollama's llamma 3.1:8B instruct, which is a light language model, with only 8 billion parameteres. While other models were available, we opted to use a model accessible to organizations with less compute power. Additionally, a general instruction trained model can be used for tasks other than simply just location extraction. We then called the model with the following prompt:


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

This prompt makes use of both few-shot prompting and a very conservative instruction that attempts to restrain the model from hallucinating, restricting its role to extraction, not inference. Similar instructions were given to the human volunteers.

Having crowdsourced evaluation helped speed up the labeling process and account for individual human bias. More importantly, we wanted another set of human responses as a comparison between each other. We procured the data by hosting a datathon across several campuses [Seattle University, Hamline University, and Carlton College]. We had succeeded in recruiting 48 different volunteers to label the articles as a news article or as a non-news article designation, such as police report. To streamline the workflow, volunteers, primarily students, were asked to only perform data labeling on news articles once identified, reading through the article to label relevant officers, civilians, and incident details. Utilizing Baserow and Zooniverse, we made use of two different environments for our datathon, exploring ways to more quickly crowdsource data.

Baserow is a platform that allows users to build databases and share permissions, similar to an online spreadsheet tool. The initial benefits to Baserow were first, its open-source structure, allowing for high degrees of accessibility to community organizations, and second, its ability to display various views. While coordinating large, synchronous data entry, there is the risk that entries will overlap, such as when two users try to make an edit simultaneously on a Google cloud document. While the risk remains for a tabular database such as Baserow to fail, Baserow allows for multiple frontend views, including card formatted data that allow subsets of data to be assigned to teams of volunteers. This data segmentation helps mitigate simultaneous data entry.

The first datathon successfully labeled 460 articles and allowed double verification of 60 articles over the course of 2 hours. Double verification is particularly important, as this will allow us to analyze the variance and noise in the responses, as human responses are very unlikely to be deterministic due to the nature of responses. For example, some volunteers may opt to offer more or less information, such as additional information about landmarks, in their data entry. Additionally, certain volunteers may provide further information about the address such as the city or state.

Despite the enhanced structure that the advanced views gave, our datathon still faced strong limits, namely continued doubled verification issues. The latency of the database prevented completely real time updates to be made. Additionally, because there was no easy way to preview articles in Baserow. Consequently, tasks could be rather taxing and navigating several pages, even with the more helpful views, was quite demanding for volunteers.

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

Baserow’s issues led us to use Zooniverse, an online platform designed specifically for data annotation, as opposed to being a more formal database as Baserow was. Crucially, it is designed as a survey rather than as a typical tabular database. Users are given items that they provide answers about. In our case, we took the first 5 pages of each article and converted them into 5 images as a preview, providing users a way in-app to read through the articles and answer relevant questions. We reaped several advantages to Zooniverse compared to Baserow, namely:

* Streamlined work environment: Volunteers no longer need to manually open an URL and instead simply need to work from a single window without interruptions.
* Lack of latency issues: A major issue was that volunteers in Baserow continued to accidentally fill in the same data rows. This was because Baserow had a small bit of latency where there was a delay between user data entries. This resulted in frustration from the volunteers. Zooniverse does not face this challenge.
* Data Integrity: Each user could not modify the responses of another volunteer. Additionally, volunteers would not be influenced by the responses of another volunteer.


```{figure} ./Zooniverse_Interface.png
:name: ZooniverseVolunteerView
:alt: Article to be read with questions along the side.
:align: center
Zooniverse Volunteer View
```


---

### Evaluation Metrics

Our evaluation metric is designed to capture similarity between responses. While a semantic matching metric would be ideal, the short nature of the responses make close matches separate. Instead, we are interested in text similarity. In particular, when the location is present, we want to ensure that both the human and LLM return the same chunk of text. To account for natural variance in writing and potential misspelling, we use fuzzy matching from the TheFuzz library as our measure of success. Fuzzy matching assigns a score from 0 to 100 based on how closely the the characters in one chunk of text match the characters in the other chunk of text. We additionally use subset fuzzy matching, which avoids assigning a penalty if one chunk has more text than another chunk, but both texts have overlapping text. More concretely, fuzzy matching represents the calculation of the Levenshtein distance.

The Levenshtein distance \text{lev}_{a,b}(i, j) is a measure between two strings a and b and is formulated with the following recusive definition.

$$\text{lev}_{a,b}(i, j) = \begin{cases}
  \max(i, j) & \text{if } \min(i, j) = 0, \\
  \min \begin{cases}
    \text{lev}_{a,b}(i-1, j) + 1 \\
    \text{lev}_{a,b}(i, j-1) + 1 \\
    \text{lev}_{a,b}(i-1, j-1) + c(a_i, b_j)
  \end{cases} & \text{otherwise.}
\end{cases}$$

In the first case, \min(i, j) = 0 occurs when comparing a string to an empty string. Let i and j also represent indices in a matrix whose columns indices represent the indices of one of the strings and whose row indices represent the indices of the other string. Recursively, the “distance” is represented by the amount of steps needed to convert one string to another. Each step is either a character deletion, represented by \text{lev}_{a,b}(i-1, j) + 1, a character insertion, represented by  \text{lev}_{a,b}(i, j-1) + 1, or a character substitution  \text{lev}_{a,b}(i-1, j-1) + c(a_i, b_j).

```{figure} ./LV_Matrix.png
:name: ZooniverseVolunteerView
:alt: Article to be read with questions along the side.
:align: center
The following Levenshtein distance matrix shows an optimal path cost of 6.
```


For the above example, 6 steps are the optimal number required to convert “minneapolis” to “Minnesota”. To convert these steps into a score, we normalize with the following equation.

$$\text{Score}_{\text{norm}} = \left(1 - \frac{\text{lev}_{a,b}(|a|, |b|)}{\max(|a|, |b|)}\right) \times 100$$

The current project makes use not of general fuzzy matching, but partial ratio fuzzy matching, or subset fuzzy matching. Instead of comparing the entire string, subset fuzzy matching only matches the best matching substring of equal length. In other words, there is not a penalty for deleting ends off of a longer string. For string a length m and string b length n where $m<n$ subset fuzzy matching creates an n x n matrix and computes the levenshtein distance for every potential window of length n for string a. The choice to use a text based match motivates tuning the model to replicate a human’s identification as much as possible, where the human is instructed to be as conservative as possible in their answers. While there is no hard cut off on how high a fuzzy score an answer should be to be considered a match, scores about a 70 fuzzy match score tend to contain the same content, with 70 being a rather conservative response.

[@dev2026] have previously used fuzzy matching as an evaluation metric in the past. The main competitor as an evaluation metric would be to use an LLM-as-judge. However, to produce as deterministic a response as we can, we use fuzzy matching primarily. Dev et al. remark that fuzzy matching remains strong so long as nuance does not need to be captured. The problem of location extraction at hand is simple enough to the point where an LLM-as-judge is likely unnecessary and not computationally worth it, especially given current problems with its use as an evaluation tool [@li2025]. But we cannot use fuzzy score pairs for an LLM and human by itself, however. Such a comparison would suggest that a single human response is enough to produce ground truth, which is not necessarily true.

A fuzzy score by itself cannot be the sole determiner of the success of the LLM. Such a measurement would only be true if human responses to the labeling questions are deterministic given an article. However, as mentioned above, human responses may naturally differ as well. What we do instead then is perform a fuzzy comparison between two human responses. Then, we perform a fuzzy comparison between the LLM and one of the human responses. We can then compare the distributions of those responses. This is a methodology adopted by both [@han2025] and [@wang2023]. Han et al. specifically call this evaluation by human agreement. The strategy involves, in some way, a type of Turing test where model success is not determined by closeness of model response, but rather the closeness of the distributions of the fuzzy pairs. To allow for a more concrete description of such a Turing test, we introduce both the two-sample Kolmogorov-Smirnov test and the Wasserstein distance.

The Kolmogorov-Smirnov (K-S) test and the Wasserstein difference on 1 dimensional data allow for a more nuanced analysis of the data by comparing distributions between the different label pairs. The K-S test takes two samples and provides a similarity score with p-value as to whether the difference between these distributions is statistically significant. It is analogous to a two sample t-test for means. A low statistic and a high p value for these tests would indicate that the distributions are not distinct enough to tell apart. The statistic $D$ is computed as follows:

$$D = \sup_x \vert{}F_1(x) - F_2(x)\vert{}$$

We let $F_1(x)$ and $F_2(x)$ be our two empirical cumulative distribution functions created from our two sets (human-human and human-LLM) of fuzzy pairs. We then compute the supremum of the difference between these functions to identify the divergence between the two, giving us a metric to determine with the LLM and human responses could have the same underlying distribution. However, note that this test is not meant to prove that two distributions are the same, but can only give strong evidence that distributions are distinct. In other words, the null hypothesis is that the distributions are the same and the test finds evidence to reject the null hypothesis. With that in mind, we cannot assert the null hypothesis and should hedge our conclusions.
Known as the Earth Mover's Distance and introduced by [@rubner2000], the Wasserstein distance is a metric that measures how much must change to transform one probability distribution into another. It quantifies the cost of transporting probability mass to align two distributions. The distance can be formulated as follows:

$$W_1(F_1, F_2) = \int_{-\infty}^{\infty} |F_1(x) - F_2(x)| \, dx$$

Let $W_1(F_1, F_2)$ stand for the 1st Wasserstein distance, as opposed to other Wasserstein distances which may square the differences. $F_1(x)$ and $F_2(x)$ once again represent the cumulative probability distributions for the two sets of pairs mentioned above. We then integrate the absolute difference between the two curves to identify, intuitively, the total distance between the distributions. In the context of our 0-to-100 fuzzy matching scale, the Wasserstein distance represents the average number of score points that the human-LLM fuzzy scores would need to shift to perfectly match the human-human baseline distribution. We can use this to measure the average point adjustment required to align the LLM-human distribution with the natural variance of human agreement.




---

## Results
After running both datathons, we compiled two datasets and managed to double-verify 120 articles. We have very promising data from our Baserow dataset. What we have so far is 123 LLM-human pairs and 85 human-human pairs, where human-human matches are those where we had two separate volunteers identify a location for an article. However, the data from the Zooniverse dataset is not as promising, with only 13 human-human pairs with 123 LLM-human pairs. This is likely due to the fact that Zooniverse has no native way to capture news articles only, leading to more entries for legal documents and other types of documents. We recognize the limitations of only 13 data points for the Zooniverse comparison, and leave it in the following discussions section. For transparency purposes we make the Zooniverse comparison, but rely more on our usage of the Baserow dataset.


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


At an initial glance, the data looks promising. With a median of at least 88, we can suspect at least half the data is a reasonable match. That being said, this still entails that half the data is below, with a mass of points clustering around a fuzzy score of 0.
 To add more context, we display a handful of observations.
| MAX_LOCATION | Location | url | fuzz_loc | Explanation |
|---|---|---|---|---|
| Deer River | | [source](https://assets.nationbuilder.com/cuapb/pages/1472/attachments/original/1669428027/Apitz_Itasca_Co._sheriff's_deputy_charged_with_videotaping_girl_in_bathroom.pdf?1669428027) | 0 | LLM attempts answer when there is none |
| UNKNOWN | | [source](https://assets.nationbuilder.com/cuapb/pages/1472/attachments/original/1670339558/Meemken_Longtime_Stearns_County_Deputy_Gets_9_Months_Jail.pdf?1670339558) | 0 | Negative answers expressed differently |
| "KASOTA, Minn." | LeSueur County | [source](https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1626881948/Settlement_In_LeSueur_Co._Deputy%E2%80%99s_Fatal_Shooting_Of_Man.pdf?1626881948) | 23 | Overly vague answers |
| "Minneapolis City Hall" | Eastside St Paul Checkerboard Pizza | [source](https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1585615610/Five_SPPD_Cops_Fired.pdf?1585615610) | 30 | LLM fails to locate instance |
| "Interstate Hwy. 35E and University Avenue" | Interstate Highway 35E North by University Avenue | [source](https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1509687779/Mark_Kaspszak_MPD_Arrested_for_DWI_010606.pdf?1509687779) | 72 | |
| "1800 block of Columbus Avenue S." | 1800 block of Columbus Avenue S | [source](https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1613597722/Chauvin_Shooting_Article.pdf?1613597722) | 100 | |
| Red Wing City Hall | | [source](https://d3n8a8pro7vhmx.cloudfront.net/cuapb/pages/270/attachments/original/1613839705/Red_Wing_Police_Chief_Fired.pdf?1613839705) | 33 | LLM provides wrong location |

We see that errors often populate when it is unclear what the location is in the article. However, fuzzy score matches by themselves are not enough.

Employing both the Kolmogorov-Smirnov (K-S) test and the Wasserstein difference gives us a different result. We display the results for both datathons.
| Dataset | K-S Statistic | p-value | Wasserstein Distance |
| :--- | :---: | :---: | :---: |
| **Baserow R1 and R2** | 0.1027 | 0.6158 | 3.3351 |
| **Zooniverse** | 0.2245 | 0.5192 | 9.0125 |

For the Baserow dataset, we see that the K-S statistic shows that there is not yet enough evidence that the distributions are different. More importantly, the Wasserstein distance shows that only a small shift in the data is necessary to match one distribution to the other.

For our much smaller Zooniverse dataset, the results are not as conclusive and show a larger difference, but still have a high p value, indicating that they do not have sufficient evidence to claim that the distributions are different. Additionally, we see that the Wasserstein distance is larger than the one in the Baserow dataset, but is still low.










## Discussion

We consider our results. The initial view of the data is not promising, offering only modest success. We noticed that overall fuzzy comparisons between the model and a human to be unpromising, but such comparisons are not apt for a proper evaluation of a model. We therefore opt for [@han2025]'s evaluation methodology. They handle the problem from the paradigm of "one-to-many" questions in the literature. In other words, they discuss questions for which the text generated answer is open-ended, hence "many answers" to "one question". The question of location extraction might seem at first to be a closed-ended question. After all, a police incident can only occur at a single location. However, recall that several terms can refer to a single address. Consequently, answers differing by a word or two should not be heavily penalized.


### Limitations

Encoding and reading errors:

While we had initially addressed Private Use Access Characters and text embedded in images, the solutions raised are not comprehensive. When analyzing larger and more diverse sets of articles, it is highly likely that many newer PUA characters are unreadable with the current infrastructure. More salient of an issue is that longer news articles with small snippets of text within images are not read at all. This is because our method of detecting text-based images was due to character count. In an article with high character count but many images, our data extraction pipeline is significantly weakened.

While our datathons labeled over 800 documents, only about a third of these were news articles, and even fewer of these articles were double-reviewed. This greatly limited the comparisons made in the results section. If more data were collected, it could be that the distributions compared would be different in a statistically significant manner. Most importantly, we must address the extremely limited human sample. Recall that we only have 13 human-human data points for the Zooniverse dataset and only 85 human-human data points from the Baserow dataset. While it is better, it remains far from ideal. We therefore wish to hedge our results, instead using this research as a framework for future research and community engagement.

The lack of data for human comparison is the most important area of improvement for our research. There are two ways that future research can improve on our results. First, organizers can lower the total number of articles such that they can guarantee each article in the labeling set can be labeled at least twice. Second, a different route can be to collect enough data with more time and volunteers such that the volume of data is simply higher.





### Future Work

The results derived from this research are only cursory, and seek only to lay the groundwork for later investigation to how an LLM can mimic a human's ability to identify a location in text.

While the trials collected over 400 data points, only 123 of these points were used to do an analysis of the data because the news article data was not labeled beforehand. Future analyses can benefit from human labeling of the data. Out of the 1300, we only have 460 articles labeled. This is only a third of the articles from the CUAPB database. There is already infrastruture for this work in the form of the Zooniverse datathon page. Outside of the 1300 scraped articles, researchers in the field may be interested in exploring more data sources. However, more data sources would only help given the availability of human labeling.

Even aside from data that remains unlabeled, such comparisons with the data can be done on for both new data and with new volunteers. Recall that we only had 48 total volunteers across both datathons. While this ratio remains high, with about 8 documents per volunteer, different biases among the volunteers and predilections towards certain answers may skew the results. In particular, because the datathons were organized as community events, we can be fairly confident that some labels were not made independently of each other. Finally, a replication of this data labeling process should include an explicit mechanism to ensure there are more human-human pairs of data points. Because all the volunteers labeled the data independently, there was no way to guarantee that the data would have duplicates. In addition to enriching human data, there are also many ways to improve computational results.

Aside from further data exploration, comparison and exploration of different large language models may benefit our audience. Recall that our current model only made use of Ollama's llama 3.1:8B as the data extractor. The intention of this choice of model was to use as few parameters as possible to reduce compute resources and use a general-use model to make linking to a general RAG pipeline easier. Both of these considerations were with respect to accessibility. However, researchers interested in those areas may use more powerful models to derive more accurate locations, especially in cases where the location is unclear or implied.

## Conclusion

We present a generalized outline for how to use LLMs in data extraction, especially for mass-quantity document analysis. We discuss data scraping and cleaning as well as the effectiveness of LLMs in searching through source documents. Our initial results are promising, showing that LLMs provide similar to that of a human, sparing precious manpower as well avoiding unnecessary psychological stress from having to read details about brutal events, serving as a cognitive shield.
