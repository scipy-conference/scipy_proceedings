# Style Guide for SciPy Conference Proceedings

Please refer to this guide along with the current [README](https://github.com/scipy-conference/scipy_proceedings/blob/2024/README.md) of the repository for the proceedings.

There is a page limit of 8 pages on the paper, excluding references.

For general Style Guide details please check [IEEE style guide](https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/style_references_manual.pdf). For inclusive language, please refer to [American Psychological Association’s style guide](https://www.apa.org/about/apa/equity-diversity-inclusion/language-guidelines). This style guide is based on both these references. Use [Strunk and White 4th edition](https://archive.org/details/TheElementsOfStyle4thEdition) as a grammar reference. We use [Merriam-Webster](https://www.merriam-webster.com/) as the English dictionary.


### Mamba Models a possible replacement for Transformers?

In the paper title, capitalize the first letter of the first and last word and all the nouns, pronouns, adjectives, verbs, adverbs, and subordinating conjunctions (If, Because, That, Which). Capitalize abbreviations that are otherwise lowercase (e.g., use DC, not dc or Dc) except for unit abbreviations and acronyms. Articles (a, an, the), coordinating conjunctions (and, but, for, or, nor), and most short prepositions are lowercase unless they are the first or last word. Prepositions of more than three letters (Before, Through, With, Without, Versus, Among, Under, Between) should be capitalized.


### Body of a Paper

This is the main section of your paper and should be divided into the following subsections with clear headings:

1. Introduction
  
      1. Sequence Modeling in Scientific Computing: Briefly explain the importance of sequence modeling in scientific domains like bioinformatics, time series analysis, and physical simulations. Provide examples of tasks like protein structure prediction, gene sequence analysis, and time series forecasting.
    
      2. Transformers: The Current Standard: Introduce Transformers as the leading architecture for sequence modeling. Briefly mention their core concepts like attention mechanisms. Acknowledge limitations of Transformers such as high computational complexity for scientific computing tasks.
    
      3. Mamba Models: A Promising Alternative: Briefly introduce Mamba models as a novel architecture with potential advantages for scientific computing. Highlight key features like selective matrix parameters and potential for memory efficiency.

  2. Mamba Model Architecture
    2.1 Core Concepts of Mamba Models: Explain how Mamba models learn selective matrix parameters, reducing memory usage compared to Transformers. Discuss the role of the SRAM cache (if applicable) in improving efficiency for long sequences.
        2.2 Comparison to Transformers' Architecture: Clearly explain how Mamba models differ from Transformers, particularly the absence of attention mechanisms. Discuss potential trade-offs between the two approaches.

  3. Applications in Scientific Computing
        3.1 Bioinformatics Applications: Explain how Mamba models can be used for tasks like protein structure prediction or gene sequence analysis. Emphasize how their memory efficiency is crucial for handling large biological datasets.
        3.2 Time Series Analysis Applications: Discuss how Mamba models can be used to analyze long time series data in finance, weather forecasting, or sensor readings. Highlight the advantage of their ability to handle long sequences effectively.
        3.3 Potential Applications in Physical Simulations (Optional): Briefly explore the potential of using Mamba models for simulating complex physical systems.

  4. Comparison and Results
        4.1 Performance Comparison with Transformers: Compare the performance of Mamba models with Transformers on specific scientific computing tasks mentioned in section 3 (e.g., protein structure prediction, time series forecasting). Utilize relevant metrics like accuracy, memory footprint, and training time. Present results in tables or visualizations for clarity. Discuss potential limitations of Mamba models compared to Transformers in specific scenarios.

  5. Conclusion
        Summarize the key findings on the potential of Mamba models for scientific computing.
        Discuss the future directions of Mamba model research and potential areas for improvement.
        Emphasize how Mamba models align with the focus of SciPy'24 by offering memory efficiency and handling long sequences effectively in scientific computing tasks.

6. References

Every reference should be a separate entry. Using one number for more than one reference is not allowed. Please refer to your specific publication guidelines for formatting references.


#### Abstract

Every published paper must contain an Abstract. Abstracts shall not contain numbered mathematical equations or numbered references.

We will talk about why Mamba models are better than Transformers and explain their architecture.

Framing for SciPy'24:

  Focus on scientific applications: Highlight how Mamba models benefit scientific tasks like bioinformatics or time series analysis beyond just language.
  
  Computational efficiency: Emphasize Mamba's advantages in memory usage and handling long sequences, which are crucial for scientific computing.
    
  Comparison to established methods: Clearly explain how Mamba models differ from Transformers, particularly in their architecture (no attention mechanism).
  
  Python implementation: Briefly mention the potential for implementing Mamba models using Python libraries like TensorFlow or PyTorch.

  Briefly explain sequence modeling and its importance in scientific domains like bioinformatics, time series analysis, and physical simulations.

  Introduce Transformers as the current leading architecture for sequence modeling, mentioning their limitations (e.g., computational complexity) for scientific tasks.

  Briefly introduce Mamba models, highlighting their potential to address these limitations, particularly in terms of memory efficiency and handling long sequences.

  Mamba Model Architecture with SciPy Applications in Mind:

  Deep dive into the core concepts of Mamba models, emphasizing aspects beneficial for scientific computing:
  
  Selective Matrix Parameters: Explain how Mamba models focus on learning relevant parameters, reducing memory usage compared to Transformers' full attention matrices.
  
  SRAM Cache (if applicable): Discuss how this feature (if present in the specific Mamba model you're studying) helps with caching frequently used parameters, further improving efficiency for long sequences.
  
  Comparison to Transformers' Architecture: Clearly explain how Mamba models deviate from Transformers, especially the absence of attention mechanisms.

#### Appendix

Numbering should be (Appending I) or (Appendix A)


#### Acknowledgements

They should appear after the text of the paper, before references, after appendix. Do not use Mr., Mrs., Ms., or Miss (list first initial and last name only). Use the Dr. or Prof. title with each name separately; do not use plural Drs. or Profs. with lists of names.

The acknowledgement sections should be written in the third person.


#### Author Affiliations

Abbreviations should be written out in Affiliations (e.g. LANL should be written as Los Alamos National Lab).


#### References

Every reference should be a separate entry. Using one number for more than one reference is not allowed.


#### Text Citation of Figures and Tables

All citations of figures and tables in text must be in numerical order. Citations to figures in text always carry the abbreviation "Fig." followed by the figure number. The abbreviation is used even when it begins a sentence.



### Other Text


#### Footnotes

Footnotes should be numbered in consecutive order throughout the text. The footnote numbers are superscripts in text and in the actual footnotes. In text, place the superscript footnote numbers after the punctuation such as fullstops, commas, and parentheses, but before colons, dashes, quotation marks, and semicolons in a compound sentence. The footnotes should be placed at the bottom of the text column in which they are cited.


#### List in Text

The ordering of labeling for all lists is 1), 2), 3) followed by a), b), c), and then i), ii), iii).

For example, first list goes as this: 1) first item; 2) second item; and 3) third item.


#### Editorial style


##### Acronyms

Adding abbreviations to the metadata means we have accessible abbreviations across all instances of abbreviations in the manuscript. ([reference](https://mystmd.org/guide/glossaries-and-terms#abbreviations))

Define acronyms the first time they appear in the Abstract as well as the first time they appear in the body of the paper, written out as part of the sentence, followed by the acronym in parentheses. If the acronym is not repeated in the Abstract, do not include the acronym in parentheses. Coined plurals or plurals of acronyms do not take the apostrophe (e.g., FETs).

Possessive forms of the acronym do take the apostrophe (e.g., CPU’s speed). Indefinite articles are assigned to abbreviations to fit the sound of the first letter (e.g., an FCC regulation; a BRI).


##### Plurals

Plurals of units of measure usually do not take the "s". For example, the plural form of 3 mil is 3 mil, but 3 bits/s instead of 3 bit/s. Plural forms of calendar years do not take the apostrophe (e.g., 1990s). To avoid confusion, plural forms of variables in equations do take the apostrophe (e.g., x’s).


### Inclusive language

This section of the style guide is copied from the APA - [American Psychological Association’s style guide](https://www.apa.org/about/apa/equity-diversity-inclusion/language-guidelines). Refer to that for more details.

Avoid using identity-first language while talking about disabilities, either a person is born with or they are imposed later. This is not applicable for chosen identities, e.g, educators, programmers, etc.


| Terms to avoid      | Suggested alternative        |
| ------------------- | ---------------------------- |
| elderly             | senior citizen               |
| subject             | particiant                   |
| wheel-chair bound <br> confined to a wheelchair    | person who uses a wheelchair <br> wheelchair user|
| confined to a wheelchair | wheelchair user         |
| mentally ill <br>crazy <br>insane <br>mental defect <br>suffers from or is afflected with [condition]| person living with a mental illness <br>person with a preexisting mental health disorder <br>person with a behavioral health disorder <br>person with a diagnosis of a mental illness/mental health disorder/behavioral health disorder |
| asylum              | psychiatric hospital/facility |
| drug user / abuser <br>addict  | person who uses drugs <br>person who injects drugs <br> person with substance use disorder|
| alcoholic <br> alcohol abuser  | person with alcohol use disorder <br> person in recovery from substance use/alcohol disorder |
| person who relapsed | person who returned to use   |
| smoker             | person who smokes             |
| homeless people <br> the homeless <br> transient population | people without housing <br>people experiencing homelessness <br>people experiencing unstable housing/housing insecurity/people who are not securely housed <br>people experiencing unsheltered homelessness <br>clients/guests who are accessing homeless services <br>people experiencing houselessness <br> people experiencing housing or food insecurity |
| prostitute         | person who engages in sex work <br> sex worker (abbreviated as SWer) |
| prisoner <br>convict | person who is/has been incarcerated |
| slave                | person who is/was enslaved |


### General Language Suggestions

(Thanks to Chris Calloway and Renci for the following instructions.)

Shortest sentences are the best. Some ways to shorten the sentences and make those professional are as follows:



1. Avoid "which" and "that". For example, don’t use – "the model that we trained". Instead use – "we trained a model"
2. Avoid using pronouns like "I", "we", etc. For example, avoid "we trained the model"; instead use "the trained model …"
3. Avoid passive voice.
4. Avoid using questions, use statements instead. For example, avoid "which metrics would be useful for success"; instead use "the success metrics here are …"
5. Avoid words, verbs with emotions. For example, avoid "reflecting on; theme"; use "in short, or summarizing; topic"