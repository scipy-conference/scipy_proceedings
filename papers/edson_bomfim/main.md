---
title: "A Practical Guide to Data Quality: Problems, Detection, and Strategy"
abstract: |
  The proliferation of data-driven systems, from machine learning models to business intelligence platforms, has placed unprecedented importance on the quality of the underlying data. The principle of "garbage in, garbage out" has never been more relevant, as poor data quality can lead to flawed models, incorrect business decisions, and a fundamental lack of trust in data systems.

  This paper serves as a practical guide for software developers and data practitioners to understand, identify, and address data quality issues. We present a comprehensive taxonomy of 23 common data quality problems, synthesizing definitions and contexts from empirical software engineering and database literature. Subsequently, we catalogue 22 distinct methods for detecting these issues, categorizing them into actionable strategies: Rule-Based & Constraint Enforcement, Statistical & Machine Learning Analysis, Data Comparison & Standards, and Human & Manual Review.

  By mapping specific problems to their detection-method categories, we provide a structured framework to help teams develop robust, efficient, and targeted data validation strategies.

  We conclude by discussing how the systematic practice of "data testing," analogous to software testing, not only improves pipeline maintainability and dataset trustworthiness but also enhances the overall design and reliability of data-intensive systems.
---

## Introduction

In the modern technological landscape, data is the foundational asset that powers innovation, from scientific research, training sophisticated machine learning (ML) models to driving critical business analytics. A dataset can be defined as a structured collection of individual data points, or instances, that hold information about a set of entities sharing some common characteristics. The utility and reliability of any system built upon this data are inextricably linked to its quality.

This paper treats the quality of data as the degree to which it accurately and faithfully represents the real-world phenomena it purports to describe [@anchoring-data-quality] [@data-linter]. This is a crucial distinction. For example, a common challenge in machine learning is class imbalance, where one class is significantly underrepresented. However, if this imbalance accurately reflects the real world (e.g., fraudulent transactions are rare), the dataset itself is not of poor quality; rather, it is a perfect representation of a difficult problem. The challenge then lies with the modeling technique, not the data's fidelity. Our focus is on errors where the data *fails* to represent the world correctly.

The consequences of poor data quality are severe, leading to the well-known "garbage in, garbage out" paradigm. Flawed data can introduce subtle biases in algorithms, generate misleading analytical reports, and erode stakeholder trust, ultimately undermining the value of data-driven initiatives [@other-face-of-big-data] [@data-quality-info-and-decision-making]. Despite its importance, a systematic, developer-focused approach to identifying and mitigating these issues is often lacking.

This paper aims to fill that gap by providing a practical, empirical guide for data practitioners. We synthesize a comprehensive taxonomy of data quality problems and a corresponding catalogue of detection methods, drawing from extensive research in empirical software engineering, data cleaning, database management and the author own experience. The goal is to equip developers with a structured framework to reason about, test for, and strategically address data quality within their systems.

The paper is structured as follows:
*   **Section 2: Data Quality Problems** details a taxonomy of 23 common data quality problems, providing concise definitions and relevant context for each, and organizes them into practical categories.
*   **Section 3: Methods to Detect Data Quality Problems** presents a catalogue of 22 methods used to detect these problems, discusses their automation potential, and groups them into four strategic categories. This section includes a mapping of which method categories are effective for which errors.
*   **Section 4: Conclusions** discuss with the strategic implications of adopting a proactive data quality assurance mindset, drawing parallels between data testing and established software testing practices to argue for a more robust approach to building data systems.


## Data Quality Problems

A data quality problem, or "dirty data," is any instance where the data fails to correctly represent the real-world entity or event it describes [@taxonomy-of-dity-data] [@problems-methods-data-cleasing] [@survey-of-data-quality-tools]. These problems can be subtle or overt, but all carry the risk of corrupting analysis and undermining model performance. Below is a taxonomy of common data quality problems compiled from the literature.

### Taxonomy of Data Quality Problems

1.  **Missing Data / Incompleteness / Empty Values:** Records or fields that should contain a value but are null, empty, or otherwise not populated. This is one of the most common issues, forcing decisions about imputation or removal [@taxonomy-data-quality-challenges] [@comments-on-software-defects].
2.  **Duplicates / Redundancy:** The same real-world entity is represented by two or more records in a dataset. These can be exact duplicates or approximate (fuzzy) duplicates with minor variations [@survey-of-data-quality-tools] [@ajax] [@the-field-matching-problem].
3.  **Inconsistency / Contradicting Records / Semantic Consistency Violations:** A single real-world entity is described by multiple records that contain conflicting information (e.g., the same customer with two different birth dates). This also includes violations of semantic rules (e.g., a patient's discharge date is before their admission date) [@survey-of-data-quality-tools] [@HEINRICH201895].
4.  **Wrong / Incorrect Data / Invalid Data:** Data values that are syntactically valid but factually wrong (e.g., an age of "30" for a person who is actually 40). This is one of the hardest problems to detect without an external source of truth [@problems-methods-data-cleasing] [@ajax].
5.  **Misspellings:** Typographical errors in textual data, which can hinder matching and categorization (e.g., "Jhon" instead of "John") [@ajax] [@problems-methods-data-cleasing].
6.  **Outdated Temporal Data / Timeliness / Currency / Volatility:** The data was correct at the time of recording but is no longer valid due to real-world changes (e.g., a customer's address before they moved). This dimension measures how up-to-date the data is [@RIDZUAN2024341] [@problems-methods-data-cleasing].
7.  **Non-standardized / Non-Conforming Data / Irregularities:** The use of different formats or units to represent the same information (e.g., "USA", "U.S.A.", "United States"; or measurements in both Celsius and Fahrenheit) [@problems-methods-data-cleasing] [@problems-methods-data-cleasing].
8.  **Ambiguous Data / Incomplete Context:** Data that can be interpreted in multiple ways without additional context. This can arise from abbreviations ("J. Stevens" for John or Jack) or homonyms ("Miami" in Florida vs. Ohio) [@problems-methods-data-cleasing] [@ajax].
9.  **Embedded / Extraneous Data / Value Items Beyond Attribute Context:** A single field contains multiple pieces of information that should be separate (e.g., a "name" field containing "President John Stevens") or information that belongs in another field (e.g., a zip code in an address line) [@ajax] [@problems-methods-data-cleasing].
10. **Misfielded Values:** Correct data is stored in the wrong field (e.g., the value "Portugal" in a "city" attribute) [@ajax].
11. **Outliers / Noise:** Erroneous data points that lie significantly outside the typical distribution of values, often due to measurement or entry errors. They are distinct from valid but extreme data points [@taxonomy-data-quality-challenges].
12. **Insufficient / Imprecise Metadata:** The metadata (data about the data) is missing or does not adequately describe the metrics, entities, or collection process, leading to a high risk of misinterpretation [@doi:10.1007/s10664-024-10518-9] [@RIDZUAN2024341].
13. **Schema Violations / Wrong Data Type / Structural Conflicts:** Data that violates the formal schema of the database, such as incorrect data types (e.g., "XY" in an age field), or different structural representations for the same object across systems [@ajax] [@survey-of-data-quality-tools].
14. **Dangling Data / Referential Integrity Violation:** A record refers to another record that does not exist, breaking a relational link (e.g., an order record with a `customer_id` that does not exist in the customers table) [@ajax] [@problems-methods-data-cleasing].
15. **Concurrency Control / Transaction Issues:** Data corruption arising from the lack of proper transaction management in databases, leading to issues like lost updates or dirty reads. While typically handled by the DBMS, it is a source of poor data quality [@problems-methods-data-cleasing].
16. **Wrong Categorical Data:** A value for a categorical attribute is outside the predefined set of valid categories (e.g., `shipping_method` is "air" when the only valid options are "ground" or "sea") [@problems-methods-data-cleasing].
17. **Name Conflicts / Homonyms / Synonyms:** When integrating data, the same name is used for different objects (homonyms) or different names are used for the same object (synonyms), causing structural ambiguity [@survey-of-data-quality-tools].
18. **Different Aggregation Levels:** In data integration, the same entity is represented at different levels of detail across sources (e.g., sales data aggregated by day in one source and by month in another) [@formal-definition].
19. **Circularity Among Tuples in Self-Relationship:** A logical impossibility where entities in a self-referencing relationship form a cycle (e.g., Product A is a sub-product of B, and Product B is a sub-product of A) [@formal-definition].
20. **Semi-empty Tuple:** A record where a large number of its attributes are empty, but not all. This may indicate a partial data entry failure or a distinct sub-type of entity that was not modeled correctly [@formal-definition].
21. **Data Miscoding:** The incorrect assignment of data types during processing, such as treating a numerical feature as a string. This is a process-induced error rather than a source data error [@bhatia2024dataqualityantipatternssoftware].
22. **Distribution Shift / Data Drift:** The statistical properties of the data change over time or between different contexts (e.g., training vs. production), rendering models trained on older data less accurate [@bhatia2024dataqualityantipatternssoftware].
23. **Plausibility:** Data values that are syntactically correct and within a valid range, but are highly improbable in the given context [@comparison-of-string-distances].


#### A Note on Excluded Concepts

*   **Trust/Trustworthiness:** While critically important, trust is a property of the relationship between a data consumer and a data source, not an intrinsic property of the dataset itself. It is often established over time through repeated positive interactions and is influenced by the provenance and perceived reliability of the source [@taxonomy-data-quality-challenges]. Therefore, it is considered outside the scope of direct data quality measurement.
*   **Accessibility/Availability/Security:** These dimensions relate to the systems and procedures governing access to the data, not the content of the data itself [@RIDZUAN2024341]. While a dataset that cannot be accessed is useless, we focus here on the quality of the data once it has been accessed.

### Detection Scalability

Data quality problems can be broadly divided into two groups based on the scope required for their detection:
*   **Single-Datapoint Errors:** These errors can be identified by examining a single record or data point in isolation. Examples include `Missing Data`, `Schema Violations`, and `Misspellings`. Detection for these errors is highly parallelizable and scales linearly with the size of the dataset, as each data point can be checked independently.
*   **Multi-Datapoint Errors:** These errors require the context of other records, or even the entire dataset, to be identified. `Duplicates`, `Outliers`, and `Distribution Shift` fall into this category. Detection can be computationally expensive, often requiring full-dataset statistics. While optimizations like sorting or indexing can help, adding new data may necessitate a full re-evaluation of the entire dataset.

### Categorizing Data Quality Problems

While a flat list of problems is useful, categorization helps in forming a mental model for tackling them. A common approach in the literature is to group them by the aspect of quality they violate [@RIDZUAN2024341]:

*   **Content Quality (Accuracy & Integrity):** Problems related to the correctness and internal consistency of the data. Includes *Inconsistency*, *Wrong Data*, *Misspellings*, *Outliers*, *Schema Violations*, and *Dangling Data*.
*   **Completeness (Sufficiency of Information):** Problems concerning missing information. Includes *Missing Data*, *Semi-empty Tuple*, and *Insufficient Metadata*.
*   **Representational & Structural Quality (Form & Understandability):** Problems with how the data is structured and represented. Includes *Non-standardized Data*, *Embedded Data*, *Misfielded Values*, *Name Conflicts*, and *Wrong Categorical Data*.
*   **Timeliness & Relevance (Fitness for Current Context):** Problems related to the data being out-of-date or contextually inappropriate. Includes *Outdated Temporal Data* and *Distribution Shift*.

While this categorization is useful for understanding the *impact* of poor data quality, a more practical approach is to categorize problems based on the *methods required to detect them*. This pragmatic view directly informs the implementation of a data quality assurance strategy, which we explore in the next section.


## Methods to Detect Data Quality Problems

Detecting the problems outlined in Section 2 requires a diverse toolkit of methods, ranging from simple, deterministic rules to complex statistical models. Below, we catalogue these methods and discuss some aspects of their application.

```{list-table} Mapping Detection Methods to Data Quality Problems
:label: tbl:detection-to-problems
:header-rows: 1
* - Method
  - Applicable Data Quality Problems Found
* - **Database Constraints** (Not Null, Unique, Primary Key, Foreign Key)
  - Missing Data, Duplicates, Dangling Data, Schema Violations
* - **User-Specified Integrity Constraints & Triggers**
  - Inconsistency, Wrong Data, Wrong Categorical Data, Circularity
* - **Direct Checks** (for Null, Empty Strings, Dummy Values)
  - Missing Data, Semi-empty Tuple
* - **Quantitative & Statistical Metrics**
  - Missing Data (e.g., completion rate), Distribution Shift
* - **Comparison & Similarity** (Direct, Distance, String Metrics)
  - Duplicates, Inconsistency, Misspellings, Non-standardized Data
* - **Sorted Neighbourhood Method**
  - Duplicates (approximate)
* - **Attribute Cardinality Checks**
  - Duplicates (where cardinality should equal row count)
* - **Redundancy Analysis**
  - Redundancy (correlated features)
* - **Semantic & Rule-Based Testing** (incl. Regex, Set Validation)
  - Inconsistency, Wrong Data, Misfielded Values, Schema Violations, Non-standardized Data
* - **Probability-based Metrics & Statistical Tests**
  - Inconsistency (Semantic), Plausibility
* - **Association Rule Mining**
  - Inconsistency, Wrong Data
* - **Comparison to Reference Data** (Real World, Legal Values)
  - Wrong / Incorrect Data, Outdated Temporal Data
* - **Explicit Type Checking & Domain Format Checks**
  - Schema Violations, Non-standardized Data
* - **Spell Checker**
  - Misspellings
* - **Temporal Constraint Checks**
  - Outdated Temporal Data
* - **Distribution Analysis** (L-infinity, Jensen-Shannon, etc.)
  - Distribution Shift, Outliers
* - **Outlier Detection Algorithms** (ML, Statistical, Distance-based)
  - Outliers, Noise
* - **Clustering Algorithms**
  - Outliers, Duplicates (by grouping similar records)
* - **Lookup Tables & Dictionaries** (Thesaurus, Name/Address Dirs)
  - Non-standardized Data, Ambiguous Data, Misspellings
* - **Metadata & Schema Evaluation** (Metamodels, DQRs)
  - Insufficient / Imprecise Metadata, Schema Violations, Name Conflicts
* - **Cycle Detection Algorithms**
  - Circularity Among Tuples
* - **Data Profiling & Human Review**
  - All (serves as a general-purpose discovery and verification method)
```

### Automation and Maintenance:

The methods above vary significantly in their potential for automation. Database constraints are fully automated, while methods like statistical analysis and rule-based testing can be incorporated into automated data pipelines. Others, such as resolving ambiguous data using a thesaurus or verifying incorrect data against the real world, fundamentally require human-in-the-loop intervention.

Just as software testing is an integral part of CI/CD pipelines, data quality tests should be an integral part of data pipelines. Whenever data is updated, ingested, or transformed, a corresponding suite of data tests should be executed. Single-datapoint checks can be run efficiently on new or changed data. However, for multi-datapoint checks, an incremental update may not be sufficient; a full-dataset re-evaluation is often necessary to ensure issues like new duplicates or shifts in the overall distribution are caught.


### Categories of Detection Methods

To provide a clear, actionable framework, we first present a table mapping individual detection methods to the data quality problems they are primarily used to find. A more strategic approach is to group these methods into broader categories.
We propose four categories of data quality detection methods, which group the 22 techniques into strategic approaches:


```{list-table} Categories of Data Quality Detection Methods
:label: tbl:cat-detection
:header-rows: 1
* - Category
  - Methods Included
* - **Rule-Based & Constraint Enforcement**
  - Not Null Constraints, Unique/Primary Key Constraints, General DB Integrity Constraints, Triggers, Rule-based Testing, Regex Pattern Verification, Set Validation, Type Checking/Constraints, Temporal Constraints, Domain Format Checks, Cycle Detection Algorithms, Formal Specifications (DQRs, DPRs).
* - **Statistical & Machine Learning Analysis**
  - Quantitative Metrics, Statistical Criterion/Sampling, Redundancy Analysis, Probability-based Metrics, Association Rule Mining, Distribution Analysis (L-infinity, Jensen-Shannon, etc.), All Outlier Detection Methods (ML, Statistical, Distance-based), Clustering, Trimmed Means.
* - **Data Comparison & Standards**
  - Direct Comparison, Distance Metrics, String Similarity Metrics, Sorted Neighbourhood, Comparison to Reference Data, Standardization/Normalization, Lookup Tables/Dictionaries, Thesaurus, Name/Address Directories.
* - **Human & Manual Review**
  - Identifying Dummy Values, Data Profiling, Metadata Evaluation, Human Verification/Manual Inspection, Domain Expert Intervention.
```

This categorization allows for a structured approach to building a data quality firewall. A team can start with the most automatable category, "Rule-Based & Constraint Enforcement," and progressively add more sophisticated "Statistical & ML Analysis" and "Data Comparison" methods, while reserving "Human & Manual Review" for exceptions and complex cases.

Also we present the @tbl:cat-detection-to-problems which provides a high-level map for practitioners, showing which categories of methods are effective at identifying specific data quality problems.

```{list-table} Linking Data Problems to Detection Method Categories
:label: tbl:cat-detection-to-problems
:header-rows: 1
* - Data Quality Problem
  - Rule-Based & Constraint
  - Statistical & ML
  - Data Comparison & Standards
  - Human & Manual
* - Missing Data / Incompleteness
  -  ✔
  -  ✔
  - 
  -  ✔
* - Duplicates / Redundancy
  -  ✔
  -  ✔
  -  ✔
  -  ✔
* - Inconsistency / Contradictions
  -  ✔
  -  ✔
  - 
  -  ✔
* - Wrong / Incorrect Data
  -  ✔
  -  ✔
  -  ✔
  -  ✔
* - Misspellings
  - 
  - 
  -  ✔
  -  ✔
* - Outdated Temporal Data
  -  ✔
  - 
  -  ✔
  -  ✔
* - Non-standardized Data
  -  ✔
  - 
  -  ✔
  -  ✔
* - Ambiguous Data
  - 
  - 
  -  ✔
  -  ✔
* - Embedded / Extraneous Data
  -  ✔
  - 
  - 
  -  ✔
* - Misfielded Values
  -  ✔
  - 
  - 
  -  ✔
* - Outliers / Noise
  - 
  -  ✔
  - 
  -  ✔
* - Insufficient / Imprecise Metadata
  - 
  - 
  - 
  -  ✔
* - Schema Violations
  -  ✔
  - 
  - 
  -  ✔
* - Dangling Data / Referential Integrity
  -  ✔
  - 
  - 
  - 
* - Concurrency / Transaction Issues
  -  ✔
  - 
  - 
  - 
* - Wrong Categorical Data
  -  ✔
  -  ✔
  - 
  -  ✔
* - Name Conflicts
  - 
  - 
  - 
  -  ✔
* - Different Aggregation Levels
  - 
  - 
  -  ✔
  -  ✔
* - Circularity Among Tuples
  -  ✔
  - 
  - 
  - 
* - Semi-empty Tuple
  -  ✔
  - 
  - 
  -  ✔
* - Data Miscoding
  -  ✔
  - 
  - 
  -  ✔
* - Distribution Shift / Data Drift
  - 
  -  ✔
  - 
  -  ✔
* - Plausibility
  - 
  -  ✔
  - 
  -  ✔
```


## Conclusions

This paper has provided a comprehensive, practical-focused framework for understanding, identifying, and addressing data quality problems. By cataloguing 23 distinct types of "dirty data" and 22 corresponding detection methods, we have created a guide that can the turned into actionable strategies.

The key takeaway is the need to treat data quality assurance as a first-class citizen in the development of data systems. Understanding the fundamentals—the specific ways data can be dirty and the array of tools available to detect it—allows a team to optimize its validation strategy. Instead of applying a generic, one-size-fits-all approach, they can select the most efficient methods for their specific risks, whether that is enforcing a strict schema with **Rule-Based** methods, discovering novel anomalies with **Statistical** analysis, or standardizing inputs with **Data Comparison** techniques.

Integrating these methods into automated "data testing" suites within data pipelines is critical for long-term maintainability and trust. Just as unit tests verify code logic, data tests verify data integrity. This practice provides continuous feedback, catches regressions when data sources or processing logic change, and builds confidence in the final datasets used for researches, analytics, and machine learning.

Perhaps the most profound benefit of this approach lies in the design phase. The very process of defining data quality requirements and designing data tests forces a team to deeply consider potential failure modes. It requires them to ask critical questions: What are the invariants of our data? What are the semantic rules that must hold? What are the statistical norms? This process, much like writing the design of a prospective study or writing a formal specification for software, clarifies assumptions and exposes ambiguities *before* a single line of data processing code is written. The resulting data system is not only more robust but also better designed, because it is built on a foundation of clearly articulated expectations about the data it will handle.





