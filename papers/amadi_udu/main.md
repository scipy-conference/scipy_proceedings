---
# Ensure that this title is the same as the one in `myst.yml`
title: Computational Resource Optimisation in Feature Selection under Class Imbalance Conditions
abstract: |
  Feature selection is crucial for reducing data dimensionality as well as enhancing model interpretability and performance in machine learning tasks. However, selecting the most informative features in large dataset often incurs high computational costs. This study explores the possibility of performing feature selection on a subset of data to reduce the computational burden. The study uses five real-life datasets with substantial sample sizes and severe class imbalance ratios between 0.09 – 0.18. The results illustrate the variability of feature importance with smaller sample fractions in different models. In this cases considered, light gradient-boosting machine exhibited the least variability, even with reduced sample fractions, while also incurring the least computational resource.

---
(sec:introduction)=
## Introduction

In the development of prediction models for real-world applications, two key challenges often arise: high-dimensionality resulting from the numerous features, and class-imbalance due to the rarity of samples in the positive class. Feature selection methods are  utilised to address issues of high-dimensionality by selecting a smaller subset of relevant features, thus reducing noise, increasing interpretability, and enhancing model performance [@Cai2018; @Dhal2022; @Udu2023a]. 

Studies [@Yin2013; @Tsai2020; @deHaro-Garcia2020; @Matharaarachchi2021] on the performance of feature selection methods with class imbalance data have been undertaken on using synthetic and real-life datasets. A significant drawback noted was the computational cost of their approach on large sample sizes.  While experimental investigations of feature selection amid class imbalance conditions have been studied in the literature, there is a need to further understand the effect of sample size on performance degradation of feature selection methods. This would offer valuable insights into tackling the associated resource expense involved in undertaking feature selection with respect to large sample sizes where class-imbalance exists, for a wide range of applications. 

This study investigates the impact of performing feature selection on a reduced dataset on feature importance and model performance, using five real-life datasets characterised by large sample sizes and severe class imbalance structures. We employ a feature selection process that utilises permutation feature importance (PFI) and evaluate the feature importance on three selected models; namely light gradient-boosting machine (Light GBM), random forest (RF) and support vector machines (SVM). These models are popular in real-world machine learning (ML) studies and also serve as a benchmark for comparing novel models [@bonaccorso2018; @feng2019; @sarker2021; @paleyes2022; @udu2024a]. Feature importance was evaluated using the area under the Receiver Operator Characteristics (ROC) curve, commonly referred to as AUC owing to its suitability in class imbalance problems [@Luque2019; @Temraz2022]. The development of the ML framework and data visualisation in this study was facilitated by several key Python libraries. Pandas [@pandas1] and NumPy [@numpy] were used for data loading and numerical computations, respectively. Scikit-learn [@sklearn1] provided tools for data preprocessing, model development, and evaluation. Matplotlib [@matplotlib] was employed for visualising data structures. Additionally, the SciPy [@scipy] library's cluster, spatial, and stats modules were crucial for hierarchical clustering, Spearman rank correlation, and distance matrix computations.

The rest of the paper is organised as follows: @sec:methodology briefly outlines the methodology adopted, while @sec:results presents the results and discussion. The conclusion of the study is provided in @sec:conclusion.

(sec:methodology)=
## Methodology
### Description of datasets
Five real-life datasets from different subject areas were considered in this study. Four of the datasets were obtained from the UC Irvine ML repository, including CDC Diabetes Health Indicator [@diabetes], Census Income [@census_income], Bank Marketing [@bank_marketing], and Statlog (Shuttle) [@statlog]. The fifth dataset is Moisture Absorbed Composite [@Osa-uwagboe2024] from a damage morphology study. The datasets are presented in @tbl:dataset_summary. Notably, all datasets exhibited high class imbalance ratios from 0.09 - 0.18 (i.e., the ratio of the number of samples in the minority class over that of the majority class).

:::{table} Summary of datasets used in the study
:label: tbl:dataset_summary
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th style="text-align: center;">Features</th>
      <th style="text-align: center;">Instances</th>
      <th>Subject Area</th>
      <th style="text-align: center;">Imbalance Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Diabetes health indicator</td>
      <td style="text-align: center;">20</td>
      <td style="text-align: center;">253,680</td>
      <td>Health and Medicine</td>
      <td style="text-align: center;">0.16</td>
    </tr>
    <tr>
      <td>Census income</td>
      <td style="text-align: center;">14</td>
      <td style="text-align: center;">48,842</td>
      <td>Social Science</td>
      <td style="text-align: center;">0.09</td>
    </tr>
    <tr>
      <td>Bank marketing</td>
      <td style="text-align: center;">17</td>
      <td style="text-align: center;">45,211</td>
      <td>Business</td>
      <td style="text-align: center;">0.13</td>
    </tr>
    <tr>
      <td>Statlog (shuttle)</td>
      <td style="text-align: center;">7</td>
      <td style="text-align: center;">58,000</td>
      <td>Physics and Chemistry</td>
      <td style="text-align: center;">0.18</td>
    </tr>
    <tr>
      <td>Moisture absorbed composite</td>
      <td style="text-align: center;">9</td>
      <td style="text-align: center;">295,461</td>
      <td>Mechanics of Materials</td>
      <td style="text-align: center;">0.11</td>
    </tr>
  </tbody>
</table>
:::

Building data-driven models in the presence of  high dimensionality includes several steps such as data preprocessing, feature selection, model training and evaluation. To address class imbalance issues during model training, an additional resampling step may be performed to adjust the uneven distribution of class samples [@Udu2023b; @rezvani2023; @Udu2024b]. This paper, however, focuses on the feature selection method, model training, and  the evaluation metrics adopted.

###  Feature selection and model training
To maintain a model-agnostic approach that is not confined to any specific ML algorithm, this study employed PFI for feature selection. PFI assesses how each feature affects the model's performance by randomly shuffling the values of a feature and noting the resulting change in performance. In essence, if a feature is important, shuffling its values should significantly reduce the model's performance since the model relies on that feature to make predictions. A positive importance score suggests that a feature is useful for the model's prediction as permuting the values of the feature led to a decrease in the model’s performance. Conversely, a negative importance score suggests that a feature might be introducing noise and the model might perform better without it. Thus, PFI interrupts the link between a feature and its predicted outcome, enabling us determine the extent to which a model relies on a particular feature [@Li2017; @sklearn1; @Kaneko2022]. It is noteworthy that the effect of permuting one feature could be negligible when features are collinear. Hence, an important feature may report a low score. To tackle this, a hierarchical cluster on a Spearman rank-order correlation can be adopted, with a threshold taking from visual inspection of the dendrograms in grouping features into clusters and selecting the feature to retain.

Datasets were loaded using pandas, and categorical features were encoded using one-hot encoding. The Spearman correlation matrix was computed and then converted into a distance matrix. Hierarchical clustering was subsequently performed using Ward’s linkage method, and  a threshold for grouping features into clusters was determined through visual inspection of the dendrograms, allowing for the selection of features to retain. Subsequently, the investigation proceeded in two steps. In step 1,  all samples of the respective dataset was used. The dataset was split into training and test sets based on a test-size of 0.25. The respective classifiers were initialised using their default hyper-parameter settings and fitted on the training data. Thereafter, PFI was computed on the fitted model with number of times a feature is permuted set to 30 repeats. Lastly, the change in AUC was evaluated on the test set.

In the second step, we initiate three for-loops to handle the different features, fractions of samples, and repetition of the PFI process undertaken in step 1. Sample fraction sizes were taken from 10% – 100% in increments of 10%, with the entire process randomly repeated 10 times. This provided an array of 300 AUC scores for each sample fraction and respective feature of the PFI process. To ensure reproducibility,  the random state for the classifiers, sample fractions, data split, and permutation importance were predefined.  Computation processes were accelerated using the joblib parallel library on the Sulis High Performance Computing platform.  A sample source code of step 2 is presented:

```python
# Define the function for parallel execution
def process_feature(f_no, selected_features, df):
    for frac in np.round(np.arange(0.1, 1.1, 0.1), 1).tolist():  #loop for sample fractions
        for rand in range(10): #loop for 10 repeats of the process
            df_new = df.sample(frac=frac, random_state=rand)
...
            pfi = permutation_importance(model, X_val, y_val, n_repeats=30,
                                       random_state=rand, scoring='roc_auc', n_jobs=-1)
    return final_df
# Parallelise computation 
results = Parallel(n_jobs=-1)(delayed(process_feature)(f_no, selected_features, df) for f_no in range(len(selected_features)))
```

(sec:results)=
## Results and Discussions
The hierarchical cluster and Spearman’s ranking for moisture absorbed composite dataset is shown in [Figure 1a](#hiercorr-a) and [b](#hiercorr-a) respectively (Frequency Centroid – FC, Peak Frequency – PF, Rise Time – RT, Initiation Frequency – IF, Average Signal Level – ASL, Duration – D, Counts – C, Amplitude – A and Absolute Energy – AE). Based on the visual inspection of the hierarchical cluster, a threshold of 0.8 was selected, thus, retaining features RT, C, ASL, and FC. 

:::{figure} 
:alt: Hierarchical cluster and Spearman correlation for GSVS
:width: 30%
:align: center
:label: fig:hiercorr
(hiercorr-a)=
![](./images/gsvs_hierclus_cmap.jpg)

Feature relationship for moisture absorbed composite dataset; (a) hierarchical cluster, (b) Spearman correlation ranking.
:::

As observed in [Figure 1a](#hiercorr-a), Frequency Centroid and Peak Frequency are in the same cluster with a highly correlated value of 0.957 shown in [Figure 1b](#hiercorr-a). Similarly, Rise Time and Initiation Frequency are clustered with a highly negative correlation of -0.862. Amplitude and Absolute Energy also exhibited a high positive correlation of 0.981. 

@tbl:result_table gives the median and interquartile (IQR) feature importance scores based on change in AUC for the LightGBM, RF and SVM models. These scores were obtained using all samples in the PFI process. Values emphasised in bold fonts represent the  highest ranked feature for the respective models based on their median change in AUC.

:::{table} Median and IQR feature importance scores based on change in AUC for LightGBM, RF and SVM models, (values in bold fonts represent the  highest ranked feature for the respective models).
:label: tbl:result_table
<table border="1">
  <tr>
    <th colspan="11">Census Income</th>    <th> </th>    <th colspan="11">Bank Marketing</th>
  </tr>
  <tr>
    <th colspan="1">ID</th>    <th colspan="1">Feature</th>    <th colspan="3">LightGBM</th>    <th colspan="3">RF</th>    <th colspan="3">SVM</th>    <th></th>    <th colspan="1">ID</th>    <th colspan="1">Feature</th>    <th colspan="3">LightGBM</th>    <th colspan="3">RF</th>    <th colspan="3">SVM</th>
  </tr>
  <tr>
    <th> </th>    <th> </th>     <th>Med</th>    <th colspan="2">IQR </th>    <th>Med</th>    <th colspan="2">IQR </th>    <th>Med</th>    <th colspan="2">IQR </th>    <th> </th>    <th> </th>    <th> </th> 
    <th>Med</th>    <th colspan="2">IQR </th>    <th>Med</th>    <th colspan="2">IQR </th>    <th>Med</th>    <th colspan="2">IQR </th>
  </tr> 
  <tr>
    <th> </th>    <th> </th>     <th> </th>     <th>25<sup>th</sup></th>    <th>75<sup>th</sup></th>    <th> </th>    <th>25<sup>th</sup></th>    <th>75<sup>th</sup></th>    <th> </th>    <th>25<sup>th</sup></th>
    <th>75<sup>th</sup></th>    <th> </th>    <th> </th>    <th> </th>     <th> </th>     <th>25<sup>th</sup></th>    <th>75<sup>th</sup></th>    <th> </th>    <th>25<sup>th</sup></th>    <th>75<sup>th</sup></th>
    <th> </th>    <th>25<sup>th</sup></th>    <th>75<sup>th</sup></th>  </tr>  
<tr>
    <td>0</td><td>Age</td> <th>0.117</th> <th>0.114</th> <th>0.121</th> <th>0.066</th> <th>0.061</th> <th>0.069</th> <td><10<sup>-3</sup></td> <td><10<sup>-3</sup></td> <td><10<sup>-3</sup></td> <td> </td>
    <td>0</td><td>Age</td> <td>0.016</td> <td>0.015</td> <td>0.017</td> <td>0.026</td> <td>0.024</td> <td>0.027</td> <td>-0.001</td> <td>-0.001</td> <td>0.002</td>
 </tr>
 <tr>
    <td>1</td> <td>Final weight</td> <td>-0.002</td> <td>-0.003</td> <td>-0.001</td> <td><10<sup>-3</sup></td> <td>-0.003</td> <td>0.004</td> <td>0.003</td> <td><10<sup>-3</sup></td> <td>0.011</sup></td>
    <td></td><td>1</td>
    <td>Balance</td><td>0.013</td> <td>0.011</td> <td>0.014</td> <td>0.011</td> <td>0.009</td><td>0.012</td><td>0.026</td> <td>0.021</td> <td>0.027</td>
</tr>
 <tr>
    <td>2</td><td>Education-num</td><td>0.085</td><td>0.080</td><td>0.087</td><td>0.063</td><td>0.061</td><td>0.068</td><td><10<sup>-3</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td></td>
    <td>2</td><td>Day of week</td><td>0.012</td><td>0.011</td><td>0.013</td><td>0.014</td><td>0.014</td><td>0.016</td><td>0.001</td><td>0.001></td> <td>0.001</td>
 </tr>
 <tr>
    <td>3</td><td>Capital-gain</td><td>0.049</td><td>0.048</td><td>0.052</td><td>0.047</td><td>0.046</td><td>0.050</td><th>0.029</th><th>0.026</th><th>0.030</th><td></td>
    <td>3</td><td>Duration</td><th>0.256</th><th>0.253</th><th>0.261</th><th>0.211</th><th>0.209</th><th>0.215</th><th>0.154</th><th>0.148</th> <th>0.157</th>
 </tr>
 <tr>
    <td>4</td><td>Workclass</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td>0.001</td><td>0.003</td><td>0.001</td><td>0.004</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td></td>
    <td>4</td><td>PDays</td><td>0.051</td><td>0.051</td><td>0.052</td><td>0.054</td><td>0.052</td><td>0.055</td><td>0.053</td><td>0.050</td> <td>0.055</td>
 </tr>
 <tr>
    <td>5</td><td>Race</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td>0.001</td><td>0.001</th><td>0.002</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td></td>
    <td>5</td><td>Job_b</td><td>0.003</td><td>0.003</td><td>0.004</td><td>0.002</td><td>0.001</th><td>0.002</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td>
 </tr>
 <tr>
    <td colspan="11", rowspan="2"></td><td></td><td>6</td><td>Job_m</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.002</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td>
  </tr>
  <tr>
    <td></td></td><td>7</td><td>Housing</td><td>0.026</td><td>0.025</td><td>0.027</td><td>0.032</td><td>0.031</td><td>0.034</td><td>0.001</td><td>0.001</td><td>0.001</td>
  </tr>
  <tr>
    <th colspan="23"></th>
  </tr>
  <tr>
    <th colspan="11">Statlog (Shuttle)</th>
    <th> </th>
    <th colspan="11">Diabetes</th>
  </tr>
   <tr>
     <td>0</td><td>Rad Flow</td><th>0.355</th><th>0.350</th><th>0.360</th><th>0.387</th><th>0.383</th><th>0.389</th><td>0.253</td><td>0.249</td><td>0.259</td><td></td>
    <td>0</td><td>HighBP</td><th>0.128</th><th>0.127</th><th>0.129</th><th>0.128</th><th>0.128</th><th>0.130</th><th>0.066</th><th>0.065</th><th>0.067</th>
  </tr>
  <tr>
    <td>1</td><td>Fpv Close</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.012</td><td>0.011</td><td>0.013</td><td><10<sup>-3</sup></td><td><10<sup>-3</sup></td><td>0.001</td>
    <td></td><td>1</td><td>CholCheck</td><td>0.009</td><td>0.009</td><td>0.010</td><td>0.011</td><td>0.010</td><td>0.011</td><td>-0.001</td><td>-0.001</td><td>-0.001</td>
  </tr>
  <tr>
    <td>2</td><td>Fpv Open</td><td>0.241</td><td>0.239</td><td>0.244</td><td>0.274</td><td>0.270</td><td>0.277</td><th>0.319</th><th>0.316</th><th>0.322</th>
    <td></td><td>2</td><td>BMI</td><td>0.080</td><td>0.078</td><td>0.081</td><td>0.079</td><td>0.077</td><td>0.080</td><td>-0.073</td><td>-0.074</td><td>-0.072</td>
  </tr>
  <tr>
  <td colspan="11"></td><td></td><td>3</td><td>Smoker</td><td>0.004</td><td>0.004</td><td>0.005</td><td>0.004</td><td>0.004</td><td>0.005</td><td>0.026</td><td>0.025</td><td>0.027</td>
  </tr>
  <tr>
  <td colspan="23"> </td>
  </tr>
  <tr>
  <th colspan="11">Moisture Absorbed Composites</th>
      <th> </th>
  </tr>
  <tr>
   <td>0</td><td>Risetime</td> <td>0.005</td>    <td>0.005</td>    <td>0.005</td>    <td>0.009</td>    <td>0.008</td>    <td>0.009</td><td>0.004</td><td>0.003</td><td>0.004</td>
  <td></td>
  </tr>							
  <tr>
   <td>1</td><td>Counts</td>    <td>0.037</td>    <td>0.037</td>    <td>0.037</td>    <td>0.075</td>    <td>0.073</td>    <td>0.075</td>    <td>0.009</td>    <td>0.009</td>    <td>0.009</td>
  <td></td>
  </tr>
  <tr> 
   <td>2</td>    <td>ASL</td>    <td>0.034</td>    <td>0.034</td>    <td>0.034</td>    <td>0.072</td>    <td>0.071</td>    <td>0.073</td>    <td><10<sup>-3</sup></td> <td><10<sup>-3</sup></td>    <td><10<sup>-3</sup></td>
  <td></td>
  </tr>
  <tr> 
   <td>3</td>    <td>Freq. Centroid</td>    <th>0.468</th>    <th>0.466</th>    <th>0.470</th>    <th>0.463</th>    <th>0.461</th>    <th>0.465</th>    <th>0.422</th> <th>0.421</th>    <th>0.425</th>
  <td></td>
  </tr>
</table>
:::
   
From @tbl:result_table, SVM tended to be have very low scores in some datasets, possibly due to its reliance of support vectors in determining the decision boundaries. Thus, features with strong influence at the decision boundary but not directly affecting the support vectors may seem less important.   For the Moisture Absorbed Composite dataset, the three classifiers reported similar scores for Frequency Centroid of 0.468, 0.466 and 0.422 respectively in @tbl:result_table. 

However, in Bank Marketing dataset, LightGBM and RF identified Feature 1 as a relatively important feature, while SVM considered it insignificant. The mutability of importance scores for the classifiers considered underscores the need to explore multiple classifiers when undertaking a comprehensive investigation of feature importance for feature selection purposes. 

[Figure 2](#db_time-a) shows the PFI process time and corresponding sample fractions for the Diabetes dataset, which has a substantial sample size of 253,680 instances. The results are based on one independent run, with PFI set at 30 feature-permuted repeats. For LightGBM and RF, the PFI process time increased linearly with larger sample fractions, whereas SVM experienced an exponential growth. LightGBM had the lowest computational cost, with CPU process times of 3.9 seconds and 28.8 seconds for 10% and 100% sample fractions, respectively. SVM required 21,263 seconds to process the entire dataset, reflecting a 9,345% increase in CPU computational cost compared to using a 10% sample fraction. SVM's poor performance relative to LightGBM and RF is likely due to its poor CPU parallelisability.

:::{figure}
:alt: PFI process time and corresponding sample fractions for the Diabetes dataset.
:label: fig:db_time
:width: 30%
:align: center
(db_time-a)= 
![](./images/time_plot.png)

PFI process time and corresponding sample fractions for the Diabetes dataset.
:::

[Figure 3a](#ci_boxplot-a) - [c](#ci_boxplot-c) present the PFI for Final Weight feature of Census Income dataset, evaluated across different sample fractions using LightGBM, RF, and  SVM models, respectively. The change in AUC indicates the impact on model performance when Final Weight feature is permuted.  Generally, for smaller sample fractions, there was a higher variability in AUC and prominence of outliers. This could be attributed to the increased influence of randomness, fewer data points, and sampling fluctuations for smaller sample fractions across the datasets.

:::{figure} 
:alt: Sample fractions and corresponding change in AUC for Final Weight feature of Census Income dataset
:width: 20%
:align: center
:label: fig:ci_boxplot
(ci_boxplot-a)=
![](./images/lgbm_census_income_feature_1.png)
(ci_boxplot-b)=
![](./images/rf_census_income_feature_1.png)
(ci_boxplot-c)=
![](./images/svm_census_income_feature_1.png)

Sample fractions and corresponding change in AUC for Final Weight feature of Census Income dataset; (a) LightGBM, (b) RF, and (c) SVM.
:::

:::{figure} 
:alt: Sample fractions and corresponding change in AUC for Duration feature of Bank Marketing dataset
:width: 20%
:align: center
:label: fig:bm_boxplot
(bm_boxplot-a)=
![](./images/lgbm_bank_marketing_feature_3.png)
(bm_boxplot-b)=
![](./images/rf_bank_marketing_feature_3.png)
(bm_boxplot-c)=
![](./images/svm_bank_marketing_feature_3.png)

Sample fractions and corresponding change in AUC for Duration feature of Bank Marketing dataset; (a) LightGBM, (b) RF, and (c) SVM.
:::

:::{figure} 
:alt: Sample fractions and corresponding change in AUC for Rad Flow feature of Statlog
:width: 20%
:align: center
:label: fig:ss_boxplot
(ss_boxplot-a)=
![](./images/lgbm_statlog_shuttle_feature_0.png)
(ss_boxplot-b)=
![](./images/rf_statlog_shuttle_feature_0.png)
(ss_boxplot-c)=
![](./images/svm_statlog_shuttle_feature_0.png)

Sample fractions and corresponding change in AUC for Rad Flow feature of Statlog (Shuttle) dataset; (a) LightGBM, (b) RF, and (c) SVM.
:::


For LightGBM model in [Figure 3a](#ci_boxplot-a), the median change in AUC was close to  zero,  indicating that Final Weight had minimal impact on model performance, as noted in @tbl:result_table. Similar results were recorded in [Figure 4a](#bm_boxplot-a) - [c](#bm_boxplot-c) for the Duration feature of Bank Marketing dataset, where all models exhibited similarly high feature importance scores. Even for sample fractions of 0.5, LightGBM and RF appeared to give similar importance scores to using the entire data sample. On the other hand, SVM exhibited a higher median change in AUC, indicating that the Final Weight feature had a more significant impact on its performance. Additionally, SVM showed the greatest variability and the most prominent outliers, particularly at lower sample fractions. This was noticeable in [Figure 5a](#ss_boxplot-a) - [c](#ss_boxplot-c), where all classifiers reported similar importance scores as noted in @tbl:result_table. This variability and the presence of outliers suggest that the model's performance is less stable when features are permuted. 

PFI can provide insights into the importance of features, but it is susceptible to variability, especially with smaller sample sizes. Thus, complementary feature selection methods could be explored to validate feature importance. Future work could investigate the variability of features under particular models and sample sizes, with a view to evolving methods of providing a more stable information to the models. 

(sec:conclusion)=
## Conclusion
Feature selection for large datasets incurs considerable computational cost in the model development process of various ML tasks. This study undertakes a preliminary investigation into the influence of sample fractions on feature importance and model performance in datasets characterised by class imbalance. Five real-life datasets with large sample sizes from different subject fields which exhibited high class imbalance ratios of 0.09 – 0.18 were utilised. 

Due to its model-agnostic nature, PFI was adopted for feature selection process with feature importance evaluated on Light GBM, RF and SVM. The models were chosen due to their widespread use in real-world ML studies and their role as benchmarks for comparing new models. Cluster, spatial, and stats sub-packages of SciPy were instrumental in tackling the multicollinearity effects associated with PFI. Using a PFI approach, the study revealed the variability of feature importance with smaller sample fractions in LightGBM, random forest and SVM models. In the cases explored, LightGBM showed the lowest variability, while SVM exhibited the highest variability in feature importance. Also, Light GBM had the least CPU process time across the cases considered, while SVM showed the highest computational cost.

In future work, this investigation would be expanded to substantially larger datasets and introduce some quantitative measure of the variability of various models and feature selection methods. An understanding of the variability of feature importance can inform feature engineering efforts that provides means of alleviating the variability of feature importance in samples fractions under class imbalance conditions.


## Acknowledgement
This work was supported by the Petroleum Technology Development Fund under grant PTDF/ED/OSS/PHD/AGU/1076/17 and NISCO UK Research Centre. Computations were  performed using the Sulis Tier 2 HPC platform hosted by the Scientific Computing Research Technology Platform at the University of Warwick. Sulis is funded by EPSRC Grant EP/T022108/1 and the HPC Midlands+ consortium.
