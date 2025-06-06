---
# Ensure that this title is the same as the one in `myst.yml`
title: Explaining ML predictions with SHAP
abstract: |
  As machine learning models become increasingly accurate and complex, 
  explainability has become essential to ensure trust, transparency, and informed decision-making. 
  SHapley Additive exPlanations (SHAP) provide a rigorous and intuitive approach for 
  interpreting model predictions, delivering consistent and theoretically grounded feature attributions. 
  This article demonstrates the application of SHAP across two representative model types: 
  boosted decision trees and neural networks. 
  
  We utilize the UCI Adult Income dataset with an XGBoost model to predict if a client will subscribe 
  to a term deposit in a bank and the Human Activity Recognition Using Smartphones dataset 
  with a convolutional neural network (CNN) to classify activities (e.g., walking, sitting) 
  based on sensor data. The paper concludes with a discussion of SHAP's practical utility, strengths, 
  and limitations, guiding readers on effective usage in real-world scenarios.
---

## Introduction

Machine Learning models have advanced to the point where they are being used in high-stakes decision-making processes.
Finance, healthcare and technology are just a few examples of industries that are 
using machine learning to make decisions that affect people's lives. However, the increase in performance of these 
models have come at the cost of interpretability. This is especially true for models with complex architectures
such as deep neural networks.

In the industry, model decisions can influence loan approvals, medical diagnoses or hiring decisions. Therefore, an
inability to explain why a model produced a certain output can hinder adoption, erode user trust and raise ethical
or compliance concerns. Deep neural networks and Large Language Models (LLMs) despite being powerful tools, are often
treated as black boxes. 

SHapley Additive exPlanations (SHAP [@DBLP:journals/corr/LundbergL17]) is one such explainability method 
that aims to make models more interpretable. It uses a game-theoretic approach that provides a way to interpret 
the predictions of any machine learning model. SHAP provides a mechansim to understand the contribution of each feature
to the prediction of a model. It provides different levels of interpretability ranging from a global view of the model
to local explanations for individual predictions.

In this article, we demonstrate how SHAP can be used to interpret two classes of models that are especially common in industry:

- **Gradient Boosted Decision Trees** (GBDTs), which are widely used for structured tabular data due to their strong 
performance and relative robustness
- **Convolutional Neural Networks** (CNNs), which are popular in domains like image and sensor data where 
spatial relationships matter.

We illustrate the use of SHAP with two representative datasets:

- The Bank Marketing dataset [@bank_marketing_222] for the GBDT model, where the task is to predict whether a 
customer will subscribe to a term deposit based on demographic and interaction data.

- The Human Activity Recognition Using Smartphones dataset [@human_activity_recognition_using_smartphones_240] 
for the CNN model, where the goal is to classify 6 different physical activities from time-series sensor 
data collected from mobile devices.

Through these use cases, we will demonstrate the application of SHAP to two different model types and data modalities.
Finally, we will discuss the practical utility, strengths, and limitations of SHAP, and provide guidelines for its effective
use in real-world scenarios.


## Core Concepts

In this section, we shall briefly go over the key concepts necessary to understand the approach of SHAP,
and its application to neural networks and GBDTs.

### Shapley Values

Shapley values, originating from cooperative game theory, were introduced by Lloyd Shapley [@shapley1953value]. 
They provide a consistent method to fairly distribute credit or reward among players in a cooperative game. 
In the context of machine learning, each "player" corresponds to a feature, and the "game" is the prediction 
task. The Shapley value for a feature quantifies its average marginal contribution to the prediction across 
all possible coalitions of features.

Formally, the Shapley value for a feature $i$ is defined as:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \; (|N| - |S| - 1)!}{|N|!} \left[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_{S}(x_{S}) \right]
$$

where 
- $N$ is the set of all features
- $S$ is a subset of features excluding feature $i$
- $f_S$ is the model's prediction function when only the features in $S$ are used
- $x_S$ is the input vector with only the features in $S$.


### SHAP Values

SHAP values are a specific implementation of Shapley values for machine learning models. They provide a way to 
attribute the prediction of a model to its input features. SHAP values are calculated for each feature in the input, 
and the sum of the SHAP values for all features is equal to the difference between the model's prediction for the 
input and the average prediction of the model.


## SHAP for GBDTs

Gradient Boosted Decision Trees (GBDTs) are a popular class of models in industry due to their strong performance
and relative robustness. They are also amenable to interpretation using SHAP.

### Dataset

The Bank Marketing dataset [@bank_marketing_222] contains information about direct marketing campaigns (phone calls)
of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe (1/0)
to a term deposit (variable y).

### Model

We use an XGBoost model to predict whether a customer will subscribe to a term deposit based on demographic and 
interaction data. The model is trained on the Bank Marketing dataset.

### SHAP Explanations

TODO: Add SHAP explanations for GBDTs


## SHAP for CNNs

Convolutional Neural Networks (CNNs) are a popular class of models for image and sensor data. They are also amenable
to interpretation using SHAP.

### Dataset

The Human Activity Recognition Using Smartphones dataset [@human_activity_recognition_using_smartphones_240] 
contains sensor data collected from mobile devices. The goal is to classify 6 different physical activities 
from time-series sensor data.

### Model

We use a convolutional neural network to classify activities from time-series sensor data. The model is trained on the 
Human Activity Recognition Using Smartphones dataset.

### SHAP Explanations

TODO: Add SHAP explanations for CNNs


## Practical Utility, Strengths, and Limitations

SHAP provides a consistent and theoretically grounded way to interpret model predictions. It is applicable to a wide range of
model types and can provide both global and local explanations. However, it can be computationally expensive to compute
SHAP values for large models or datasets. Additionally, SHAP values are based on the assumption that the model is a 
coalitional game, which may not always be the case.

## Conclusion

In this article, we have demonstrated the application of SHAP to two different model types and data modalities.
We have shown that SHAP can provide both global and local explanations for model predictions. We have also discussed
the practical utility, strengths, and limitations of SHAP. We hope that this article will serve as a guide for
practitioners to effectively use SHAP in real-world scenarios.
