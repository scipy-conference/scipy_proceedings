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

SHAP values as introduced by Lundberg and Lee [@DBLP:journals/corr/LundbergL17] are a specific implementation of 
Shapley values to explain predictions from machine learning models. It defines the SHAP values as a linear model of 
feature contributions [@molnar2025].

The SHAP explanation model $g$ is defined as:

$$
g(z) = \phi_0 + \sum_{i=1}^{M} \phi_i(i) z_i
$$

where 
- $z_i$ is a binary coalition variable indicating whether feature $i$ is present (1) or absent (0)
- $\phi_i(i)$ is the SHAP value for feature $i$ when it is present
- $\phi_0$ is the baseline prediction
- $M$ is the number of features

For a specific input $x$, the above equation simplifies to:

$$
g(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)
$$

where 
- $\phi_0$ is the baseline prediction, typically the mean prediction over the dataset
- $\phi_i(x)$ is the SHAP value for feature $i$ for input $x$
- $M$ is the number of features


### Estimation of SHAP Values

The exact calculation of SHAP values is computationally infeasible for most models due to the combinatorial 
nature of the Shapley value formula. The computational complexity is $O(2^M)$, where $M$ is the number of features. 
Therefore, various approximation methods are used to estimate the SHAP values, each tailored to different model types.

- **Kernel SHAP**: Uses a weighted linear regression to approximate shap values for arbitrary models.
Although it is applicable to any model type, it can be computationally expensive for large datasets.

- **Tree SHAP**: This method [@DBLP:journals/corr/abs-1802-03888] is specifically designed for tree-based models 
such as decision trees, random forests, and gradient boosted decision trees. It exploits the herarchical structure of 
decision trees to efficiently compute SHAP values [@lundberg2020local2global].

- **Deep SHAP**: Approximation method that uses a modified version of the DeepLIFT method [@DBLP:journals/corr/ShrikumarGK17] 
to estimate SHAP values for deep neural networks including CNNs. It leverages backpropagation to efficiently compute 
feature attributions.

In this paper, using the `shap` Python library, we will specifically employ Tree SHAP for GBDTs and Deep SHAP for CNNs.

## SHAP for Decision Trees

Gradient Boosted Decision Trees (GBDTs) [@friedman2001greedy; @DBLP:journals/corr/ChenG16; @ke2017lightgbm] are 
powerful, robust and interpretable models that are widely used in the industry. They are highly effective for 
structured tabular data, and can handle both numerical and categorical features in a seamless manner. GBDTs in general are 
interpretable models, and can be interpreted by analyzing the nodes and splits in the decision trees. However, 
as the complexity of the model as well as the number of features increases, it becomes more valuable to use a 
more principled approach like SHAP for interpretation.

### Dataset

The Bank Marketing dataset [@bank_marketing_222] contains information about direct marketing campaigns (phone calls)
of a Portuguese banking institution. It contains customer demographic information, financial details and interation
history. The goal is to predict whether a customer will subscribe to a term deposit [@Moro2014ADA]. The dataset is 
available on the UCI Machine Learning Repository.

The key details of the dataset are:

- **Number of samples**: 45,211
- **Number of features in the dataset**: 16
  - Number of features used in the model: 15
  - Removed feature: `duration` (in seconds)
- **Number of classes**: 2 (yes: 1, no: 0)
  - yes: the client will subscribe to a term deposit
  - no: the client will not subscribe to a term deposit
- **Objective**: Binary classification

Note that the `duration` feature is removed from the model training since it is not available before the call is made,
and hence cannot be used for prediction. Moreover, it is highly correlated with the target variable.

The dataset can be downloaded from the UCI Machine Learning Repository using the `ucimlrepo` Python library.

```{code-block} python
:linenos: true
:emphasize-lines: 4
:caption: Downloading the Bank Marketing dataset

import pandas as pd
from ucimlrepo import fetch_ucimlrepo

bank_marketing_ds = fetch_ucimlrepo(id=222)

x_df: pd.DataFrame = bank_marketing_ds.data.features
y_df: pd.DataFrame = bank_marketing_ds.data.targets
```

### Model

We use an XGBoost [@DBLP:journals/corr/ChenG16] model to predict whether a customer will subscribe to a term deposit
or not. Specifically, we follow the following steps to train the model:

- Convert the categorical features into pandas [@pandas1; @pandas2] categorical data type
  ```{code-block} python
  :caption: Converting categorical features to pandas categorical data type
  
  # x_df is the dataframe containing the features
  # Convert all columns to categorical type
  for col in x_df.columns:
      x_df[col] = pd.Categorical(x_df[col])
  ```
- Split the dataset into train and test sets
  - 80% for training and 20% for testing with random shuffling using `scikit-learn` [@sklearn1; @sklearn2]
  ```{code-block} python
  :caption: Splitting the dataset into train and test sets
  from sklearn.model_selection import train_test_split
    
  x_train, x_test, y_train, y_test = train_test_split(
      x_df, y_df, test_size=0.2, random_state=42
  )
  ```
- Train an XGBoost model on the train set
  - Use default hyperparameters
  - Use categorical features using the `enable_categorical` parameter
  ```{code-block} python
  :caption: Training an XGBoost model
  from xgboost import XGBClassifier

  xgb_model = XGBClassifier(
      enable_categorical=True, 
      objective="binary:logistic", 
      seed=42
  )
  xgb_model.fit(x_train, y_train)
  ```
- Evaluate the model on the test set

Note that we do not perform any feature engineering or hyperparameter tuning in this example, since the goal is to
demonstrate the use of SHAP for interpretation rather than building a high-performing model.


:::{figure} roc_xgb.png
:label: fig:bank-marketing-performance
Performance of the XGBoost model on the test set.
:::

The performance of the model on the test set is shown in {numref}`fig:bank-marketing-performance`.

### SHAP Explanations

We use the `shap` Python library [@lundberg2020local2global] to compute the SHAP values for the XGBoost model.

```{code-block} python
:linenos: true
:caption: Computing SHAP values for the XGBoost model

import shap

# xgb_model is the trained XGBoost model
# x_test is the test set

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test)
```

#### Global

The global explanations provide a high-level summary of the model's behavior. They are useful for understanding
the overall impact of each feature on the model's predictions.

The shap library provides a number of plotting functions to visualize the SHAP values. The code below uses the
`beeswarm` plot to visualize the global SHAP values.

```{code-block} python
:linenos: true
:caption: Computing global SHAP values for the XGBoost model

import matplotlib.pyplot as plt

ax = plt.subplot()
ax.grid(True)
ax = shap.plots.beeswarm(shap_values, max_display=16, show=False, log_scale=False)
ax.set_title("Global SHAP Values for XGBoost Classifier")
plt.tight_layout()
```

:::{figure} beeswarm_xgb.png
:label: fig:beeswarm-xgb
Global SHAP values for the XGBoost model on the Bank Marketing dataset.
:::

Note that in the {numref}`fig:beeswarm-xgb`, the features are ordered by their mean absolute SHAP values in
descending order. The x-axis shows the SHAP values, and the y-axis shows the features. 
Each point represents a sample from the test set. 

For numerical features (e.g., age and balance), the dots are color-coded based on the feature values—red signifies 
high values, and blue denotes low values. This color gradient helps visualize the relationship between the 
actual feature values and their contribution to predictions. For categorical features (e.g., marital and job), 
dots are typically displayed in gray because categorical variables do not have a natural ordering. Here, the focus 
is primarily on their SHAP value distribution rather than the specific category value.

The X-axis of the beeswarm plot represents SHAP values, which quantify the impact each feature has on the model's 
predictions. A positive SHAP value indicates that the feature pushes the model's prediction higher i.e. towards the 
positive class (yes), while a negative SHAP value implies the feature reduces the model's predicted value, 
i.e. towards the negative class (no). The distance from the center (zero) indicates the magnitude of this 
impact. The further away a point is from zero, the stronger the feature's influence on that specific prediction.

Thus, features with wide distributions across the X-axis have substantial variation in their impact on 
different instances, while features tightly clustered near zero exert relatively minor influence on the predictions.

In terms of insights, the beeswarm plot {numref}`fig:beeswarm-xgb` shows some interesting observations. For example,
customers with more balance (`balance` feature) in their account are more likely to subscribe to a term deposit. 
On the other hand, customers who have been contacted more frequently in the current compaign (`campaign` feature) 
are less likely to subscribe to a term deposit. However, the plot is not able to convey a deeper insight into the
top categorical features and how they affect the model's predictions. For that, we need to look at the dependency
plots.

#### Dependency

The dependency plots show the relationship between a feature and the model's predictions. They are useful for
understanding how the model uses a particular feature to make predictions.

The code block below defines a function to plot the dependency plot for a categorical feature.

```{code-block} python
:linenos: true
:caption: Computing categorical dependency plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def plot_shap_categorical(
  shap_values: shap.Explanation, 
  x: pd.DataFrame, 
  feature_name: str
 ) -> plt.Axes:
 
    # Extract SHAP values and feature values
    feature_idx = x.columns.tolist().index(feature_name)
    feature_values = x.iloc[:, feature_idx]
    shap_values_feature = shap_values[:, feature_idx].values

    # Map categories to numeric values for plotting
    categories = feature_values.unique()
    category_to_num = {cat: num for num, cat in enumerate(categories)}
    feature_values_numeric = feature_values.map(category_to_num)

    # Create scatter plot with categories on x-axis
    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        feature_values_numeric,
        shap_values_feature,
        alpha=0.7,
        s=120,
    )

    # Replace numeric x-ticks with category labels
    ax.set_xticks(
      ticks=np.arange(len(categories)), labels=categories, rotation=45, fontsize=12
    )

    # Reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="--")

    # Labels and title
    ax.grid(True)
    ax.set_xlabel(f"Categorical Feature: {feature_name}")
    ax.set_ylabel("SHAP Value")
    ax.set_title(f"SHAP Dependence Plot for Categorical Feature '{feature_name}'")

    plt.tight_layout()
    return ax
```

The dependency plot for the categorical feature `contact` can be plotted as:

```{code-block} python
:linenos: true
:caption: Plotting the dependency plot for the categorical feature `contact`

ax = plot_shap_categorical(shap_values, x_test, "contact")
```

:::{figure} dep_contact_xgb.png
:label: fig:contact-dependency
SHAP dependency plot for the categorical feature `contact` on the Bank Marketing dataset.
:::

The X-axis of the categorical dependency plot represents the categories of the feature, while the y-axis represents the 
SHAP values. Similar to the beeswarm plot, the positive SHAP values indicate that the feature pushes the model's 
prediction higher (towards the positive/yes class), while the negative SHAP values imply the feature reduces the 
model's predicted value (towards the negative/no class).

The {numref}`fig:contact-dependency` shows that the `contact` feature has a considerable impact on the 
model's predictions. Customers who were contacted via cellular (`cellular`) are more 
likely to subscribe to a term deposit than those who were contacted via telephone (`telephone`).

:::{figure} dep_categorical_xgb.png
:label: fig:dep_categorical_xgb
SHAP dependency plot for some of the other categorical features.
:::

The {numref}`fig:dep_categorical_xgb` shows the dependency plots for some of the other categorical features. 
On the top left, we see the dependency plot for the `housing` feature. Interestingly, customers who have a housing
loan (`yes`) are less likely to subscribe to a term deposit than those who do not have a housing loan (`no`).
On the bottom right, we see the dependency plot for the feature `poutcome`, which represents the outcome of the 
previous marketing campaign. Customers who were previously contacted and the outcome was a `success` are 
more likely to subscribe to a term deposit than those who were previously contacted and the outcome was a `failure`.


## SHAP for CNNs

Convolutional Neural Networks (CNNs) are a powerful class of deep learning models particularly well-suited for 
tasks involving spatial or temporal data, such as image recognition, speech recognition, and time-series 
classification. CNNs leverage convolutional operations and hierarchical feature extraction to capture 
intricate patterns within data, making them exceptionally effective for complex datasets 
[@lecun1998gradient; @krizhevsky2012imagenet].

However, despite being so effective, CNNs as with most deep learning models, are often treated as black boxes.
In this section, we will demonstrate how SHAP can be used to interpret the predictions of a one-dimensional CNN model.

### Dataset

The Human Activity Recognition Using Smartphones dataset [@human_activity_recognition_using_smartphones_240] 
contains sensor data collected from a smartphone. The data has been collected from a group of 30 volunteers
within the age group of 19-48 years. The goal is to classify 6 different physical activities from 
multivariate timeseries sensor data.

The key details of the dataset are:

- **Data type**: Multivariate timeseries
- **Number of training samples**: 7352
- **Number of testing samples**: 2947
- **Number of features in the dataset**: 563
- **Feature types**: Numeric
- **Number of classes**: 6
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying
- **Objective**: Multi-class classification

### Model

For this task, we are going to be constructing a 1D convolutional network using PyTorch [@paszke2017automatic].
Prior to model training, we need to perform some important preprocessing steps.

#### Preprocessing

1. Scale the data so that data is centered around zero mean and a unit variance.

   ```{code-block} python
   :linenos: true
   :caption: Data scaling
  
   from sklearn.preprocessing import StandardScaler
  
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)
   x_test = scaler.transform(x_test)
   ```

2. Create sliding window sequences

   ```{code-block} python
   :linenos: true
   :caption: Creating sliding window sequences
   
   from torch import Tensor

   # Convert to PyTorch tensors
   x_train = torch.from_numpy(x_train.to_numpy()).float()
   y_train = torch.from_numpy(y_train.to_numpy()).float()
   x_test = torch.from_numpy(x_test.to_numpy()).float()
   y_test = torch.from_numpy(y_test.to_numpy()).float()


   def create_sequences(
       data: Tensor, labels: Tensor, seq_length: int = 64
   ) -> tuple[Tensor, Tensor]:
       """Create sequences from the data and labels."""

       sequences = []
       labels_seq = []
       for i in range(len(data) - seq_length):
           sequences.append(data[i : i + seq_length])
           labels_seq.append(labels[i + seq_length])
       x_seq, y_seq = torch.stack(sequences), torch.stack(labels_seq)
       return x_seq.permute(0, 2, 1), y_seq


   SEQ_LENGTH = 64
   x_train_seq, y_train_seq = create_sequences(x_train, y_train, seq_length=SEQ_LENGTH)
   x_test_seq, y_test_seq = create_sequences(x_test, y_test, seq_length=SEQ_LENGTH)
   ```

#### Model Definition

Since we are focused on demonstrating the use of SHAP for interpretation, we are going to skip fine-tuning the
model for optimal performance. Instead, we are going to use 2 layer CNN architecture with a full connected layer
at the end for classification.

The following code block defines the CNN model class.

```{code-block} python
:linenos: true
:caption: Defining the CNN model class

import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 6):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
model = CNNClassifier(input_dim=x_train_seq.shape[1], num_classes=6)
```

#### Training

1. Define training parameters

   ```{code-block} python
   :linenos: true
   :caption: Defining training parameters
   
   NUM_EPOCHS = 50
   BATCH_SIZE = 32
   LEARNING_RATE = 1e-3
   
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
   ```

2. Create data loaders

   ```{code-block} python
   :linenos: true
   :caption: Creating data loaders
   
   from torch.utils.data import DataLoader, TensorDataset
   
   # Subtract 1 from the labels since the classes are 1-indexed
   train_ds = TensorDataset(x_train_seq, (y_train_seq - 1).long().squeeze())
   test_ds = TensorDataset(x_test_seq, (y_test_seq - 1).long().squeeze())
   
   # Avoid shuffling since we are using a time series dataset
   train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
   test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
   ```

3. Train the model

   ```{code-block} python
   :linenos: true
   :caption: Training the CNN model
  
   for epoch in range(NUM_EPOCHS):
       model.train()
       epoch_loss = 0.0
   
       for inputs, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(inputs)
          
           loss = criterion(outputs, labels)
           loss.backward()
          
           optimizer.step()
           epoch_loss += loss.item()
  
       avg_loss = epoch_loss / len(train_loader)
       print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.5f}")
   ```

#### Evaluate performance

The performance of the model on the test set is shown below.


| Class              | Precision | Recall | F1-Score | Support  |
|--------------------|-----------|--------|----------|----------|
| Walking            | 0.94      | 0.94   | 0.94     | 496      |
| Walking Upstairs   | 0.86      | 0.86   | 0.86     | 471      |
| Walking Downstairs | 0.83      | 0.81   | 0.82     | 420      |
| Sitting            | 0.81      | 0.82   | 0.82     | 467      |
| Standing           | 0.85      | 0.91   | 0.88     | 501      |
| Laying             | 0.95      | 0.88   | 0.91     | 528      |
| **Accuracy**       |           |        | **0.87** | **2883** |
| **Macro Avg**      | 0.87      | 0.87   | 0.87     | 2883     |
| **Weighted Avg**   | 0.88      | 0.87   | 0.87     | 2883     |

### SHAP Explanations

TODO: Add SHAP explanations for CNNs

#### Global

TODO

#### Dependency

TODO

#### Local

TODO


## Strengths and Limitations

SHAP is one of the prominent ways to explain ML model predictions. As with any other method, it has its own set of
strengths and limitations.

### Strengths

1. SHAP offers a consistent framework for interpreting model predictions, which has its roots from game theory. It is
   model agnostic and can be applied to any machine learning model, whether it is a simple linear regression model, a
   decision tree or a deep neural network.

2. SHAP can explain predictions for both global and local interpretability. Global explanations provide a high-level
   summary of the model's behavior, while local explanations provide insights into the factors influencing individual
   predictions.

3. The python SHAP library provides a number of plotting functions, e.g. beeswarm, force, waterfall, etc. that
   enable users to visualize the SHAP values in a number of ways. They communicate the insights effectively to
   both technical and non-technical stakeholders.


### Limitations

1. Exact calculation of SHAP values becomes exponentially expensive with increasing feature counts. 
   Specialized explainers e.g. TreeExplainer, GPUTreeExplainer [@mitchell2022gputreeshapmassivelyparallelexact], 
   and DeepExplainer mitigate this but can still be resource-intensive for complex models or very large datasets.

2. SHAP values can be misleading when features are highly correlated, and may allocate credit arbitrarily among them. 
   In such cases, the correct interpretation can be complicated, and may require domain expertise or 
   additional feature engineering.

3. While SHAP values quantify the impact of features on model predictions, it is important to understand that they 
   do not directly indicate causality.

## Conclusion

In this paper, we demonstrated SHAP’s practical application to two important classes of models: 
Gradient Boosted Decision Trees (GBDTs) and Convolutional Neural Networks (CNNs). Our case studies highlighted 
how SHAP facilitates explanations for both structured and unstructured data on a global and local level.
While SHAP effectively addresses many interpretability needs, users should be aware of challenges related to 
computational efficiency, correlated features, and the potential for over-interpretation.

Future work might focus on addressing these limitations, particularly through improved computational 
methods and strategies for handling feature correlations. Enhancing SHAP's usability further will contribute 
significantly to more transparent, reliable, and trusted machine learning applications in industry.
