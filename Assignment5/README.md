# Airbnb Superhost Prediction Model

## Overview

This project builds a machine learning classification model to predict whether an Airbnb host is a **Superhost** based on listing and host-related features. The goal is to develop an end-to-end machine learning workflow, including data preparation, model training, evaluation, feature selection, and model persistence.

This project was completed as part of the **Break Through Tech AI Program** and demonstrates the machine learning lifecycle from problem definition to deployment preparation.

---

## Problem Statement

Airbnb Superhosts are hosts who meet higher standards for guest experience, reliability, and activity. Predicting Superhost status can help identify characteristics associated with successful hosts and support data-driven decisions in the hospitality industry.

**Machine Learning Task:**

* Type: Supervised Learning
* Problem Type: Binary Classification
* Target Variable: `host_is_superhost`

  * `True` → Host is a Superhost
  * `False` → Host is not a Superhost

---

## Dataset

The project uses the Airbnb listings dataset (`airbnbData_train`).

The dataset includes:

* Host information
* Listing characteristics
* Review information
* Availability details
* Other engineered features

The dataset was already preprocessed with:

* One-hot encoding for categorical variables
* Feature scaling for numerical variables
* Missing value imputation

---

## Machine Learning Approach

### 1. Data Preparation

* Loaded the Airbnb dataset into a Pandas DataFrame
* Defined:

  * Features (`X`): all columns except `host_is_superhost`
  * Label (`y`): `host_is_superhost`
* Split the dataset into training and testing sets

### 2. Model Training

Two Logistic Regression models were developed:

#### Baseline Model

* Logistic Regression with default hyperparameter values
* `C = 1.0`
* Maximum iterations: 1000

#### Optimized Model

* Used `GridSearchCV` with 5-fold cross-validation
* Tested multiple values of regularization parameter `C`
* Selected the best-performing hyperparameter value

---

## Model Evaluation

Models were evaluated using:

* Confusion Matrix
* Precision-Recall Curves
* ROC Curves
* Area Under the Curve (AUC)

### Feature Selection

Used `SelectKBest` with ANOVA F-test (`f_classif`) to identify the most important features.

Results:

* Performance changed as the number of selected features increased
* Best AUC achieved: **0.8232**
* Best feature count: **45 out of 49 features**

This suggests that retaining most features improved model performance while balancing complexity.

---

## Model Persistence

The final Logistic Regression model was saved using Python's `pickle` library:

```
BTT_Assignment_5_Superhost_Model.pkl
```

The saved model was successfully loaded and used to generate predictions on new test data.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle
* Jupyter Notebook

---

## Key Takeaways

* Built a complete machine learning pipeline for binary classification
* Applied hyperparameter tuning using GridSearchCV
* Evaluated model performance using multiple classification metrics
* Explored feature selection to improve model efficiency
* Saved a trained model for future predictions

---

## Future Improvements

Potential improvements include:

* Testing additional classification algorithms such as Random Forest, XGBoost, or Neural Networks
* Performing additional feature engineering
* Optimizing classification thresholds based on business goals
* Deploying the model through an API for real-time predictions
