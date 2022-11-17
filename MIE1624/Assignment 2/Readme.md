MIE1624 Assignment 2
====================

The goal of this homework is to implement ordinal logistic regression to predict the yearly compensation of each participant. The dataset used is from the  [2022 Kaggle ML & DS Survey](": https://www.kaggle.com/competitions/kaggle-survey2022").

The code includes the following:

* Data Cleaning
    * Remove features with too many missing values
    * Fill in some missing values
    * Encode data
* EDA & Feature Selection
    * Distribution by Features
    * Correlation Heat Map
    * Random Forest
    * Feature Importance
* Model Implementation
    * Ordinal Logistic Regression with cross validation
    * Tune best regularization strength (C) based on bias/error
* Model Tuning
    * GridSearch to find best set of hyperparameter for each sub models
* Discussion
    * Test model on training and testing set
    * Compare oringal distribution vs training and testing distribution
    



