# Sentiment Analysis on Ukraine's War
The purpose of this project is to compute the sentiment of text information – in our case, social media posts/tweets posted recently on the war of Ukraine against russia - and answer the research question: “What can public opinion tell us about the russia’s war against Ukraine in 2022?”

## Project 

This project was divided in three part and used thee dataset. The first one was **sentiment_analysis,csv** to train our model. Then once our model was trained, we tested the results on **tw_reply.csv** and **30K Tweets with russiaukrainewar hashtag**. 

## Project Part 1

The first step of the project was to develop a model to classify positive and negative sentiment of tweets. **Sentiment_analysis.csv** already have some labelled associated to those tweets. In this part we followed the following steps:

* Data Preprocessing 
* Vectorization of words
* Algorithm Implementation and hyperparameter tuning :
    * Logistic Regression
    * Naive Bayes
    * SVM
    * Decision Tree
    * XGBoost

## Project Part 2

Once the best model was determined, we used it to predict the sentiment of replies from Elon Musk Tweet. The original 5 tweets can be found in **tw.csv** and all the replies in **tw_reply.csv**. The model was also tested on the following dataset **30K Tweets with russiaukrainewar hashtag.csv**. The code can be modified to add your own custom dataset.  In this part we did the following:

* Applied best ML model
* Viewed predictions
* Benchmarked model to state-of-the-art model:
    * ONE AI
    * FlairNLP

## Project Part 3 

In this section, we looked at the most frequent word and generated word cloud to draw insight on what might cause those sentiments.

## Initial Setup \ Dependencies

* Run requirements.txt
* All datasets can be found in the Data folder
* We followed our analysis with some quick Power BI analysis which can be found there Project_Visualisation.pbix
* The report we wrote based on our analysis can be found there MIE1624-Report-Group18.pdf
