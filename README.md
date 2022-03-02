# Udacity-Data-Scientist-Nanodegree-project

The code and data in this repository is part of my project submissions for the Udacity Data Scientist Nanodegree.

The objective is set as follows:

In this notebook, we will use use regression, a supervised learning technique used to predict continuous outcomes - Housing Prices.

In summary, I intend to cover the following:
1. Prepare the data
        - Gather necessary data to answer our questions
        - Handle categorical and missing data
        - Provide insight into the methods you chose and why you chose them

2. Analyze, Model, and Visualize
        Provide a clear connection between your business questions and how the data answers them.

For specific business questions, I will attempt to answer the follwing questions:

1. Find out if any of features affect housing pries using exloratory data analysis
2. Develop a model to predict the final price of each home
3. Evaluate our model to ascetain that it is suitable for predicting housing prices


The dataset used is a competition dataset used for data science education, and is hosted on Kaggle.

You can obtain the dataset via this [link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

The blogpost related to this notebook is found [hear](https://medium.com/@asilavictor/predicting-house-prices-using-linear-regression-381be1bfd200)

## Data

***Brief Descrition of the Datasets used**

I have downloaded the data and saved it on my local drive. The dataset is a zipped file and will have to be unzipped for on to acces the 
datasets for interest. The zipped file contains three .csv files - train.csv, test.csv, sample_submission.csv and data_description.txt .
Train.csv contains the training sample, while test.csv contains the test samples. Detailed descriptions for the datasets are contained 
in the data_descriptions.txt
For this notebook, I wil not be overly interested in the sample_sbumission.csv


## Libraries used
The following libraries have been used:
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. zipfile
6. warnings
7. sklearn

## Summary

One of the most important types of data analysis is regression analysis. Regression analysis is a way of mathematically sorting out which of those variables does indeed have an impact. It answers the questions: Which factors matter most? Which can we ignore? How do those factors interact with each other? And, perhaps most importantly, how certain are we about all of these factors?
Before performig regression analysis, one could have an idea of the answers to these questions by performing some exploratory data analysis, the includes some basic descptives like correlations measures, averages, etc, while also having a visual sense of the data.

## References

References have been made from the book "Hands-on Machine Learning with Sklearn, Keras and Tensorflow 2nd Edition, by Aurélien Géron", "Python Machine Learning 3rd Edition by Sebastian Raschka & Vahid Mirjalili", stackoverflow and kaggle.com
For the last two, I have borrowed code snippets that I have used to build on mine.
