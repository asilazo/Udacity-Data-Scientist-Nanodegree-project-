# Recommendations With IBM project

## Introduction

In this project, I analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they'll like.

## Motivation

For this project I looked at the interactions that users have with articles on the IBM Watson Studio platform.

In order to determine which articles to show to each user, I performed a study of the data available from the IBM Watson Studio platform.

## Python Version and Libraries Used

    python 3.6 and above
    pandas
    numpy
    matplotlib
    pickle
    re
    nltk
    sklearn
    jupyter notebook

## Data

    user-item-interactions.csv: Interactions between users and articles.
    articles_community.csv: Contents of articles.

## Overview
  
1. Exploratory Data Analysis

   Before making recommendations, one needs to explore the data they are working on for the particular project. Explore to see what you can find. There are some basic, 
   required questions to be answered about the data throughout the rest of the notebook. This space is used for exploratory activities.

2. Rank Based Recommendations

   To build recommendations, first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to 
   assume the articles with the most interactions are the most popular. These are the articles we might recommend to new or existing users depending on what we know about them.

3. User-User Based Collaborative Filtering

   To build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could 
   then be recommended to the similar users. This would be a step in the right direction towards more personalized recommendations.

4. Content Based Recommendations

   Given the amount of content available for each article, there are a number of different ways in which one might choose to implement a content based recommendations system. 
   Using NLP skills, you might come up with some extremely creative ways to develop a content based recommendation system. Be encouraged to complete a content based              recommendation 
   system, but not required to do so to complete this project.

5. Matrix Factorization

   Finally, I complete a machine learning approach to building recommendations. Using the user-item interactions, I built matrix decomposition. Using the
   decomposition, I got an idea of how well I can predict new articles an individual might interact with. I finally discuss the methods I might use moving forward, 
   and how I might test how well my recommendations are working for user engagement.
