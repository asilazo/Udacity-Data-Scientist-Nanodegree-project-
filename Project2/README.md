# Disaster Response Pipeline Project

## Table of Contents

    *Project Motivation
    *File Descriptions
    *Components
    *Instructions of How to Interact With Project
    *Licensing, Authors, Acknowledgements, etc.

## Project Motivation

In this project, tests data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. 
I have created a machine learning pipeline to categorize real messages that were sent during disaster events so that the messages could be sent to an 
appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in 
several categories. The web app will also display visualizations of the data.


## File Descriptions

### app

| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

### data

|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to

### models

|- train_classifier.py # machine learning pipeline
|- classifier.pkl # saved model

README.md
Components

There are three parts for this project are:
1. ETL Pipeline

  A Python script, process_data.py, writes a data cleaning pipeline that:

    *Loads the messages and categories datasets
    *Merges the two datasets
    *Cleans the data
    *Stores it in an SQLite database

  A jupyter notebook ETL Pipeline Preparation was used to perform the EDA to prepare the process_data.py python script.

2. ML Pipeline

  A Python script, train_classifier.py, writes a machine learning pipeline that:

    *Loads data from the SQLite database
    *Splits the dataset into training and test sets
    *Builds a text processing and machine learning pipeline
    *Trains and tunes a model using GridSearchCV
    *Outputs results on the test set
    *Exports the final model as a pickle file

  A jupyter notebook ML Pipeline Preparation was used to do EDA to prepare the train_classifier.py python script.

3. Flask Web App

  The project includes a web app that can be used by an emergency worker to input a message and get classification results in several categories. The web app will also display 
  visualizations of the data. The outputs are shown below:

