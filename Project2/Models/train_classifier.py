# import libraries
import sys
import pandas as pd
import numpy as np

import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import pickle


nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("Clean_Data", con=engine)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis = 1)
    return X,Y


def tokenize(text):
    """
    Tokenizes and lemmatizes text.
        
    """
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens=[]
    for token in tokens:
        # lemmatize, normalise case, and remove leading/trailing white space
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def model_pipeline_cv():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'clf__estimator__max_depth': [10, 20],
        'vect__ngram_range' : ((1, 1), (1,2)),
        'clf__estimator__n_estimators': [50, 100]
    }
    
    scorer = make_scorer(f1_metric,greater_is_better = True)
    
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring= scorer, verbose= 5, n_jobs=1)

    return cv



def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))
        
        
def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()