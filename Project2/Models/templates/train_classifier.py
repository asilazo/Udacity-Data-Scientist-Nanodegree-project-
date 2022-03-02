import sys
import pickle
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

from scipy.stats import hmean
from scipy.stats.mstats import gmean


nltk.download(['punkt', 'wordnet', 'stopwords','wordnet'])


# Function definitions

def load_data(database_filepath):
    database_filepath = create_engine('sqlite:///DisasterMessages.db')
    df = pd.read_sql('Clean_Messages', con = database_filepath)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis = 1)
    # Get the label names
    category_names = Y.columns

    return X, Y


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens=word_tokenize(text)
    stopwords = stopwords.tokens('English')
    lemmatizar = WordNetLemmatizer()
    cleaned_tokens = []
    
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_token)
        
    return cleaned_tokens


def f1_metric(y_true, y_pred):
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average= 'micro')
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf':[True, False],
        'clf__estimator__max_depth' : [4,8,10]
    }
    
    scorer = make_scorer(f1_metric, greater_is_better = True)
    
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring= scorer, verbose= 5)

    return cv



def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    for column in Y_test.columns:
        print('column_name: {}\n'.format(category_names=column))
        print(classification_report(y_test[column], y_pred_dataframe_cv[column]))


def save_model(model, model_filepath):
    model_filepath = 'ML_Model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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