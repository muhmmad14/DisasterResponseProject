# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'MessagesCategories'
    df = pd.read_sql_table(table_name, engine)

    X = df['message']
    y = df.iloc[:,4:]

    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if tok in stopwords.words("english"):
            continue
            
        tok = PorterStemmer().stem(tok)
        
        tok = lemmatizer.lemmatize(tok).lower().strip()

        clean_tokens.append(tok)
    
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    parameters = {
<<<<<<< HEAD
        #'vect__max_df':[0.75,1.0],
        #'clf__estimator__n_estimators': [10, 20]
        'clf__estimator__min_samples_split': [2, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=7)
=======
        'vect__max_df':[0.75,1.0],
        'clf__estimator__n_estimators': [20, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
>>>>>>> be24fe82d6e967fbf20b4ab7d7e5a823d8ff12af

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    for ix, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,ix]))

    
    acc = (Y_pred == Y_test).mean().mean()
    print("Accuracy Overall:")
    print(acc)


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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