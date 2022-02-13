###Baseline classifiers libraries###

#Keras Neural Network (Feedforward)
from keras.metrics import top_k_categorical_accuracy
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense
#Computations 
import numpy as np
#Tables
import pandas as pd
from google.colab.data_table import DataTable
#Sklearn classifiers and utils
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

import new_doc_utils

def SVM(x_train,y_train,x_test,y_test,dataset):
    best_svm_score = 0
    svm_models_params = list()
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            for idf in [True, False]:
                #Training the SVM for the hyperparam  + preprocessing combinations
                svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                                ('tfidf', TfidfTransformer(use_idf=idf)),
                                ('clf', SVC(C=C, gamma=gamma)),
                                ])    
                svm.fit(x_train, y_train)
                # evaluate the SVM on the test set
                score = svm.score(x_test, y_test)

                #keeping track of the combinations
                params = {'C': C, 'gamma': gamma, 'idf': idf}
                svm_models_params.append((score, params))

                # if we got a better score, store the score and parameters
                if score > best_svm_score:
                    best_svm_score = score
                    best_svm_parameters = {'C': C, 'gamma': gamma, 'idf': idf}

    idf = best_svm_parameters['idf']
    C = best_svm_parameters['C']
    gamma = best_svm_parameters['gamma']

    svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer(use_idf=idf)),
                    ('clf', SVC(C=C, gamma=gamma)),
                    ])    
    svm.fit(x_train, y_train)

    y_test_preds_class = svm.predict(x_test)
    y_test_preds_proba = svm.decision_function(x_test)

    new_doc_utils.top2_acc(y_test_preds_proba, y_test, verbose=1)

    new_doc_utils.plotConfMatrix(y_test, y_test_preds_class, model="SVM", dataset_type=dataset)
    
    
    
def MNB(x_train,y_train,x_test,y_test,dataset):
    best_mnb_score = 0
    mnb_models_params = list()
    for alpha in [0.001, 0.01, 0.1, 1]:
        for idf in [True, False]:
            # for each combination of parameters, train a MNB
            mnb = Pipeline([('vect', CountVectorizer(stop_words='english')),
                            ('tfidf', TfidfTransformer(use_idf=idf)),
                            ('clf', MultinomialNB(alpha=alpha)),
                            ])    
            mnb.fit(x_train, y_train)
            # evaluate the MNB on the test set
            score = mnb.score(x_test, y_test)

            params = {'alpha': alpha, 'idf': idf}
            mnb_models_params.append((score, params))
            # if we got a better score, store the score and parameters
            if score > best_mnb_score:
                best_mnb_score = score
                best_mnb_parameters = {'alpha': alpha, 'idf': idf}
    
    idf = best_mnb_parameters['idf']
    alpha = best_mnb_parameters['alpha']

    mnb = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer(use_idf=idf)),
                    ('clf', MultinomialNB(alpha=alpha)),
                    ])    
    mnb.fit(x_train, y_train) 

    y_test_preds_class = mnb.predict(x_test)
    y_test_preds_proba = mnb.predict_proba(x_test)
    
    new_doc_utils.top2_acc(y_test_preds_proba, y_test, verbose=1)
    
    new_doc_utils.plotConfMatrix(y_test, y_test_preds_class, model="NB", dataset_type=dataset)
    
    
    
#def    
    
    
