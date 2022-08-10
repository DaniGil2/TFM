###Baseline classifiers libraries###

#Keras Neural Network (Feedforward)
from keras.metrics import top_k_categorical_accuracy
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense
from tqdm.notebook import tqdm
from keras.models import model_from_json
import functools
top2_accuracy = functools.partial(top_k_categorical_accuracy, k=2)
top2_accuracy.__name__ = 'TOP2_ACC' #Defining TOP-2 accuracy 
#Computations 
import numpy as np
#Tables
import pandas as pd
from google.colab.data_table import DataTable
#Sklearn classifiers and utils
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import new_doc_utils

def SVM(x_train,y_train,x_test,y_test,dataset):
    best_svm_score = 0
    svm_models_params = list()
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            for kernel in ['linear', 'rbf', 'sigmoid']:
                #Training the SVM for the hyperparam  + preprocessing combinations
                svm = Pipeline([('clf', SVC(C=C, gamma=gamma, kernel=kernel))])    
                svm.fit(x_train, y_train)
                # evaluate the SVM on the test set
                score = svm.score(x_test, y_test)

                #keeping track of the combinations
                params = {'C': C, 'gamma': gamma, 'kernel': kernel}
                svm_models_params.append((score, params))

                # if we got a better score, store the score and parameters
                if score > best_svm_score:
                    best_svm_score = score
                    best_svm_parameters = {'C': C, 'gamma': gamma, 'kernel': kernel}
    
    C = best_svm_parameters['C']
    gamma = best_svm_parameters['gamma']
    kernel = best_svm_parameters['kernel']
    
    svm = Pipeline([('clf', SVC(C=C, gamma=gamma, kernel=kernel))])    
    svm.fit(x_train, y_train)

    y_test_preds_class = svm.predict(x_test)
    y_test_preds_proba = svm.decision_function(x_test)

    new_doc_utils.top2_acc(y_test_preds_proba, y_test, verbose=1)
    
    new_doc_utils.plotConfMatrix(y_test, y_test_preds_class, model="SVM", dataset_type=dataset)
    
    print("C: " + str(C) + " gamma: " + str(gamma) + " kernel: " + str(kernel))
    
    
def MNB(x_train,y_train,x_test,y_test,dataset):
    best_mnb_score = 0
    mnb_models_params = list()
    for alpha in [0, 0.001, 0.01, 0.1, 1 , 2, 5]:
        # for each combination of parameters, train a MNB
        mnb = Pipeline([('clf', MultinomialNB(alpha=alpha))])    
        mnb.fit(x_train, y_train)
        # evaluate the MNB on the test set
        score = mnb.score(x_test, y_test)

        params = {'alpha': alpha}
        mnb_models_params.append((score, params))
        # if we got a better score, store the score and parameters
        if score > best_mnb_score:
            best_mnb_score = score
            best_mnb_parameters = {'alpha': alpha}
                
    alpha = best_mnb_parameters['alpha']

    mnb = Pipeline([('clf', MultinomialNB(alpha=alpha))])    
    mnb.fit(x_train, y_train) 

    y_test_preds_class = mnb.predict(x_test)
    y_test_preds_proba = mnb.predict_proba(x_test)
    
    new_doc_utils.top2_acc(y_test_preds_proba, y_test, verbose=1)
    
    new_doc_utils.plotConfMatrix(y_test, y_test_preds_class, model="NB", dataset_type=dataset)
    
    print("alpha: " + str(alpha))
    
