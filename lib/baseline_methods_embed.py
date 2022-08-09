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
    
    
    
def NN(x_train,y_train,x_test,y_test,dictionary,dataset):
    top1_acc_list = []
    top2_acc_list = []
    n_runs = 30

    for i in tqdm(range(n_runs)):
        clear_session()
        #Neural Architecture Definition
        model = Sequential()
        model.add(Dense(512, activation='tanh', input_shape=(len(dictionary),)))
        model.add(Dense(256, activation='tanh'))
        if dataset=="wiki":
            l = len(new_doc_utils.ALL_TOPICS)
        else:
            l = len(new_doc_utils.ARXIV_WIKI_TOPICS)
        model.add(Dense(l, activation='softmax'))


        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy', top2_accuracy])
        hist = model.fit(x_train, y_train, epochs=5,verbose=0)
        model_top1_acc = model.evaluate(x_test, y_test, verbose=0)[1]
        model_top2_acc = model.evaluate(x_test, y_test, verbose=0)[2]
            
        if top2_acc_list and model_top2_acc >= max(top2_acc_list):
            #save best model to date
            model_json = model.to_json()
            with open("model.json", "w") as f:
                f.write(model_json)
            model.save_weights("model.h5")
        
        top1_acc_list.append(model_top1_acc)
        top2_acc_list.append(model_top2_acc)

    json_f = open('model.json', 'r')
    best_model_json = json_f.read()
    json_f.close()
    best_model = model_from_json(best_model_json)
    best_model.load_weights("model.h5")#loading weights

    # evaluate loaded model on test data
    best_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy', top2_accuracy])

    predict_x=model.predict(x_test) 
    predictions=np.argmax(predict_x,axis=1)

    print("TOP-1 acc.:",best_model.evaluate(x_test, y_test, verbose=0)[1])
    print("TOP-2 acc.:",best_model.evaluate(x_test, y_test, verbose=0)[2])

    new_doc_utils.plotConfMatrix(y_test, predictions, model = "NN", dataset_type=dataset)
    
    
