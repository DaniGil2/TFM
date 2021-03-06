#################################################################
# Utils library with useful functions developed for the Project #
# Part of my Bachelor Thesis Project @ UC3M                     #
#                                                               #
# Author: Andres Carrillo Lopez                                 #
# GitHub: AndresC98@github.com                                  #
#                                                               #
#################################################################

# Data processing
import numpy as np
import nltk, gensim
import string, time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

download('punkt')
download('wordnet')
download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Model evaluation and Visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# NN Preprocessing
from tensorflow.keras.utils import to_categorical

# Removed topics (unavailable in wiki)
# "Materials engineering",
# "Financial engineering", 

ALL_TOPICS = ["Chemical engineering",
              "Biomedical engineering",
              "Civil engineering",
              "Electrical engineering",
              "Mechanical engineering",
              "Aerospace engineering",
              "Software engineering",
              "Industrial engineering",
              "Computer engineering"]

# "Mat",
# "Fin", 

ENG_TOPICS_ABVR = ["Chem",
                   "Biomd",
                   "Civil",
                   "Elec",
                   "Mech",
                   "Aero",
                   "SW",
                   "Ind",
                   "Comp"]

# For Arxiv parser (Topics)
ARXIV_WIKI_TOPICS = ["Computer science",
                    "Economics",
                    "Systems engineering",
                    "Mathematics",
                    "Astrophysics",
                    "Computational biology",
                    "Statistics"]


import nltk
from nltk import pos_tag
from nltk.corpus import wordnet

def vectSeq(sequences, max_dims=10000):
    '''
    Source: "Deep Learning with Python - Fran??ois Cholet"
    Vectorizes a sequence of text data (supposed cleaned).
    Returns numpy vector version of sequence text data, ready 
    for Feedforward Neural Network input.
    '''

    results = np.zeros((len(sequences), max_dims))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results


def get_wordnet_pos(treebank_tag):
    """
    Funci??n que devuelve un tag reconocible para el lematizador de WordNet.
    Se implementa a partir del c??digo del siguiente enlace:
    https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

# se define una nueva funci??n para el preprocesado del texto que emplea PoS tagging
def preprocess_PoS(text,lemmatizer,stopwords,punctuation):
        if text!='':
          token_pos = pos_tag([text])
          token = token_pos[0][0]
          token_tag = token_pos[0][1]
          if text not in stopwords and text not in punctuation:
            if get_wordnet_pos(token_tag) != '':
                token = lemmatizer.lemmatize(token, get_wordnet_pos(token_tag))
                token = token.lower()
                return token

def cleanText(text,preprocess = 'simple',full_page=False, topic_defs=True):
    '''
    Given a raw text input , tokenizes into words and performs stopword
    and punctuation removal operations; text thus loses structure and is grouped.
    If 'full_page' specified, takes into account cleaning full content.
    Returns cleaned version of text (list of cleaned words).
    '''
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)  # obtaining punctuation library

    n_words = 0

    corpus = list()

    if topic_defs:  # processing topic definitions
        if full_page:
            for topic in text:
                corpus.append(word_tokenize(topic.content))
        else:
            for topic in text:
                corpus.append(word_tokenize(topic))

    if not topic_defs:  # processing test data
        for topic in text:
            corpus.append(word_tokenize(topic))
    
    if preprocess in "simple": #just return tokenized corpus
        return corpus

    else:
        lemmatizer  = LEMMATIZER
        stopwords = STOP_WORDS
        punctuation = string.punctuation

        cleaned_corpus = list()

        for topic in corpus:
            cleaned_corpus_topic = list()
            for word in topic:
                    if '.' in word:  # solving wiki bug
                        for w in word.split('.'):
                            ap = preprocess_PoS(w,lemmatizer,stopwords,punctuation)
                    else:
                        ap = preprocess_PoS(word,lemmatizer,stopwords,punctuation)
                        
                    if ap and ap!='==' and ap!='===':
                        cleaned_corpus_topic.append(ap)
                        n_words += 1
            cleaned_corpus.append(cleaned_corpus_topic)

    return cleaned_corpus


def processClassifData(train_data, test_data, dataset_type ,preprocess = 'simple',full_page=False, debug=False, clasif='NN', subcat_labels=None):
    '''
    Given a dataset (wikipedia or arxiv) cleans training and testing sets.
    Creates doc2bow dictionary of full corpus, and sequences input data into suitable form for NeuralNet Classifier.
    Returns training and test vectors.
    '''
    test_data_clean_pairs = list()  # has labels too
    test_data_clean = list()

    if dataset_type in "wiki":
        for topic_cat in test_data:
            if not topic_cat:
                # for empty (not found) topics:
                continue
            topic_id = topic_cat[1]

            cleaned_test_corpus = cleanText(topic_cat[0],full_page=full_page, preprocess=preprocess,topic_defs=False)

            if debug:
                print("Cleaning all articles from TopicID:", topic_id)
                print(cleaned_test_corpus)
            for article in cleaned_test_corpus:
                test_data_clean_pairs.append((article, topic_id))
                test_data_clean.append(article)

    elif dataset_type in "arxiv":
        for topic_cat in test_data:

            cleaned_test_corpus = list()
            topic_id = topic_cat["label"]

            for paper in topic_cat["papers"]:
                if preprocess in 'simple':
                    tokens = gensim.utils.simple_preprocess(paper["title"] + " : " + paper["abstract"])
                else:
                    tokens = custom_preprocess(paper["title"] + " : " + paper["abstract"])
                
                cleaned_test_corpus.append(tokens)

            if debug:
                print("Cleaning all articles from TopicID:", topic_id)
                print(cleaned_test_corpus)
            for processed_paper in cleaned_test_corpus:
                test_data_clean_pairs.append((processed_paper, topic_id))
                test_data_clean.append(processed_paper)

    else:
        print("ERROR: A dataset type must be specified: either 'wiki' or 'arxiv' datasets.")
        return -1

    # Clean topic defs (train data) and obtain dictionary of full corpus
    train_data_clean = cleanText(train_data,preprocess = preprocess,topic_defs=True,full_page=True)

    foo = train_data_clean.copy()  # placeholder memory allocation
    
    dict_only_train = gensim.corpora.Dictionary(foo)
    foo_only_train = foo.copy()
    
    for page in test_data_clean:  # appending test data for dictionary creation
       foo.append(page)

    # Doc2Bow dictionary of full corpus
    dictionary = gensim.corpora.Dictionary(foo)
    
    
    
    if debug:
        print(dictionary.token2id)
        print("Total number of unique words in corpus:", len(dictionary))
    
    if clasif=='NN':
        # Tf-idf representation of training data
        # Test data as BoW
        
        dictionary.filter_extremes(10, 0.7)

        vocab = dictionary.values()

        vectorizer = CountVectorizer(vocabulary=vocab)
        Bow_matrix = vectorizer.fit_transform([' '.join(doc) for doc in train_data_clean])
        Tf_idf_matrix = TfidfTransformer().fit_transform(Bow_matrix)

        x_train = Tf_idf_matrix.toarray()
            
        #else:
        #    # Data sequencing/encoding NO TF-IDF
        #    train_model_input = list()
        #    for topic in train_data_clean:
        #        train_model_input.append(dictionary.doc2idx(topic))
        #    train_model_input = np.array(train_model_input)
        #    x_train = vectSeq(train_model_input, max_dims=len(dictionary))
        
        test_model_input = list()
        for test_page in test_data_clean:
            test_model_input.append(dictionary.doc2idx(test_page))
        test_model_input = np.array(test_model_input)
        
        x_test = vectSeq(test_model_input, max_dims=len(dictionary))
        
    else:
      
        x_train = train_data_clean
        x_test = test_data_clean
    
    
    
    train_labels = list()
    test_labels = list()


    if dataset_type in "wiki":
        topics = ALL_TOPICS
    elif dataset_type in "arxiv":
        topics = ARXIV_WIKI_TOPICS

    for i, topic in enumerate(topics):
        train_labels.append(i)
    
    if subcat_labels:
        train_labels = train_labels + subcat_labels

    for test_page in test_data_clean_pairs:
        test_labels.append(test_page[1])

    if clasif=='NN':
        # Generating labels (one hot encoding)
        y_train = to_categorical(train_labels)
        y_test = to_categorical(test_labels)
    else:
        # Making texts compatible with Andres methods
        x_train_copy=x_train
        x_train=[]
        for doc in x_train_copy:
          x_train.append(' '.join(doc))

        x_test_copy=x_test
        x_test=[]
        for doc in x_test_copy:
          x_test.append(' '.join(doc))

        # Generating labels (numerical)
        y_train = train_labels
        y_test = test_labels
        
    return x_train, y_train, x_test, y_test, dictionary, foo, dict_only_train, foo_only_train


#def processClassifierData(train_raw_data, test_raw_data, topics, dataset_type="wiki"):
#    """
#    Given raw wikipedia pages (topic defs) as train data, and raw string articles as test data,
#    Generates (unprocessed text) train / test pairs suitable for Sklearn-compatible Classifiers.
#    """
#    x_train = []
#    y_test = []
#    x_test = []
#
#    # Note: this supposes topic definition is full page
#    if dataset_type in "wiki":
#        for wikipage in train_raw_data:
#            x_train.append(wikipage.content)
#
#        y_train = [i for i in range(len(topics))]
#
#        for article_class in test_raw_data:
#            for article in article_class[0]:
#                x_test.append(article)
#                y_test.append(article_class[1])
#    else:  # arxiv dataset
#        for wikipage in train_raw_data:  # also gets topics defs form wiki
#            x_train.append(wikipage.content)
#
#        y_train = [i for i in range(len(topics))]
#        for subject in  test_raw_data:
#            for paper in subject["papers"]:
#                x_test.append(paper["title"] + " : " + paper["abstract"])
#                y_test.append(subject["label"])
#
#    return x_train, y_train, x_test, y_test


def plotConfMatrix(y_test, predictions, model, dataset_type="wiki", conversion=None):
    '''
    Given a one-hot encoded test labels and predictions [class labels]
    computes and plots confusion matrix of model classification result.
    '''
    
    if conversion:
        corrected_predictions = np.copy(predictions)
        for i in range(len(predictions)):
            corrected_predictions[i]=conversion[predictions[i]]
      
        predictions = corrected_predictions
    
    if model in "MSC":
        print(accuracy_score(y_test, predictions))
    
    if model in "NN":  # onehot encoded output of NN
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions)
    else:
        conf_matrix = confusion_matrix(y_test, predictions)

    if dataset_type in "wiki":
        df_cm = pd.DataFrame(conf_matrix, index=[top for top in ENG_TOPICS_ABVR],
                             columns=[top for top in ENG_TOPICS_ABVR])
    else:  # arxiv
        df_cm = pd.DataFrame(conf_matrix, index=[top for top in ARXIV_WIKI_TOPICS],
                             columns=[top for top in ARXIV_WIKI_TOPICS])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix of topic classification")
    plt.show()

    return

def plotDefinitionLengths(definitions, dataset_type):
    '''
    Given topic definitions ("x_train") from either Wikipedia or Arxiv dataset (dataset_type)
    , plots horizontal bar plot of topic definitions length (number of words).
    '''
    total_len = 0
    topic_lengths = list()
        
    for topic in definitions:
        n_words = len( (topic.content).split(" ") )
        total_len += n_words        
        topic_lengths.append(n_words)

    print("Mean number of words per topic definition:", total_len/len(definitions))

    if dataset_type in "arxiv":
        plt.barh(ARXIV_WIKI_TOPICS, topic_lengths, align='center')
    elif dataset_type in "wiki":
        plt.barh(ALL_TOPICS, topic_lengths, align='center')

    plt.title("Length of topic definitions of {} dataset".format(dataset_type))
    plt.xlabel("Number of words")
    plt.ylabel("Topic")

    plt.show()

    return 

def custom_preprocess(doc):
    '''
    Applies tokenization + lemmatization + punctuation removal + stopwords filtering to a document.
    Returns tokens of processed document.
    '''

    lemmatizer  = LEMMATIZER
    stopwords = STOP_WORDS
    punctuation = string.punctuation

    tokenized_doc = word_tokenize(doc)
    tokens = []

    for word in tokenized_doc:
        token = preprocess_PoS(word,lemmatizer,stopwords,punctuation)
        if token:
          tokens.append(token)

    return tokens


def prepare_corpus(X, train_data=True, preprocess='simple', dataset_type="wiki"):
    '''
    For GENSIM  model (MaxSimClassifier).
    Given a raw array of texts (either test data or training topics),
    performs text preprocessing and outputs processed text.
    '''
    if not train_data:
        if dataset_type in "wiki":  
            for article in X:  
                if preprocess in 'simple':
                    tokens = gensim.utils.simple_preprocess(article)
                else:
                    tokens = custom_preprocess(article)
                yield tokens
        else:  # arxiv
            for paper in X:
                if preprocess in 'simple':
                    tokens = gensim.utils.simple_preprocess(paper)
                else:
                    tokens = custom_preprocess(paper)
                yield tokens
    else:
        for i, raw_topic_def in enumerate(X):
            if preprocess in 'simple':
                tokens = gensim.utils.simple_preprocess(raw_topic_def)
            else:
                tokens = custom_preprocess(raw_topic_def)
            # we also add topic class id for training data
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def prepare_train_articles(X, y_text, preprocess='simple'):
    '''
    For GENSIM  model (MaxSimClassifier) SUPERVISED training on articles version.
    Given a raw array of training articles,
    performs text preprocessing and outputs processed tagged text.
    '''
    for i, raw_topic_articles in enumerate(X):
        if preprocess in 'simple':
            tokens = gensim.utils.simple_preprocess(raw_topic_articles)
        else:
            tokens = custom_preprocess(raw_topic_articles)
        # we also add topic class id for training data
        yield gensim.models.doc2vec.TaggedDocument(tokens, [y_text[i]])


def evaluate_model(model, test_corpus, test_labels, eval="binary"):
    '''
    (Testing function)
    Given a doc2vec trained model from GENSIM and a test labeled corpus,
    performs similarity queries of test corpus vs topic definitions.
    
    Returns predictions array and accuracy list.
    '''
    accuracy_list = list()
    predictions = list()
    for doc_id, doc in enumerate(test_corpus):

        inferred_vector = model.infer_vector(doc)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        most_similar_label = sims[0][0]  # index 0 === most similar
        predictions.append(most_similar_label)
        second_most_similar_label = sims[1][0]
        if most_similar_label == test_labels[doc_id]:
            accuracy_list.append(1)
        elif (second_most_similar_label == test_labels[doc_id] and "weighted" in eval):
            accuracy_list.append(0.5)
        elif (second_most_similar_label == test_labels[doc_id] and "top2" in eval):
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)

    accuracy_list = np.array(accuracy_list)
    print("Model {} accuracy over {} test documents: {}%.".format(eval, len(test_labels), np.mean(accuracy_list) * 100))

    return predictions, accuracy_list

def top2_acc(probabilities, true_classes,verbose=0):
    '''
    Given a probability output of an model (MSC, FFNN, or Sklearn classifiers), 
    outputs the Top-2 accuracy and the top_1 (classical accuracy) of the results, 
    given the true_classes of the classification problem.
    '''
    top1_acc_list = []
    top2_acc_list = []
    top2_classes_list = [] 

    for i, probs in enumerate(probabilities):
        top_2_preds = np.argsort(probs, axis=-1)[::-1][:2]
        top2_classes_list.append(top_2_preds)

        if true_classes[i] not in top_2_preds:
            top1_acc_list.append(0)
            top2_acc_list.append(0)
        elif true_classes[i] == top_2_preds[0]:
            top1_acc_list.append(1)
            top2_acc_list.append(1)
        else:
            top1_acc_list.append(0)
            top2_acc_list.append(1)

    top1_acc = np.mean(top1_acc_list)
    top2_acc = np.mean(top2_acc_list)

    if verbose:
        print("TOP-1 acc.: \t{:.3f}\nTOP-2 acc.: \t{:.3f}".format(top1_acc, top2_acc))
    
    return top1_acc, top2_acc
