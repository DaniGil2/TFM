# Wiki API libraries
import wikipedia
import wikipediaapi
from wikipedia.exceptions import PageError
from wikipedia.exceptions import DisambiguationError
from new_doc_utils import ARXIV_WIKI_TOPICS

# Multithreading
import threading

ALL_TOPICS = ["Chemical engineering",
              "Biomedical engineering",
              "Civil engineering",
              "Electrical engineering",
              #"Mechanical engineering",
              "Aerospace engineering",
              "Software engineering",
              "Industrial engineering",
              "Computer engineering"]

WIKI = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

import requests


def getSubcatLabels(defs):
    subcat_labels=[]
    for sub in range(len(defs)):
      for n_sub in range(len(defs[sub][1])):
        subcat_labels.append(sub)

    return subcat_labels
  
def appendTopicSubcatDefs(topics, subcats):
  data = topics

  for sub in range(len(subcats)):
      data = data + subcats[sub][1]

  return data

def getSubcategories(category):
    # Author: Daniel Gil
    # Idea taken from: https://www.mediawiki.org/wiki/API:Categorymembers#Python_3

    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"
    
    subcategories = []

    PARAMS = {
        "action": "query",
        "cmtitle": "Category:" + category,
        "cmtype": "subcat",
        "cmlimit": "50",
        "list": "categorymembers",
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA["query"]["categorymembers"]

    for page in PAGES:
        subcategories.append(page["title"])

    return subcategories

  
def getSubcatArticles(dataset):
    subcat_set = []
    index=0
    if dataset == 'wiki':
      all_topics = ALL_TOPICS
    else:
      all_topics = ARXIV_WIKI_TOPICS
    
    for topic in all_topics:
      subcat = getSubcategories(topic)
      subcat = [x.replace('Category:', '') for x in subcat]
      subcat_pages = concurrentGetWikiFullPage(topics_list = subcat)
      subcat_pages = [s for s in subcat_pages if s != '']
      subcat_set.append([index, subcat_pages])
      index += 1

    return subcat_set
  
  
def getWikiSummaries(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    Downloads and parses all summary definitions of the <topics_list> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''

    summaries = list()
    for i, topic in enumerate(topics_list):
        print("Obtaining wikipedia summary for the topic: {}. (Class #[{}])".format(topic, i))
        summaries.append(wikipedia.summary(topic))
    if (target_article):
        # Also return target article requested.
        print("\nObtaining wikipedia summary for target article:", target_article)
        target = wikipedia.summary(target_article)
        return target, summaries
    else:
        return summaries


def getWikiFullPage(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    Downloads and parses the full page of definitions of the <topics_list> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''
    full_pages = list()
    for i, topic in enumerate(topics_list):
        print("Obtaining full wikipedia page for the topic: {}. (Definition of Class #[{}])".format(topic, i))
        full_pages.append(wikipedia.page(topic))
    if (target_article):
        # Also return target article requested.
        print("\nObtaining wikipedia summary for target article:", target_article)
        target = wikipedia.summary(target_article)
        return target, full_pages
    else:
        return full_pages

    return


def concurrentGetWikiFullPage(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    MULTITHREADING VERSION
    Downloads and parses the full page of definitions of the <topics> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''
    global lock
    global raw_dataset

    lock = threading.Lock()
    full_pages = ["" for elem in topics_list]

    def getWikiDefinitionPage(topic_id, topic):
        """wrapper function to start the job in the child process"""
        print("Obtaining full wikipedia page for the topic: {}. (Definition of Class #[{}])".format(topic, topic_id))
        lock.acquire()
        PageError

        try:
          full_pages[topic_id] = wikipedia.page(topic)
        except PageError:
          print("Going to next cat...")
        except DisambiguationError:
          print("Going to next cat (D)...")

        lock.release()

    thread_list = []

    for topic_id, topic in enumerate(topics_list):
        thread = threading.Thread(target=getWikiDefinitionPage, args=(topic_id, topic,))
        thread.daemon = True  # so that closes when disc
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    return full_pages


def getCatMembersList(topic):
    '''
    Returns for a given topic a list of its category members title pages.
    '''
    category = WIKI.page("Category:" + topic)

    cat_members_list = []
    for c in category.categorymembers.values():
        if "Category:" in c.title:
            break
        elif c.ns == 0:
            cat_members_list.append(c.title)

    return cat_members_list


def getCatMembersTexts(cat_members_list, section="Summary"):
    '''
    Retrieves either the summaries or the full wiki text of 
    all pages in a given category members list.
    '''
    c_members_texts = []

    for c_member in cat_members_list:

        c_page = WIKI.page(c_member)
        if "all" in section:
            # Obtain full wikipedia text from page
            c_members_texts.append(c_page.text)
        else:
            # Obtain only Summary section of wiki article
            c_members_texts.append(c_page.summary)

    return c_members_texts


def getAllCatArticles(topics_list, full_text_test=False):
    '''
    Retrieves all articles from categories pages given a list of topics.
    Raw text Dataset structure: [ [topic_j_cat_pages], topic_j_label]
    Returns raw text dataset and the total number of articles retrieved.
    '''

    raw_dataset = list()
    total_num_articles = 0

    for topic_id, topic in enumerate(topics_list):

        cat_members_list = getCatMembersList(topic)

        if full_text_test:
            test_pages = getCatMembersTexts(cat_members_list, section="all")
        else:
            test_pages = getCatMembersTexts(cat_members_list)

        print("Retrieved {} articles from category topic '{}'[TopicID:{}]".format(len(test_pages), topic, topic_id))
        total_num_articles += (len(test_pages) - 1)

        raw_dataset.append((test_pages[1:], topic_id))  # first summary is the topic definition, needs to be exluded

    return raw_dataset, total_num_articles


def concurrentGetAllCatArticles(topics_list, full_text_test=True):
    '''
    MULTITHREADED VERSION. Faster, but may contain bugs.
    Retrieves all articles from categories pages given a list of topics.
    Raw text Dataset structure: [ [topic_j_cat_pages], topic_j_label]
    Returns raw text dataset and the total number of articles retrieved.
    '''
    global lock
    global raw_dataset

    lock = threading.Lock()

    total_num_articles = 0
    raw_dataset = ["" for elem in topics_list]

    def getCategoryArticles(topic_id, topic):
        """wrapper function to start the job in the child process"""
        cat_members_list = getCatMembersList(topic)

        if full_text_test:
            test_pages = getCatMembersTexts(cat_members_list, section="all")
        else:
            test_pages = getCatMembersTexts(cat_members_list)

        if (len(test_pages) == 0):
            print("Could not retrieve articles from category topic:'{}'[TopicID:{}]\n".format(topic, topic_id))
        else:
            print("Retrieved {} articles from category topic '{}'[TopicID:{}]".format(len(test_pages) - 1, topic,
                                                                                      topic_id))
            lock.acquire()
            raw_dataset[topic_id] = (
            test_pages[1:], topic_id)  # first summary is the topic definition, needs to be exluded
            lock.release()

    thread_list = []

    for topic_id, topic in enumerate(topics_list):
        thread = threading.Thread(target=getCategoryArticles, args=(topic_id, topic,))
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    for topic in raw_dataset:
        if len(topic) != 0:
            total_num_articles += len(topic[0])

    return raw_dataset, total_num_articles
  
  
def getSubCategoryArticles(topic, num_arts_cat, target):
  all_articles = list()
  all_subcats = list()
  num_articles = 0
  num_articles_to_get = target - num_arts_cat

  subcat = getSubcategories(topic)
  subcat = [x.replace('Category:', '') for x in subcat]
  all_subcats.append(subcat)
  
  for cat in all_subcats[0]:
      if num_articles < num_articles_to_get:
          cat_members_list = getCatMembersList(cat)

          #if full_text_test:
          test_pages = getCatMembersTexts(cat_members_list, section="all")
          #else:
          #   test_pages = getCatMembersTexts(cat_members_list)

          if (len(test_pages) == 0):
              print("Could not retrieve articles from category topic:'{}'\n".format(cat))
          else:
              if (num_articles + len(test_pages[1:])) < num_articles_to_get:
                  print("Retrieved {} articles from category topic '{}'".format(len(test_pages) - 1, cat))
                  all_articles.append(test_pages[1:])  # first summary is the topic definition, needs to be exluded
                  num_articles = num_articles + len(test_pages[1:])
              else:
                  arts = num_articles_to_get - num_articles
                  print("Retrieved {} articles from category topic '{}'".format(arts, cat))
                  all_articles.append(test_pages[1:(arts+1)])  # first summary is the topic definition, needs to be exluded
                  num_articles = num_articles + len(test_pages[1:(arts+1)])
  
  list_articles=list()
  for articles in all_articles:
      list_articles += articles

  return list_articles, num_articles


def getAllSubCategoryArticles(topics_list, cat_arts_dataset, target):
  list_articles = ["" for elem in topics_list]
  for topic_id in range(len(topics_list)):
    arts, num = getSubCategoryArticles(topics_list[topic_id], len(cat_arts_dataset[topic_id][0]), target)
    list_articles[topic_id] = (arts, topic_id)
  
  return list_articles


def appendArticlesByTopic(cat_arts, subcat_arts):
  all_articles = ["" for elem in cat_arts]

  for topic_id in range(len(cat_arts)):
    all_articles[topic_id] = (cat_arts[topic_id][0] + subcat_arts[topic_id][0], topic_id)

  return all_articles
