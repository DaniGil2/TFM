#################################################################
# arXiv papers parser / data extractor developed for the Project#
# Part of my Bachelor Thesis Project @ UC3M                     #
#                                                               #
# Author: Andres Carrillo Lopez                                 #
# GitHub: AndresC98@github.com                                  #
#                                                               #
#################################################################

from bs4 import BeautifulSoup
import urllib.request
import time

#For Arxiv parser (Topics)
ARXIV_SUBJECTS = ["computer_science",
                "economics",
                "eess",
                "mathematics",
                "physics",
                "q_biology",
                "statistics"]
ARXIV_WIKI_TOPICS = ["Computer science",
                    "Economics",
                    "Systems engineering",
                    "Mathematics",
                    "Astrophysics",
                    "Computational biology",
                    "Statistics"]

def init_arxiv_parser(test_size = 50, fromdate = 2010, todate = 2018):
    '''
    Auxiliary function for initializing the arxiv parser.
    Returns the queries (urls) feeded later to the parser.
    '''
    if test_size not in [25,50,100,200,300,350,400,450,500]:
        #Arxiv url query only allows [25, 50, 100, 200, 500]
        print("[FAILED] test_size.") 
        return -1

    queries = list()

    #query parts for readability:
    base = "https://arxiv.org/search/advanced?advanced="
    q1 = "terms-0-operator=AND&terms-0-term=&terms-0-field=title"
    q2 = "classification-physics_archives=all&classification-include_cross_list=exclude"
    q3 = "date-year=&date-filter_by=date_range&date-from_date="+str(fromdate)+"&date-to_date="+str(todate)+"&date-date_type=submitted_date"
    q4 = "abstracts=show&size="+str(test_size)+"&order=-announced_date_first"
    #TODO: Add more parameters to tweak (year range, crosslisted...)

    for subject in ARXIV_SUBJECTS:
        subject_query = "{b}&{q1}&classification-{subject}=y&{q2}&{q3}&{q4}".format(b=base,q1=q1,subject=subject,q2=q2,q3=q3,q4=q4)
        queries.append({"subject":subject, "url":subject_query})

    return queries



def arxiv_parser(test_size = 50, fromdate = 2010, todate = 2018, debug = False):
    '''
    Test_size: number of articles per topic to obtain.

    Returns :
    - dataset: [] a list of dictionaries representing each subject.
        Each subject is a dictionary containing:
            - "Subject": str - arxiv subject
            - "label": integer - id representing that subject
            - "papers": [{paper1}, {paper2},...] - list of papers for that subject.
                -> Each {paper} is a dictionary containing:
                    "title": str
                    "abstract": str
    - paperslist: [] a flattened list of the retrieved papers
    '''
    queries = init_arxiv_parser(test_size, fromdate, todate)
    if queries == -1: #Query not accepted
        return None 

    arxiv_responses = list() #Will store here the BS4 formatted responses
    
    for query in queries:
        print("Retrieving papers for subject:" ,query['subject'])   
        arxiv_request = urllib.request.urlopen(query['url'])
        if debug:
            print("Request code: ", arxiv_request.code)
        arxiv_response = arxiv_request.read()
        arxiv_responses.append({"subject":query['subject'], "resp":BeautifulSoup(arxiv_response, 'html.parser')})
        time.sleep(1)

    papers_dataset = list()
    papers_list = list()

    for subject_label, subject in enumerate(arxiv_responses):
        subject_xml_contents = list()
        for result in subject["resp"].find_all("li", {"class": "arxiv-result"}):
            subject_xml_contents.append(result)
    
        subject_papers = list()

        for result in subject_xml_contents:
            raw_title = result.find_all("p", {"class": "title is-5 mathjax"})[0].getText()
            title = " ".join(raw_title.split())
            raw_abstract = result.find_all("span", {"class": "abstract-full has-text-grey-dark mathjax"})[0].getText()
            abstract = " ".join(raw_abstract.split())
            paper = {"title":title, "abstract":abstract}
            papers_list.append(paper)
            subject_papers.append(paper)
        
        papers_dataset.append({"Subject":subject["subject"],"label":subject_label,"papers":subject_papers})

    data_size = 0
    for subject in papers_dataset:
        data_size += len(subject["papers"])
    print("Retrieved {} papers in total from {} subjects ({} from each).".format(data_size,len(papers_dataset), int(data_size/len(papers_dataset))))

    return papers_dataset, papers_list
