## for each PPP, select the most similar paper published the same year and in the same journal that is NOT a PPP. 


from pySankey.sankey import sankey
import pandas as pd
import json, requests 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
import json, requests 
import time
import unicodedata
from metaphone import doublemetaphone
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import re
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import datetime 
from datetime import date
import psycopg2

## import database username and password
main_path = "/home/fs01/spec1142/Emma/PPPs/"

f = open(main_path + "database.txt", "r")
user , password = f.read().split()

PPPs = pd.read_csv(main_path + "PPPs_v2.tsv" , delimiter = "\t")
set_paper_id = set(list(PPPs['paper_id']))

f = open(main_path + "PPP_analysis/dic_papers_same_journal_year.json","r")
import json 
dic_papers_same_journal_year = json.load(f)


df_similar_papers = pd.read_csv(main_path + "PPP_analysis/similar_papers.tsv" , delimiter = "\t", index_col = 0 )
papers_done = set(list(df_similar_papers.index))


def clean_encoding(encoded_text):
    if encoded_text == None:
        return None
    else:
        if "\n" in encoded_text:
            encoded_text = encoded_text.replace("\n" , "")
        encoded_text = encoded_text[1:-1]
        encoded_text = list(map(float , encoded_text.split()))
        return encoded_text


## calculate efficiently the dot product between two vectors

def norm(vector):
    return sqrt(sum(x * x for x in vector))    

def cosine_similarity2(vec_a, vec_b):
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
        
        
    if norm_a == 0 or norm_b == 0 :
        return None 
    
    else:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return dot / (norm_a * norm_b)


def similar_papers( i):

    """
    This function finds similar papers for each paper in a list of PPP papers by comparing their encoded titles and abstracts using cosine similarity. It stores the results in a dictionary and saves it to a TSV file.

    Parameters:
    i (int): The starting index for selecting papers from the list of PPP papers.

    Note:
    - The function assumes that the `user`, `password`, `main_path`, `PPPs`, `dic_papers_same_journal_year`, `set_paper_id`, `clean_encoding`, and `cosine_similarity2` variables and functions are defined elsewhere in the code.
    - The function establishes a connection to a PostgreSQL database using the `psycopg2` library and executes SQL queries to fetch the required data.
    - The function calculates the cosine similarity between the encoded titles and abstracts of each pair of papers and stores the most similar paper (i.e., the twin) and the similarity score in a dictionary.
    - The function saves the dictionary to a TSV file named "similar_papers_i.tsv" in the specified directory, where i is the starting index for selecting papers.
    - The function closes the database connection after fetching the data.
    """


    workers = 24
    
    dic_similar_papers = {}
    counter = 0 

    #establishing the connection with the database 
    conn = psycopg2.connect("user=" + user + " password=" + password) 
    cursor = conn.cursor()

    PPP_papers = set(PPPs["paper_id"].tolist())
    papers = list(PPP_papers)
    index_papers = [ k for k in range(i,len(papers) , workers)]

    for k in index_papers:

        try:

            work_id = papers[k]

            if work_id not in papers_done:

                ## query PPP paper abstract and title if the journal of the paper is not missing
                if work_id in dic_papers_same_journal_year:
        
        
        
                    text = """ SELECT we.encoded_abstract , we.encoded_title
                                   FROM encoded_works_OpenAlex AS we
                                   WHERE we.work_id = '"""+ work_id + """';"""
            
            
            
                    cursor.execute(text)
                    res = cursor.fetchall()
            
                    if len(res) > 0: 
            
                        line = res[0]
                        dic_similar_papers[work_id] = {}
                        dic_similar_papers[work_id]["date"] = dic_papers_same_journal_year[work_id]['data']['date']
                            
                        title = clean_encoding(line[0])
                        abstract = clean_encoding(line[1])
        
                        max_score = -1
                        twin = None

                        ## query the abstract and title of the papers from the same year and journal
        
                        for paper in dic_papers_same_journal_year[work_id]['papers']:
    
                            ## check that the paper is not a PPP
                            if paper not in set_paper_id:
        
                                text = """  SELECT we.encoded_title , we.encoded_abstract
                                            FROM encoded_works_OpenAlex AS we
                                            WHERE work_id = '""" + paper + """';"""
                
                
                                cursor.execute(text)
                                res = cursor.fetchall()
                
                                
                
                                if len(res) > 0:
                
                                    line = res[0]
                
                                    similar_title = clean_encoding(line[0])
                                    similar_abstract = clean_encoding(line[1])
                
                                    score_title = cosine_similarity2(similar_title , title)
                                    score_abstract = cosine_similarity2(similar_abstract , abstract)

                                    ## the title counts as 1/3 of the similarity score, and the abstract 2/3 
                                    if 1/3*score_title + 2/3*score_abstract > max_score:
                                        
                                        max_score = 1/3*score_title + 2/3*score_abstract
                                        
                                        twin = paper
                                    
                        ## keep the most similar paper and similarity score 
                        dic_similar_papers[work_id]['twin'] = twin
                        dic_similar_papers[work_id]['score'] = max_score
        
        
                ## if the journal of the PPP is missing, query the date, abstract and title 
                else:
    
    
                    text = """ SELECT we.encoded_abstract , we.encoded_title, w.publication_date
                               FROM encoded_works_OpenAlex AS we
                               JOIN works_OpenAlex AS w ON w.work_id = we.work_id
                               WHERE we.work_id = '"""+ work_id + """';"""
    
        
                    cursor.execute(text)
                    res = cursor.fetchall()
            
                    if len(res) > 0: 

                        ## query the title and abstract of 1000 papers published the same year 
            
                        line = res[0]
                        dic_similar_papers[work_id] = {}
                        dic_similar_papers[work_id]["date"] = line[2]
                        year = int(line[2].year)
                            
                        title = clean_encoding(line[0])
                        abstract = clean_encoding(line[1])
    
    
                        text = """  SELECT we.work_id , we.encoded_title , we.encoded_abstract
                                    FROM ( SELECT work_id , publication_date , venue_or_source  , concepts , referenced_works 
                                           FROM works_OpenAlex
                                           WHERE extract(year from publication_date) = '"""+ str(year) + """'
                                           LIMIT 1000 ) AS w
                                    JOIN encoded_works_OpenAlex AS we ON we.work_id = w.work_id
                                ;"""
    
    
                        cursor.execute(text)
                        res = cursor.fetchall()
    
                        max_score = -1
                        twin = None
    
                        for line in res:
        
                            paper = line[0]

                            ## check that the paper is not a PPP 
                            if paper not in set_paper_id:
        
                                similar_title = clean_encoding(line[1])
                                similar_abstract = clean_encoding(line[2])
            
                                score_title = cosine_similarity2(similar_title , title)
                                score_abstract = cosine_similarity2(similar_abstract , abstract)
                                
                                ## the title counts as 1/3 of the similarity score, and the abstract 2/3 
                                if 1/3*score_title + 2/3*score_abstract > max_score:
                                    max_score = 1/3*score_title + 2/3*score_abstract
                                    twin = paper

                        ## keep the most similar paper and similarity score 
                        dic_similar_papers[work_id]['twin'] = twin
                        dic_similar_papers[work_id]['score'] = max_score
                
                counter += 1

                ## save the output file as a flat file every 50 iterations 
                if counter % 50 == 0:
                    table = pd.DataFrame(dic_similar_papers).T
                    table.to_csv(main_path + "PPP_analysis/similar_papers_" + str(i) + ".tsv" , sep = "\t")
        
        except:
            file_error = main_path + "PPP_analysis/error.txt"
            file_object = open(file_error, 'a')
            file_object.write(work_id + "\n")
            file_object.close()

    
    ## save the final outfile as a flat file 
    table = pd.DataFrame(dic_similar_papers).T
    table.to_csv(main_path + "PPP_analysis/similar_papers_" + str(i) + ".tsv" , sep = "\t")
    


## run the code using 24 cpus 
import warnings
from multiprocessing import Process


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        
        processes = [Process(target=similar_papers, args=(k,)) for k in range(24)]
        
        for process in processes:
            process.start()
            
        for process in processes:
            process.join()
            



