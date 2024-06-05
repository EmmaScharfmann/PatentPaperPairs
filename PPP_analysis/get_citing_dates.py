## query relevant data (date, institution) of the citing papers. Store the citing paper's date and wheather it's a "self-citation" (citation from the same institution). 



## load packages and database username/password
import pandas as pd
from tqdm import tqdm
import datetime 
from datetime import date
import psycopg2

## import database username and password
main_path = "/home/fs01/spec1142/Emma/PPPs/"

f = open(main_path + "database.txt", "r")
user , password = f.read().split()





## load dictionary with PPP papers and twins citing papers 
f = open(main_path + "twins/dic_citing_papers_loose_twins.json" ,"r")
import json
dic = json.load(f)


def get_citations( i):

    """
    This function retrieves citation information for a list of papers from the OpenAlex database and saves it to a JSON file.

    Parameters:
    workers (int): The number of worker processes to use for parallel processing.
    i (int): The starting index for selecting papers from the list.

    Note:
    - The function assumes that the `user`, `password`, `dic`, and `main_path` variables are defined elsewhere in the code.
    - The function establishes a connection to a PostgreSQL database using the `psycopg2` library and executes SQL queries to fetch the required data.
    - The function stores the citation information in a dictionary `dic_citations`, which includes the citing paper, publication date, and self-citation flag (indicating whether the citing paper shares at least one institution with the cited paper).
    - The function saves the `dic_citations` dictionary to a JSON file named "dic_loose_twins_citations\_i.json" in the specified directory, where i is the starting index for selecting papers.
    - The function closes the database connection after fetching the data.
    - The function saves the output dictionary every 500 iterations.
    """


    kk = 0 

    ## output file: save the data on citing papers 
    dic_citations = {}

    ## intermediate file: keep in memory the citing papers already queried not to have to query them twice
    dic_memory = {}

    #establishing the connection with the database 
    conn = psycopg2.connect("user=" + user + " password=" + password) 
    cursor = conn.cursor()
    
    list_index = [ k for k in range(i,len(dic) , 96)]
    list_items = list(dic.items())
    

    for k in list_index:

        key , value = list_items[k]
        
        kk += 1

        paper_id = key

        
        ## query the institutions for the focal paper
        if paper_id not in dic_citations:

            dic_citations[paper_id] = {}

            institution_ids = set()


            text = """ SELECT  wa.institution_id
                        FROM works_authors_OpenAlex  AS wa 
                        WHERE wa.work_id ='""" + paper_id + """';"""


            cursor.execute(text)  
            res = cursor.fetchall()

            for line in res:

                if line[0] != '':
                    institution_ids.add(line[0])


            ## the date and institution of the citing paper
            for citing_paper in value:

                ## check if the citing paper was already queried 
                if citing_paper not in dic_memory:

                    text = """ SELECT  w.publication_date , array_agg(wa.institution_id) 
    
                               FROM works_OpenAlex AS w
                               
                               LEFT JOIN works_authors_OpenAlex AS wa ON w.work_id = wa.work_id
                               
                               WHERE w.work_id = '""" + citing_paper.replace(' ','') + """' 
                               
                               GROUP BY w.work_id,w.publication_date ;"""
    
                    cursor.execute(text)  
                    res = cursor.fetchall()


                    dic_memory[citing_paper] = res

                ## load query results if the citing paper was already queried
                else:
                    res = dic_memory[citing_paper] 
                    

                ## save citing paper's date and check if it's a self citation (citing paper from the same institution).
                for line in res:
                    dic_citations[paper_id][citing_paper] = {}

                    citing_institution = set()

                    if line[0] != None:
                        dic_citations[paper_id][citing_paper]["date"] = line[0].strftime("%Y-%m-%d")

                    if line[1] != None and line[1] != '':
                        citing_institution = citing_institution.union(set(line[1]))
                    

                    if len(institution_ids & citing_institution) > 0:
                        dic_citations[paper_id][citing_paper]["self_citation"] = 1
                    else:
                        dic_citations[paper_id][citing_paper]["self_citation"] = 0


        ## save the output dictionary every 500 iterations 
        if kk % 500 == 0:
            import json
            json = json.dumps(dic_citations)
            f = open(main_path + "twins/dic_loose_twins_citations_" + str(i) + ".json","w")
            f.write(json)
            f.close()
            
            
            
    cursor.close()

    ## save the final output dictionary  
    import json
    json = json.dumps(dic_citations)
    f = open(main_path + "twins/dic_loose_twins_citations_" + str(i) + ".json","w")
    f.write(json)
    f.close()



## run the code using 96 cpus 
import warnings     
from multiprocessing import Process

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        
        processes = [Process(target=get_citations, args=(k,)) for k in range(96)]
        
        for process in processes:
            process.start()
            
        for process in processes:
            process.join()
            

