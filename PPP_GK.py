################ generate scientist-inventor based PPPs ################


## import packages and database username and password
import pandas as pd
import json, requests 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import unicodedata
from metaphone import doublemetaphone
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import re
from mpi4py import MPI
from math import radians, cos, sin, asin, sqrt
import datetime 
from datetime import date
import psycopg2


main_path  = '/home/fs01/spec1142/Emma/PPPs/'


f = open(main_path + "database.txt", "r")
user_emma , password_emma = f.read().split()

## load institutions data
dic_institutions = pd.read_csv(main_path + "institutions_up_to_20230817.tsv" , delimiter = "\t", index_col = 0).to_dict("index")


## import predictive model trained using self-plagiarism
import pickle
with open(main_path + 'random_forest_PPPs.pkl','rb') as f:
    rf = pickle.load(f)


## transform the numerical representation of the text stored as a string into a vector
def clean_encoding(encoded_text):
    if encoded_text == None:
        return None
    else:
        if "\n" in encoded_text:
            encoded_text = encoded_text.replace("\n" , "")
        encoded_text = encoded_text[1:-1]
        encoded_text = list(map(float , encoded_text.split()))
        return encoded_text
    


#suppress all the unwanted suffixes from a string. 
#name_del file can be modified if more or less suffixes want to be suppressed 

name_del = set(["2nd", "3rd", "Jr", "Jr.", "Junior", "Sr", "Sr.", "Senior"])

def name_delete(string):
    for elmt in name_del:
        if f" {elmt}" in string:
            string =  string.replace(f" {elmt}","")
    return string


#merge the nobiliary particles with the last name
#ln_suff file can be modified if more or less nobiliary particles want to be suppressed

ln_suff= set(["oster", "nordre", "vaster", "aust", "vesle", "da", "van t", "af", "al", "setya", "zu", "la", "na", "mic", "ofver", "el", "vetle", "van het", "dos", "ui", "vest", "ab", "vste", "nord", "van der", "bin", "ibn", "war", "fitz", "alam", "di", "erch", "fetch", "nga", "ka", "soder", "lille", "upp", "ua", "te", "ni", "bint", "von und zu", "vast", "vestre", "over", "syd", "mac", "nin", "nic", "putri", "bet", "verch", "norr", "bath", "della", "van", "ben", "du", "stor", "das", "neder", "abu", "degli", "vre", "ait", "ny", "opp", "pour", "kil", "der", "oz",  "von", "at", "nedre", "van den", "setia", "ap", "gil", "myljom", "van de", "stre", "dele", "mck", "de", "mellom", "mhic", "binti", "ath", "binte", "snder", "sre", "ned", "ter", "bar", "le", "mala", "ost", "syndre", "sr", "bat", "sndre", "austre", "putra", "putera", "av", "lu", "vetch", "ver", "puteri", "mc", "tre", "st"])

def ln_suff_merge(string):
    for suff in ln_suff:
        if f"{' ' + suff + ' '}" in string or string.startswith(f"{suff + ' '}"):
            print(suff)
            string =  string.replace(f"{suff + ' '}", suff.replace(" ",""))
    return string




#normalize a string dat that represents often a name. 
def normalize(data):
    normal = unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore')
    val = normal.decode("utf-8")
    # delete unwanted elmt
    val = name_delete(val)
    # Lower case
    val = val.lower()
    # remove special characters
    val = re.sub('[^A-Za-z ]+', ' ', val)
    # remove multiple spaces
    val = re.sub(' +', ' ', val)
    # remove trailing spaces
    val = val.strip()
    return val



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
    
    
    
    
    
#return a ratio of similarity of letters between two strings (to handle in the first names errors)
def match_ratio(string1,string2):
    return fuzz.ratio(string1, string2)


#return 4 if string1 and string2 are the same
#return 3 if string1 and string2 sound the same
#otherwise, return less

def metaphone(string1,string2):
    if string1==string2:
        return 4
    tuple1 = doublemetaphone(string1)
    tuple2 = doublemetaphone(string2)
    if tuple1[0] == tuple2[0]:
        return 3
    elif tuple1[0] == tuple2[1] or tuple1[1] == tuple2[0]:
        return 2
    elif tuple1[1] == tuple2[1]:
        return 1
    else:
        return 0
    
    
## return 1 if name1 and name2 potentially represent the same individual
## else return 0

def comparison(name1 , name2):
    
    #if there is no first name, retrun it's a match
    if name1 == "" or name2 == "":
        return 0
    
    #if some first names exist:
    list_name1 = name1.split()
    list_name2 = name2.split()
    
    #minimum number of first names to match
    number_match = min( len(list_name1) , len(list_name2))
    
    #for each name, check if there is a match
    count_match = 0
    for elem1 in list_name1:
        match = 0
        
        #if we just have the initial:
        if len(elem1) == 1:
            for elem2 in list_name2:
                if elem1[0] == elem2[0]:
                    match = 1
                    
        #if we have the entire first name:
        else:
            for elem2 in list_name2:
                #if we just have the initial:
                if len(elem2) == 1 and elem1[0] == elem2[0]:
                    match = 1
                    
                #if elem1 and elem2 are entire first names that sound the same and have a ratio of common letters higher thsan 85%, it's a match
                elif len(elem2) > 1 and (metaphone(elem1,elem2) > 2 or match_ratio(elem1 , elem2) > 85 ) :
                    match = 1
                    
        #count the number of first names that match    
        count_match += match
        
    #check if we have enough first names that match 
    if count_match < number_match:
        return 0
    else:
        return 1


    
    
    
## get number of in common authors

def number_of_in_common_authors(inventors , authors ):

    #count the number of names in common, and store the names in common
    count = 0
    list_in_common_authors = []

    #compare all the inventors with all the authors
    for name_inventor in inventors:
        
        for name_author in authors:
                        
            #to be more efficient, compare only the names with a common first or last name. 
            if len(set(name_inventor.split()) & set( name_author.split())) > 0:
            
                match = comparison(name_author , name_inventor)

                #if the first names match, we store the first names that are matching and their index 
                if match == 1:
                    count += 1 
                    list_in_common_authors.append(name_author + "-" + name_inventor)
                    


    #return 1) the number of names in common, 2) the list of names in common, 3) their index 
    return  count ,  list_in_common_authors 





## calculate efficiently the geographic distance between two points on the earth

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km




def distance_assignees(coords1 , coords2):
     
    distance = np.inf
    
    #get minimum geographic distance between the paper and the patent institutions
    
    #to be more efficient, check if coords1 and coords2 share common elements. if so, the distance is null. 
    if len(set(coords1) & set(coords2)) > 0:
        return 0
   

    #otherwise, compare the coordinates and get the minimal distance.
    else:
        for elem1 in coords1:
            for elem2 in coords2:
                dist = haversine(elem1[0], elem1[1], elem2[0], elem2[1])
                if dist < distance:
                    distance = dist
                    
    if distance == np.inf:
        distance = None
                    
    return distance
        
    


##function that compare a patent and a paper and create the features that will be used as input of the predictive model
def paper_patent_comparison(patent_id , paper_id , dic_comparison):

    
    ##the features are stored as a dictionay in the input dictonary
    dic_comparison["comparisons"][patent_id][paper_id] = {}
    
    ##compare the title and the abstract with by taking the cosinus of the numerical representations of the text
    title_OA = dic_comparison["OA"][paper_id]["title"]
    title_PV = dic_comparison["PV"][patent_id]["title"]
    abstract_OA = dic_comparison["OA"][paper_id]["abstract"]
    abstract_PV = dic_comparison["PV"][patent_id]["abstract"]
    
    if title_OA != None and title_PV != None:
        dic_comparison["comparisons"][patent_id][paper_id]["title_similarity"] = cosine_similarity2( title_OA , title_PV ) 
    else:
        dic_comparison["comparisons"][patent_id][paper_id]["title_similarity"] = None
        
    if abstract_PV != None and abstract_OA != None:
        dic_comparison["comparisons"][patent_id][paper_id]["abstract_similarity"] = cosine_similarity2( abstract_PV , abstract_OA ) 
    else:
        dic_comparison["comparisons"][patent_id][paper_id]["abstract_similarity"] = None
        
        
    ##calucate the difference (in days) between the patent grant date and the paper publication date
    if   dic_comparison["OA"][paper_id]["publication_date"] != None and dic_comparison["PV"][patent_id]["patent_date"] != None and dic_comparison["PV"][patent_id]["patent_date"]!= '':
        dic_comparison["comparisons"][patent_id][paper_id]["difference_patent_grant_paper"] =   (dic_comparison["PV"][patent_id]["patent_date"] - dic_comparison["OA"][paper_id]["publication_date"] ).days
        
    else:
        dic_comparison["comparisons"][patent_id][paper_id]["difference_patent_grant_paper"] = None
     
    
    ##calucate the difference (in days) between the patent application date and the paper publication date
    if dic_comparison["OA"][paper_id]["publication_date"] != None and dic_comparison["PV"][patent_id]["filing_date"] != None:
        dic_comparison["comparisons"][patent_id][paper_id]["difference_patent_app_paper"] =  (dic_comparison["PV"][patent_id]["filing_date"] - dic_comparison["OA"][paper_id]["publication_date"]).days
    else:
        dic_comparison["comparisons"][patent_id][paper_id]["difference_patent_app_paper"] = None
        
        
        
    ##store the paper's type (article, book-chapter etc...) 
    dic_comparison["comparisons"][patent_id][paper_id]["paper type"] = dic_comparison["OA"][paper_id]["type"]  
    
    ##count the number of author's name in common between the paper and the patent 
    dic_comparison["comparisons"][patent_id][paper_id]["authors in common"] = number_of_in_common_authors(dic_comparison["PV"][patent_id]["co_inventors"] , dic_comparison["OA"][paper_id]["co_authors"])[0]
    
    ##calculate what proportion of the patent inventors the number of author's name in common represents 
    dic_comparison["comparisons"][patent_id][paper_id]["proportion inventors"] = 100*number_of_in_common_authors(dic_comparison["PV"][patent_id]["co_inventors"] , dic_comparison["OA"][paper_id]["co_authors"])[0] / len(dic_comparison["PV"][patent_id]["co_inventors"])
    
    ##calculate the (minimal) geographic distance between the patent assignee and the paper institution (in km)
    dic_comparison["comparisons"][patent_id][paper_id]["distance_inst_assignee"] = distance_assignees(dic_comparison["PV"][patent_id]["coordinates"], dic_comparison["OA"][paper_id]["coordinates"] )
  
    ##return the updated dic_comparison
    return dic_comparison
        
        




import warnings
warnings.filterwarnings("ignore")



df_gatekeepers = pd.read_csv(main_path + "gatekeepers_clean_v2.tsv", delimiter = "\t")

data = df_gatekeepers[["inventor_id" , "author_id"]].to_numpy()

clusters = []
G = nx.Graph()
G.add_edges_from(data)
count = 0 
for connected_component in nx.connected_components(G):
    clusters.append(connected_component)
    
#print("Number of gatekeepers:" , len(clusters))




dic_clusters = {}
count = 0 

for cluster in clusters:
    dic_clusters[count] = {}
    dic_clusters[count]["OA"] = []
    dic_clusters[count]["PV"] = []
    for elem in cluster:
        
        if elem[0] == "A":
            dic_clusters[count]["OA"].append(   elem  )
        else:
            dic_clusters[count]["PV"].append( elem )
        
    count += 1



def PPPs_GK(i, workers):
    
    #establishing the connection
    conn = psycopg2.connect(database="spec1142", user=user_emma , password=password_emma , host="192.168.100.54")
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
        
    ##number of gatekeeper to analyse 
    size = len(dic_clusters)
    ##new output file: will store the potential PPPs. 
    dic_features = {}
    
    ##count the number of iteration realized - dic_features will be saved at 100 iterations 
    k = 0
    
    for count in range(i,size,workers):
        
        k += 1 
        
        try:
            
            
            ##dic_comparison stores all the papers (key: "OA"), patents data (key: "PV") and comparison between papers and patents ("comparisons")  corresponding the gatekeeper
            dic_comparison = {}
            dic_comparison["PV"] = {}
            dic_comparison["OA"] = {}
            dic_comparison["comparisons"] = {}


            ##query the papers corresponding to the author id
            ##paper_ids stores the papers written by the gatekeepers
            paper_ids  = set()
            for author_id in dic_clusters[count]["OA"]:

                new_paper_ids  = set()

                text = """ SELECT work_id
                            FROM works_authors_OpenAlex 
                            WHERE author_id = '""" + author_id + """';"""

                cursor.execute(text)
                res = cursor.fetchall()

                for line in res:
                    new_paper_ids.add(line[0])
                paper_ids = paper_ids.union(new_paper_ids)

            
            ##query the patents corresponding to the inventor id
            ##patent_ids stores the patents written by the gatekeepers
            patent_ids = set()
            for inventor_id in dic_clusters[count]["PV"]:

                new_patent_ids  = set()


                text = """ SELECT patent_id
                            FROM inventors_PatentsView  
                            WHERE inventor_id = '""" + inventor_id + """';"""

                cursor.execute(text)  
                res = cursor.fetchall()

                for line in res:
                    new_patent_ids.add(line[0])

                    patent_ids = patent_ids.union(new_patent_ids)


            

            ##query the patents data, and store the data in dic_comparison["PV"]
            for patent_id in patent_ids:

                dic_comparison_new = {}
        

                ##query the patents data, and store the data in dic_comparison["PV"]
                dic_comparison_new = {}
                
                text = """ SELECT   p.patent_date , 
                                    ap.filing_date , 
                                    string_agg( CONCAT(disambig_inventor_name_first , ' ' , disambig_inventor_name_last  ) , '#') , 
                                    pe.encoded_title , 
                                    pe.encoded_abstract , 
                                    string_agg( CONCAT(a.disambig_assignee_organization  , '%' , l.longitude , '%' , l.latitude)  , '#')
                            FROM patents_PatentsView AS p 
                            JOIN applications_PatentsView AS ap ON ap.patent_id = p.patent_id
                            JOIN inventors_PatentsView AS i ON i.patent_id =  p.patent_id
                            JOIN assignees_PatentsView AS a ON a.patent_id =  p.patent_id
                            JOIN locations_PatentsView AS l ON a.location_id = l.location_id
                            JOIN encoded_patents_PatentsView AS pe ON pe.patent_id = p.patent_id
                            WHERE p.patent_id = '""" + patent_id  + """'
                            GROUP BY p.patent_date , 
                                    ap.filing_date , 
                                    pe.encoded_title , 
                                    pe.encoded_abstract ;"""
                        
                cursor.execute(text)
                res = cursor.fetchall()
                
                for line in res:
                    dic_comparison_new["patent_date"] = line[0]
                    dic_comparison_new["filing_date"] = line[1]
                    if line[2] != None:
                        dic_comparison_new["co_inventors"] = [ normalize(elem) for elem in set(line[2].split("#")) ] 
                    else:
                        dic_comparison_new["co_inventors"] = []
                        
                    try:
                        dic_comparison_new["title"] = clean_encoding(line[3])
                    except:
                        dic_comparison_new["title"] = None
                    try:
                        dic_comparison_new["abstract"] = clean_encoding(line[4])
                    except:
                        dic_comparison_new["abstract"] = None
                    if line[5] != None:
                        locations = [ elem.split("%") for elem in set(line[5].split("#"))] 
                        dic_comparison_new["assignees"] = ", ".join([ elem[0] for elem in locations])
                        dic_comparison_new["coordinates"] = [ (float(elem[1]) , float(elem[2]))for elem in locations if len(elem) > 2 ]
                    else:
                        dic_comparison_new["assignees"] = None
                        dic_comparison_new["coordinates"] = []
                dic_comparison["PV"][patent_id] = dic_comparison_new


        
            ##query papers data and store the data in dic_comparison["OA"]
            for paper_id in paper_ids:


                dic_comparison_new = {}
                ##query papers data and store the data in dic_comparison["OA"]
                dic_comparison_new = {}
                text = """ SELECT   we.encoded_title , 
                            we.encoded_abstract , 
                            w.publication_date , 
                            w.type , 
                            string_agg(a.display_name , '#') , 
                            string_agg(wa.institution_name , '#') , 
                            string_agg(wa.institution_id , '#'),
                            w.doi
            
                        
                   FROM works_authors_OpenAlex AS wa
                   JOIN works_OpenAlex AS w ON w.work_id = wa.work_id
                   JOIN authors_OpenAlex AS a ON wa.author_id = a.author_id
                   JOIN encoded_works_OpenAlex AS we ON we.work_id = wa.work_id
                   WHERE wa.work_id =  '""" + paper_id + """'
                   GROUP BY we.encoded_title , 
                            we.encoded_abstract , 
                            w.publication_date , 
                            w.type,
                            w.doi;"""
                
                cursor.execute(text)
                res = cursor.fetchall()
                
                for line in res:
                    try:
                        dic_comparison_new["title"]  = clean_encoding(line[0])
                    except: 
                        dic_comparison_new["title"] = None
                    try: 
                        dic_comparison_new["abstract"]  = clean_encoding(line[1])
                    except: 
                        dic_comparison_new["abstract"] = None
                    dic_comparison_new["publication_date"] = line[2]
                    dic_comparison_new["type"] = line[3]
                    dic_comparison_new["co_authors"] = [ normalize(elem) for elem in set(line[4].split("#") )] 
            
                    dic_comparison_new["institutions"] = set()
                    if line[5] != None:
                        dic_comparison_new["institutions"] = set( line[5].split("#") )
                        
                    if line[6] != None:
                        inst_ids = set(line[6].split("#"))
                    else:
                        inst_ids = []
                        
                    dic_comparison_new["coordinates"] = []
                    for inst_id in inst_ids:
                            if inst_id in dic_institutions:
                                dic_comparison_new["coordinates"].append((dic_institutions[inst_id]["longitude"] , dic_institutions[inst_id]["latitude"]))
                                dic_comparison_new["institutions"].union(dic_institutions[inst_id]["display_name"])
                    dic_comparison_new["institutions"] = "; ".join(list(dic_comparison_new["institutions"]))

                    dic_comparison_new["doi"] = line[7]
            
                
                dic_comparison["OA"][paper_id]  = dic_comparison_new



            for patent_id in patent_ids:
                ##compare all the patents and papers together
                dic_comparison["comparisons"][patent_id] = {}
                for paper_id in paper_ids:

                    ##compare patent_id with paper_id
                    if dic_comparison["OA"][paper_id] != {} and dic_comparison["PV"][patent_id] != {}:
                        dic_comparison = paper_patent_comparison(patent_id , paper_id , dic_comparison)
                        

                if len(dic_comparison["comparisons"][patent_id]) > 0:
                    table = pd.DataFrame(dic_comparison["comparisons"][patent_id]).T[['title_similarity', 'abstract_similarity', 'authors in common', 'proportion inventors','distance_inst_assignee']]
                    
                    ##if the distance is missing (if the assignee or institution location is missing) 
                    ##replace the missing value with the average distance between the papers and the patents
                    mean = table["distance_inst_assignee"].mean()
                    table["distance_inst_assignee"] = table["distance_inst_assignee"].fillna(mean)
                    
                    ##drop all the papers and patents comparison with missing values (abstract, title, etc...)
                    table = table.dropna()
                    
                    ##predict if the patents and papers are PPPs
                    if len(table) > 0:
                
                        table["predictions"] = rf.predict_proba(table)[:,1]
                        
                        ##keep only keep the prediction higher than 0.2 and the highest prediction (with a precision of 0.05)
                        max_pred = max(table["predictions"])
                        PPPs_paper_ids = table[ ( table["predictions"] > max_pred - 0.05 ) &  ( table["predictions"] > 0.7 )][["predictions"]].to_dict("index")
                        
                        ##write the selected PPPs in the output file dic_features 
                        for paper_id in PPPs_paper_ids:
                            dic_features[paper_id + " US-" + patent_id] = dic_comparison["comparisons"][patent_id][paper_id]
                            dic_features[paper_id + " US-" + patent_id]["paper_id"] = paper_id
                            dic_features[paper_id + " US-" + patent_id]["patent_id"] = patent_id
                            dic_features[paper_id + " US-" + patent_id]["doi"] = dic_comparison["OA"][paper_id]["doi"]
                            dic_features[paper_id + " US-" + patent_id]["prediction"] = PPPs_paper_ids[paper_id]["predictions"]
        
        except:
            file_error = main_path + "PPPs_GK/errors_" + str(i) 
            file_object = open(file_error, 'a')
            file_object.write(str(count) + "\n")
            file_object.close()
            continue

        ##save the output every 1000 iterations.
        if k % 500 == 0:
    
            res = pd.DataFrame(dic_features).T
            res.to_csv(main_path + "PPPs_GK/table_pred_" + str(i) + ".tsv"  , sep = "\t")
            
    #Closing the connection
    conn.close() 


    ##save the file as a tsv file
    res = pd.DataFrame(dic_features).T
    res.to_csv(main_path + "PPPs_GK/table_pred_" + str(i) + ".tsv" , sep = "\t" )

        


    
workers = 64

#run the code using 64 CPUs   
import warnings    
from multiprocessing import Process

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        
        processes = [Process(target=PPPs_GK, args=(k,)) for k in range(workers)]
        
        for process in processes:
            process.start()
            
        for process in processes:
            process.join()

