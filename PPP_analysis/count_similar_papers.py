######################################################################################################################################################## generate counts of similar papers in a window between the 5 years before and 10 years after the paoper publication ##############################################################################################################################################################################################


## load packages and databse username/password
import pandas as pd
import psycopg2
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
from collections import Counter
import time 
main_path = '/home/fs01/spec1142/Emma/'

f = open(main_path + "PPPs/database.txt", "r")
user_emma , password_emma = f.read().split()



## tranform string into vector
def clean_encoding(encoded_text):
    if encoded_text == None:
        return None
    else:
        if "\n" in encoded_text:
            encoded_text = encoded_text.replace("\n" , "")
        encoded_text = encoded_text[1:-1]
        encoded_text = list(map(float , encoded_text.split()))
        return encoded_text




## load array with all paper's abstract embedding (per year) 
path = "/home/fs01/spec1142/Emma/GateKeepers/novelty_measure/"

def load_array(year ):
    
    array_abstract = np.load(path + 'data/papers_abstracts/abstracts' + str(year) + '.npy')
    array_patents = np.load(path + 'data/papers_abstracts/papers' + str(year) + '.npy')

    return array_abstract, array_patents



## query paper's abstracts
def get_abstract(year,workers, i):

    list_abstract = []
    list_papers = []
    
    papers_year = list(file[file['paperpubyear'] == year]['paper_id'])   
    list_index = [ k for k in range(i,len(papers_year),workers) ] 
    
    
    conn = psycopg2.connect(database="spec1142", user=user_emma , password=password_emma , host="192.168.100.54")
    cursor = conn.cursor()
    
    for k in list_index:
        work_id = "W" + str(papers_year[k])
        
            
        text = """select  work_id , encoded_abstract
                  from encoded_works_OpenAlex 
                  where work_id = '""" + work_id + """'
                    ;"""
        cursor.execute(text)
        res = cursor.fetchall()
        if len(res) > 0:
            list_papers.append(res[0][0])
            list_abstract.append(clean_encoding(res[0][1]))
    
    cursor.close()

    return list_papers , list_abstract
    
    


## count the number of papers published in a given year with a similarity score > threshold 
def count_similarity( list_abstracts , list_papers, threshold1 , threshold2,threshold3 , year):

    """
    This function calculates the similarity between a list of paper abstracts and a list of patent abstracts for a given year, using a matrix multiplication approach.

    Parameters:
    list_abstracts (list): A list of paper abstracts represented as numerical vectors.
    list_papers (list): A list of paper IDs corresponding to the abstracts in `list_abstracts`.
    threshold1 (float): The first similarity threshold to be used for counting.
    threshold2 (float): The second similarity threshold to be used for counting.
    threshold3 (float): The third similarity threshold to be used for counting.
    year (int): The year for which to retrieve patent abstracts.

    Note:
    - The function assumes that the `load_array` and `matrix_multiplication` functions are defined elsewhere in the code.
    - The function loads the patent abstracts for the specified year as numerical vectors using the `load_array` function.
    - The function performs matrix multiplication between the paper abstracts and patent abstracts using the `matrix_multiplication` function, and counts the number of similarities above the specified thresholds.
    - The function returns three arrays containing the counts of similarities above the first, second, and third thresholds, respectively.
    """
    
    k=0    
    array_abstract, array_patents = load_array(year)
    size = len(array_abstract)
    print(size)

    array_return1 = np.zeros((len(list_abstracts)))
    array_return2 = np.zeros((len(list_abstracts)))
    array_return3 = np.zeros((len(list_abstracts)))

    for k in range(1+size//100000):
        knowlegde = np.transpose(array_abstract[100000*k:100000*(k+1),:])
        count1 , count2, count3 = matrix_multiplication(list_abstracts, knowlegde, threshold1 , threshold2,threshold3 )
        array_return1 += count1
        array_return2 += count2
        array_return3 += count3
        
    print(year)
    return array_return1 , array_return2, array_return3



## calculate the dot product between two matrix and return the number of similarity scores > threshord.  
def matrix_multiplication(a,b,threshold1,threshold2,threshold3):
    a = np.array(a,dtype = np.float32)
    b = np.array(b,dtype = np.float32)
    similarity = a.dot(b)
    list_counts1 = np.sum(similarity > threshold1, axis=1)
    list_counts2 = np.sum(similarity > threshold2, axis=1)
    list_counts3 = np.sum(similarity > threshold3, axis=1)
    similarity=0
        
    return list_counts1 , list_counts2,list_counts3



## load file with the twins (similar papers)
controls = pd.read_csv(main_path + "PPPs/twins/loose_twins_cites_1patent.tsv", sep = "\t")

## keep relevant features
file = controls[['paper_id','pair_id','paper_date','application_date','grant_date', 'publication_date']]
file = file[file['paper_id'].notnull()]
file['paper_id'] = [ elem[1:] for elem in file['paper_id']]
file['paper_id'] = file['paper_id'].astype('int')
file['paperpubyear'] = [ int(elem[:4]) if pd.isna(elem) == False else elem for elem in file['paper_date'] ] 


## define 3 thresholds
threshold1 = 0.6
threshold2 = 0.64
threshold3 = 0.68

## define number of cpus to use. Note that if workers too high, the code can exceed the available RAM. 
workers = 12


for year in range(2017,2021):

    ##get the abstracts of the papers published in 'year'
    p = Pool(processes=workers)
    func = partial(get_abstract, year,workers)
    abstracts = p.map(func, [ i  for i in range(workers)])
    p.close()

    ##organize the abstract into an array
    list_papers = []
    list_abstracts = []
    for elem in abstracts:
        list_papers += elem[0]
        list_abstracts += elem[1]

    if len(list_papers) > 0:
    
        list_papers = np.array(list_papers)
        list_abstracts = np.array(list_abstracts, dtype=np.float32)    

        print('step1') ##the abstracts are in the array
        

        ##count the number of similar papers between year and year + 10
        p = Pool(len([ i  for i in range(year+1,min(2023,year+11))]))
        func = partial(count_similarity,list_abstracts,list_papers, threshold1,threshold2,threshold3)
        count_similarities_a = p.map(func, [ i  for i in range(year+1,min(2023,year+11))])
        p.close()

        ##count the number of similar papers between year - 5 and year
        p = Pool(len([ i  for i in range(year-5,year+1)]))
        func = partial(count_similarity,list_abstracts,list_papers, threshold1,threshold2,threshold3)
        count_similarities_b = p.map(func, [ i  for i in range(year-5,year+1)])
        p.close()
        
        
        print('step2') ##we have the counts of similar papers

        ## array to store the counts of similar papers AFTER the paper publication 
        count_similarities_a = np.array(count_similarities_a)
        count_similarities1_a = count_similarities_a[:,0,:]
        count_similarities2_a = count_similarities_a[:,1,:]
        count_similarities3_a = count_similarities_a[:,2,:]

        ## array to store the counts of similar papers BEFORE the paper publication 
        count_similarities_b = np.array(count_similarities_b)
        count_similarities1_b = count_similarities_b[:,0,:]
        count_similarities2_b = count_similarities_b[:,1,:]
        count_similarities3_b = count_similarities_b[:,2,:]
        
        ## keep papers published the given year
        file_year = file[file['paperpubyear'] == year]
        df = pd.DataFrame()
        df['paper_id'] = [ int(elem[1:]) for elem in list_papers ] 

        ## store the counts of similar papers in the 10 years after paper publication (publication year + 1 to publication year + 11)
        for k in range(len(count_similarities_a)):
            df['count_after_' + str(k+1) + '_' + str(threshold1)] = count_similarities1_a[k]
            df['count_after_' + str(k+1) + '_' + str(threshold2)] = count_similarities2_a[k]
            df['count_after_' + str(k+1) + '_' + str(threshold3)] = count_similarities3_a[k]

        last = 5
        ## store the counts of similar papers in the 5 years before paper publication (publication year - 5 to publication year)
        for k in range(len(count_similarities_b)):
            df['count_before_' + str(last-k) + '_' + str(threshold1)] = count_similarities1_b[k]
            df['count_before_' + str(last-k) + '_' + str(threshold2)] = count_similarities2_b[k]
            df['count_before_' + str(last-k) + '_' + str(threshold3)] = count_similarities3_b[k]


        ## merge the twin file and the counts
        file_year = file_year.merge(df , on = 'paper_id' , how = 'left')
        file_year = file_year.drop_duplicates()

        ## save the flat file
        file_year.to_csv(main_path + 'PPPs/twins/count_similar_papers/counts_' + str(year) + '.tsv' , sep = "\t")
        


