######################################################################################################################################################## generate counts of similar papers in a window between the 5 years before and 10 years after the paoper publication ##############################################################################################################################################################################################


## load packages 
import pandas as pd
import psycopg2
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
from collections import Counter
import time 
main_path = '/home/fs01/spec1142/Emma/'

f = open(main_path + "database.txt", "r")
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
path = "/home/fs01/spec1142/Emma/test/Website/"

def load_array(year ):
    
    array_abstract = np.load(path + 'data/paper_abstract_array/abstracts' + str(year) + '.npy')
    array_patents = np.load(path + 'data/paper_abstract_array/papers' + str(year) + '.npy')

    return array_abstract, array_patents


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
    
    



def count_similarity( list_abstracts , list_papers, threshold1 , threshold2,threshold3 , year):
    
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




def matrix_multiplication(a,b,threshold1,threshold2,threshold3):
    a = np.array(a,dtype = np.float32)
    b = np.array(b,dtype = np.float32)
    similarity = a.dot(b)
    list_counts1 = np.sum(similarity > threshold1, axis=1)
    list_counts2 = np.sum(similarity > threshold2, axis=1)
    list_counts3 = np.sum(similarity > threshold3, axis=1)
    similarity=0
        
    return list_counts1 , list_counts2,list_counts3



## load file with the twins (loose and tight twins)
controls = pd.read_stata(main_path + 'PPPs_twins/anticommonsredux_emma.dta')

file = controls[['paper_id','pair_id','paperpubyear','app_pub_year','grant_year']]
file['paper_id'] = file['paper_id'].astype('int')


## define threshold
threshold1 = 0.6
threshold2 = 0.64
threshold3 = 0.68

workers = 12

for year in range(1980,2017):

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
        
        
        print('step2') ##we have the count of similar papers
        
        count_similarities_a = np.array(count_similarities_a)
        count_similarities1_a = count_similarities_a[:,0,:]
        count_similarities2_a = count_similarities_a[:,1,:]
        count_similarities3_a = count_similarities_a[:,2,:]

        count_similarities_b = np.array(count_similarities_b)
        count_similarities1_b = count_similarities_b[:,0,:]
        count_similarities2_b = count_similarities_b[:,1,:]
        count_similarities3_b = count_similarities_b[:,2,:]
        
        
        file_year = file[file['paperpubyear'] == year]
        df = pd.DataFrame()
        df['paper_id'] = [ int(elem[1:]) for elem in list_papers ] 
        
        for k in range(len(count_similarities_a)):
            df['count_after_' + str(k+1) + '_' + str(threshold1)] = count_similarities1_a[k]
            df['count_after_' + str(k+1) + '_' + str(threshold2)] = count_similarities2_a[k]
            df['count_after_' + str(k+1) + '_' + str(threshold3)] = count_similarities3_a[k]

        last = 5
        for k in range(len(count_similarities_b)):
            df['count_before_' + str(last-k) + '_' + str(threshold1)] = count_similarities1_b[k]
            df['count_before_' + str(last-k) + '_' + str(threshold2)] = count_similarities2_b[k]
            df['count_before_' + str(last-k) + '_' + str(threshold3)] = count_similarities3_b[k]


        file_year = file_year.merge(df , on = 'paper_id' , how = 'left')
        file_year = file_year.drop_duplicates()
        
        file_year.to_csv(main_path + 'PPPs/counts_' + str(year) + '.tsv' , sep = "\t")
        


