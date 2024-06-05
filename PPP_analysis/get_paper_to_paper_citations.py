## this function load into the postgres database the citing papers of a given list of papers. The data are load into the table "citations_OpenAlex"

## load packages and database username/password
import pandas as pd
import psycopg2
from tqdm import tqdm 

main_path = '/home/fs01/spec1142/Emma/PPPs/twins/'

f = open('/home/fs01/spec1142/Emma/GateKeepers/' + "database.txt", "r")
user , password = f.read().split()


## load PPP papers and twins  
loose_twins = pd.read_csv(main_path + "similar_papers.tsv" , delimiter = "\t", index_col = 0 )
list_works = [ elem for elem in list(loose_twins.index) + list(loose_twins["twin"]) if pd.isna(elem) == False ] 
len(list_works)


## load the citing papers flat file by chunks (size=50000000).
path = "/home/fs01/shared/foremma/works_referenced_works.csv"
df_chunks = pd.read_csv(path , on_bad_lines = "warn" , chunksize = 50000000 ,  names = [ "work_id" , "referenced_work_id"] ) 
count = 0 


## connect to the database 
conn = psycopg2.connect("user=" + user + " password=" + password)
cursor = conn.cursor()

## for each chunk, only select the citing papers of the given papers 
for data in tqdm(df_chunks):
    
    count += 1
            
    ## load citing data as a dataframe
    df = pd.DataFrame(data , dtype = str)

    ## transform given papers into the correct format
    list_papers = set([ 'https://openalex.org/' + elem for elem in list_works])

    ## keep only the citing papers of the given papers 
    df = df[df["referenced_work_id"].isin(list_papers)]
    df["work_id"] = [ elem[21:] for elem in df["work_id"]]
    df["referenced_work_id"] = [ elem[21:] for elem in df["referenced_work_id"]]

    ## save the citing papers 
    df.to_csv("/home/fs01/spec1142/Emma/Download_OpenAlex/citation_chunk.tsv" , sep = "\t" , index = False )

    ## load the citing papers into the database
    with open("/home/fs01/spec1142/Emma/Download_OpenAlex/citation_chunk.tsv") as f:
        cursor.copy_expert("COPY citations_OpenAlex FROM STDIN WITH DELIMITER E'\t' CSV HEADER", f)
          
## commit the action           
conn.commit()

        
#Closing the connection
conn.close()


