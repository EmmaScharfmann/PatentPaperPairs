## This folder provides the code 1) pair each PPP with a control paper and 2) to create the files used for the regressions (cites and similar papers)

This section requires the full data (OpenAlex + PatentsView) to be loaded into a Postgres database (username and password are required to run the codes, please see folders download_OpenAlex and download_PatentsView from https://github.com/EmmaScharfmann/scientists-inventors)as well as the patent and paper titles and abstracted to be encoded with a pre-trained model (please see folder text_encoding https://github.com/EmmaScharfmann/scientists-inventors). It also requires the PPP file to be generated. 

* The notebook "paired_papers.ipynb" provides the code to pair each PPP with a control paper ("similar paper"). It generates the file "loose_twins.tsv". The section "Getting all papers from the same journal" provides the code to query and store all papers by year and journal.  The section "Identify most similar paper" provides the code to identify and pair each PPP with the most similar paper published in the same journal and year. The last section "Additional data on the PPPs / paired paper" provides the code to add the paper's dates and patent's date to the control file, as well as a pair id and the PPP confidence score.

* The notebook "number_of_cites_PPPs.ipynb" provides the code to query the papers (PPP and control) number of cites at different time. It generates the files "loose_twins_cites_year_by_year.tsv", "loose_twins_cites_1patent_byyear.tsv", "loose_twins_cites_1patent.tsv". The section  "Query citing papers and dates" provides the code to query the paper's forward citations. At this step, the python file "get_citing_dates.py" needs to be run to query the data (date, authors, institutions) of the citing papers. The section "Count citations" provides the code to count the citations at different time (before/same year/after patent application/publication/grant). The subsection "Citations - exact date" uses the exact date, "Citations - yearly dates" uses only the year of publication and "Count citation by year - Murray Stern replication" uses Murray and Stern setup.
* The python file "control_papers.py" provides the code to query the data (date, authors, institutions) of the citing papers. 
* The python file "count_similar_papers.py" provides the code to count the number of similar papers (for 3 different "similarity" thresholds, each can be adjusted) in the 5 years before the paper publication and 10 years after.
* The notebook "number_of_similar_papers.ipynb" provides the code to count the number of papers similar to the PPP paper or control, a different time. It generates the file "similar_papers_1980_2020_1patent.tsv". After running "count_similar_papers.py", the section "Count similar papers before/after app/pub/grant" provides the code to count the number of similar papers (for different similarity threshold) before/same year/after patent application/publication/grant. 