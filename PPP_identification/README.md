# This folder describes how the identify to citation based and scientist-inventors based PPPs.

This folder provides the code to create the PPP dataset.

This section requires the full data (OpenAlex + PatentsView) to be loaded into a Postgres database (username and password are required to run the codes, please see folders download_OpenAlex and download_PatentsView from https://github.com/EmmaScharfmann/scientists-inventors) as well as the patent and paper titles and abstracted to be encoded with a pre-trained model (please see folder text_encoding https://github.com/EmmaScharfmann/scientists-inventors). It also requires the section "predictive_model" to be run and to save the predictive classification models "random_forest_PPPs.pkl". 


* The notebook "PPPs_with_citations.ipynb" provides the code to generate the citation based PPPs. The section "Generate citation based PPPs: run this section OR the python file PPP_citations.py" provides the code to identify the work ids and patent ids corresponding to the citation based PPPs. The last section "Add PPP additional data" provides the code to query and add to the PPP file additional data on the PPPs.
* The python file "PPP_citations.py" provides the code to identify the work ids and patent ids corresponding to the citation based PPPs. The user can run this python file instead of the section "Generate citation based PPPs: run this section OR the python file PPP_citations.py" of the notebook "PPPs_with_citations.ipynb" (to run it in background).
* The notebook "PPPs_with_scientist_inventors.ipynb" provides the code to generate the scientist-inventors based PPPs. The section "Generate scientist-inventors based PPPs: run this section OR the python file PPP_with_scientist_inventors.py" provides the code to identify the work ids and patent ids corresponding to the scientist-inventors based PPPs. The last section "Add PPP additional data" provides the code to query and add to the PPP file additional data on the PPPs.
* The python file "PPP_with_scientist_inventors.py" provides the code to identify the work ids and patent ids corresponding to the scientist-inventors based PPPs. The user can run this python file instead of the section "Generate scientist-inventors based PPPs: run this section OR the python file PPP_with_scientist_inventors.py" of the notebook "PPPs_with_scientist_inventors.ipynb" (to run it in background).


