1) Run the following command:  pip install lightgbm to install the lightGBM classifier and install the XGBooster and SGD Classifiers if these are not installed.
   - Also install RLScore  
   git clone https://github.com/aatapa/RLScore.git
   python setup.py install
   python setup.py install --home=<dir>
   python setup.py build_ext --inplace

2) If you don't have the following. Then run the commands below on the terminal.
  import nltk
  nltk.download('all')

Install any other related dependencies you might be missing.
Execute the command below from inside the root folder of the project. 

python scripts/run.py --seedDoc-folder train/seedDocs/ --tfidf-folder train/tfidf/ --qrels-folder qrels/ --l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ --projDir /home/pfb16181/NetBeansProjects/PubMed-CAL-AutoTar/ --topic-list tl --classifier clf

- Where clf is one of the following: svm, lgb (lightGBM)
- Where tl is one of the following: TOPIC_LIST_SMALL, TOPIC_LIST_2017, TOPIC_LIST_2018, TOPIC_LIST_2019, TOPIC_LIST_UWA_UWB, TOPIC_LIST_A_B_RANK_THRESH_NORMAL, TOPIC_LIST_UWA_UWBC
- After the --projDir the user needs to enter his/her own path to the extracted PubMed-CAL-AutoTar project.

An example to run a project

python scripts/run.py --seedDoc-folder train/seedDocs/ --tfidf-folder train/tfidf/ --qrels-folder qrels/ --l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ --projDir /home/pfb16181/NetBeansProjects/PubMed-CAL-AutoTar/ --topic-list TOPIC_LIST_2017 --classifier svm

List of libraries you might need to install in python in order to use MeLIR:
collections, os, itertools, re, string, time, defaultdict, matplotlib.pyplot, logging,
sklearn, xgboost, rlscore, nltk, operator, math, random, save_npz, numpy, scipy.sparse,
coo_matrix, csr_matrix, TfidfVectorizer, sys, csv. Some libraries may have other
dependencies which the user will need to install on his system.
