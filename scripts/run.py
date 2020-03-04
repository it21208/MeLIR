# !/usr/bin/python
# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

# Copyright Alexandros Ioannidis
# Thesis for Information Retrieval and machine learning for conducting systematic review.
# University of Strathclyde

import argparse
import collections
import itertools
import logging
import math
import os
import re
import string
import sys
import time
# import pkg_resources
# pkg_resources.require("numpy==`1.16.2") # modified to use specific numpy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, load_npz
from sklearn import linear_model, neural_network
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from build_docid_idx_and_label import build_docid_idx_and_label
from build_vocab_tfidf_dict import build_vocab_tfidf_dict
from build_vocab_idf_dict import build_vocab_idf_dict
from main import run_main
from readL4IRresultsFilepath import readL4IRresultsFilepath
from to_sparse import to_sparse
from write_feature import write_feature
from writeResults import writeResults

# topic_list contains the unique train and test topics of the CLEF TAR e-Health Task 2 from 2017, 2018 and 2019  train topics = 100, 
# test topics = 28, train and test topics = 125. Because 3 topics (CD011571, CD011686, CD012164) are included in both train and test sets 

TOPIC_LIST_2017 = [
  'CD005139','CD005253','CD006715','CD007431','CD007868','CD008018','CD008081','CD008170','CD008201','CD010019','CD010355','CD010502','CD010526','CD010657','CD010680','CD010771',
  'CD010772','CD010775','CD010778','CD010783','CD010860','CD010864','CD010896','CD011053','CD011126','CD011145','CD011380','CD011420','CD011431','CD011436','CD011515','CD011571',
  'CD011602','CD011686','CD011912','CD011926','CD012009','CD012010','CD012083','CD012120','CD012164','CD012165','CD012179','CD012216','CD012223','CD012281','CD012347','CD012521',
  'CD012599','CD012930'
  ]

TOPIC_LIST_2018 = [
  'CD007394','CD007427','CD008054','CD008122','CD008587','CD008643','CD008686','CD008691','CD008759','CD008760','CD008782','CD008803','CD008892','CD009020','CD009135','CD009175',
  'CD009185','CD009263','CD009323','CD009372','CD009519','CD009591','CD009579','CD009551','CD009593','CD009647','CD009694','CD009786','CD009925','CD009944','CD010023','CD010173',
  'CD010213','CD010276','CD010296','CD010339','CD010386','CD010409','CD010438','CD010542','CD010632','CD010633','CD010653','CD010705','CD011134','CD011548','CD011549','CD011975',
  'CD011984','CD012019'
  ]

TOPIC_LIST_2019 = [
  'CD000996','CD001261','CD004414','CD006468','CD007867','CD008874','CD009044','CD009069','CD009642','CD010038','CD010239','CD010558','CD010753','CD011140','CD011768','CD011977',
  'CD012069','CD012080','CD012233','CD012342','CD012455','CD012551','CD012567','CD012669','CD012768'  
  ]

# 30 topics Waterloo 2018 CLEF
TOPIC_LIST_UWA_UWB = [
  'CD008122','CD008587','CD008759','CD008892','CD009175','CD009263','CD009694','CD010213','CD010296','CD010502','CD010657','CD010680','CD010864','CD011053','CD011126','CD011420',
  'CD011431','CD011515','CD011602','CD011686','CD011912','CD011926','CD012009','CD012010','CD012083','CD012165','CD012179','CD012216','CD012281','CD012599'
  ]

# 3 topics for quick testing
TOPIC_LIST_UWA_UWBC = ['CD008122', 'CD008759', 'CD008892']

# 30 topics Waterloo 2017 CLEF
TOPIC_LIST_A_B_RANK_THRESH_NORMAL = [
    'CD007431','CD008081','CD008760','CD008782','CD008803','CD009135','CD009185','CD009372','CD009519','CD009551',
    'CD009579','CD009647','CD009786','CD009925','CD010023','CD010173','CD010276','CD010339','CD010386','CD010542',
    'CD010633','CD010653','CD010705','CD010772','CD010775','CD010783','CD010860','CD010896','CD011145','CD012019'
 ]

TOPIC_LIST_SMALL = ['CD008760'] #'CD010705' #'CD007431', 'CD008081', 'CD008760', 'CD008782'

#topic_seedDoc = 'CD005139'    

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S ')
    
    start_time = time.time()

    # Dictionary with the different options of topic collections supported. 
    topic_list_dict = {'TOPIC_LIST_SMALL': TOPIC_LIST_SMALL,
                       'TOPIC_LIST_2019': TOPIC_LIST_2019,
                       'TOPIC_LIST_2018': TOPIC_LIST_2018,
                       'TOPIC_LIST_2017': TOPIC_LIST_2017,
                       'TOPIC_LIST_UWA_UWB': TOPIC_LIST_UWA_UWB,
                       'TOPIC_LIST_A_B_RANK_THRESH_NORMAL': TOPIC_LIST_A_B_RANK_THRESH_NORMAL,
                       'TOPIC_LIST_UWA_UWBC': TOPIC_LIST_UWA_UWBC}

    parser = argparse.ArgumentParser()
    parser.add_argument("--seedDoc-folder", '-s', type=str, help='path to seedDoc file folder', required=True)
    parser.add_argument("--tfidf-folder", '-t', type=str, help='path to tfidf file folder', required=True)
    parser.add_argument("--qrels-folder", '-q', type=str, help='path to qrels file folder', required=True)
    parser.add_argument("--l4ir-results-folder", '-l', type=str, help='path to l4ir-results file folder', required=True)
    parser.add_argument("--output-folder", '-o', type=str, help='output folder to dump every file into', required=True)
    parser.add_argument("--projDir", '-p', type=str, help='directory path to project', required=True)
    parser.add_argument("--topic-list", '-tl', type=str, help='choose topic list of your preference', required=True)
    parser.add_argument("--idf-folder", '-i', type=str, help='path to idf file folder', required=True)
    parser.add_argument("--classifier", '-c', type=str, help='classifier used to train, choose from lr, smote, rfc, kne, dtc, rls, xgb, sgd, lgb and svm', default='svm')
    
    # argument parse
    args = parser.parse_args() 
    seedDoc_folder = args.seedDoc_folder
    tfidf_file_folder = args.tfidf_folder
    qrels_file_folder = args.qrels_folder
    l4ir_results_folder = args.l4ir_results_folder
    out_folder = args.output_folder
    projDir = args.projDir
    topic_list = topic_list_dict[args.topic_list]
    idf_file_folder = args.idf_folder
    clf = args.classifier 
    #print(topic_list)
    #print(type(topic_list))
    #sys.exit()

    # Concatenate certain strings to form the right directory paths.
    qrels_filepath = os.path.join( qrels_file_folder, 'full.train.abs.2017.2018.2019_and_full.test.abs.2019.qrels' )
    qrels_content_filepath = os.path.join( qrels_file_folder, 'full.train.content.2017.2018.2019_and_full.test.content.2019.qrels' )
    l4ir_results_filepath = os.path.join( l4ir_results_folder, 'RAS.bm25_b0.75_k1.2pubmed_5_tar.title.query6.qe.2017-2019train.full.train.abs.2017.2018.2019.qrels.res' ) # bm25_b0.75_k1.2pubmed_5_tar.query6.full.train.test.abs.2019.qrels.res
    out_file_features = os.path.join( out_folder, 'features' )
    
    docid_idx_dict, topic_docid_label, cur_idx, docid_idx_dict_content, topic_docid_label_content = build_docid_idx_and_label(qrels_file_folder, topic_list)
    vocab_idx_dict, tfidf_dict, total_words = build_vocab_tfidf_dict(tfidf_file_folder, docid_idx_dict)
    ''' ********** do the same procedure as above for the idf values only ********* '''
    vocab_idx_dict_2, idf_dict, dummy_var, idf_word_dict = build_vocab_idf_dict(idf_file_folder, docid_idx_dict)
    ''' ************************************************************************************ '''
    tfidf_sp = to_sparse(tfidf_dict, docid_idx_dict, vocab_idx_dict)
    write_feature(topic_docid_label, tfidf_sp, docid_idx_dict, out_path=out_file_features)
  
    dict_initialScoreRankingResults = readL4IRresultsFilepath(l4ir_results_filepath)
    
    ''' to verify output '''
    #print( dict_initialScoreRankingResults['CD007431'] )
    #print( len( dict_initialScoreRankingResults['CD007431']) )  
    #sys.exit()

    # Each item in the list 'list_doc_score' will contain an ordered dictionary with the doc scores for each topic in the form {docid : docscore} .
    list_doc_score = []
    
    # Loop through all the topics in the topic_list.
    for topic in topic_list:
      
      # Get directory path for the seed document of the current topic.  
      seedDoc_filepath = os.path.join(seedDoc_folder, topic)   
      
      # Implement CAL with Supervised Machine Learning for the current topic & receive the prediction scores for the documents of the current topic.  
      list_doc_score.append(run_main(projDir, seedDoc_filepath, qrels_filepath, tfidf_dict,
                                     vocab_idx_dict, topic_docid_label, clf, tfidf_sp, out_file_features, docid_idx_dict,
                                     topic, qrels_content_filepath, idf_dict, vocab_idx_dict_2, cur_idx, total_words, idf_word_dict, dict_initialScoreRankingResults))

    
    #LambdaParam_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    LambdaParam_list = [0, 0.4, 0.5, 1]
    list_doc_score_copies = [None]*len(LambdaParam_list)
    for i in range(len(LambdaParam_list)):
        list_doc_score_copies[i] = list_doc_score
    
    # Do a Lambda parameter sweep and write results.
    for idx, LambdaParam in enumerate(LambdaParam_list):
      
      writeResults(out_folder, clf, dict_initialScoreRankingResults, LambdaParam, topic_list, list_doc_score_copies[idx])
    
    logging.info(f'Run finished in {time.time() - start_time} seconds')
