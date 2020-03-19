# !/usr/bin/python
# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

import warnings
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import unittest
import argparse
import collections
# from test import autotest
import itertools
import logging
import time
import math
import os
import re
import string
import sys
import logging
# import pkg_resources
# pkg_resources.require("numpy==`1.16.2") # modified to use specific numpy
import numpy as np
from collections import defaultdict
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, load_npz
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer
from build_docid_idx_and_label import build_docid_idx_and_label
from build_vocab_tfidf_dict import build_vocab_tfidf_dict
from build_vocab_idf_dict import build_vocab_idf_dict
from main import run_main
from readL4IRresultsFilepath import readL4IRresultsFilepath
from to_sparse import to_sparse
from write_feature import write_feature
from writeResults import writeResults
import warnings
import json
from os import listdir
from os.path import isfile, join

# complete CLEF 2017 topics list
TOPIC_LIST_2017=['CD005139','CD005253','CD006715','CD007431','CD007868','CD008018','CD008081','CD008170','CD008201','CD010019','CD010355','CD010502','CD010526','CD010657','CD010680','CD010771',
'CD010772','CD010775','CD010778','CD010783','CD010860','CD010864','CD010896','CD011053','CD011126','CD011145','CD011380','CD011420','CD011431','CD011436','CD011515','CD011571',
'CD011602','CD011686','CD011912','CD011926','CD012009','CD012010','CD012083','CD012120','CD012164','CD012165','CD012179','CD012216','CD012223','CD012281','CD012347','CD012521',
'CD012599','CD012930']
# complete CLEF 2018 topics list
TOPIC_LIST_2018=['CD007394','CD007427','CD008054','CD008122','CD008587','CD008643','CD008686','CD008691','CD008759','CD008760','CD008782','CD008803','CD008892','CD009020','CD009135','CD009175',
'CD009185','CD009263','CD009323','CD009372','CD009519','CD009591','CD009579','CD009551','CD009593','CD009647','CD009694','CD009786','CD009925','CD009944','CD010023','CD010173',
'CD010213','CD010276','CD010296','CD010339','CD010386','CD010409','CD010438','CD010542','CD010632','CD010633','CD010653','CD010705','CD011134','CD011548','CD011549','CD011975',
'CD011984','CD012019']
# complete CLEF 2019 topics list
TOPIC_LIST_2019=['CD000996','CD001261','CD004414','CD006468','CD007867','CD008874','CD009044','CD009069','CD009642','CD010038','CD010239','CD010558','CD010753','CD011140','CD011768','CD011977',
'CD012069','CD012080','CD012233','CD012342','CD012455','CD012551','CD012567','CD012669','CD012768']
# 30 topics (test topics) Waterloo 2018 CLEF
TOPIC_LIST_UWA_UWB=['CD008122','CD008587','CD008759','CD008892','CD009175','CD009263','CD009694','CD010213','CD010296','CD010502','CD010657','CD010680','CD010864','CD011053','CD011126','CD011420',
'CD011431','CD011515','CD011602','CD011686','CD011912','CD011926','CD012009','CD012010','CD012083','CD012165','CD012179','CD012216','CD012281','CD012599']
# 3 topics for quick testing
TOPIC_LIST_UWA_UWBC=['CD008122','CD008759','CD008892']
# 30 topics (test topics) Waterloo 2017 CLEF
TOPIC_LIST_A_B_RANK_THRESH_NORMAL=['CD007431','CD008081','CD008760','CD008782','CD008803','CD009135','CD009185','CD009372','CD009519','CD009551',
'CD009579','CD009647','CD009786','CD009925','CD010023','CD010173','CD010276','CD010339','CD010386','CD010542','CD010633','CD010653','CD010705','CD010772','CD010775','CD010783','CD010860','CD010896','CD011145','CD012019']
# 'CD010705' #'CD007431', 'CD008081', 'CD008760', 'CD008782'
TOPIC_LIST_SMALL=['CD007431','CD008081','CD008760','CD011686','CD008759']


class run():
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S ')
    ################################################################################################################
    # def suite_TestRunMethods():
    #     suite_TestRunMethods=unittest.TestSuite()
    #     suite_TestRunMethods.addTest(TestRunMethods('string1'))
    #     return(suite_TestRunMethods)
    ################################################################################################################
    # def suite_Test_build_docid_idx_and_label_Methods():
    #     suite_Test_build_docid_idx_and_label_Methods=unittest.TestSuite()
    #     suite_Test_build_docid_idx_and_label_Methods.addTest(Test_build_docid_idx_and_label_Methods('string1'))
    #     return(suite_Test_build_docid_idx_and_label_Methods)
    ################################################################################################################
    # runner.run(suite_TestMainMethods())
    # runner.run(suite_TestRunMethods())
    # runner.run(suite_Test_build_docid_idx_and_label_Methods())
    # Dictionary with the different options of topic collections supported.
    topic_list_dict = {'TOPIC_LIST_SMALL': TOPIC_LIST_SMALL,
                       'TOPIC_LIST_2019': TOPIC_LIST_2019,
                       'TOPIC_LIST_2018': TOPIC_LIST_2018,
                       'TOPIC_LIST_2017': TOPIC_LIST_2017,
                       'TOPIC_LIST_UWA_UWB': TOPIC_LIST_UWA_UWB,
                       'TOPIC_LIST_A_B_RANK_THRESH_NORMAL': TOPIC_LIST_A_B_RANK_THRESH_NORMAL,
                       'TOPIC_LIST_UWA_UWBC': TOPIC_LIST_UWA_UWBC,
                       'FULL_TOPIC_LIST': TOPIC_LIST_2017+TOPIC_LIST_2018+TOPIC_LIST_2019}
    ################################################################################################################

    def parse_args():
        
        while True:
            try:
                text = input("Please enter the command line arguments so that I can parse them: ")
            except ValueError:
                print(
                    "SORRY, AN ERROR OCCURED AFTER YOU PASSED THE ARGUMENTS."
                    "Please enter the correct command line arguments ("
                    "scripts/run.py --seedDoc-folder train/seedDocs_title_and_processedQuery_queryExpansionTop20terms/ "
                    "--tfidf-folder train/tfidf/ --qrels-folder qrels/ "
                    "--l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ "
                    "--projDir /home/pfb16181/NetBeansProjects/MeLIR/ --topic-list TOPIC_LIST_SMALL --idf-folder idf/ "
                    "--classifier svm) so that I can parse them:. "
                    "=============================================")
                continue
            
            bool_condition = "--seedDoc-folder " not in text or "--tfidf-folder " not in text or "--qrels-folder " not in text or "--l4ir-results-folder " not in text or "--output-folder " not in text or "--projDir " not in text or "--topic-list " not in text or "--idf-folder " not in text or "--classifier " not in text

            if bool_condition == True:
                print(
                    "Sorry, please enter the correct command line arguments ("
                    "scripts/run.py --seedDoc-folder train/seedDocs_title_and_processedQuery_queryExpansionTop20terms/ "
                    "--tfidf-folder train/tfidf/ --qrels-folder qrels/ "
                    "--l4ir-results-folder resources/abs_results_retrievalAppSubset/ --output-folder output/ "
                    "--projDir /home/pfb16181/NetBeansProjects/MeLIR/ --topic-list TOPIC_LIST_SMALL --idf-folder idf/ "
                    "--classifier svm) so that I can parse them:. "
                    "=============================================")
                continue
            else:
                # Params are successfully entered - Ready to exit the loop.
                break
        
        # Proceed to split the arguments passed by the user.
        args_splited = text.split("--")
        removed_empty_string_items = [i for i in args_splited if i]
        print('=================================')
        print(removed_empty_string_items)

        for idx, item in enumerate(removed_empty_string_items):
            if idx == 0:
                seedDoc_folder = item.split(" ")[1]
            elif idx == 1:
                tfidf_file_folder = item.split(" ")[1]
            elif idx == 2:
                qrels_file_folder = item.split(" ")[1]
            elif idx == 3:
                l4ir_results_folder = item.split(" ")[1]
            elif idx == 4:
                out_folder = item.split(" ")[1]
            elif idx == 5:
                projDir = item.split(" ")[1]
            elif idx == 6:
                topic_list = item.split(" ")[1]
            elif idx == 7:
                idf_file_folder = item.split(" ")[1]
            else:
                clf = item.split(" ")[1]
        return(seedDoc_folder, tfidf_file_folder, qrels_file_folder, l4ir_results_folder, out_folder, projDir, topic_list, idf_file_folder, clf)
    ################################################################################################################

    def save_dicts_to_output(dir_path, dicts_list):
        filenames = ['tfidf_dict', 'docid_idx_dict', 'vocab_idx_dict']
        for idx, dict_data in enumerate(dicts_list):
            f = open(dir_path+filenames[idx], "w")
            f.write(str(dict_data))
            f.close()
    ################################################################################################################
    # Read command-line arguments from user.
    seedDoc_folder, tfidf_file_folder, qrels_file_folder, l4ir_results_folder, out_folder, projDir, topic_list, idf_file_folder, clf = parse_args()
    topic_list = topic_list_dict[topic_list]
    # Concatenate certain strings to form the right directory paths.
    qrels_filepath = os.path.join(
        qrels_file_folder, 'full.train.abs.2017.2018.2019_and_full.test.abs.2019.qrels')
    qrels_content_filepath = os.path.join(
        qrels_file_folder, 'full.train.content.2017.2018.2019_and_full.test.content.2019.qrels')
    # bm25_b0.75_k1.2pubmed_5_tar.query6.full.train.test.abs.2019.qrels.res
    l4ir_results_filepath = os.path.join(
        l4ir_results_folder, 'RAS.bm25_b0.75_k1.2pubmed_5_tar.title.query6.qe.2017-2019train.full.train.abs.2017.2018.2019.qrels.res')
    out_file_features = os.path.join(out_folder, 'features')
    docid_idx_dict, topic_docid_label, cur_idx, docid_idx_dict_content, topic_docid_label_content = build_docid_idx_and_label(
        qrels_file_folder, topic_list)
    # construct some required dictionary data structures
    vocab_idx_dict, tfidf_dict, total_words = build_vocab_tfidf_dict(
        tfidf_file_folder, docid_idx_dict)
    vocab_idx_dict_2, idf_dict, dummy_var, idf_word_dict = build_vocab_idf_dict(
        idf_file_folder, docid_idx_dict)
    tfidf_sp = to_sparse(tfidf_dict, docid_idx_dict, vocab_idx_dict)
    # Save dictionaries to output/src_outputs__test_inputs/ of the MeLIR project to read it from the test script
    dir_src_outputs__test_inputs = 'output/src_outputs__test_inputs/dicts/'
    dir_path = projDir + dir_src_outputs__test_inputs
    save_dicts_to_output(
        dir_path, [tfidf_dict, docid_idx_dict, vocab_idx_dict])
    logging.info('=================================')
    logging.info(
        "Saved dictionaries: tfidf_dict, docid_idx_dict, vocab_idx_dict")

    #runner = unittest.TextTestRunner()
    #runner.run(suite_test_to_sparse())
################################################################################################################
if __name__ == '__main__':
    run_test = run()
