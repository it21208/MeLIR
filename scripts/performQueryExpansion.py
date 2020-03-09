# '!/usr/bin/python
# '-*-'coding:utf-8'-*-
# 'author'='Alexandros'Ioannidis
import os
import logging
import argparse
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from build_docid_idx_and_label import build_docid_idx_and_label
from build_vocab_tfidf_dict import build_vocab_tfidf_dict
import sys
from collections import Counter
import itertools

path = '/home/pfb16181/NetBeansProjects/MeLIR/output/features/'
qrels_file_folder = '/home/pfb16181/NetBeansProjects/MeLIR/qrels/'
tfidf_file_folder = '/home/pfb16181/NetBeansProjects/MeLIR/train/tfidf/'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S ')
TOPIC_LIST = ['CD012930']
'''
['CD000996','CD001261','CD004414','CD005139','CD005253','CD006468','CD006715','CD007394','CD007427','CD007431','CD007867','CD007868','CD008018','CD008054',
'CD008081','CD008122','CD008170','CD008201','CD008587','CD008643','CD008686','CD008691','CD008759','CD008760','CD008782','CD008803','CD008874','CD008892',
'CD009020','CD009044','CD009069','CD009135','CD009175','CD009185','CD009263','CD009323','CD009372','CD009519','CD009551','CD009579','CD009591','CD009593',
'CD009642','CD009647','CD009694','CD009786','CD009925','CD009944','CD010019','CD010023','CD010038','CD010173','CD010213','CD010239','CD010276','CD010296',
'CD010339','CD010355','CD010386','CD010409','CD010438','CD010502','CD010526','CD010542','CD010558','CD010632','CD010633','CD010653','CD010657','CD010680',
'CD010705','CD010753','CD010771','CD010772','CD010775','CD010778','CD010783','CD010860','CD010864','CD010896','CD011053','CD011126','CD011134','CD011140',
'CD011145','CD011380','CD011420','CD011431','CD011436','CD011515','CD011548','CD011549','CD011571','CD011602','CD011686','CD011768','CD011912','CD011926',
'CD011975','CD011977','CD011984','CD012009','CD012010','CD012019','CD012069','CD012080','CD012083','CD012120','CD012164','CD012165','CD012179','CD012216',
'CD012223','CD012233','CD012281','CD012342','CD012347','CD012455','CD012521','CD012551','CD012567','CD012599','CD012669','CD012768','CD012930']

parser = argparse.ArgumentParser()
parser.add_argument("--topic", '-t', type=str, help='provide topic to perform query expansion', required=True)
args = parser.parse_args() 
topic = args.topic
'''
docid_idx_dict, topic_docid_label, cur_idx,  docid_idx_dict_content, topic_docid_label_content = build_docid_idx_and_label(
    qrels_file_folder, TOPIC_LIST)
vocab_idx_dict, tfidf_dict, total_words = build_vocab_tfidf_dict(
    tfidf_file_folder, docid_idx_dict)
for topic in TOPIC_LIST:
    cx = coo_matrix(load_npz(os.path.join(path, f'{topic}.npz')))
    topic_term_tfidf_dict = {}
    for i, j, v in zip(cx.row, cx.col, cx.data):
        try:
            if topic_docid_label[topic][i] == 1:
                topic_term_tfidf_dict[vocab_idx_dict[j]] = v
        except Exception as e:
            pass

    str_output = ''
    for i in Counter(topic_term_tfidf_dict).most_common(20):
        str_output += (i[0]+' ')

    with open(os.path.join('/home/pfb16181/NetBeansProjects/lucene4ir-master/data/pubmed/query_parsers_2017_2018_2019/query/new_results_from_queryExpansion_top20Terms', topic), "w") as file1:
        file1.write(str_output)
    file1.close()
    print('wrote ', topic)
