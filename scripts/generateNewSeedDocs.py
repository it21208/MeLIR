# !/usr/bin/python
# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis
import os
import xml.etree.ElementTree
from xml.etree import ElementTree as et
from xml.etree.ElementTree import fromstring, tostring
import re


QRELS_PATH = '/home/pfb16181/NetBeansProjects/MeLIR/qrels/full.train.abs.2017.2018.2019_and_full.test.abs.2019.qrels'

PATH_TO_SEED_DOCS_1_REL_DOC = '/home/pfb16181/NetBeansProjects/MeLIR/train/seedDocs_1_rel_doc/'
PATH_TO_SEED_DOCS_ALL_REL_DOCS = '/home/pfb16181/NetBeansProjects/MeLIR/train/seedDocs_all_rel_documents/'


# all 125 topics from 2017, 2018, 2019
TOPIC_LIST = ['CD005139', 'CD005253', 'CD006715', 'CD007431', 'CD007868', 'CD008018', 'CD008122', 'CD008170', 'CD008201', 'CD008587',
              'CD008759', 'CD008892', 'CD009175', 'CD009263', 'CD009694', 'CD010019', 'CD010213', 'CD010296', 'CD010355', 'CD010502',
              'CD010526', 'CD010657', 'CD010680', 'CD010771', 'CD010772', 'CD010775', 'CD010778', 'CD010783', 'CD010860', 'CD010864',
              'CD010896', 'CD011053', 'CD011126', 'CD011145', 'CD011380', 'CD011420', 'CD011431', 'CD011436', 'CD011515', 'CD011571',
              'CD011602', 'CD011686', 'CD011912', 'CD011926', 'CD012009', 'CD012010', 'CD012083', 'CD012120', 'CD012164', 'CD012165',
              'CD012179', 'CD012216', 'CD012223', 'CD012281', 'CD012347', 'CD012521', 'CD012599', 'CD012930', 'CD007394', 'CD007427',
              'CD008054', 'CD008081', 'CD008643', 'CD008686', 'CD008691', 'CD008760', 'CD008782', 'CD008803', 'CD009020', 'CD009135',
              'CD009185', 'CD009323', 'CD009372', 'CD009519', 'CD009591', 'CD009579', 'CD009551', 'CD009593', 'CD009647', 'CD009786',
              'CD009925', 'CD009944', 'CD010023', 'CD010173', 'CD010276', 'CD010339', 'CD010386', 'CD010409', 'CD010438', 'CD010542',
              'CD010632', 'CD010633', 'CD010653', 'CD010705', 'CD011134', 'CD011548', 'CD011549', 'CD011975', 'CD011984', 'CD012019',
              'CD000996', 'CD001261', 'CD004414', 'CD006468', 'CD007867', 'CD008874', 'CD009044', 'CD009069', 'CD009642', 'CD010038',
              'CD010239', 'CD010558', 'CD010753', 'CD011140', 'CD011768', 'CD011977', 'CD012069', 'CD012080', 'CD012233', 'CD012342',
              'CD012455', 'CD012551', 'CD012567', 'CD012669', 'CD012768'
              ]

PATH_TO_PARSED_DATA = '/home/pfb16181/NetBeansProjects/extractTitleAbstractPubMedData_generateWordDoc2vec/parsed_data/parsed_pubmed_filter_2017_2018_2019/'
temp_list = []


def readFeedbackQRELS(topic_seedDoc, qrels_filepath):
    f = open(qrels_filepath, "r")
    topic_qrels_dict = {}
    for line in f:
        topic = line.split()[0]
        if topic == topic_seedDoc:
            doc = line.split()[2]
            # relevancy 0 or 1
            relevancy = line.split()[3]
            '''
            if relevancy == '0' or relevancy == '1':
                relevancy = int(relevancy)
            # Check if relevancy is 2 and make it 1
            if relevancy == '2':
                #relevancy = '1'
                relevancy = 1
            '''
            if relevancy == '1':
                return(doc)


def readFeedbackQRELS2(topic_seedDoc, qrels_filepath):
    list_of_rel_docs = []
    f = open(qrels_filepath, "r")
    topic_qrels_dict = {}
    for line in f:
        topic = line.split()[0]
        if topic == topic_seedDoc:
            doc = line.split()[2]
            # relevancy 0 or 1
            relevancy = line.split()[3]
            '''
            if relevancy == '0' or relevancy == '1':
                relevancy = int(relevancy)
            # Check if relevancy is 2 and make it 1
            if relevancy == '2':
                #relevancy = '1'
                relevancy = 1
            '''
            if relevancy == '1':
                list_of_rel_docs.append(doc)

    return(list_of_rel_docs)


for topic in TOPIC_LIST:

    '''
    seed_rel_doc_for_topic = readFeedbackQRELS(topic, QRELS_PATH)
    full_path_for_seed_rel_doc_for_topic = PATH_TO_PARSED_DATA+seed_rel_doc_for_topic+".txt"

    with open(full_path_for_seed_rel_doc_for_topic, 'r') as file:
        data = file.read().replace('\n', '')

    completeName = os.path.join(PATH_TO_SEED_DOCS_1_REL_DOC, topic)
    file_seed = open(completeName, "w")
    file_seed.write( " ".join(data.split()) )
    print('writting 1 relevant document as seed doc for topic '+topic)
    file_seed.close()
    '''

    list_of_rel_docs = readFeedbackQRELS2(topic, QRELS_PATH)
    for idx, rel_doc in enumerate(list_of_rel_docs):
        list_of_rel_docs[idx] = PATH_TO_PARSED_DATA+rel_doc+".txt"

    string_data = ''
    for full_path_for_seed_rel_doc_for_topic in list_of_rel_docs:
        with open(full_path_for_seed_rel_doc_for_topic, 'r') as file:
            string_data += file.read().replace('\n', '')

    completeName = os.path.join(PATH_TO_SEED_DOCS_ALL_REL_DOCS, topic)
    file_seed = open(completeName, "w")
    file_seed.write(" ".join(string_data.split()))
    print('writting all relevant documents as one seed doc for topic '+topic)
    file_seed.close()
