# !/usr/bin/python
# -*- coding:utf-8 -*-
# author = Alexandros Ioannidis

import os

PATH_TO_SEED_DOCS_1_REL_DOC = '/home/pfb16181/NetBeansProjects/MeLIR/train/seedDocs_1_rel_doc/'
PATH_TO_SEED_DOCS_TITLE_AND_PROCESSED_QUERY = '/home/pfb16181/NetBeansProjects/MeLIR/train/seedDocs_title_and_processedQuery/'

PATH_TO_SEED_DOCS_TITLE_AND_PROCESSED_QUERY_AND_1_REL_DOC = '/home/pfb16181/NetBeansProjects/MeLIR/train/seedDocs_title_and_processedQuery_rel_document/'

# all 125 topics from 2017, 2018, 2019
TOPIC_LIST=['CD005139','CD005253','CD006715','CD007431','CD007868','CD008018','CD008122','CD008170','CD008201','CD008587',
            'CD008759','CD008892','CD009175','CD009263','CD009694','CD010019','CD010213','CD010296','CD010355','CD010502',
            'CD010526','CD010657','CD010680','CD010771','CD010772','CD010775','CD010778','CD010783','CD010860','CD010864',
            'CD010896','CD011053','CD011126','CD011145','CD011380','CD011420','CD011431','CD011436','CD011515','CD011571',
            'CD011602','CD011686','CD011912','CD011926','CD012009','CD012010','CD012083','CD012120','CD012164','CD012165',
            'CD012179','CD012216','CD012223','CD012281','CD012347','CD012521','CD012599','CD012930','CD007394','CD007427',
            'CD008054','CD008081','CD008643','CD008686','CD008691','CD008760','CD008782','CD008803','CD009020','CD009135',
            'CD009185','CD009323','CD009372','CD009519','CD009591','CD009579','CD009551','CD009593','CD009647','CD009786',
            'CD009925','CD009944','CD010023','CD010173','CD010276','CD010339','CD010386','CD010409','CD010438','CD010542',
            'CD010632','CD010633','CD010653','CD010705','CD011134','CD011548','CD011549','CD011975','CD011984','CD012019',
            'CD000996','CD001261','CD004414','CD006468','CD007867','CD008874','CD009044','CD009069','CD009642','CD010038',
            'CD010239','CD010558','CD010753','CD011140','CD011768','CD011977','CD012069','CD012080','CD012233','CD012342',
            'CD012455','CD012551','CD012567','CD012669','CD012768'  
            ]


for topic in TOPIC_LIST:
    completeName1 = os.path.join(PATH_TO_SEED_DOCS_TITLE_AND_PROCESSED_QUERY, topic)
    completeName2 = os.path.join(PATH_TO_SEED_DOCS_1_REL_DOC, topic)

    string_data = ''

    with open(completeName1, 'r') as file1:
        string_data = file1.read().replace('\n', '')

    string_data += ' '

    with open(completeName2, 'r') as file2:
        string_data += file2.read().replace('\n', '')


    completeName = os.path.join(PATH_TO_SEED_DOCS_TITLE_AND_PROCESSED_QUERY_AND_1_REL_DOC, topic)
    file_seed = open(completeName, "w")
    file_seed.write( " ".join(string_data.split()) )
    print('writting seed doc for topic '+topic)
    file_seed.close()