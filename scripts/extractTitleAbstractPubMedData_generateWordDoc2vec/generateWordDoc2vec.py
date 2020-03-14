# !/usr/bin/python
# Alexandros Ioannidis
# Python script to generate word vectors using Word2Vec 
  
# importing all necessary modules 
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
warnings.filterwarnings(action = 'ignore') 
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import dill
from readFeedbackQRELS import readFeedbackQRELS
from collections import OrderedDict
import numpy as np
import pandas as pd
from itertools import chain

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

user_path = '/home/pfb16181/NetBeansProjects/'

directory_tmp_tokenised = user_path+'extractTitleAbstractPubMedData_generateWordDoc2vec/dict/tokenised/'
directory_tmp_word2vec = user_path+'extractTitleAbstractPubMedData_generateWordDoc2vec/dict/word2vec/'

qrels_filepath = user_path+'extractTitleAbstractPubMedData_generateWordDoc2vec/qrels/full.train.abs.2017.2018.2019_and_full.test.abs.2019.qrels'
directory = user_path+"extractTitleAbstractPubMedData_generateWordDoc2vec/parsed_data/parsed_pubmed_filter_2017_2018_2019/"
filenames = os.listdir(directory) 

def save_obj(obj, name, directory_tmp, topic_current):
    print('writting object ...')
    with open(directory_tmp + name + '.' + topic_current + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def load_obj(name, directory_tmp, topic_current):
    with open(directory_tmp + name + '.'+topic_current+'.pkl', 'rb') as f:
        return pickle.load(f)

def remove_empty_keys(d):
    new_d = {}
    for k, v in d.items():
        if not d[k]:
            continue
        else:
            new_d[k] = v 
    return(new_d)

# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def LabeledLineSentence(doc_list, labels_list):        
    for idx, doc in enumerate(doc_list):
        yield gensim.models.doc2vec.LabeledSentence(doc, [labels_list[idx]])

files = {}
for filename in filenames:
    with open(directory+filename, "r") as file:
        #if filename in files:
        #    continue
        s = file.read()
        f = s.replace("\n", " ")
        f = re.sub(r"([^\s\w]|_)+", "",  f)
        data = [] 
        # iterate through each sentence in the file 
        for i in sent_tokenize(f): 
            temp = []
            # tokenize the sentence into words 
            for j in word_tokenize(i): 
                temp.append(j.lower()) 
            data.append(temp)
        temp_idx = filename.partition('.txt')[0]
        files[temp_idx] = data
        print('read ', temp_idx)

print('Done building dictionary')

# I need to remove from the files dictionary the pairs where the value is an empty list
new_files = remove_empty_keys(files)
print('removed empty keys')

topics_data = []
topics_data_labels = []
topics_dict_data_doc_labels = []

for topic_current in TOPIC_LIST:
    topic_doc_list, topic_qrels_dict = readFeedbackQRELS(topic_current, qrels_filepath)
    temp_dict = {}
    data = []
    data_for_df = []
    data_labels = []
    dict_data_doc_labels = OrderedDict()
    # by the end of this loop I need to have trained word2vec models saved in a dictionary as values and keys the corresponding number of documents
    # and I need to have a sparse matrix for the whole topic in the form below
    # (num of doc i, word i)  
    # (num of doc i, word i+1)
    # (num of doc i, word i+2)
    # (num of doc i+1, word i)
    # (num of doc i+1, word i+1)
    # (num of doc i+2, word i)
    # (num of doc i+2, word i+1)
    # (num of doc i+2, word i+2)

    for key, value in new_files.items():
        if key in topic_doc_list:
            value_only_for_word2vec = value
            temp_value_only_for_word2vec = list(chain(*value_only_for_word2vec))
            value = list(chain(*value))
            data.append(value)
            data_labels.append(topic_qrels_dict[key])
            dict_data_doc_labels[key] = value
            #list_of_sizes = [64, 128, 256, 512]
            list_of_sizes = [512] 
            for size_i in list_of_sizes:
                # Compute the word2vec for each topic
                Word2Vec_dict = {}               
                # Create model for every document of the collection. 4 variation sizes = 64, 128, 256, 512
                try:
                    temp_model = gensim.models.Word2Vec(value_only_for_word2vec, min_count = 1, size = size_i, window = 5)                 
                    #temp_model.save("word2vec.model_"+str(size_i)+"_"+key)
                    #temp_model.load("word2vec.model_"+str(size_i)+"_"+key)
                    temp_model.train(value_only_for_word2vec, total_examples = 1, epochs=1)
                    #for term in temp_value_only_for_word2vec:
                    #    vector_for_current_term = temp_model.wv[term]
                    #print("Cosine similarity between 'alice' " + "and 'wonderland' - CBOW : ", model1.similarity('alice', 'wonderland')) # Print results
                    print('Creating Word2Vec for ', key)
                    Word2Vec_dict[key] = temp_model
                except Exception as e:
                    print('failed: ',e)
            
    save_obj(Word2Vec_dict, 'Word2Vec_dict_'+str(size_i), directory_tmp_word2vec+str(size_i)+"/", topic_current)
    print('saved Word2Vec_dict_'+str(size_i) + ' dictionary for topic ', topic_current)
    topics_data.append(data)
    topics_data_labels.append(data_labels)
    topics_dict_data_doc_labels.append(dict_data_doc_labels)
   
    
print('End of program')
