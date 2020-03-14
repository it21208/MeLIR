# !/usr/bin/python
# Alexandros Ioannidis

import os
import xml.etree.ElementTree
from xml.etree import ElementTree as et
from xml.etree.ElementTree import fromstring, tostring

path = '/home/pfb16181/NetBeansProjects/extractTitleAbstractPubMedData_generateWordDoc2vec/data/pubmed_filter_2017_2018_2019/'
temp_list = []
for filename in os.listdir(path):
    print('parsing '+filename)
    temp_str = ''
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)

    et = xml.etree.ElementTree.parse(fullname)
    
    tree = et.parse(fullname)
    string_tree = tostring(tree)
    n_string_tree = str(string_tree, 'utf-8')
    root = xml.etree.ElementTree.fromstring(n_string_tree)
    
    try:
        temp_str += et.find('.//ArticleTitle').text
        temp = root.findall('.//AbstractText')
        for j in temp:
            temp_str += j.text
    except:
        print('error')
        pass

    text_file = open("/home/pfb16181/NetBeansProjects/extractTitleAbstractPubMedData_generateWordDoc2vec/parsed_data/parsed_pubmed_filter_2017_2018_2019/"+
                                                                                                           "".join(filename.partition('.xml')[0].split())+".txt", "w")
    text_file.write(temp_str)
    text_file.close()

print('Done')
