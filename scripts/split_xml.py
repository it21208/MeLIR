# Run this command gunzip * .xml.gz from the terminal to unzip all the .gz files in a directory
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup

index = 0
directoryPath_read = '/home/pfb16181/all_PubMed_Data_1/'
directoryPath_write = '/home/pfb16181/all_PubMed_Data_2/'
XMLfilenames = [join(directoryPath_read, f) for f in listdir(directoryPath_read) if isfile(join(directoryPath_read, f))]

for XMLfilename in XMLfilenames:
    context = ET.iterparse(XMLfilename, events=('end', ))
    tree = ET.parse(XMLfilename)
    root = tree.getroot()
    xmlstr = ET.tostring(root, encoding='utf8', method='xml')
    soup = BeautifulSoup(xmlstr, 'lxml')
    extracted_ids = []
    for i in soup.select('ArticleId[IdType="pubmed"]'):
        extracted_ids.append(i.text)        
        
    for event, elem in context:
        if elem.tag == 'ArticleId':
            if elem.attrib['IdType'] == "pubmed":
                index = elem.text
        
        if elem.tag == 'PubmedArticle':
            filename = format(str(index) + ".xml")
            print('saving ', filename, ' from ', XMLfilename)
            try:
                with open(join(directoryPath_write, filename), 'wb+') as f:
                    f.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n")
                    f.write(b"<PubmedArticleSet>\n")
                    f.write(ET.tostring(elem))
                    f.write(b"</PubmedArticleSet>")
            except Exception as e:
                print('failed to save ', filename, ' from ', XMLfilename)
    
