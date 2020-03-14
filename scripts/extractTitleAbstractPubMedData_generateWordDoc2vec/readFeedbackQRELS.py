# !/usr/bin/python
# author = Alexandros Ioannidis

def readFeedbackQRELS(topic_current, qrels_filepath):
    
    f = open(qrels_filepath, "r")
    
    topic_doc_list = []

    topic_qrels_dict = {}

    for line in f:

        topic = line.split()[0]

        if topic == topic_current:

            doc = line.split()[2]

            topic_doc_list.append(doc)

            # relevancy 0 or 1
            relevancy = line.split()[3]

            if relevancy == '0' or relevancy == '1':
                
                relevancy = int(relevancy)

            # Check if relevancy is 2 and make it 1
            if relevancy == '2':
                
                #relevancy = '1'
                relevancy = 1

            temp_dictionary = {doc: relevancy}
            topic_qrels_dict.update(temp_dictionary)

    return(topic_doc_list, topic_qrels_dict)
