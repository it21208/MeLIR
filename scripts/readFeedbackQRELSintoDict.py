# author = Alexandros Ioannidis


def readFeedbackQRELSintoDict(topic_seedDoc, qrels_filepath):
    
    f = open(qrels_filepath, "r")
    
    topic_qrels_dict = {}

    for line in f:

        topic = line.split()[0]

        if topic == topic_seedDoc:

            doc = line.split()[2]

            # relevancy 0 or 1
            relevancy = line.split()[3]

            if relevancy == '0' or relevancy == '1':
                
                relevancy = int(relevancy)

            # Check if relevancy is 2 and if yes and make it 1.
            if relevancy == '2':
                
                # relevancy = '1'
                relevancy = 1

            temp_dictionary = {doc: relevancy}
            
            topic_qrels_dict.update(temp_dictionary)
            
    return(topic_qrels_dict)
