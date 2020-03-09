# author = Alexandros Ioannidis
import logging
import os


# all files rank file folder
def build_docid_idx_and_label(rank_file_folder, topic_list):
    logging.info(f'start building docid idx dict and topic docid label dict')
    cur_idx = 0
    docid_idx_dict = {}
    topic_docid_label = {topic: {} for topic in topic_list}
    rank_files = os.listdir(rank_file_folder)

    for idx, rank_file in enumerate(rank_files):

        logging.info(f'building docid idx dict for {rank_file}')
        with open(os.path.join(rank_file_folder, rank_file), 'r') as f:

            for line in f:
                topic, _, docid, label = line.strip().split()
                if topic in topic_list:
                    if docid not in docid_idx_dict:
                        # Build docid_idx_dict
                        docid_idx_dict[docid] = cur_idx
                        docid_idx_dict[cur_idx] = docid
                        cur_idx += 1
                    # update topic_docid_label
                    if label == '0':
                        topic_docid_label[topic][docid_idx_dict[docid]] = 0
                    else:
                        topic_docid_label[topic][docid_idx_dict[docid]] = 1

        if idx == 0:
            docid_idx_dict_abs = docid_idx_dict
            topic_docid_label_abs = topic_docid_label
        else:
            docid_idx_dict_content = docid_idx_dict
            topic_docid_label_content = topic_docid_label

    logging.info(f'finish building docid idx dict and topic docid label dict.')
    logging.info(f'{cur_idx} files found in total.')
    return(docid_idx_dict_abs, topic_docid_label_abs, cur_idx, docid_idx_dict_content, topic_docid_label_content)
