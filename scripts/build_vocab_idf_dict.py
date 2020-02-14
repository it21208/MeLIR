# author = Alexandros Ioannidis
import logging
import os

from utils import TfidfTgzReader


def build_vocab_idf_dict(idf_raw_folder, docid_idx_dict):

  logging.info('start building vocab idf dict')
  vocab_idx_dict, idf_dict, idf_word_dict = {}, {}, {} 
  cur_idx = 0
  idf_raw_files = os.listdir(idf_raw_folder)

  for idf_raw in idf_raw_files:
    count = 0
    reader = TfidfTgzReader(os.path.join(idf_raw_folder, idf_raw))

    while reader.hasnextdoc():
      docid = reader.getnextdoc().strip()
      count += 1
      if count % 100000 == 0:
        logging.info(f'{count} files have been processed...')
      if docid not in docid_idx_dict:
        reader.skipdoc()
        continue
      doc_idx = docid_idx_dict[docid]

      while reader.hasnexttfidf():
        word, idf = reader.getnexttfidf()
        #print(word, idf)
        if word not in vocab_idx_dict:
            vocab_idx_dict[word] = cur_idx
            vocab_idx_dict[cur_idx] = word
            cur_idx += 1
        idf_dict[(doc_idx, vocab_idx_dict[word])] = float(idf)
        idf_word_dict[word] = float(idf)

  logging.info(f'finish building vocab idf dict, idf_word_dict, {cur_idx} words in total.')
  
  total_words = cur_idx

  return(vocab_idx_dict, idf_dict, total_words, idf_word_dict)
