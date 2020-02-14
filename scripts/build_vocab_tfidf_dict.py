# author = Alexandros Ioannidis
import logging
import os

from utils import TfidfTgzReader


def build_vocab_tfidf_dict(tfidf_raw_folder, docid_idx_dict):

  logging.info('start building vocab tfidf dict')
  vocab_idx_dict, tfidf_dict = {}, {}
  cur_idx = 0
  tfidf_raw_files = os.listdir(tfidf_raw_folder)

  for tfidf_raw in tfidf_raw_files:
    count = 0
    reader = TfidfTgzReader(os.path.join(tfidf_raw_folder, tfidf_raw))

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
        word, tfidf = reader.getnexttfidf()
        if word not in vocab_idx_dict:
            vocab_idx_dict[word] = cur_idx
            vocab_idx_dict[cur_idx] = word
            cur_idx += 1
        tfidf_dict[(doc_idx, vocab_idx_dict[word])] = float(tfidf)

  logging.info(f'finish building vocab tfidf dict, {cur_idx} words in total.')
  
  total_words = cur_idx

  return(vocab_idx_dict, tfidf_dict, total_words)
