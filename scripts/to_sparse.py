# author = Alexandros Ioannidis
import logging

import numpy as np
import scipy.sparse
import sklearn.preprocessing
from scipy.sparse import csr_matrix


def to_sparse(tfidf_dict, docid_idx_dict, vocab_idx_dict):

    logging.info("Converting tf-idf info to sparse matrix")

    num_docs, num_vocabs = len(docid_idx_dict) // 2, len(vocab_idx_dict) // 2

    indices = tuple(zip(*tfidf_dict.keys()))

    values = list(tfidf_dict.values())

    tfidf_sp = csr_matrix((values, indices), shape=(
        num_docs, num_vocabs), dtype=np.float32)

    return sklearn.preprocessing.normalize(tfidf_sp, norm='l2')
