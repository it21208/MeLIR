# author = Alexandros Ioannidis
import logging
import os

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, save_npz


def write_feature(topic_docid_label: dict, feature_matrix: csr_matrix, docid_idx_dict: dict, out_path: str):
    
    logging.info("Writting training data...")

    for topic, labels in topic_docid_label.items():

        logging.info(f'Writing topic {topic}')

        X_idx, y = list(zip(*labels.items()))

        X, y = feature_matrix[X_idx, :], np.array(y)

        save_npz(os.path.join(out_path, f'{topic}.npz'), X)
        
        np.save(os.path.join(out_path, f'{topic}.npy'), np.array(y))
