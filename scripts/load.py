# author = Alexandros Ioannidis

import os

import numpy as np
import scipy.sparse
from scipy.sparse import load_npz, coo_matrix, csr_matrix


def load(topic, path):
    X = load_npz(os.path.join(path, f'{topic}.npz'))
    y = np.load(os.path.join(path, f'{topic}.npy'))
    #save_npz("/home/pfb16181/NetBeansProjects/scr_matrix_CD008782.npz", X)
    cx = X.tocoo()
    return(cx, y)
