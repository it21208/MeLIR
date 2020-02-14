# author = Alexandros Ioannidis

import os

import numpy as np
from scipy.sparse import load_npz


def load(topic, path):
    X = load_npz(os.path.join(path, f'{topic}.npz'))
    y = np.load(os.path.join(path, f'{topic}.npy'))
    return X, y
