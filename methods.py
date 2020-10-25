from fastdtw import fastdtw
from multiprocessing import Pool
import numpy as np
import time

N_CPUS = 5


def _apply_dtw(x, y, x_idx, y_idx):
    distance = fastdtw(x, y)[0]
    # print('computing distance...')
    return distance, x_idx, y_idx


def apply_dtw(data):
    args = [(data[i], data[j], i, j) for i, _ in enumerate(data) for j, _ in enumerate(data)]
    # start = time.time()
    with Pool(N_CPUS) as pool:
        returns = pool.starmap(_apply_dtw, args)
    distance_matrix = np.zeros((len(data), len(data)), dtype=np.float)
    for d, i, j in returns:
        distance_matrix[i, j] = distance_matrix[j, i] = d
    # end = time.time()
    # print(end-start)
    return distance_matrix
