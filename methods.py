from fastdtw import fastdtw
from multiprocessing import Pool
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

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

def split_cross_validation(model,X,y,success,k):
    rk_fold=RepeatedKFold(n_splits=k, n_repeats=k)
    score_list=[]
    y = np.array(y)
    for train_index, test_index in rk_fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_s = [c for c, s in zip(X_train, success) if s]
        y_train_s = [t for t, s in zip(y_train, success) if s]
        clf = model.fit(X_train_s,y_train_s)
        score = clf.score(X_test, y_test)
        score_list.append(score)
    return score_list



