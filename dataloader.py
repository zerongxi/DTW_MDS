import pickle
import numpy as np

DATA_PATH = "./dtw_mds.pkl"


def read_data(path=DATA_PATH):
    """

    :param path:
    :return: controller: a list of 2D array, time_index x (elevator, aileron, throttle)
    task: task id
    success: success (1.0) or failure (0.0)
    """
    with open(path, "rb") as fp:
        controller, task, success = pickle.load(fp)
    task = [int(u.split("_")[-1]) for u in task]
    success = np.array(success, dtype=np.bool)
    return controller, task, success
