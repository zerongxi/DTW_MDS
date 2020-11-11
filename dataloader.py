import pickle
import numpy as np

DATA_PATH = "./data/dtw_mds.pkl"
KNOWLEDGE_PATH = "./data/knowledge.pkl"


def read_data(path=DATA_PATH):
    """

    :return: controller: a list of 2D array, time_index x (elevator, aileron, throttle)
    aircraft: a list of 2D array, time_index x (airspeed, altitude, direction)
    task: task id
    uid: user id
    success: success (1.0) or failure (0.0)
    """
    with open(path, "rb") as fp:
        controller, aircraft, task, uid, success = pickle.load(fp)
    task = [int(u.split("_")[-1]) for u in task]
    success = np.array(success, dtype=np.bool)
    return controller, aircraft, task, uid, success


def read_knowledge(path=KNOWLEDGE_PATH):
    """

    :return: knowledge: 2D array, #users x #questions
    uid: user id
    """
    with open(path, "rb") as fp:
        knowledge, uid = pickle.load(fp)
    return knowledge, uid