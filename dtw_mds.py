from dataloader import read_data
from methods import *
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.manifold import MDS
from imblearn.over_sampling import SMOTE
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class Model():
    def __init__(self):
        self.svc = SVC()
        self.knn = KNeighborsClassifier(n_neighbors=10)
        self.logreg = LogisticRegression()
        self.gnb = GaussianNB()
        self.rf = RandomForestClassifier()
        self.lda = LinearDiscriminantAnalysis()


def arr_pad_align(arr1, arr2):
    arr1_len = len(arr1)
    arr2_len = len(arr2)
    pad_len = np.max([arr1_len, arr2_len]) - np.min([arr1_len, arr2_len])
    if arr1_len < arr2_len:
        output1 = np.pad(arr1, ((0,pad_len),(0,0)), 'constant', constant_values=0)
        output2 = arr2
    else:
        output2 = np.pad(arr2, ((0,pad_len),(0,0)), 'constant', constant_values=0)
        output1 = arr1
    return output1, output2


def compute_eud_distance(data):
    distance_matrix = np.zeros((len(data), len(data)), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data)):
            if len(data[i]) != len(data[j]):
                arr1, arr2 = arr_pad_align(data[i], data[j])
            else:
                arr1 = data[i]
                arr2 = data[j]
            distance_matrix[i,j] = np.sqrt(np.sum(np.square(arr1-arr2)))
    return distance_matrix


def _predict_success(x, y, model):
    # distance_matrix = apply_dtw(x)
    distance_matrix = compute_eud_distance(x)
    features = MDS(n_components=3, dissimilarity="precomputed").fit_transform(distance_matrix)
    features, labels = SMOTE(k_neighbors=2).fit_resample(features, y)
    score = np.mean(cross_val_score(model, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    return score


def predict_success():
    """Predict task preformance (success or failure)"""
    # controller, task, success = read_data()
    controller, aircraft, task, uid, success= read_data()
    unique_tasks = sorted(list(set(task)))
    scores = list()
    # model = SVC()
    model = Model()
    for t in unique_tasks:
        idx = [u == t for u in task]
        controller_ = [c for c, i in zip(controller, idx) if i]
        success_ = [s for s, i in zip(success, idx) if i]

        # skip task if all subjects succeed or fail because only one class is represented
        success_rate = np.sum(success_) / len(success_)
        if success_rate < 1e-8 or success_rate > 1 - 1e-8:
            continue
        scores.append(_predict_success(controller_, success_, model.svc))
        str = "Prediction accurate on task %d is %.3f" % (t, scores[t-1])
        print(str)
    print("Predict success: {:.3f}".format(np.mean(scores)))
    return


def predict_task(success_only=False):
    """Predict which task the controller data is associated with"""
    # controller, task, success = read_data()
    controller, aircraft, task, uid, success = read_data()
    if success_only:
        controller = [c for c, s in zip(controller, success) if s]
        task = [t for t, s in zip(task, success) if s]

    distance_matrix = compute_eud_distance(controller)

    # distance_matrix = apply_dtw(controller)
    features = MDS(n_components=8, dissimilarity="precomputed").fit_transform(distance_matrix)
    # feat_pca = PCA(n_components=8).fit_transform(distance_matrix)
    features, labels = SMOTE(k_neighbors=2).fit_resample(features, task)

    model=Model()
    svm_pre = np.mean(cross_val_score(model.svc, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with SVM: {:.3f}".format(svm_pre))

    knn_pre = np.mean(cross_val_score(model.knn, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with KNN: {:.3f}".format(knn_pre))

    logreg_pre = np.mean(cross_val_score(model.logreg, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with Log: {:.3f}".format(logreg_pre))

    gnb_pre = np.mean(cross_val_score(model.gnb, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with GaussanNB: {:.3f}".format(gnb_pre))

    rf_pre = np.mean(cross_val_score(model.rf, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with RF: {:.3f}".format(rf_pre))

    lda_pre = np.mean(cross_val_score(model.lda, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with LDA: {:.3f}".format(lda_pre))
    return


if __name__ == "__main__":
    predict_success()
    # predict_task(False)
