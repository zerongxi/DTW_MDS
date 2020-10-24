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


def _predict_success(x, y, model):
    distance_matrix = apply_dtw(x)
    features = MDS(n_components=3, dissimilarity="precomputed").fit_transform(distance_matrix)
    features, labels = SMOTE(k_neighbors=2).fit_resample(features, y)
    score = np.mean(cross_val_score(model, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    return score


def predict_success():
    """Predict task preformance (success or failure)"""
    controller, task, success = read_data()
    unique_tasks = sorted(list(set(task)))
    scores = list()
    model = SVC()
    for t in unique_tasks:
        idx = [u == t for u in task]
        controller_ = [c for c, i in zip(controller, idx) if i]
        success_ = [s for s, i in zip(success, idx) if i]

        # skip task if all subjects succeed or fail because only one class is represented
        success_rate = np.sum(success_) / len(success_)
        if success_rate < 1e-8 or success_rate > 1 - 1e-8:
            continue

        scores.append(_predict_success(controller_, success_, model))
    print("Predict success: {:.3f}".format(np.mean(scores)))
    return


def predict_task(success_only=False):
    """Predict which task the controller data is associated with"""
    controller, task, success = read_data()
    if success_only:
        controller = [c for c, s in zip(controller, success) if s]
        task = [t for t, s in zip(task, success) if s]
    # if os.path.exists("distance_matrix.txt"):
    #     distance_matrix = np.loadtxt("distance_matrix.txt")
    # else:
    distance_matrix = apply_dtw(controller)
    features = MDS(n_components=8, dissimilarity="precomputed").fit_transform(distance_matrix)
    features, labels = SMOTE(k_neighbors=2).fit_resample(features, task)
    svm_pre = np.mean(cross_val_score(SVC(), features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with SVM: {:.3f}".format(svm_pre))
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn_pre = np.mean(cross_val_score(knn, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with KNN: {:.3f}".format(knn_pre))
    logreg = LogisticRegression()
    logreg_pre = np.mean(cross_val_score(logreg, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with Log: {:.3f}".format(logreg_pre))
    gnb = GaussianNB()
    gnb_pre = np.mean(cross_val_score(gnb, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with GaussanNB: {:.3f}".format(gnb_pre))
    rf = RandomForestClassifier()
    rf_pre = np.mean(cross_val_score(rf, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with RF: {:.3f}".format(rf_pre))
    lda = LinearDiscriminantAnalysis()
    lda_pre = np.mean(cross_val_score(lda, features, labels, scoring="accuracy", cv=5, n_jobs=5))
    print("Predict task with LDA: {:.3f}".format(lda_pre))
    return


if __name__ == "__main__":
    # predict_success()
    predict_task(False)
