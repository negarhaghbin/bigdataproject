from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np


def kfold_cv(X, y):
    rs = ShuffleSplit(n_splits=10, random_state=0)
    for trainIndex, testIndex in rs.split(X, y):
        yield trainIndex, testIndex


#{windowsize:score}
def KFoldCV(windows_data, alg):
    scores_dict = {}
    for window_data in windows_data:
        custom_cv = kfold_cv(windows_data[window_data][0], windows_data[window_data][1])
        score = cross_val_score(alg, windows_data[window_data][0], windows_data[window_data][1], cv=custom_cv,
                                scoring='f1_micro')
        scores_dict[window_data]=score.mean()
    print(scores_dict)
    return scores_dict


def subject_cv(X_index):
    i = 0
    while i < len(X_index):
        testIndex = np.arange(X_index[i][1][0], X_index[i][1][1] + 1, dtype=int)
        trainIndex = np.concatenate((np.arange(0, X_index[i][1][0], dtype=int),
                                     np.arange(X_index[i][1][1] + 1, X_index[len(X_index) - 1][1][1] + 1, dtype=int)),
                                    axis=None)
        yield trainIndex, testIndex
        i += 1


def SubjectCV(windows_data, alg):
    scores_dict = {}
    for window_data in windows_data:
        custom_cv = subject_cv(windows_data[window_data][2])
        score = cross_val_score(alg, windows_data[window_data][0], windows_data[window_data][1], cv=custom_cv,
                                scoring='f1_micro')
        scores_dict[window_data]=score.mean()
    print(scores_dict)
    return scores_dict
