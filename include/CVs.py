from sklearn.model_selection import cross_val_score
import numpy as np


def KFoldCV(windows_data,alg):
    scoreList=[]
    for window_data in windows_data:
        # for 10-fold
        score = cross_val_score(alg, windows_data[window_data][0], windows_data[window_data][1], cv=10, scoring='f1_micro')
        scoreList.append((score.mean(),window_data))
    print(scoreList)
    return scoreList

def subject_cv(X_index):
    i = 0
    while i < len(X_index):
        testIndex = np.arange(X_index[i][1][0],X_index[i][1][1]+1, dtype=int)
        trainIndex = np.concatenate((np.arange(0,X_index[i][1][0], dtype=int),np.arange(X_index[i][1][1]+1,X_index[len(X_index)-1][1][1]+1, dtype=int)),axis=None)
        # print(trainIndex)
        # print(testIndex)
        yield trainIndex, testIndex
        i += 1

def SubjectCV(windows_data,alg):
    scoreList=[]
    for window_data in windows_data:
        custom_cv = subject_cv(windows_data[window_data][2])
        score = cross_val_score(alg, windows_data[window_data][0], windows_data[window_data][1], cv=custom_cv, scoring='f1_micro')
        scoreList.append((score.mean(), window_data))
    print(scoreList)
    return scoreList
