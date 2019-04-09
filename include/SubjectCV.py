import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from dataPrepration import data_preparation


def subject_cv(X_index):
    i = 0
    while i < len(X_index):
        testIndex = np.arange(X_index[i][1][0],X_index[i][1][1]+1, dtype=int)
        trainIndex = np.concatenate((np.arange(0,X_index[i][1][0], dtype=int),np.arange(X_index[i][1][1]+1,X_index[len(X_index)-1][1][1]+1, dtype=int)),axis=None)
        # print(trainIndex)
        # print(testIndex)
        yield trainIndex, testIndex
        i += 1

def SubjectCV(datafiles,alg):
    scoreList=[]
    for i in np.arange(0.25, 7.25, 0.25):
        X,y,X_index= data_preparation(datafiles,i)
        custom_cv = subject_cv(X_index)
        score = cross_val_score(alg, X, y, cv=custom_cv, scoring='f1_micro')
        scoreList.append((score.mean(), i))
    print(scoreList)
    return scoreList
