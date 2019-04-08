import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from include.dataPrepration import data_preparation


def subject_cv(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        idy = np.arange(100, dtype=int)
        yield idy, idx
        i += 1

def SubjectCV(datafiles,alg):

    scorList=[]
    for i in np.arange(0.25, 7.25, 0.25):
        X,y= data_preparation(datafiles,i)
        custom_cv = subject_cv(X)
        score = cross_val_score(alg, X, y, cv=custom_cv, scoring='f1_micro')
        scorList.append((score.mean(), i))
    print(scorList)
