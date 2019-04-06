from include.dataPrepration import data_preparation
from sklearn.model_selection import cross_val_score


def KFoldCV(datafiles,alg):
    X,y= data_preparation(datafiles)
    # for 10-fold
    score = cross_val_score(alg, X, y, cv=10, scoring='f1_micro')
    print(score.mean())