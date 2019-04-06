from include.dataPrepration import data_preparation,data_preparation2
from sklearn.model_selection import cross_val_score
import numpy as np


def KFoldCV(datafiles,alg):
    scorList=[]
    for i in np.arange(0.25, 7.25, 0.25):
        # X,y= data_preparation(datafiles,i)
        X, y = data_preparation2(datafiles, i)
        # for 10-fold
        score = cross_val_score(alg, X, y, cv=10, scoring='f1_micro')
        scorList.append((score.mean(),i))
    print(scorList)