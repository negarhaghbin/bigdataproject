from include.dataPrepration import data_preparation,data_preparation2
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt


def KFoldCV(datafiles,alg):
    scorList=[]
    for i in np.arange(0.25, 7.25, 0.25):
        # X,y= data_preparation(datafiles,i)
        X, y = data_preparation2(datafiles, i)
        # for 10-fold
        score = cross_val_score(alg, X, y, cv=10, scoring='f1_micro')
        scorList.append((score.mean(),i))
    print(scorList)
    resultScore = []
    resultWindow = []
    for i in range(0, len(scorList)):
        resultScore.append(scorList[i][0])
        resultWindow.append((i + 1) * 0.25)
    plt.plot(resultWindow, resultScore)
    plt.ylim(0.2, 1)
    plt.xlabel('window sizes')
    plt.ylabel('f1-scores')
    plt.show()