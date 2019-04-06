import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from include.dataPrepration import data_preparation,data_preparation2


def LeaveOneOutCV(datafiles, alg):
    scorList=[]
    loo = LeaveOneOut()
    for i in np.arange(0.25, 7.25, 0.25):
        # X,y= data_preparation(datafiles,i)
        X, y = data_preparation2(datafiles, i)
        n=X.shape[0]

        print("########")
        print(X.shape)
        print("########")
        print(y.shape)

        y = y.reshape(-1, 1)

        print("########")
        print(X.shape)
        print("########")
        print(y.shape)


        loo.get_n_splits(X)
        crossvalidation = KFold(n_splits=n, random_state=None, shuffle=False)
        score = cross_val_score(alg, X, y, scoring="f1_micro", cv=crossvalidation , n_jobs=1)
        scorList.append((score.mean(),i))
    print(scorList)







