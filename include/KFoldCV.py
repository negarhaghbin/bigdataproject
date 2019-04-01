from include.dataPrepration import data_preparation
from sklearn.model_selection import cross_val_score


def KFoldCV(datafiles,alg):
    scoresList=[]
    for datafile in datafiles:
        X,y= data_preparation(datafile)
        #for 10-fold
        scores = cross_val_score(alg, X, y, cv=10, scoring='f1_weighted')
        #Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label)
        scoresList.append(scores.mean())
    avg=sum(scoresList)/len(scoresList)
    print(avg)