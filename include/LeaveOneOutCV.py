# doesnt work########
import numpy as np
from sklearn.model_selection import LeaveOneOut as loo
from include.dataPrepration import data_preparation


def LeaveOneOutCV(datafiles, alg):
    square_error_sum: float = 0.0
    scoresList = []
    X, y = data_preparation(datafiles)
    scores= []
    #print(loo.get_n_splits(X, y))
    #print(loo.split(X,y))
    for train_index, test_index in loo.split(X.reshape(-1,1), y.reshape(-1,1)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = alg.fit(X_train, y_train.ravel())
        predicted_y = model.predict(X_test)
        square_error_sum += float(y_test[0] - predicted_y) ** 2
        scores.append(model.score(X_test, y_test))
    mse = square_error_sum / X.shape[0]
    scoresList.append(np.array(scores).mean())
    print('-----------------------')
    print('Leave One Out?mse ', mse)
    print('-----------------------')

    avg = sum(scoresList) / len(scoresList)
    print(avg)
