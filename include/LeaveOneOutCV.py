#doesnt work########

from sklearn.model_selection import LeaveOneOut as loo
from include.dataPrepration import data_preparation


def LeaveOneOutCV(datafiles,alg):
    square_error_sum = 0.0
    for datafile in datafiles:
        X, y = data_preparation(datafile)
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = alg.fit(X_train, y_train.ravel())
            predicted_y = model.predict(X_test)
            square_error_sum += float(y_test[0] - predicted_y) ** 2
        mse = square_error_sum / X.shape[0]
        print ('-----------------------')
        print (('Leave One Out?mse ') , mse)
        print ('-----------------------')

