import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from include.CVs import KFoldCV,SubjectCV
from matplotlib import pyplot as plt
import time
import json
from include.dataPrepration import data_preparation


def plot(result_dict):
    plt.rcParams['font.size'] = 12
    print(result_dict)
    for key in result_dict:
        print(key + " starting.")
        fig, ax = plt.subplots()
        ax.plot(result_dict[key]['KNN'][1], result_dict[key]['KNN'][0], 'y-',
                result_dict[key]['DT'][1], result_dict[key]['DT'][0], 'b-')
        print(np.argmax(result_dict[key]['KNN'][0]))
        ax.plot((np.argmax(result_dict[key]['KNN'][0]))*0.25 + 0.25, max(result_dict[key]['KNN'][0]), 'r*', label='peak')
        ax.plot((np.argmax(result_dict[key]['DT'][0])) * 0.25 + 0.25, max(result_dict[key]['DT'][0]), 'r*',
                label='peak')
        ax.legend(('KNN','DT','peak'),
                   loc=(1.004, .72))
        ax.set(ylim=[0.2,1], title=key , xlabel='Windows Size(s)', ylabel='f1-scores')
        plt.savefig(key)
        plt.show()

def start(classifiers,CVs):
    datafiles=[]
    for i in range(1,18):
        datafiles.append("../data/subject"+str(i)+"_ideal.log")

    windows_data={}
    for i in np.arange(0.25, 7.25, 0.25):
        X,y, X_index= data_preparation(datafiles,i)
        windows_data[i]=(X,y,X_index)
        #if data is prepared uncomment these
        # prepared_data=json.load(open("./windowed_data/" + str(i)))
        # windows_data[i] = (prepared_data['X'], prepared_data['y'], prepared_data['X_Index'])

    result_dict = {}
    run_times = []
    for cv in CVs:
        print(cv.__name__ + " starting.")
        temp={}
        for classifier in classifiers:
            print(classifier + " starting.")
            start = time.time()
            scores_dict=cv(windows_data,classifiers[classifier])
            end = time.time()
            run_times.append((cv.__name__,classifier, (end - start)/60))
            resultScore = []
            resultWindow = []
            for window_size in scores_dict:
                resultScore.append(scores_dict[window_size])
                resultWindow.append(window_size)
            temp[classifier]=(resultScore,resultWindow)

        result_dict[cv.__name__]=temp
    print(run_times)
    plot(result_dict)



classifiers={'KNN':KNeighborsClassifier(n_neighbors=3),
             'DT':DecisionTreeClassifier(criterion="entropy", random_state=0)}
CVs=[KFoldCV,SubjectCV]
start(classifiers,CVs)