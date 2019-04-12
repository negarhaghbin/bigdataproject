import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from include.CVs import KFoldCV,SubjectCV
from matplotlib import pyplot as plt
import time
import json



from include.dataPrepration import data_preparation

def plot(result_dict):
    for key in result_dict:
        fig, ax = plt.subplots()
        ax.plot(result_dict[key]['KNN'][1], result_dict[key]['KNN'][0], 'r.-',
                result_dict[key]['DT'][1], result_dict[key]['DT'][0], '.-')
        ax.legend(('KNN', 'DT'),
                   loc='upper right', bbox_to_anchor=(1, 0.5))
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set(ylim=[0.2,1], title=key , xlabel='Windows Size(s)', ylabel='f1-scores')
        # ax.title('Decision Tree Classifier')
        # ax.ylim(0.2, 1)
        # ax.xlabel('window sizes')
        # plt.ylabel('f1-scores')
        plt.savefig(key)
        # plt.savefig('KNN')
        plt.show()

def start(classifiers):
    CVs=[KFoldCV,SubjectCV]
    datafiles=[]
    result_dict = {}
    prepared_data={}
    windows_data={}
    run_times=[]
    for i in range(1,18):
        datafiles.append("../data/subject"+str(i)+"_ideal.log")

    for i in np.arange(0.25, 7.25, 0.25):
        X,y, X_index= data_preparation(datafiles,i)
        # prepared_data=json.load("./windowed_data" + open(str(i)))
        windows_data[i]=(X,y,X_index)
        # windows_data[i] = (prepared_data['X'], prepared_data['y'], prepared_data['X_Index'])

    for cv in CVs:
        temp={}

        for index, classifier in enumerate(classifiers):
            start = time.time()
            scoreList=cv(windows_data,classifier)
            end = time.time()
            run_times.append((cv.__name__,"KNN" if index==0 else "DT", end - start))
            resultScore = []
            resultWindow = []
            for i in range(0, len(scoreList)):
                resultScore.append(scoreList[i][0])
                resultWindow.append((i + 1) * 0.25)
            temp["KNN" if index==0 else "DT"]=(resultScore,resultWindow)
            print(run_times)

        result_dict[cv.__name__]=temp

    plot(result_dict)



classifiers=[]
classifiers.append(KNeighborsClassifier(n_neighbors=3))
classifiers.append(DecisionTreeClassifier(criterion="entropy", random_state=0))
start(classifiers)


