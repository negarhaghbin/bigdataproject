from sklearn.neighbors import KNeighborsClassifier
from include.CVs import KFoldCV,SubjectCV
from matplotlib import pyplot as plt

def start(clf):
    CVs=[KFoldCV,SubjectCV]
    datafiles=[]
    data = {}
    for i in range(1,18):
        datafiles.append("../data/subject"+str(i)+"_ideal.log")

    for cv in CVs:
        scoreList=cv(datafiles,clf)
        resultScore = []
        resultWindow = []
        for i in range(0, len(scoreList)):
            resultScore.append(scoreList[i][0])
            resultWindow.append((i + 1) * 0.25)
        data[cv.__name__]=(resultScore,resultWindow)

    plt.plot(data['SubjectCV'][1], data['SubjectCV'][0],'r--', data['KFoldCV'][1], data['KFoldCV'][0], '-')
    plt.ylim(0.2, 1)
    plt.xlabel('window sizes')
    plt.ylabel('f1-scores')
    # plt.savefig('DT')
    plt.savefig('KNN')
    plt.show()


clf = KNeighborsClassifier(n_neighbors=3)
# clf = DecisionTreeClassifier(random_state=0)
start(clf)


