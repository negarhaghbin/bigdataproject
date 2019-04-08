from sklearn.neighbors import KNeighborsClassifier
#from include.LeaveOneOutCV import LeaveOneOutCV
from KFoldCV import KFoldCV
from matplotlib import pyplot as plt

datafiles=[]
for i in range(1,10):
    datafiles.append("../data/subject"+str(i)+"_ideal.log")

clf = KNeighborsClassifier(n_neighbors=3)
#LeaveOneOutCV(datafiles,clf)
scoreList=KFoldCV(datafiles,clf)
# scoreList=SubjectCV(datafiles,clf)
resultScore = []
resultWindow = []
for i in range(0, len(scoreList)):
    resultScore.append(scoreList[i][0])
    resultWindow.append((i + 1) * 0.25)
plt.plot(resultWindow, resultScore)
plt.ylim(0.2, 1)
plt.xlabel('window sizes')
plt.ylabel('f1-scores')
plt.show()


