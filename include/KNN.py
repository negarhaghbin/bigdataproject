from sklearn.neighbors import KNeighborsClassifier
from include.LeaveOneOutCV import LeaveOneOutCV
from include.KFoldCV import KFoldCV

datafiles=[]
for i in range(1,18):
    datafiles.append("../data/subject"+str(i)+"_ideal.log")

clf = KNeighborsClassifier(n_neighbors=3)
LeaveOneOutCV(datafiles,clf)
#KFoldCV(datafiles,clf)

