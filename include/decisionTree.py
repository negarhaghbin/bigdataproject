from sklearn.tree import DecisionTreeClassifier
# from include.LeaveOneOutCV import LeaveOneOutCV
from KFoldCV import KFoldCV
from matplotlib import pyplot as plt
from SubjectCV import SubjectCV

datafiles=[]
for i in range(1,18):
    datafiles.append("../data/subject"+str(i)+"_ideal.log")

clf = DecisionTreeClassifier(random_state=0)
# LeaveOneOutCV(datafiles,clf)
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
plt.savefig('DT-KFold')
# plt.savefig('DT-Subject')

# scorList=[(0.9476157038155899, 0.25), (0.9674642839556007, 0.5), (0.9700095860587095, 0.75), (0.974421440703023, 1.0), (0.9780703905507224, 1.25), (0.9817596876637495, 1.5), (0.9805509723569832, 1.75), (0.9771505420245538, 2.0), (0.9771232786979761, 2.25), (0.9834989325992106, 2.5), (0.9796457196902688, 2.75), (0.9831108516964286, 3.0), (0.9808368003715104, 3.25), (0.9817631769299455, 3.5), (0.981483004645136, 3.75), (0.9829826105312307, 4.0), (0.9837724945862277, 4.25), (0.9814969460036436, 4.5), (0.9839565369641079, 4.75), (0.985607416454599, 5.0), (0.9836442663881806, 5.25), (0.9836264185286773, 5.5), (0.9836289167796426, 5.75), (0.9828646908774019, 6.0), (0.9838199904576455, 6.25), (0.9797666381715958, 6.5), (0.9829712305833361, 6.75), (0.9831905571509532, 7.0)]
# resultScore=[]
# resultWindow=[]
# for i in range(0, len(scorList)):
#     resultScore.append(scorList[i][0])
#     resultWindow.append((i+1)*0.25)
# plt.plot(resultWindow,resultScore)
# plt.ylim(0.2, 1)
# plt.xlabel('window sizes')
# plt.ylabel('f1-scores')
# plt.show()


#train, test=loo.split(data)
#cross_validate(clf, rddFeatures, rddLabels, cv=10)
#print(loo.split(data))
# dataset = init_spark().createDataFrame(
#      [(Vectors.dense([0.0]), 0.0),
#       (Vectors.dense([0.4]), 1.0),
#       (Vectors.dense([0.5]), 0.0),
#       (Vectors.dense([0.6]), 1.0),
#       (Vectors.dense([1.0]), 1.0)] * 10,
#      ["features", "label"])
# lr = LogisticRegression()
# grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
# evaluator = BinaryClassificationEvaluator()
# cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,parallelism=2)
# cvModel = cv.fit(dataset)
# cvModel.avgMetrics[0]
# print(evaluator.evaluate(cvModel.transform(dataset)))


# # Load and parse the data file into an RDD of LabeledPoint.
# data = MLUtils.loadLibSVMFile(sc, '../data/heart.txt')
# # Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = data.randomSplit([0.7, 0.3])
#
# # Train a DecisionTree model.
# #  Empty categoricalFeaturesInfo indicates all features are continuous.
# model = DecisionTree.trainClassifier(trainingData,2, {})
#
# # Evaluate model on test instances and compute test error
# predictions = model.predict(testData.map(lambda x: x.features))
# labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
# testErr = labelsAndPredictions.filter(
#     lambda lp: lp[0] != lp[1]).count() / float(testData.count())
# print('Test Error = ' + str(testErr))
# print('Learned classification tree model:')
# print(model.toDebugString())
#
# # Save and load model
# model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
# sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")