from sklearn.tree import DecisionTreeClassifier
from include.LeaveOneOutCV import LeaveOneOutCV
from include.KFoldCV import KFoldCV

datafiles=[]
for i in range(1,3):
    datafiles.append("../data/subject"+str(i)+"_ideal.log")

clf = DecisionTreeClassifier(random_state=0)
# LeaveOneOutCV(datafiles,clf)
KFoldCV(datafiles,clf)






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