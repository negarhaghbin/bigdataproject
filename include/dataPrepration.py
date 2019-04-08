from pyspark.sql import SparkSession
import numpy as np
from collections import Counter

timestamp = 0.02


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def addList(x, y):
    result = []
    label = []
    for i in range(0, len(x[0][0])):
        result.append(x[0][0][i] + y[0][0][i])
    count = x[1] + y[1]
    for i in range(0, len(x[0][1])):
        label.append(x[0][1][i])
    for i in range(0, len(y[0][1])):
        label.append(y[0][1][i])

    returnVal = ((result, label), count)
    return returnVal


def mean(x):
    result = []

    label = Counter(x[0][1]).most_common(1)[0][0]
    for i in range(0, len(x[0][0])):
        result.append(x[0][0][i] / x[1])

    returnVal = (result, label)
    return returnVal


# (window_size,list of(list of(windows)))
def data_preparation(filenames, window_size):
    """
    in this function we only select timestamp, 9 sensors
    acceleration and label columns.
    """
    sc = init_spark().sparkContext
    X=sc.parallelize([])
    data_count = int(window_size / timestamp)
    start=0
    X_Index = []
    for index,filename in enumerate(filenames):
        rdd = sc.textFile(filename) \
                .map(lambda row: row.split())
        X_temp = rdd.map(lambda x: ((float(x[0]), float(x[1])), ([float(x[2]), float(x[3]),float(x[4])
                                    , float(x[15]), float(x[16]), float(x[17])
                                    , float(x[28]), float(x[29]), float(x[30])
                                    , float(x[41]), float(x[42]), float(x[43])
                                    , float(x[54]), float(x[55]), float(x[56])
                                    , float(x[67]), float(x[68]), float(x[69])
                                    , float(x[80]), float(x[81]), float(x[82])
                                    , float(x[93]), float(x[94]), float(x[95])
                                    , float(x[106]), float(x[107]), float(x[108])],
                                        [int(x[119])]))).zipWithIndex()
        X_temp = X_temp.map(lambda x: (int(x[1] / data_count), x[0][1])) \
                .map(lambda x: (x[0], (x[1], 1)))\
                .reduceByKey(lambda x, y: addList(x, y)) \
                .map(lambda x: (index,mean(x[1])))
        end=start+X_temp.count()-1
        X_Index.append((index,[start,end]))
        start=end+1
        X=X.union(X_temp)

    y = X.map(lambda x: x[1][1])
    X=X.map(lambda x: x[1][0])
    # print("###############")
    # print(X_Index)
    return np.array(X.collect()).astype(np.float64), np.array(y.collect()).astype(np.float64), X_Index

def data_preparation2(filenames, window_size):
    """
    in this function we first remove the data which doesnt have label and then only select timestamp, 9 sensors
    acceleration and label columns.
    """
    sc = init_spark().sparkContext
    X=sc.parallelize([])
    rdd = sc.parallelize([])
    data_count = int(window_size / timestamp)
    for filename in filenames:
        rdd_temp = sc.textFile(filename) \
                .map(lambda row: row.split())

        print(rdd_temp.count())
        rdd=rdd.union(rdd_temp)

    X_temp = rdd.map(lambda x: ((float(x[0]), float(x[1])), ([float(x[2]), float(x[3]),float(x[4])
                                    , float(x[15]), float(x[16]), float(x[17])
                                    , float(x[28]), float(x[29]), float(x[30])
                                    , float(x[41]), float(x[42]), float(x[43])
                                    , float(x[54]), float(x[55]), float(x[56])
                                    , float(x[67]), float(x[68]), float(x[69])
                                    , float(x[80]), float(x[81]), float(x[82])
                                    , float(x[93]), float(x[94]), float(x[95])
                                    , float(x[106]), float(x[107]), float(x[108])],
                                        [int(x[119])]))).sortByKey().zipWithIndex()
    X_temp = X_temp.map(lambda x: (int(x[1] / data_count), x[0][1])) \
                .map(lambda x: (x[0], (x[1], 1)))\
                .reduceByKey(lambda x, y: addList(x, y)) \
                .map(lambda x: mean(x[1]))

    y = X_temp.map(lambda x: x[1])
    X = X_temp.map(lambda x: x[0])
    return np.array(X.collect()).astype(np.float64), np.array(y.collect()).astype(np.float64)
