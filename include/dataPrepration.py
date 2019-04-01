from pyspark.sql import SparkSession

#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def data_preparation(filename):
    '''
    in this function we first remove the data which doesnt have label and then only select timestamp, 9 sensors
    acceleration and label columns.
    '''
    sc = init_spark().sparkContext
    rdd = sc.textFile(filename) \
        .map(lambda row: row.split())\
        .filter(lambda x:x[119]!='0')
    X=rdd.map(lambda x:[x[0],x[1],x[2],x[3],x[4]
                                 ,x[15],x[16],x[17]
                                 ,x[28],x[29],x[30]
                                 ,x[41],x[42],x[43]
                                 ,x[54],x[55],x[56]
                                 ,x[67],x[68],x[69]
                                 ,x[80],x[81],x[82]
                                 ,x[93],x[94],x[95]
                                 ,x[106],x[107],x[108]])
    y=rdd.map(lambda x:x[119])
    return X.collect(),y.collect()