# spark_mllib_basic_machine_learning_code

# Running basic logistic regression on spark mllib

I am using an aws EMR spark cluster
emr-4.7.1
Spark: Spark 1.6.1 on Hadoop 2.7.2 YARN with Ganglia 3.7.2
m4.large

## copying file from local to aws ubuntu / linux 

scp  -i ~/***.pem sample_svm_data.txt hadoop@ec2-**.compute-1.amazonaws.com:/home/hadoop/temp_files

## copying from linux to hdfs 

hadoop fs -put /home/hadoop/temp_files/sample_svm_data.txt /user/hadoop/sample_svm_data.txt



## getting into pyspark
pyspark

## basic code


from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "myModelPath")
sameModel = LogisticRegressionModel.load(sc, "myModelPath")
