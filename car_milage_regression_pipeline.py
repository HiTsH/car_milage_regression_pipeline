import findspark
findspark.init()


# Importing required libraries
import subprocess
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

#Create SparkSession
spark = SparkSession.builder.appName('ML Pipeline for a Regression Project').getOrCreate()

### Extracting, Transformation and Loading ###
# Download CSV file
download_command = 'curl -o mpg-raw.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/mpg-raw.csv'
subprocess.run(download_command.split())

# Load dataset
df = spark.read.csv('mpg-raw.csv', header=True, inferSchema=True)

# Print dataset schema, top 5 rows, count of cars per origin
df.printSchema()
df.show(5)
df.groupBy('Origin').count().orderBy('count').show()
rowcount1 = df.count()

# Drop duplicates
df = df.dropDuplicates()
rowcount2 = df.count()

# Drop rows with null values
df = df.dropna()
rowcount3 = df.count()

# Rename a column
df = df.withColumnRenamed('Engine Disp', 'Engine_Disp')


# Save dataset to parquet file
df.write.mode('overwrite').parquet('mpg-cleaned.parquet')

# ### Testing row count ###
# print("ETL - Evaluation")
# print("Total rows = ", rowcount1)
# print("Total rows after dropping duplicate rows = ", rowcount2)
# print("Total rows after dropping duplicate rows and rows with null values = ", rowcount3)
# print("Renamed column name = ", df.columns[2])
# print("mpg-cleaned.parquet exists :", os.path.isdir("mpg-cleaned.parquet"))


### ML Pipeline Creation ###
# Load data from parquet file
df = spark.read.parquet('mpg-cleaned.parquet')
rowcount4 = df.count()

# StringIndexer Pipeline Stage
indexer = StringIndexer(inputCol='Origin', outputCol='OriginIndex')

# VectorAssembler Pipeline Stage
assembler = VectorAssembler(inputCols=['Cylinders', 'Engine_Disp', 'Horsepower', 'Weight', 'Accelerate', 'Year'], outputCol='features')

# StandardScaler pipeline Stage
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

# LinearRegression Pipeline Stage
lr = LinearRegression(featuresCol='scaledFeatures', labelCol='MPG')

# Build Pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Split the data into training and testing sets with 70:30 split. Use 42 as seed
(trainingData, testingData) = df.randomSplit([0.7,0.3], seed=42)

# Fit the pipeline using the training data
pipelineModel = pipeline.fit(trainingData)

# ### testing Pipeline ###
# print("ML Pipeline Evaluation")
# print("Total rows = ", rowcount4)
# ps = [str(x).split("_")[0] for x in pipeline.getStages()]
# print("Pipeline Stage 1 = ", ps[0])
# print("Pipeline Stage 2 = ", ps[1])
# print("Pipeline Stage 3 = ", ps[2])
# print("Label column = ", lr.getLabelCol())


### Model Evaluation ###
# Make predictions on testing data
predictions = pipelineModel.transform(testingData)

# Mean Squared Error (mse)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='MPG', metricName='mse')
mse = evaluator.evaluate(predictions)
print(mse)

# Mean Absolute Error (mae)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='MPG', metricName='mae')
mae = evaluator.evaluate(predictions)
print(mae)

# R Squared (r2)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='MPG', metricName='r2')
r2 = evaluator.evaluate(predictions)
print(r2)

# Save the pipeline model
pipelineModel.write().save('car_milage_model')

# ### Testing Model Evaluation ###
# print("Model Evaluation")
# print("Mean Squared Error = ", round(mse,2))
# print("Mean Absolute Error = ", round(mae,2))
# print("R Squared = ", round(r2,2))
# lrModel = pipelineModel.stages[-1]
# print("Intercept = ", round(lrModel.intercept,2))


### Model Persistance for Reuse ###
# Load the pipeline model
loaded_pipelineModel = PipelineModel.load('car_milage_model')

# Use the loaded pipeline model for predictions
predictions = loaded_pipelineModel.transform(testingData)

# Show Prediction
predictions.select('MPG', 'prediction').show()


### Stop Spark Session ###
spark.stop()
