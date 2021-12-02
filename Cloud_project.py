#!/usr/bin/env python
# coding: utf-8

# In[99]:


get_ipython().run_cell_magic('sh', '', '\n# preview the data file\n\nhead -n 5 Downloads/training_1.csv')


# In[12]:


from pyspark.sql import SparkSession


# In[13]:


spark = SparkSession     .builder     .appName("how to read csv file")     .getOrCreate()


# In[14]:


inputDF = spark.read.csv('Downloads/training_1.csv',header='true', inferSchema='true', sep=';')


# In[15]:


inputDF.printSchema()
print("Rows: %s" % inputDF.count())


# In[16]:


display(inputDF.limit(5))


# In[17]:


from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputDF.columns if c != 'quality']

# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")

# transform the original data
dataDF = assembler.transform(inputDF)
dataDF.printSchema()


# In[18]:


display(dataDF.limit(3))


# In[19]:


from pyspark.ml.regression import LinearRegression


# In[20]:


lr = LinearRegression(maxIter=30, regParam=0.3, elasticNetParam=0.3, featuresCol="features", labelCol="quality")
lrModel = lr.fit(dataDF)


# In[22]:


for t in zip(featureColumns, lrModel.coefficients):
    print(t)


# In[23]:


predictionsDF = lrModel.transform(dataDF)
display(predictionsDF.limit(3))


# In[24]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[25]:


evaluator = RegressionEvaluator(
    labelCol='quality', predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictionsDF)
print("Root Mean Squared Error (RMSE) = %g" % rmse)


# In[26]:


from pyspark.sql.functions import *


# In[27]:


avgQuality = inputDF.groupBy().avg('quality').first()[0]
print(avgQuality)


zeroModelPredictionsDF = dataDF.select(col('quality'), lit(avgQuality).alias('prediction'))


zeroModelRmse = evaluator.evaluate(zeroModelPredictionsDF)
print("RMSE of 'zero model' = %g" % zeroModelRmse)


# In[28]:


(trainingDF, testDF) = inputDF.randomSplit([0.7, 0.3])


# In[29]:


from pyspark.ml import Pipeline


# In[30]:


pipeline = Pipeline(stages=[assembler, lr])


# In[31]:


lrPipelineModel = pipeline.fit(trainingDF)


# In[32]:


traningPredictionsDF = lrPipelineModel.transform(trainingDF)
testPredictionsDF = lrPipelineModel.transform(testDF)


# In[33]:


print("RMSE on traning data = %g" % evaluator.evaluate(traningPredictionsDF))

print("RMSE on test data = %g" % evaluator.evaluate(testPredictionsDF))


# In[34]:


from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator


# In[35]:


search_grid = ParamGridBuilder()     .addGrid(lr.regParam, [0.0, 0.3, 0.6])     .addGrid(lr.elasticNetParam, [0.4, 0.6, 0.8]).build()


# In[36]:


cv = CrossValidator(estimator = pipeline, estimatorParamMaps = search_grid, evaluator = evaluator, numFolds = 3)
cvModel = cv.fit(trainingDF)


# In[37]:


cvTestPredictionsDF = cvModel.transform(testDF)
print("RMSE on test data with CV = %g" % evaluator.evaluate(cvTestPredictionsDF))


# In[38]:


cvTestPredictionsDF = cvModel.transform(testDF)
print("RMSE on test data with CV = %g" % evaluator.evaluate(cvTestPredictionsDF))


# In[39]:


print(cvModel.avgMetrics)


# In[40]:


from pyspark.ml.regression import RandomForestRegressor


# In[41]:


rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20,                            minInstancesPerNode=5, seed=33)
rfPipeline = Pipeline(stages=[assembler, rf])


# In[42]:


rfPipelineModel = rfPipeline.fit(trainingDF)


# In[43]:


rfTrainingPredictions = rfPipelineModel.transform(trainingDF)
rfTestPredictions = rfPipelineModel.transform(testDF)
print("Random Forest RMSE on traning data = %g" % evaluator.evaluate(rfTrainingPredictions))
print("Random Forest RMSE on test data = %g" % evaluator.evaluate(rfTestPredictions))


# In[44]:


rfModel = rfPipelineModel.stages[1]
rfModel.featureImportances


# In[45]:


rfPipelineModel.write().overwrite().save('output/rf.model')


# In[46]:


from pyspark.ml import PipelineModel
loadedModel = PipelineModel.load('output/rf.model')
loadedPredictionsDF = loadedModel.transform(testDF)


# In[47]:


print("Loaded model RMSE = %g" % evaluator.evaluate(loadedPredictionsDF))


# In[66]:


from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

all_assembler = VectorAssembler(
    inputCols=featureColumns,
    outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="norm_features")
pca = PCA(k=2, inputCol="norm_features", outputCol="pca_features")

pca_pipeline = Pipeline(stages=[all_assembler, normalizer, pca])

pca_model = pca_pipeline.fit(inputDF)

display(pca_model.transform(inputDF).select('features', 'norm_features', 'pca_features').limit(3))


# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# In[100]:


df = pd.read_csv('Downloads/training_1.csv',sep=';')


# In[92]:


df.info()


# In[93]:


df.isnull().sum()


# In[94]:


df.describe()


# In[95]:


df['quality'].value_counts()


# In[96]:


plt.figure(figsize=(10,8.5))
sns.countplot(df['quality'])
plt.xticks(rotation='vertical',size=15)
plt.show()


# In[97]:


plt.figure(figsize=(15,12))
sns.boxplot(x=df['quality'],data=df)
plt.show()


# In[98]:


sns.pairplot(df)


# In[ ]:




