# Databricks notebook source
# MAGIC %run your_import_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import module only for this notebook

# COMMAND ----------

# MAGIC %pip install tensorflow

# COMMAND ----------

dbutils.widgets.text("colum_to_splitBy_to_predict", "")
colum_to_splitBy_to_predict = dbutils.widgets.get("colum_to_splitBy_to_predict")

if(~isinstance(colum_to_splitBy_to_predict, list)):  
    print("Not a list ... just 1 colum_to_splitBy_to_predict")
    colum_to_splitBy_to_predict = colum_to_splitBy_to_predict.split(",")
    
print(colum_to_splitBy_to_predict)

# COMMAND ----------
# Use all data 
dbutils.widgets.text("all_data", "")
all_data = dbutils.widgets.get("all_data")
print("all_data = ", all_data)

# COMMAND ----------

dbutils.widgets.text("ref_list_column_to_splitBy", "")
ref_list_column_to_splitBy = dbutils.widgets.get("ref_list_column_to_splitBy")


if(len(ref_list_column_to_splitBy) == 0):
    #reference colum_to_splitBy list (If ALL colum_to_splitBy) maybe the Kernel becomes unresponsive)
    ref_list_column_to_splitBy = ['value1', 'value2', 'value3']

    
elif(~isinstance(ref_list_column_to_splitBy, list)):  
    print("Not a list ... just 1 value")
    ref_list_column_to_splitBy = ref_list_column_to_splitBy.split(",")
    
print(ref_list_column_to_splitBy)

# COMMAND ----------

import numpy as np 
import pandas as pd 

import itertools
import os

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from pyspark.sql import functions as F

# COMMAND ----------

layers = keras.layers
models = keras.models

print("import successful")

#%%
# This code was tested with TensorFlow v1.8
print("You have TensorFlow version", tf.__version__)


import re

_surrogates = re.compile(r"[\uDC80-\uDCFF]")


# COMMAND ----------

# MAGIC %run your_import_functions

# COMMAND ----------

#%%
# category = 'label' --> Value to be predicted
# text --> what is is used to predict the label

ks_source = spark.read.format("csv").option("header","true")\
    .option("inferSchema", "true")\
    .option("delimiter", ";")\
    .load('paht_in' + train_data_colum_to_splitBy) # From notebook 1 "1_Create_Train_toBePred_Sets_name"

# Detect the environment you are working on
env = env_detection()
print(env)

# Select ONLY some value in your colum_to_splitBy (If ALL Kernel becomes unresponsive)
if((env == 'UAT') or (all_data == 0)):
    ref_list_column_to_splitBy.append(colum_to_splitBy_to_predict[0])
    print("ref_list_column_to_splitBy:")
    print(ref_list_column_to_splitBy)
    
    ks_sp = ks_source.where(F.col("field_colum_to_splitBy").isin(ref_list_column_to_splitBy)).selectExpr("text","category")
    
#PROD Env have enough power to handle all 
else:
     ks_sp = ks_source.selectExpr("text","category")

data = ks_sp.toPandas()


# COMMAND ----------

# Convert to string to 'float' object has no attribute 'lower'
data["text"] = data["text"].str.lower().fillna("")
data["category"] = data["category"].str.lower().fillna("OTHER")

# COMMAND ----------

#%%
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

# COMMAND ----------

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

#%%

train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)

#%%
# Data preparation
    
#First we'll split the data into training and test sets.
#Then we'll tokenize the words (text), and then convert them to a numbered index.
#Next we'll do the same for the labels (categories), by using the LabelEncoder utility.
#Finally, we'll convert the labels to a one-hot representation.
#from sklearn.cross_validation import train_test_split
#train_cat, test_cat = train_test_split(data['category'], train_size, stratify=data['category'])
#train_text, test_text = train_test_split(data['text'], train_size, stratify=data['category'])


#from sklearn import cross_validation
#train_cat, test_cat = cross_validation.train_test_split(data['category'], train_size, stratify=True)
#train_text, test_text = cross_validation.train_test_split(data['text'], train_size, stratify=True)

# COMMAND ----------

# DBTITLE 1,Handle never seen before values
# https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

# COMMAND ----------

#%%
max_words = 1000

tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)


tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data

x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
# encoder = LabelEncoder()
encoder = LabelEncoderExt() #Use NEW Clase for Label Encoder which handles UNKNOWN values (or not previously seen values)

encoder.fit(train_cat)

y_train = encoder.transform(train_cat)

# ---------------------------------------------------------- #
y_test = encoder.transform(test_cat) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Store tokenizer

# COMMAND ----------

import joblib
import sklearn
scikit_ver = sklearn.__version__
print(scikit_ver)

joblib.dump(tokenize, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_tokenize_{version}.pkl".format(version=scikit_ver)) 
joblib.dump(tokenize, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_tokenize.pkl")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Store enconder

# COMMAND ----------

joblib.dump(encoder, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_encoder_{version}.pkl".format(version=scikit_ver)) 
joblib.dump(encoder, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_encoder.pkl")

# COMMAND ----------

#%%
# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# COMMAND ----------

# DBTITLE 1,TRAIN THE MODEL
#%%

## TRAIN DE MODEL

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 3
drop_ratio = 0.5

# Build the model
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
# model.add(layers.Dropout(drop_ratio))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the model

# COMMAND ----------

# DBTITLE 1,Evaluate the model
#%%
# Evaluate the model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

#%%
# Hyperparameter tuning

def run_experiment(batch_size, epochs, drop_ratio):
    print('batch size: {}, epochs: {}, drop_ratio: {}'.format(
          batch_size, epochs, drop_ratio))
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_split=0.1)
    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=0)
    print('\tTest loss:', score[0])
    print('\tTest accuracy:', score[1])
    
batch_size = 16
epochs = 4
drop_ratio = 0.4
run_experiment(batch_size, epochs, drop_ratio)

# COMMAND ----------

#%%
# Hyperparameter Search
#You can also automate this process using for-loops and more sophiscated methods of deciding which combinations of hyperparameter values to try out.
#
#Exhaustive search is generally not the most elegant way, this is mostly just for illustrative purposes.

# Note: The data processed for this output had the max text length set to 400.
# for batch_size in range(10,31,10
# ):
#   for epochs in range(3,15,5):
#     for drop_ratio in np.linspace(0.1, 0.5, 3):
#       run_experiment(batch_size, epochs, drop_ratio)


# COMMAND ----------

# MAGIC %md
# MAGIC # Store the model

# COMMAND ----------

import joblib
import sklearn
scikit_ver = sklearn.__version__
print(scikit_ver)

joblib.dump(model, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_Model_{version}.pkl".format(version=scikit_ver)) 
joblib.dump(model, "path_out/PREDICTIONS/ML_MODEL/modelName_ML_Model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC # Make some predictions - manual checks

# COMMAND ----------

# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_ 

# for i in range(100):
for i in range(20):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
#     print(test_text.iloc[i][:50], "...")
    print('Actual label:' + test_cat.iloc[i])
    print("Predicted label: " + predicted_label + "\n")  


# COMMAND ----------



dbutils.notebook.exit("Sucess")

# COMMAND ----------

