# Databricks notebook source
# MAGIC %run your_import_functions

# COMMAND ----------

dbutils.widgets.text("colum_to_splitBy_to_predict", "")
colum_to_splitBy_to_predict = dbutils.widgets.get("colum_to_splitBy_to_predict")


if(~isinstance(colum_to_splitBy_to_predict, list)):  
    print("Not a list ... just 1 colum_to_splitBy_to_predict")
    colum_to_splitBy_to_predict = colum_to_splitBy_to_predict.split(",")
    
print(colum_to_splitBy_to_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load ML model

# COMMAND ----------

import joblib
import sklearn

model = joblib.load("/path/PREDICTIONS/ML_MODEL/name_ML_Model.pkl") # From notebook 2 _ "2_Train_Store_ML_Model_Name"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Label Encoder

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create new classe to load label encoder which handle missing values

# COMMAND ----------

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

encoder = joblib.load("/path/PREDICTIONS/ML_MODEL/name_ML_encoder.pkl")

# COMMAND ----------

text_labels = encoder.classes_ 

# COMMAND ----------

# MAGIC %md
# MAGIC ## The data to be predicted is the Missing or incorrect

# COMMAND ----------

data_to_pred = pd.read_csv("/path/PREDICTIONS/INPUT/data_toBePredicted.csv", delimiter = ';',  encoding = 'utf-8', dtype=str) # From notebook 1 - "1_Create_train_toBePred_Sets_name"
data_to_pred_orig = pd.read_csv("/path/PREDICTIONS/INPUT/data_toBePredicted.csv", delimiter = ';',  encoding = 'utf-8', dtype=str) 

print("colum_to_splitBy_to_predict = ", colum_to_splitBy_to_predict)

data_to_pred = data_to_pred[data_to_pred['colum_to_splitBy'].isin(colum_to_splitBy_to_predict)]

data_to_pred.drop('colum_to_splitBy', axis=1, inplace=True)  
data_to_pred.drop('PK_Column', axis=1, inplace=True)  

data_to_pred.text=data_to_pred.text.astype(str)


# COMMAND ----------

# print(data_to_pred.count())
# print(data_to_pred.head())

# COMMAND ----------

# DBTITLE 1,Remove duplicates
data_to_pred=data_to_pred.drop_duplicates()


# COMMAND ----------

print(data_to_pred.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load tokenizer

# COMMAND ----------

tokenize = joblib.load("/path/PREDICTIONS/ML_MODEL/name_ML_tokenize.pkl")

# COMMAND ----------

print(data_to_pred.shape[0]) # number of rows
test_text_n = data_to_pred['text']

x_test_n = tokenize.texts_to_matrix(test_text_n)

output_dataset = data_to_pred

# COMMAND ----------

print(output_dataset.shape[0]) # number of rows

# COMMAND ----------

# MAGIC %md
# MAGIC # PREDICT labels and create final dataset

# COMMAND ----------

#%%
for i in range(data_to_pred.shape[0]):
    prediction = model.predict(np.array([x_test_n[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
#     print(test_text_n.iloc[i][:50], "...")
#     print("Predicted label: " + predicted_label + "\n")  
    if(i % 100 == 0):
        print(i)
    output_dataset.loc[output_dataset['text'] ==  test_text_n.iloc[i], 'category'] = predicted_label

# COMMAND ----------

output_dataset["colum_to_splitBy"] = str(colum_to_splitBy_to_predict[0])

# COMMAND ----------

# output_dataset

#convert each Series to a DataFrame
text_df = output_dataset['text'].to_frame(name='text')
category_df = output_dataset['category'].to_frame(name='category')
colum_to_splitBy_to_predict_df = output_dataset['colum_to_splitBy'].to_frame(name='colum_to_splitBy')

#concatenate three Series into one DataFrame
output_df = pd.concat([text_df, category_df, colum_to_splitBy_to_predict_df], axis=1)

# COMMAND ----------

pred_out = output_df.merge(data_to_pred_orig, on='text', how='left', indicator=True)

# COMMAND ----------

display(pred_out)

# COMMAND ----------

pred_out.count()

# COMMAND ----------

pred_out.rename(columns = {'category_x':'ML_colname_Prediction', 'colum_to_splitBy_x':'colum_to_splitBy'}, inplace = True)
pred_out = pred_out[['text','ML_colname_Prediction','colum_to_splitBy','PK_Column']]
pred_out['ML_colname_Prediction'] = pred_out['ML_colname_Prediction'].str.upper()

# COMMAND ----------

# DBTITLE 1,Remove duplicates
pred_out = pred_out.drop_duplicates()

# COMMAND ----------

pred_out.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store final dataset resutls

# COMMAND ----------

output_path = 'outputpath'
output_filename = 'name_Data_Predicted_' + str(colum_to_splitBy_to_predict[0]) + '.csv'

pred_out.to_csv(output_path + output_filename, index = False, header=True)

# COMMAND ----------

# DBTITLE 1,Display some of the predictions
display(pred_out)

# COMMAND ----------

dbutils.notebook.exit("Sucess")

# COMMAND ----------


