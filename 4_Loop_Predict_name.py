# Databricks notebook source
# MAGIC %run your_import_functions

# COMMAND ----------

dbutils.widgets.text("colum_to_splitBy_to_predict","")
colum_to_splitBy_to_predict = dbutils.widgets.get("colum_to_splitBy_to_predict")

list_param = list(colum_to_splitBy_to_predict.split(","))
print(list_param)

print(len(list_param))

# COMMAND ----------

if(len(list_param) <= 1):
    if(len(list_param[0]) == 0): #empty list
        colum_to_splitBy_to_predict_list = [
          'value1',
        'value2',
        'value3'
        ]
    else:
        colum_to_splitBy_to_predict_list = list_param
else:
        colum_to_splitBy_to_predict_list = list_param


# COMMAND ----------

for colum_to_splitBy in colum_to_splitBy_to_predict_list:
    print(colum_to_splitBy)
    dbutils.notebook.run("3_Load_ML_Model_Predict_name", 20000, arguments = {"colum_to_splitBy_to_predict": colum_to_splitBy})
    

# COMMAND ----------


