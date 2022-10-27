# Databricks notebook source
# MAGIC %run your_import_functions

# COMMAND ----------

dbutils.widgets.text("colum_to_splitBy_to_predict", "")
colum_to_splitBy_to_predict = dbutils.widgets.get("colum_to_splitBy_to_predict")
print("colum_to_splitBy_to_predict = ", colum_to_splitBy_to_predict)

# COMMAND ----------

env = env_detection()
print(env)

# COMMAND ----------

# DBTITLE 1, data

    
folderpath = "your_path"
    
filename = "your_filename.csv.gz"

df = spark.read.format("csv").option("header","true")\
    .option("delimiter", ";")\
    .option("inferSchema", "true")\
    .load(folderpath + filename)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read all PREDICTION files in directoy (one file per colum_to_splitB prediction) and load into final DF

# COMMAND ----------

filesToRead = []
for item in dbutils.fs.ls("directory_with_All_colum_to_splitBy_predictions"):
#     print(item.name)
    if((len(item.name) > 15) and item.name[0:8] == 'some_string_to_identifgyFiles'):
        filesToRead.append(item.path)

# COMMAND ----------

df_pred = spark.read.format("com.databricks.spark.csv") \
  .option("header", "true") \
  .option("dateFormat", "dd/MM/yyyy HH:mm:ss") \
  .option("inferSchema","false") \
  .option("multiLine", "true") \
  .option("escape", '"')\
  .option("mergeSchema", "true")\
  .option("delimiter", ",")\
  .csv(filesToRead)
  

# COMMAND ----------

display(df_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join with original data source to get ALL source fields

# COMMAND ----------

# display(ks_sp)
# display(df_pred)

# COMMAND ----------

df2 = df.join(df_pred, on=['PK_Column'], how='left').select(df["*"],df_pred["ML_col_name"])

# COMMAND ----------
#Identify discrepancies
df2 = df2.withColumn("ML_Disc", F.when( F.col("colSource") != F.col("ML_col_name") , 1).otherwise(0))

# COMMAND ----------
folderpath = "final_output_path"
filename = "final_output_filename"

store_as_gzip(folderpath, filename, df_final_pred, ";")

# COMMAND ----------

dbutils.notebook.exit("Sucess")
