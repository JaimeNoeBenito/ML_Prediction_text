# Databricks notebook source
# MAGIC %run your_import_functions

# COMMAND ----------

folderpath = <your_folderpath>
    
filename = "your_filename"

df = spark.read.format("csv").option("header","true")\
    .option("delimiter", ";")\
    .option("inferSchema", "true")\
    .load(folderpath + filename)\
    . withColumn("PK_Column", F.concat(F.concat(F.col("field1"), F.lit('_')), F.col("field2") )) # Concatenation of fields to create unique PK

# COMMAND ----------

# MAGIC %md
# MAGIC ## Only use for training data you are sure are OK

# COMMAND ----------


train_data_colum_to_splitBy = df.selectExpr("field_concatenation (field1||'_'||field2 ... ) as text", "category", "colum_to_splitBy") 

# Define function to store as single CSV file (usually in your import functions
store_as_csv("output_train_data_path",  "output_train_data_name_colum_to_splitBy", train_data_colum_to_splitBy, ';') 

data_train = train_data_colum_to_splitBy.drop("colum_to_splitBy")

store_as_csv("output_train_data_path",  "output_train_data_name", data_train, ';') 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incorrect values (or missing) are the ones that need to be predicted

# COMMAND ----------

data_toBePredicted = df.where((F.col("field_incorrect_indicator") == 1) | (F.col("colname").isNull() )).selectExpr("field_concatenation (field1||'_'||field2 ... ) as text", "fiel_used_asCatergor as category", "colum_to_splitBy", "PK_Field") 

store_as_csv("output_path", "data_toBePredicted", data_toBePredicted, ';')

# COMMAND ----------

dbutils.notebook.exit("Sucess")

# COMMAND ----------


