import os
import csv
import pickle
import pyspark
import pandas as pd

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row, Window, functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id
from local_configuration import *




def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	icd9_str = str(icd9_object)
	if icd9_str[0] == 'E' :
		converted = icd9_str[:4]
	else:
		converted = icd9_str[:3]

	return converted


def build_codemap(df_icd9, df_drg_code, df_lab_item, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	convert_icd9_udf = udf(lambda icd9: convert_icd9(icd9))
	df_digits = df_icd9.withColumn("ICD9_CODE", convert_icd9_udf(df_icd9.ICD9_CODE))

	#Build CODE MAP for Diagnoses Codes
	df = df_digits.drop_duplicates().withColumn("index",row_number().over(Window.orderBy(monotonically_increasing_id()))-1)
	codemap_icd = dict(df.rdd.map(lambda x : (x[0], x[1])).collect())

	#Build CODE MAP for Drugs Codes
	max_idx= df.select(F.max("index")).collect()[0].asDict()["max(index)"]
	df = df_drg_code.withColumn("index",row_number().over(Window.orderBy(monotonically_increasing_id()))+max_idx)
	codemap_drg = dict(df.rdd.map(lambda x : (x[0], x[1])).collect())

	#Build CODE MAP for LAB Labels
	max_idx= df.select(F.max("index")).collect()[0].asDict()["max(index)"]
	df = df_lab_item.withColumn("index",row_number().over(Window.orderBy(monotonically_increasing_id()))+max_idx)
	codemap_itm = dict(df.rdd.map(lambda x : (x[0], x[1])).collect())

	return codemap_icd, codemap_drg, codemap_itm


def create_dataset(path, codemap_icd, codemap_drg, codemap_itm, transform, spark):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.

	df_mortality = spark.read.csv(path+"MORTALITY.csv",header=True, inferSchema="true")
	df_admissions = spark.read.csv(path+"ADMISSIONS.csv",header=True, inferSchema="true")
	df_diagnoses = spark.read.csv(path+"DIAGNOSES_ICD.csv",header=True, inferSchema="true").select('SUBJECT_ID', 'HADM_ID', 'ICD9_CODE')
	df_medications = spark.read.csv(path+"MEDICATIONS.csv",header=True, inferSchema="true").select('SUBJECT_ID', 'HADM_ID', 'DRG_CODE')
	df_lab = spark.read.csv(path+"LAB.csv",header=True, inferSchema="true").select('SUBJECT_ID', 'HADM_ID', 'ITEMID')

	#Convert ICD9 CODES
	convert_icd9_udf = udf(lambda icd9:  convert_icd9(icd9))
	df_diagnoses = df_diagnoses.filter(df_diagnoses.ICD9_CODE.isNotNull()).withColumn("ICD9_CODE", convert_icd9_udf(df_diagnoses.ICD9_CODE))
	#MAP ICD9 CODE
	map_udf =  udf (lambda x: codemap_icd.get(x), IntegerType())
	df_diagnoses = df_diagnoses.withColumn("ICD9_CODE",map_udf(df_diagnoses.ICD9_CODE))
	df_diagnoses = df_diagnoses.filter(df_diagnoses.ICD9_CODE.isNotNull())

	#MAP DRG CODE
	map_udf =  udf (lambda x: codemap_drg.get(x), IntegerType())
	df_medications = df_medications.withColumn("DRG_CODE",map_udf(df_medications.DRG_CODE))
	df_medications = df_medications.filter(df_medications.DRG_CODE.isNotNull())

	#MAP LAB ITEMS
	map_udf =  udf (lambda x: codemap_itm.get(x), IntegerType())
	df_lab = df_lab.withColumn("ITEMID",map_udf(df_lab.ITEMID))
	df_lab = df_lab.filter(df_lab.ITEMID.isNotNull())


	# MERGE ADMISSIONS WITH DIAG, MED AND LAB Dataframes
	df_admissions = df_admissions.select('SUBJECT_ID', 'HADM_ID', 'ADMITTIME')
	df_merge_diag = df_admissions.join(df_diagnoses, on=['SUBJECT_ID','HADM_ID'],how='left').dropna()
	df_merge_med = df_admissions.join(df_medications, on=['SUBJECT_ID','HADM_ID'],how='left').dropna()
	df_merge_lab = df_admissions.join(df_lab, on=['SUBJECT_ID','HADM_ID'],how='left').dropna()
	df_merge_lab = df_merge_lab.withColumnRenamed("ITEMID", "CODE")

	#CONCAT ALL MERGED DF INTO ONE SINGLE DF FEATURE
	df_merge_diag = df_merge_diag.withColumnRenamed("ICD9_CODE", "CODE")
	df_merge_med = df_merge_med.withColumnRenamed("DRG_CODE", "CODE")
	df_merge_lab = df_merge_lab.withColumnRenamed("ITEMID", "CODE")
	df_union = df_merge_diag.union(df_merge_med)
	df_union = df_union.union (df_merge_lab)

	# TODO: 3. Group the diagnosis codes for the same visit.
	groupby_visits  = df_union.groupby (['SUBJECT_ID','ADMITTIME']).agg(F.collect_set("CODE").alias("FEATURES"))

	# TODO: 4. Group the visits for the same patient.
	df_sort = groupby_visits.orderBy("SUBJECT_ID", "ADMITTIME")
	groupby_patients= df_sort.groupby ('SUBJECT_ID').agg(F.collect_set("FEATURES").alias("FEATURES")).orderBy("SUBJECT_ID")

	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.
	seq_data = list(groupby_patients.rdd.map(lambda x : x[1]).collect())

	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	df_sort = df_mortality.orderBy('SUBJECT_ID')
	df = groupby_patients.join(df_sort, how ='left_outer', on='SUBJECT_ID').select('SUBJECT_ID', 'MORTALITY')
	patient_ids = list(df.rdd.map(lambda x : x[0]).collect())
	labels = list(df.rdd.map(lambda x : x[1]).collect())

	return patient_ids, labels, seq_data


def main():
	conf = SparkConf().setMaster("local[4]").setAppName("My App")
	sc = SparkContext(conf = conf)
	spark = SQLContext(sc)

	# Build a code map from the train set
	print("Build feature id maps")
	df_icd9 = spark.read.csv(PATH_TRAIN+"DIAGNOSES_ICD.csv",header=True, inferSchema="true").select("ICD9_CODE").dropna().drop_duplicates()
	df_drg_code = spark.read.csv(PATH_TRAIN+"MEDICATIONS.csv",header=True, inferSchema="true").select("drg_code").dropna().drop_duplicates()
	df_lab_item = spark.read.csv(PATH_TRAIN+"LAB.csv",header=True, inferSchema="true").select ("itemid").dropna().drop_duplicates()
	codemap_icd, codemap_drg, codemap_itm = build_codemap(df_icd9, df_drg_code, df_lab_item, convert_icd9)

	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap_icd, open(os.path.join(PATH_OUTPUT, "mortality.codemap_icd.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(codemap_drg, open(os.path.join(PATH_OUTPUT, "mortality.codemap_drg.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(codemap_itm, open(os.path.join(PATH_OUTPUT, "mortality.codemap_itm.train"), 'wb'), pickle.HIGHEST_PROTOCOL)


	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap_icd, codemap_drg, codemap_itm, convert_icd9, spark)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap_icd, codemap_drg, codemap_itm, convert_icd9, spark)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap_icd, codemap_drg, codemap_itm, convert_icd9, spark)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	sc.stop()
	print("Complete!")


if __name__ == '__main__':
	main()
