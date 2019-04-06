import os
import csv
import pickle
import pyspark
import pandas as pd

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row, Window, functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import udf, row_number, col, monotonically_increasing_id
from local_configuration import *
import csv
import math
import numpy as np

#----------VITALS------------||-------------------LAB RESULT VALUES------------------------
#HR, SBP, DBP, TEMP, RR, SP02,   Albumin, BUN, Ca, Cre, Na, K,HCO3, Glc, PH, PaC02, Platelets
#    cols = [['heartrate', 'sysbp', 'diasbp', 'tempc', 'resprate', 'spo2', 'glucose'],
#                   ['albumin', 'bun','creatinine', 'sodium', 'bicarbonate', 'platelet', 'inr'],
#                   ['potassium', 'calcium', 'ph', 'pco2', 'lactate']]


def translate(mapping):
    def translate_(col):
        return mapping.get(col)
    return udf(translate_, StringType())



def get_event_key_ids():
    #TODO (for final) aggregate all synonomous variations of each measurement --- SEE INPORTANT CONSIDERATIONS: https://mimic.physionet.org/mimictables/d_items/

    #TODO Finish this section.  There should be two item numbers that map to the same item as described in the link above.   Let's get it mostly right for the draft.

    item_mappings = {}
    item_mappings['211'] = 'HEART_RATE'   #HEART RATE
    item_mappings['220045'] = 'HEART_RATE'  # HEART RATE
    item_mappings['3313'] = 'SBP'  #BP Cuff [Systolic]
    item_mappings['0'] = 'DBP'  #
    item_mappings['0'] = 'TEMP'  #            TODO replace 0s with the correct values
    item_mappings['0'] = 'RR'  #
    item_mappings['0'] = 'SP02'  #
    item_mappings['3066'] = 'albumin'  #albumin
    item_mappings['227000'] = 'BUN_ApacheIV'  #BUN_ApacheIV
    item_mappings['227001'] = 'BunScore_ApacheIV'  #BunScore_ApacheIV
    item_mappings['1162'] = 'BUN'  # BUN
    item_mappings['225624'] = 'BUN'  #BUN
    item_mappings['44441'] = 'Calcium'  #
    item_mappings['227005'] = 'Creatinine_ApacheIV'
    item_mappings['227006'] = 'CreatScore_ApacheIV'
    item_mappings['4231'] = 'NaCl'
    item_mappings['1535'] = 'Potassium'
    item_mappings['227006'] = 'CreatScore_ApacheIV'

    return item_mappings




def filter_chart_events(spark, orig_chrtevents_file_path, admissions_csv_file_path, filtered_chrtevents_outfile_path):
    #TAKES ONLY THE RELEVANT ITEM ROWS FROM THE CHARTEVENTS.CSV file
    item_mappings = get_event_key_ids()



    #use subset of large CHARTEVENTS.csv file for faster development
    chrtevents_file_path_to_use = orig_chrtevents_file_path
    use_sample_subset_lines = True
    if use_sample_subset_lines:

        chartevents_sample_temp_file = "CHARTEVENTS_SAMPLE.csv"
        chrtevents_file_path_to_use = chartevents_sample_temp_file

        temp_file = open(chartevents_sample_temp_file, "w+")
        with open(orig_chrtevents_file_path) as orig_file:
            i = 0
            for line in orig_file:
                temp_file.write(line)
                i = i + 1
                if i > 20000:
                    break
        temp_file.close()



    df_chartevents = spark.read.csv(chrtevents_file_path_to_use, header=True, inferSchema="false")

    filtered_chartevents = df_chartevents.filter(col('ITEMID').isin(list(item_mappings.keys())))
    filtered_chartevents = filtered_chartevents.withColumn("ITEMNAME", translate(item_mappings)("ITEMID"))


    #join filtered_chartevents with ADMISSIONS.csv on HADMID --- only keep HADMID AND ADMITTIME COLUMNS FROM ADMISSIONS
    df_admissions = spark.read.csv(admissions_csv_file_path, header=True, inferSchema="false")



    #add column that contains the hour the observation occurred after admission  (0 - X)
    filtered_chartevents = filtered_chartevents.join(df_admissions, ['HADM_ID'])
    timeFmt = "yyyy-MM-dd' 'HH:mm:ss"   #2153-09-03 07:15:00
    timeDiff = F.round((F.unix_timestamp('CHARTTIME', format=timeFmt)
                - F.unix_timestamp('ADMITTIME', format=timeFmt)) / 60 / 60).cast('integer')  #calc diff, convert seconds to minutes, minutes to hours, then math.floor to remove decimal places (for hourly bin/aggregations)
    filtered_chartevents = filtered_chartevents.withColumn("HOUR_OF_OBS_AFTER_HADM", timeDiff)  #  F.round(   ).cast('integer')

    #filter out all observations where X > 48  (occurred after initial 48 hours of admission)
    filtered_chartevents = filtered_chartevents.filter(col('HOUR_OF_OBS_AFTER_HADM') <= 48)


    #TODO: REMOVE columns that are not needed (keep CHARTEVENTS cols, ITEMNAME, HOUR_OF_OBS_AFTER_HADM

    with open(filtered_chrtevents_outfile_path, "w+") as f:
        w = csv.DictWriter(f, fieldnames=filtered_chartevents.schema.names)
        w.writeheader()

        for rdd_row in filtered_chartevents.rdd.toLocalIterator():
            w.writerow(rdd_row.asDict())


def consolidateColNumbers(row):
    num_hours = 48
    new_row_dict = {}
    new_row_dict['HADM_ID'] = row.HADM_ID
    new_row_dict['ITEMNAME'] = row.ITEMNAME
    consolidated_arr = np.full(num_hours, np.nan)
    row_dict = row.asDict()
    for i in range(num_hours):
        if i in row_dict:
            consolidated_arr[i] = row_dict[i]
    new_row_dict['hourly_averages'] = consolidated_arr
    return Row(**new_row_dict)


def aggregate_temporal_features_hourly(filtered_chartevents_path):
    df_filtered_chartevents = spark.read.csv(filtered_chartevents_path, header=True, inferSchema="false")
    df_filtered_chartevents = df_filtered_chartevents.withColumn("VALUENUM", df_filtered_chartevents["VALUENUM"].cast(IntegerType()))
    hourly_averages = df_filtered_chartevents.groupBy("HADM_ID", "ITEMNAME").pivot('HOUR_OF_OBS_AFTER_HADM', range(0,48)).avg("VALUENUM")

    hourly_averages.show(n=15)

    new_rdd = hourly_averages.rdd.map(consolidateColNumbers).take(15)
    print(new_rdd)



if __name__ == '__main__':
    conf = SparkConf().setMaster("local[4]").setAppName("My App")
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)
    filtered_chart_events_path = os.path.join(PATH_OUTPUT, 'FILTERED_CHARTEVENTS.csv')

    admissions_csv_path = os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv')
    #filter_chart_events(spark, os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'CHARTEVENTS.csv'), admissions_csv_path, filtered_chart_events_path)


    aggregate_temporal_features_hourly(filtered_chart_events_path)


    #low priority- remove patient admissions that don't have enough data points during 1st 48 hours of admission  - determine "enough" may need to look at other code

    #standardize each feature as in the paper  -- in their preprocess.py this is what they did:  (values - min_feat_value) / (95thpercentile - min_value)   from their preprocessing.py:  dfs[idx][c] = (dfs[idx][c]-dfs[idx][c].min() )/ (dfs[idx][c].quantile(.95) - dfs[idx][c].min())

    #for each admission, for each hourly bin, construct feature vector

    #write feature file with list of tuple (patientid.hadmid, list patient-admission sequences 48 long each)
    #get mortality labels for admissions - if the patient died during the admission.   These are located int ADMISSIONS.csv table.  Must be in same ORDER (and length) as feature file.
