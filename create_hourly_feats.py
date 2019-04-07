import os
import csv
import pickle
import pyspark
import pandas as pd

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row, Window, functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType
from pyspark.sql.functions import udf, row_number, col, monotonically_increasing_id, pandas_udf, PandasUDFType, explode, collect_list, create_map
from local_configuration import *
import csv
import math
import numpy as np
import pandas as pd

#----------VITALS------------||-------------------LAB RESULT VALUES------------------------
#HR, SBP, DBP, TEMP, RR, SP02,   Albumin, BUN, Ca, Cre, Na, K,HCO3, Glc, PH, PaC02, Platelets
#    cols = [['heartrate', 'sysbp', 'diasbp', 'tempc', 'resprate', 'spo2', 'glucose'],
#                   ['albumin', 'bun','creatinine', 'sodium', 'bicarbonate', 'platelet', 'inr'],
#                   ['potassium', 'calcium', 'ph', 'pco2', 'lactate']]


def flatten(l, ltypes=(list, tuple)):  #from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)



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
    item_mappings['8502'] = 'DBP'  #BP Cuff [Diastolic]
    item_mappings['3312'] = 'MBP'  #BP Cuff [Mean]
    item_mappings['0'] = 'TEMP'  #            TODO replace 0s with the correct values
    item_mappings['7884'] = 'RR'  #
    item_mappings['646'] = 'SP02'  #	SpO2
    item_mappings['220277'] = 'SP02'  # saturation pulseoxymetry SpO2
    item_mappings['3066'] = 'albumin'  #albumin
    item_mappings['227000'] = 'BUN_ApacheIV'  #BUN_ApacheIV
    item_mappings['227001'] = 'BunScore_ApacheIV'  #BunScore_ApacheIV
    item_mappings['1162'] = 'BUN'  # BUN
    item_mappings['225624'] = 'BUN'  #BUN
    item_mappings['1530'] = 'INR'
    item_mappings['44441'] = 'Calcium'  #
    item_mappings['3784'] = 'PCO2'
    item_mappings['812'] = 'HCO3'
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






def aggregate_temporal_features_hourly(filtered_chartevents_path):
    num_hours = 48
    df_filtered_chartevents = spark.read.csv(filtered_chartevents_path, header=True, inferSchema="false")
    df_filtered_chartevents = df_filtered_chartevents.withColumn("VALUENUM", df_filtered_chartevents["VALUENUM"].cast(IntegerType()))
    hourly_averages = df_filtered_chartevents.groupBy("HADM_ID", "ITEMNAME").pivot('HOUR_OF_OBS_AFTER_HADM', range(0,48)).avg("VALUENUM")

    #hourly_averages.show(n=15)

    def consolidateColNumbers(row):
        new_row_dict = {}
        new_row_dict['HADM_ID'] = row.HADM_ID
        new_row_dict['ITEMNAME'] = row.ITEMNAME
        row_dict = row.asDict()
        del row_dict['HADM_ID']
        del row_dict['ITEMNAME']
        consolidated_series = pd.Series(list(row_dict.values()), index=row_dict.keys())
        consolidated_series.reindex(range(num_hours))
        consolidated_series = pd.Series.fillna(pd.Series.fillna(consolidated_series, method='ffill'),  method='bfill')  # foward fill then backward fill
        new_row_dict['hourly_averages'] = consolidated_series.tolist()
        return Row(**new_row_dict)

    rdd_hadm_individual_metrics = hourly_averages.rdd.map(consolidateColNumbers)
    #print(rdd_hadm_individual_metrics.take(15))

    df_hadm_hourly_averages_filled  = rdd_hadm_individual_metrics.flatMap(lambda x: [Row(**{'HADM_ID': x.HADM_ID, 'ITEMNAME': x.ITEMNAME, 'HOUR':y, 'VALUE':x.hourly_averages[y]}) for y in range(len(x.hourly_averages))]).toDF()
    #df_hadm_hourly_averages_filled.show(100)

    #for each itemname (temporal feature), get filtered dataframe with only that feature.   Outer join their values under a new column with that ITEMNAME onto a new aggregate dataframe
    df_distinct_hadmids = df_hadm_hourly_averages_filled.select('HADM_ID').distinct()
    hours_df = spark.createDataFrame(range(num_hours), IntegerType()).withColumnRenamed('value', 'HOUR')
    cartesian_hadm_hours = df_distinct_hadmids.crossJoin(hours_df)
    #cartesian_hadm_hours.show(150)

    itemnames = list(set(get_event_key_ids().values()))

    for itemname in itemnames:
        df_single_feature = df_hadm_hourly_averages_filled.filter(col('ITEMNAME') == itemname)
        df_single_feature = df_single_feature.drop(df_single_feature.ITEMNAME).withColumnRenamed('VALUE', itemname)
        cartesian_hadm_hours = cartesian_hadm_hours.join(df_single_feature, ['HADM_ID', 'HOUR'], how='left')

    cartesian_hadm_hours.show(150)
    df_hadm_hourly_feature_arrays = cartesian_hadm_hours.select('HADM_ID', 'HOUR', F.struct(itemnames).alias('all_temporal_feats'))
    df_hadm_hourly_feature_arrays.show(150)


    df_hadm_all_hour_feats = df_hadm_hourly_feature_arrays.groupBy("HADM_ID").agg(collect_list(create_map(col("HOUR"),col('all_temporal_feats'))).alias('all_hours_all_temporal_feats'))
    df_hadm_all_hour_feats.show(150)

    def transformMapToArrayFn(row):
        new_row_dict = {}
        new_row_dict['HADM_ID'] = row.HADM_ID
        list_of_single_entry_dicts_for_each_hr = flatten(row.all_hours_all_temporal_feats)
        dict_hour_to_feature_row = {k: v for d in list_of_single_entry_dicts_for_each_hr for k, v in d.items()}
        sequences = [None] * num_hours
        for h in range(num_hours):
            if h in dict_hour_to_feature_row:
                features_array_order_of_itemnames = []
                features_dict_for_hour = dict_hour_to_feature_row[h]
                for itemname in itemnames:
                    features_array_order_of_itemnames.append(features_dict_for_hour[itemname])
                    sequences[h] = features_array_order_of_itemnames
        return Row(**{'HADM_ID':row.HADM_ID, 'all_hours_all_temporal_feats' : sequences})

    rdd_hadm_individual_metrics_hadm_to_sequences = df_hadm_all_hour_feats.rdd.map(transformMapToArrayFn)
    print(rdd_hadm_individual_metrics_hadm_to_sequences.take(10))




if __name__ == '__main__':

    conf = SparkConf().setMaster("local[4]").setAppName("My App")
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)
    filtered_chart_events_path = os.path.join(PATH_OUTPUT, 'FILTERED_CHARTEVENTS.csv')

    admissions_csv_path = os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv')
    #filter_chart_events(sc, os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'CHARTEVENTS.csv'), admissions_csv_path, filtered_chart_events_path)


    aggregate_temporal_features_hourly(filtered_chart_events_path)


    #low priority- remove patient admissions that don't have enough data points during 1st 48 hours of admission  - determine "enough" may need to look at other code

    #standardize each feature as in the paper  -- in their preprocess.py this is what they did:  (values - min_feat_value) / (95thpercentile - min_value)   from their preprocessing.py:  dfs[idx][c] = (dfs[idx][c]-dfs[idx][c].min() )/ (dfs[idx][c].quantile(.95) - dfs[idx][c].min())

    #for each admission, for each hourly bin, construct feature vector

    #write feature file with list of tuple (patientid.hadmid, list patient-admission sequences 48 long each)
    #get mortality labels for admissions - if the patient died during the admission.   These are located int ADMISSIONS.csv table.  Must be in same ORDER (and length) as feature file.
