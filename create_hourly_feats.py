import os
import csv
import pickle
import pyspark
import pandas as pd

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame, Row, Window, functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType, StructType, StructField, ArrayType, FloatType
from pyspark.sql.functions import array, udf, row_number, col, monotonically_increasing_id, pandas_udf, PandasUDFType, explode, collect_list, create_map
from functools import reduce
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
    #Heart Rate
    item_mappings['211'] = 'HEART_RATE'   #HEART RATE
    item_mappings['220045'] = 'HEART_RATE'  # HEART RATE
    
    #Blood Pressure [Systolic]
    item_mappings['3313'] = 'SBP'  #BP Cuff [Systolic]
    item_mappings['3315'] = 'SBP'  #BP Left Arm [Systolic]
    item_mappings['3317'] = 'SBP'  #BP Left Leg [Systolic]
    item_mappings['3319'] = 'SBP'  #BP PAL [Systolic]  
    item_mappings['3321'] = 'SBP'  #BP Right Arm [Systolic] 
    item_mappings['3323'] = 'SBP'  #BP Right Leg [Systolic]
    item_mappings['3325'] = 'SBP'  #BP UAC [Systolic]
    item_mappings['6'] = 'SBP'  #ABP [Systolic]
    item_mappings['51'] = 'SBP'  #Arterial BP [Systolic]    
    item_mappings['6701'] = 'SBP'  #Arterial BP #2 [Systolic]
    item_mappings['225309'] = 'SBP'  #ART BP Systolic
    item_mappings['220050'] = 'SBP'  #Arterial Blood Pressure systolic
    item_mappings['442'] = 'SBP'  #Manual BP [Systolic]
    item_mappings['224167'] = 'SBP'  #Manual Blood Pressure Systolic Left
    item_mappings['227243'] = 'SBP'  #Manual Blood Pressure Systolic Right
    item_mappings['442'] = 'SBP'  #Manual BP [Systolic]
    item_mappings['455'] = 'SBP'  #NBP [Systolic]
    item_mappings['480'] = 'SBP'  #Orthostat BP sitting [Systolic] 
    item_mappings['482'] = 'SBP'  #OrthostatBP standing [Systolic]
    item_mappings['484'] = 'SBP'  #Orthostatic BP lying [Systolic]  
    item_mappings['220179'] = 'SBP'  #Non Invasive Blood Pressure systolic 
        
    #Blood Pressure [Diastolic]
    item_mappings['8502'] = 'DBP'  #BP Cuff [Diastolic]
    item_mappings['8503'] = 'DBP'  #BP Left Arm [Diastolic]
    item_mappings['8504'] = 'DBP'  #BP Left Leg [Diastolic]
    item_mappings['8505'] = 'DBP'  #BP PAL [Diastolic]
    item_mappings['8506'] = 'DBP'  #BP Right Arm [Diastolic]
    item_mappings['8507'] = 'DBP'  #BP Right Leg [Diastolic]
    item_mappings['8508'] = 'DBP'  #BP UAC [Diastolic]
    item_mappings['8440'] = 'DBP'  #Manual BP [Diastolic]
    item_mappings['227242'] = 'DBP'  #Manual Blood Pressure Diastolic Right
    item_mappings['224643'] = 'DBP'  #Manual Blood Pressure Diastolic Left
    item_mappings['8441'] = 'DBP'  #NBP [Diastolic]
    item_mappings['8364'] = 'DBP'  #ABP [Diastolic]
    item_mappings['8555'] = 'DBP'  #Arterial BP #2 [Diastolic]
    item_mappings['8368'] = 'DBP'  #Arterial BP [Diastolic]  
    item_mappings['225310'] = 'DBP'  #ART BP Diastolic      
    item_mappings['220051'] = 'DBP'  #Arterial Blood Pressure diastolic 
    item_mappings['220180'] = 'DBP'  #Non Invasive Blood Pressure diastolic  
    
    #Temperature
    item_mappings['676'] = 'TEMPC' #Temperature C
    item_mappings['677'] = 'TEMPC'  #Temperature C (calc)
    item_mappings['223762'] = 'TEMPC'  #Temperature Celsius    
    
    #Respiratory Rate
    item_mappings['7884'] = 'RR' #RR
    item_mappings['618'] = 'RR' #Respiratory Rate
    item_mappings['220210'] = 'RR' #Respiratory Rate
    item_mappings['619'] = 'RR' #Respiratory Rate Set
    item_mappings['224688'] = 'RR' #Respiratory Rate (Set)
    item_mappings['614'] = 'RR' #Resp Rate (Spont)
    item_mappings['224689'] = 'RR' #Respiratory Rate (spontaneous)
    item_mappings['615'] = 'RR' #Resp Rate (Total) 
    item_mappings['224690'] = 'RR' #Respiratory Rate (Total) 
    
    #Saturation Pulseoxymetry
    item_mappings['646'] = 'SP02'  #SpO2
    item_mappings['220277'] = 'SP02'  # saturation pulseoxymetry SpO2
    item_mappings['223769'] = 'SP02'  # O2 Saturation Pulseoxymetry Alarm - High
    item_mappings['223770'] = 'SP02'  # O2 Saturation Pulseoxymetry Alarm - Low 
    
    #Albumin
    item_mappings['3066'] = 'Albumin'  #albumin
    item_mappings['1521'] = 'Albumin'  #Albumin
    item_mappings['46564'] = 'Albumin'  #Albumin
    item_mappings['227456'] = 'Albumin'  #Albumin 
    item_mappings['45403'] = 'Albumin'  #albumin
    item_mappings['42832'] = 'Albumin'  #albumin 12.5%
    item_mappings['44203'] = 'Albumin'  #Albumin 12.5%
    item_mappings['772'] = 'Albumin'  #Albumin (>3.2) 
    item_mappings['3727'] = 'Albumin'  #Albumin (3.9-4.8)    
    item_mappings['30181'] = 'Albumin'  #Serum Albumin 5%
    item_mappings['44952'] = 'Albumin'  #OR Albumin 
    item_mappings['220574'] = 'Albumin'  #ZAlbumin 
    item_mappings['226981'] = 'Albumin'  #Albumin_ApacheIV
    item_mappings['226982'] = 'Albumin'  #AlbuminScore_ApacheIV     
    item_mappings['30008'] = 'Albumin'  #Albumin 5%
    item_mappings['220864'] = 'Albumin'  #Albumin 5%
    item_mappings['30009'] = 'Albumin'  #Albumin 25%  
    item_mappings['220862'] = 'Albumin'  #Albumin 25% 
    item_mappings['220861'] = 'Albumin'  #Albumin (Human) 20%
    item_mappings['220863'] = 'Albumin'  #Albumin (Human) 4%  
    
    #Bun
    item_mappings['227000'] = 'BUN'  #BUN_ApacheIV
    item_mappings['227001'] = 'BUN'  #BunScore_ApacheIV
    item_mappings['1162'] = 'BUN'  # BUN
    item_mappings['5876'] = 'BUN'  # bun
    item_mappings['225624'] = 'BUN'  #BUN
    item_mappings['781'] = 'BUN'  #BUN (6-20) 
    item_mappings['3737'] = 'BUN'  #BUN (6-20)  
    item_mappings['8220'] = 'BUN'  #Effluent BUN 
 
    #International Normalized Ratio (INR)
    item_mappings['1530'] = 'INR'
    item_mappings['227467'] = 'INR'
    
    #Platelets
    item_mappings['828'] = 'Platelets' #Platelets
    item_mappings['225170'] = 'Platelets'  #Platelets
    item_mappings['30006'] = 'Platelets'  #Platelets
    item_mappings['30105'] = 'Platelets'  #OR Platelets 
    
    #Calcium
    item_mappings['44441'] = 'Ca'  #calcium
    item_mappings['44855'] = 'Ca'  #Calcium
    item_mappings['1522'] = 'Ca'  #Calcium 
    item_mappings['227525'] = 'Ca'  #Calcium
    
    #Creatinine
    item_mappings['1525'] = 'Creatinine' #Creatinine
    item_mappings['220615'] = 'Creatinine' #Creatinine 
    item_mappings['791'] = 'Creatinine' #Creatinine (0-1.3)
    item_mappings['3750'] = 'Creatinine' #Creatinine (0-0.7)
    item_mappings['5811'] = 'Creatinine' #urine creatinine 
    item_mappings['1919'] = 'Creatinine' #Urine creat 
    item_mappings['227005'] = 'Creatinine' #Creatinine_ApacheIV
    item_mappings['227006'] = 'Creatinine' #CreatScore_ApacheIV
    item_mappings['226752'] = 'Creatinine' #CreatinineApacheIIValue
    item_mappings['226751'] = 'Creatinine' #CreatinineApacheIIScore

    
    #carbon dioxide 
    item_mappings['3784'] = 'PCO2'
    item_mappings['778'] = 'PCO2' #Arterial PaCO2 
    item_mappings['220235'] = 'PCO2' #Arterial CO2 Pressure
    item_mappings['3774'] = 'PCO2' #Mix Venous PCO2
    item_mappings['226062'] = 'PCO2' #Venous CO2 Pressure 
    item_mappings['227036'] = 'PCO2' #PCO2_ApacheIV  
    
    #Bicarbonate
    item_mappings['812'] = 'HCO3' #HCO3
    item_mappings['46362'] = 'HCO3' #bicarbonate
    item_mappings['44166'] = 'HCO3' #BICARBONATE-HCO3
    item_mappings['227443'] = 'HCO3' #HCO3 (serum)
    item_mappings['226759'] = 'HCO3' #HCO3ApacheIIValue
    item_mappings['226760'] = 'HCO3' #HCO3Score
    item_mappings['225165'] = 'HCO3' #Bicarbonate Base
    
    #Glucose
    item_mappings['1529'] = 'Glucose' #Glucose
    item_mappings['807'] = 'Glucose'
    item_mappings['1455'] = 'Glucose' #fingerstick glucose
    item_mappings['2338'] = 'Glucose' #finger stick glucose
    item_mappings['225664'] = 'Glucose' #Glucose finger stick
    item_mappings['1812'] = 'Glucose' #abg: glucose
    item_mappings['811'] = 'Glucose'
    item_mappings['227015'] =  'Glucose'
    item_mappings['227016'] =  'Glucose'
    item_mappings['3816'] = 'Glucose'  
    item_mappings['228388'] = 'Glucose'
    item_mappings['226537'] = 'Glucose'
    item_mappings['227976'] = 'Glucose' #Boost Glucose Control (1/4)
    item_mappings['227977'] = 'Glucose' #Boost Glucose Control (1/2)
    item_mappings['227978'] = 'Glucose' #Boost Glucose Control (3/4)
    item_mappings['227979'] = 'Glucose' #Boost Glucose Control (Full)
    item_mappings['3744'] = 'Glucose'
    item_mappings['3745'] = 'Glucose'
    item_mappings['3447'] = 'Glucose'
    item_mappings['220621'] = 'Glucose' #Glucose (serum)

    #Sodium
    item_mappings['1536'] = 'Na' 
    
    #Postassium
    item_mappings['1535'] = 'K' #Potassium
    item_mappings['41956'] = 'K' #Potassium
    item_mappings['44711'] = 'K' #potassium 
    item_mappings['226771'] = 'K' #PotassiumApacheIIScore
    item_mappings['226772'] = 'K' #PotassiumApacheIIValue
    
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
                if i > 5000:
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


    #REMOVE columns that are not needed (keep CHARTEVENTS cols, ITEMNAME, HOUR_OF_OBS_AFTER_HADM
    filtered_chartevents = reduce(DataFrame.drop, ["ADMITTIME", "DISCHTIME", "DEATHTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY", "EDREGTIME", "EDOUTTIME", "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG", "HAS_CHARTEVENTS_DATA"], filtered_chartevents)

    with open(filtered_chrtevents_outfile_path, "w+") as f:
        w = csv.DictWriter(f, fieldnames=filtered_chartevents.schema.names)
        w.writeheader()

        for rdd_row in filtered_chartevents.rdd.toLocalIterator():
            w.writerow(rdd_row.asDict())

            
             
#Returns a list of mortality labels and  a list of tuples of (SUBJECT_ID, HADM_ID, sequences)
def create_dataset(spark, admissions_csv_path, hadm_sequences):
    hadm_sequences= hadm_sequences.withColumnRenamed('HADMID', 'HADM_ID')
    mortality = spark.read.csv(admissions_csv_path , header=True, inferSchema="false").select('SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG')
    
    df = hadm_sequences.join(mortality, on='HADM_ID', how='left')
    labels =  df.select ('HOSPITAL_EXPIRE_FLAG').rdd.flatMap(lambda x: x).collect()
    seqs = df.rdd.map(lambda x: (x[2], x[0], x[1])).collect()
    
    return labels, seqs



#Standardize features in a new column Standardized_Value  (we can change the value in place VALUENUM)
def standardize_features (df_filtered_chartevents): 
    temp = df_filtered_chartevents.select('ITEMNAME','VALUENUM')
    min_quantile = temp.groupBy('ITEMNAME').agg(F.min('VALUENUM').alias("Min"), F.expr('percentile_approx(VALUENUM, 0.95)').alias("Quantile95"))
    cartesian_min_quantile = min_quantile.join(df_filtered_chartevents, on='ITEMNAME', how='left')
    #TODO: Verify the return value when the min and quantile are equal
    udf_standardize = F.udf( lambda x: (x[0]-x[1]) / (x[2]-x[1] ) if x[2]!=x[1] else  float(x[0]) , DoubleType()) 
    #standardized_df = cartesian_min_quantile.withColumn("Standardized_Value", udf_standardize(array("VALUENUM", "Min", "Quantile95")))
    standardized_df = cartesian_min_quantile.withColumn("VALUENUM", udf_standardize(array("VALUENUM", "Min", "Quantile95")))
    
    return standardized_df.drop("Min","Quantile95")


def aggregate_temporal_features_hourly(filtered_chartevents_path):
    num_hours = 48
    df_filtered_chartevents = spark.read.csv(filtered_chartevents_path, header=True, inferSchema="false")

    df_filtered_chartevents = df_filtered_chartevents.withColumn("VALUENUM_INT", df_filtered_chartevents["VALUENUM"].cast(IntegerType()))
    df_filtered_chartevents = df_filtered_chartevents.drop(df_filtered_chartevents.VALUENUM).withColumnRenamed('VALUENUM_INT', 'VALUENUM')
    df_filtered_chartevents = df_filtered_chartevents.na.drop(subset=["VALUENUM"])
    df_standardized_chartevents = standardize_features (df_filtered_chartevents)
    hourly_averages = df_standardized_chartevents.groupBy("HADM_ID", "ITEMNAME").pivot('HOUR_OF_OBS_AFTER_HADM', range(0,48)).avg("VALUENUM")

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

    #cartesian_hadm_hours.show(150)
    df_hadm_hourly_feature_arrays = cartesian_hadm_hours.na.fill(0.0).select('HADM_ID', 'HOUR', F.struct(itemnames).alias('all_temporal_feats'))
    #df_hadm_hourly_feature_arrays.show(150)


    df_hadm_all_hour_feats = df_hadm_hourly_feature_arrays.groupBy("HADM_ID").agg(collect_list(create_map(col("HOUR"),col('all_temporal_feats'))).alias('all_hours_all_temporal_feats'))
    #df_hadm_all_hour_feats.show(150)

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
        #return Row(**{'HADM_ID':row.HADM_ID, 'all_hours_all_temporal_feats' : sequences})
        return (row.HADM_ID, sequences)

    rdd_hadm_individual_metrics_hadm_to_sequences = df_hadm_all_hour_feats.rdd.map(transformMapToArrayFn)
    #print(rdd_hadm_individual_metrics_hadm_to_sequences.take(10))

    schema = StructType([StructField("HADMID", StringType(), True), StructField("SEQUENCES", ArrayType(ArrayType(FloatType()), containsNull=True), True)])
    df_hadm_individual_metrics_hadm_to_sequences = spark.createDataFrame(rdd_hadm_individual_metrics_hadm_to_sequences, schema=schema)

    def array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'

    array_to_string_udf = udf(array_to_string, StringType())

    df_hadm_individual_metrics_hadm_to_sequences_final = df_hadm_individual_metrics_hadm_to_sequences.withColumn('SEQUENCES_STR', array_to_string_udf(df_hadm_individual_metrics_hadm_to_sequences["SEQUENCES"]))
    df_hadm_individual_metrics_hadm_to_sequences_final = df_hadm_individual_metrics_hadm_to_sequences_final.drop("SEQUENCES")

    output_dir = os.path.join(PATH_OUTPUT, 'hadm_sequences')  #must be absolute path

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    df_hadm_individual_metrics_hadm_to_sequences.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('file:///' + output_dir)
    return df_hadm_individual_metrics_hadm_to_sequences





if __name__ == '__main__':

    conf = SparkConf().setMaster("local[4]").setAppName("My App")
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)
    filtered_chart_events_path = os.path.join(PATH_OUTPUT, 'FILTERED_CHARTEVENTS.csv')

    admissions_csv_path = os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv')
    filter_chart_events(spark, os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'CHARTEVENTS.csv'), admissions_csv_path, filtered_chart_events_path)
    
    hadm_sequences = aggregate_temporal_features_hourly(filtered_chart_events_path)
    labels, seqs = create_dataset(spark, admissions_csv_path, hadm_sequences)


    #low priority- remove patient admissions that don't have enough data points during 1st 48 hours of admission  - determine "enough" may need to look at other code

    #get mortality labels for admissions - if the patient died during the admission.   These are located int ADMISSIONS.csv table.  Must be in same ORDER (and length) as feature file.
