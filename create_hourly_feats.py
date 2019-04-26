import os
import csv
import pickle
import pyspark
import pandas as pd

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame, Row, Window, functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType, StructType, StructField, ArrayType, FloatType
from pyspark.sql.functions import array, udf, avg, row_number, col, concat, lit, monotonically_increasing_id, pandas_udf, PandasUDFType, explode, collect_list, create_map
from pyspark.ml.feature import QuantileDiscretizer
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
    item_mappings['3313'] = 'SBP_Limb'  #BP Cuff [Systolic] (cc/min)
    item_mappings['3315'] = 'SBP_Limb'  #BP Left Arm [Systolic] (cc/min)
    item_mappings['3317'] = 'SBP_Limb'  #BP Left Leg [Systolic] (cc/min)
    item_mappings['3321'] = 'SBP_Limb'  #BP Right Arm [Systolic] (cc/min)
    item_mappings['3323'] = 'SBP_Limb'  #BP Right Leg [Systolic] (cc/min)
    
    item_mappings['3319'] = 'SBP_Line'  #BP PAL [Systolic]  (Breath)
    item_mappings['3325'] = 'SBP_Line'  #BP UAC [Systolic] (Breath)
    
    item_mappings['6'] = 'SBP'  #ABP [Systolic] (mmHg)
    item_mappings['51'] = 'SBP'  #Arterial BP [Systolic]   (mmHg)
    item_mappings['6701'] = 'SBP'  #Arterial BP #2 [Systolic] (mmHg)
    item_mappings['225309'] = 'SBP'  #ART BP Systolic (mmHg)
    item_mappings['220050'] = 'SBP'  #Arterial Blood Pressure systolic (mmHg)
    item_mappings['442'] = 'SBP'  #Manual BP [Systolic] (mmHg)
    item_mappings['224167'] = 'SBP'  #Manual Blood Pressure Systolic Left (mmHg)
    item_mappings['227243'] = 'SBP'  #Manual Blood Pressure Systolic Right (mmHg)
    item_mappings['442'] = 'SBP'  #Manual BP [Systolic] (mmHg)
    item_mappings['455'] = 'SBP'  #NBP [Systolic]  (mmHg)
    item_mappings['480'] = 'SBP'  #Orthostat BP sitting [Systolic]  (mmHg)
    item_mappings['482'] = 'SBP'  #OrthostatBP standing [Systolic]   (mmHg)
    item_mappings['484'] = 'SBP'  #Orthostatic BP lying [Systolic]   (mmHg)
    item_mappings['220179'] = 'SBP'  #Non Invasive Blood Pressure systolic (mmHg) 
        
    #Blood Pressure [Diastolic]
    item_mappings['8502'] = 'DBP_Limb'  #BP Cuff [Diastolic] (cc/min)
    item_mappings['8503'] = 'DBP_Limb'  #BP Left Arm [Diastolic] (cmH20)
    item_mappings['8504'] = 'DBP_Limb'  #BP Left Leg [Diastolic] (cc/min)
    item_mappings['8506'] = 'DBP_Limb'  #BP Right Arm [Diastolic] (cc/min)
    item_mappings['8507'] = 'DBP_Limb'  #BP Right Leg [Diastolic] (cc/min)
    
    item_mappings['8508'] = 'DBP_Line'  #BP UAC [Diastolic] (Breath)
    item_mappings['8505'] = 'DBP_Line'  #BP PAL [Diastolic] (Breath)
    
    item_mappings['8440'] = 'DBP'  #Manual BP [Diastolic] (mmHg)
    item_mappings['227242'] = 'DBP'  #Manual Blood Pressure Diastolic Right (mmHg)
    item_mappings['224643'] = 'DBP'  #Manual Blood Pressure Diastolic Left (mmHg)
    item_mappings['8441'] = 'DBP'  #NBP [Diastolic] (mmHg)
    item_mappings['8364'] = 'DBP'  #ABP [Diastolic] (mmHg)
    item_mappings['8555'] = 'DBP'  #Arterial BP #2 [Diastolic] (mmHg)
    item_mappings['8368'] = 'DBP'  #Arterial BP [Diastolic]  (mmHg)
    item_mappings['225310'] = 'DBP'  #ART BP Diastolic   (mmHg)
    item_mappings['220051'] = 'DBP'  #Arterial Blood Pressure diastolic  (mmHg)
    item_mappings['220180'] = 'DBP'  #Non Invasive Blood Pressure diastolic  (mmHg)
    
    #Blood Pressure [Mean]
    
    item_mappings['456'] = 'MBP'  #NBP Mean  (mmHg)
    item_mappings['224322'] = 'MBP'  #IABP Mean (mmHg)
    item_mappings['224'] = 'MBP'  #IABP Mean  (mmHg)
    item_mappings['443'] = 'MBP'  #Manual BP Mean(calc)  (mmHg)
    item_mappings['6702'] = 'MBP'  #Arterial BP Mean #2 (mmHg)
    item_mappings['52'] = 'MBP'  #Arterial BP Mean  (mmHg)
    item_mappings['225312'] = 'MBP'  #ART BP mean (mmHg)
    item_mappings['220052'] = 'MBP'  #Arterial Blood Pressure mean   (mmHg)    
    item_mappings['220181'] = 'MBP'  #Non Invasive Blood Pressure mean (mmHg)
    
    item_mappings['3324'] = 'MBP_Line'  #BP UAC [Mean]    (Breath)
    item_mappings['3318'] = 'MBP_Line'  #BP PAL [Mean] (Breath)
    
    item_mappings['7622'] = 'MBP_Limb'  #BP Rt. Leg Mean (None)
    item_mappings['7620'] = 'MBP_Limb'  #BP Rt. Arm Mean (None)
    item_mappings['7618'] = 'MBP_Limb'  #BP Lt. leg Mean (None)
    item_mappings['3322'] = 'MBP_Limb'  #BP Right Leg [Mean] (Breath)
    item_mappings['3320'] = 'MBP_Limb'  #BP Right Arm [Mean] (Breath)
    item_mappings['3316'] = 'MBP_Limb'  #BP Left Leg [Mean] (Breath)
    item_mappings['3314'] = 'MBP_Limb'  #BP Left Arm [Mean] (Breath)    
    item_mappings['3312'] = 'MBP_Limb'  #BP Cuff [Mean]  (cc/min)  
    
    #Temperature
    item_mappings['676'] = 'TEMP' #Temperature C
    #item_mappings['677'] = 'TEMPC'  #Temperature C (calc)
    item_mappings['223762'] = 'TEMP'  #Temperature Celsius    
    item_mappings['678'] = 'TEMP' #Temperature F
    item_mappings['223761'] = 'TEMP'  #Temperature Fahrenheit
    
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
    item_mappings['646'] = 'SPO2'  #SpO2 (%)
    item_mappings['220277'] = 'SPO2'  # saturation pulseoxymetry SpO2  (%)
    item_mappings['223769'] = 'SPO2'  # O2 Saturation Pulseoxymetry Alarm - High  (%)
    item_mappings['223770'] = 'SPO2'  # O2 Saturation Pulseoxymetry Alarm - Low  (%)
    
    #Albumin
    item_mappings['3066'] = 'Albumin'  #albumin
    item_mappings['1521'] = 'Albumin'  #Albumin
    item_mappings['46564'] = 'Albumin'  #Albumin
    item_mappings['227456'] = 'Albumin'  #Albumin (g/dL)
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
    item_mappings['225624'] = 'BUN'  #BUN (mg/dL)
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
    
    #Lactate
    item_mappings['818'] = 'Lactate' #Lactic Acid(0.5-2.0)
    item_mappings['1531'] = 'Lactate' #Lactic Acid
    item_mappings['225668'] = 'Lactate' #Lactic Acid
    
    #Calcium
    item_mappings['44441'] = 'Ca'  #calcium
    item_mappings['44855'] = 'Ca'  #Calcium
    item_mappings['1522'] = 'Ca'  #Calcium 	mg/dl
    item_mappings['227525'] = 'Ca'  #Calcium
    
    #Creatinine
    item_mappings['1525'] = 'Creatinine' #Creatinine
    item_mappings['220615'] = 'Creatinine' #Creatinine  (mg/dL)
    item_mappings['791'] = 'Creatinine' #Creatinine (0-1.3)
    item_mappings['3750'] = 'Creatinine' #Creatinine (0-0.7)
    item_mappings['227005'] = 'Creatinine' #Creatinine_ApacheIV
    #item_mappings['227006'] = 'Creatinine' #CreatScore_ApacheIV
    item_mappings['226752'] = 'Creatinine' #CreatinineApacheIIValue
    #item_mappings['226751'] = 'Creatinine' #CreatinineApacheIIScore

        
    #pH 
    item_mappings['6003'] = 'pH' #ph level
    item_mappings['1673'] = 'pH'  #PH
    #item_mappings['220734'] = 'pH'  #PH (dipstick)
    #item_mappings['223830'] = 'pH'  #PH (Arterial) 
    
    #carbon dioxide 
    item_mappings['3784'] = 'PCO2'
    item_mappings['778'] = 'PCO2' #Arterial PaCO2  (mmHg)
    item_mappings['220235'] = 'PCO2' #Arterial CO2 Pressure (mmHg)
    item_mappings['3774'] = 'PCO2' #Mix Venous PCO2 
    item_mappings['226062'] = 'PCO2' #Venous CO2 Pressure (mmHg)
    item_mappings['227036'] = 'PCO2' #PCO2_ApacheIV
    
    #Bicarbonate
    item_mappings['812'] = 'HCO3' #HCO3 (mEq/l)
    item_mappings['46362'] = 'HCO3' #bicarbonate
    item_mappings['44166'] = 'HCO3' #BICARBONATE-HCO3
    item_mappings['227443'] = 'HCO3' #HCO3 (serum)
    item_mappings['226759'] = 'HCO3' #HCO3ApacheIIValue
    #item_mappings['226760'] = 'HCO3' #HCO3Score
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
    #item_mappings['227016'] =  'Glucose'
    item_mappings['3816'] = 'Glucose'  
    item_mappings['228388'] = 'Glucose'
    item_mappings['226537'] = 'Glucose' #(mg/dL)
    item_mappings['227976'] = 'Glucose' #Boost Glucose Control (1/4)
    item_mappings['227977'] = 'Glucose' #Boost Glucose Control (1/2)
    item_mappings['227978'] = 'Glucose' #Boost Glucose Control (3/4)
    item_mappings['227979'] = 'Glucose' #Boost Glucose Control (Full)
    item_mappings['3744'] = 'Glucose'
    item_mappings['3745'] = 'Glucose'
    item_mappings['3447'] = 'Glucose'
    item_mappings['220621'] = 'Glucose' #Glucose (serum) (mg/dL)

    #Sodium
    item_mappings['1536'] = 'Na' #Sodium
    item_mappings['837'] = 'Na' #Sodium (135-148)
    item_mappings['3803'] = 'Na' #Sodium (135-148)
    item_mappings['227052'] = 'Na' #Sodium_ApacheIV
    item_mappings['228389'] = 'Na' #Sodium (serum) (soft) 
    item_mappings['228390'] = 'Na' #Sodium (whole blood) (soft)
    item_mappings['220645'] = 'Na' #Sodium (serum) (mEq/L)
    item_mappings['226534'] = 'Na' #Sodium (whole blood) 
    
    #Postassium
    item_mappings['1535'] = 'K' #Potassium
    item_mappings['41956'] = 'K' #Potassium
    item_mappings['44711'] = 'K' #potassium 
    #item_mappings['226771'] = 'K' #PotassiumApacheIIScore
    item_mappings['226772'] = 'K' #PotassiumApacheIIValue

    item_mappings['829'] = 'K' #Potassium (3.5-5.3)
    item_mappings['3792'] = 'K' #Potassium (3.5-5.3)
    item_mappings['227464'] = 'K' #Potassium (whole blood) 
    item_mappings['226535'] = 'K' #ZPotassium (whole blood) 
    item_mappings['220640'] = 'K' #ZPotassium (serum)
    
    #Anion Gap
    item_mappings['3732'] = 'AG' #Anion Gap (8-20) (mEq/L)
    item_mappings['227073'] = 'AG' #Anion gap  (mEq/L)
    
    #BANDS
    item_mappings['3734'] = 'BANDS' #BANDS 
    item_mappings['795'] = 'BANDS' #Differential-Bands
    item_mappings['3738'] = 'BANDS' #Bands
    item_mappings['225638'] = 'BANDS' #Differential-Bands
    
    #White Blood Cell (WBC)
    item_mappings['3834'] = 'WBC' #WhiteBloodC 4.0-11.0
    item_mappings['1127'] = 'WBC' #WBC (4-11,000)
    item_mappings['861'] = 'WBC' #WBC (4-11,000)
    item_mappings['4200'] = 'WBC' #WBC 4.0-11.0
    item_mappings['1542'] ='WBC' #WBC
    item_mappings['220546'] = 'WBC' #WBC
    
    #Partial Thromboplastin Time (PTT) 
    item_mappings['825'] = 'PTT' #PTT(22-35) 
    item_mappings['3796'] = 'PTT' #Ptt
    item_mappings['1533'] = 'PTT' #PTT
    item_mappings['227466'] = 'PTT' #PTT
    item_mappings['220562'] = 'PTT' #ZPTT
    
    #Prothrombin Time (PT) 
    item_mappings['1286'] = 'PT' #PT
    item_mappings['824'] = 'PT' #PT(11-13.5)
    item_mappings['227465'] = 'PT' #Prothrombin time
    item_mappings['220560'] = 'PT' #ZProthrombin time
    
    #HEMOGLOBIN
    item_mappings['814'] = 'Hb' #Hemoglobin
    item_mappings['220228'] = 'Hb' #Hemoglobin (g/dl)
    item_mappings['1165'] = 'Hb' #Hgb 
    item_mappings['3759'] = 'Hb' #HGB (10.8-15.8)
    
    #HEMATOCRIT
    item_mappings['813'] = 'Ht' #Hematocrit (%)
    item_mappings['3761'] = 'Ht' #Hematocrit (35-51)
    item_mappings['226540'] = 'Ht' #Hematocrit (whole blood - calc) (%)
    item_mappings['220545'] = 'Ht' #Hematocrit (serum) (%)
    
    #Chloride    
    item_mappings['228385'] = 'Cl' #Chloride (serum) (soft)
    item_mappings['228386'] = 'Cl' #Chloride (whole blood) 
    item_mappings['220602'] = 'Cl' #Chloride (serum)
    item_mappings['226536'] = 'Cl' #Chloride (whole blood)
    item_mappings['1523'] = 'Cl' #Chloride
    item_mappings['3747'] = 'Cl' #Chloride (100-112)
    item_mappings['788'] = 'Cl' #Chloride (100-112)
            
    #BILIRUBIN
    item_mappings['4948'] = 'Bilirubin' #Bilirubin
    item_mappings['225651'] = 'Bilirubin' #Direct Bilirubin (mg/dL)
    item_mappings['225690'] = 'Bilirubin' #Total Bilirubin (mg/dL)
    
    return item_mappings 



def filter_chart_events(spark, orig_chrtevents_file_path, admissions_csv_file_path, filtered_chrtevents_outfile_path):
    #TAKES ONLY THE RELEVANT ITEM ROWS FROM THE CHARTEVENTS.CSV file
    item_mappings = get_event_key_ids()



    #use subset of large CHARTEVENTS.csv file for faster development
    chrtevents_file_path_to_use = orig_chrtevents_file_path
    use_sample_subset_lines = False
    if use_sample_subset_lines:

        chartevents_sample_temp_file = "CHARTEVENTS_SAMPLE.csv"
        chrtevents_file_path_to_use = chartevents_sample_temp_file

        temp_file = open(chartevents_sample_temp_file, "w+")
        with open(orig_chrtevents_file_path) as orig_file:
            i = 0
            for line in orig_file:
                temp_file.write(line)
                i = i + 1
                if i > 500000:
                    break
        temp_file.close()

    # LOS ***
    #*********
    
    los_path =  os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, "ICUSTAYS.csv")
    df_los = spark.read.csv(los_path, header=True, inferSchema="false")
    
    df_los = df_los.filter(col('LOS') >=1).select(['HADM_ID'])

    df_chartevents = spark.read.csv(chrtevents_file_path_to_use, header=True, inferSchema="false")

    filtered_chartevents = df_chartevents.filter(col('ITEMID').isin(list(item_mappings.keys())))
    filtered_chartevents = filtered_chartevents.withColumn("ITEMNAME", translate(item_mappings)("ITEMID"))


    #join filtered_chartevents with ADMISSIONS.csv on HADMID --- only keep HADMID AND ADMITTIME COLUMNS FROM ADMISSIONS
    df_admissions = spark.read.csv(admissions_csv_file_path, header=True, inferSchema="false").select('HADM_ID', 'ADMITTIME')

    #add column that contains the hour the observation occurred after admission  (0 - X)
    filtered_chartevents = filtered_chartevents.join(df_admissions, ['HADM_ID'])
    timeFmt = "yyyy-MM-dd' 'HH:mm:ss"   #2153-09-03 07:15:00
    timeDiff = F.round((F.unix_timestamp('CHARTTIME', format=timeFmt)
                - F.unix_timestamp('ADMITTIME', format=timeFmt)) / 60 / 60).cast('integer')  #calc diff, convert seconds to minutes, minutes to hours, then math.floor to remove decimal places (for hourly bin/aggregations)
    filtered_chartevents = filtered_chartevents.withColumn("HOUR_OF_OBS_AFTER_HADM", timeDiff)  #  F.round(   ).cast('integer')

    #filter out all observations where X > 48  (occurred after initial 48 hours of admission)
    filtered_chartevents = filtered_chartevents.filter((col('HOUR_OF_OBS_AFTER_HADM') <= 48) & (col('HOUR_OF_OBS_AFTER_HADM') >=0))
    
    filtered_chartevents = df_los.join(filtered_chartevents, ['HADM_ID'])

    #REMOVE columns that are not needed (keep CHARTEVENTS cols, ITEMNAME, HOUR_OF_OBS_AFTER_HADM
    filtered_chartevents = reduce(DataFrame.drop, ['ADMITTIME'], filtered_chartevents)

    with open(filtered_chrtevents_outfile_path, "w+") as f:
        w = csv.DictWriter(f, fieldnames=filtered_chartevents.schema.names)
        w.writeheader()

        for rdd_row in filtered_chartevents.rdd.toLocalIterator():
            w.writerow(rdd_row.asDict())

            
             
#Returns a list of admission ids, a list of mortality labels and  a list of sequences
def create_dataset(spark, admissions_csv_path, hadm_sequences):
    hadm_sequences= hadm_sequences.withColumnRenamed('HADMID', 'HADM_ID')
    mortality = spark.read.csv(admissions_csv_path , header=True, inferSchema="false").select('SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG')
    df = hadm_sequences.join(mortality, on='HADM_ID', how='left').na.drop()
    labels =  df.select ('HOSPITAL_EXPIRE_FLAG').rdd.flatMap(lambda x: x).collect()
    seqs = df.rdd.map(lambda x:  x[1]).collect()  
    hadm_ids = df.rdd.map(lambda x:  x[0]).collect()
    return hadm_ids, labels, seqs



#Standardize features values in place VALUENUM using Quantiles
def standardize_features (df_filtered_chartevents): 
    temp = df_filtered_chartevents.select('ITEMNAME','VALUENUM')
    min_quantile = temp.groupBy('ITEMNAME').agg( F.expr('percentile_approx(VALUENUM, 0.05)').alias("Quantile5"), F.expr('percentile_approx(VALUENUM, 0.95)').alias("Quantile95"))
    cartesian_min_quantile = min_quantile.join(df_filtered_chartevents, on='ITEMNAME', how='left')
    #TODO: Verify the return value when the min and quantile are equal
    udf_standardize = F.udf( lambda x: (x[0]-x[1]) / (x[2]-x[1] ) if x[2]!=x[1] else  float(x[0]) , DoubleType()) 
    #standardized_df = cartesian_min_quantile.withColumn("Standardized_Value", udf_standardize(array("VALUENUM", "Min", "Quantile95")))
    standardized_df = cartesian_min_quantile.withColumn("VALUENUM", udf_standardize(array("VALUENUM", "Quantile5", "Quantile95")))
    
    return standardized_df.drop("Quantile5","Quantile95")

#Standardize features values in place VALUENUM using Max
def standardize_features_max (df_filtered_chartevents): 
    temp = df_filtered_chartevents.select('ITEMNAME','VALUENUM')
    max_df = temp.groupBy('ITEMNAME').agg( F.max(temp.VALUENUM).alias("Max"))
    cartesian_max = max_df.join(df_filtered_chartevents, on='ITEMNAME', how='left')
    udf_standardize = F.udf( lambda x: (x[0] /x[1]) if x[1]!=0.0 else  0.0 , DoubleType()) 
    standardized_df = cartesian_max.withColumn("VALUENUM", udf_standardize(array("VALUENUM", "Max")))
    
    return standardized_df.drop("Max")


#Convert all temperature values in dataframe to Celsius
def temp_conversion (df_filtered_chartevents):
    
    def fahrenheit_celsius(itemid, value):
        if (itemid=='678') or (itemid=='223761'):
            return (float(value)-32) * (5 /9)
        else:
            return float(value)
 
    udf_conversion  = udf(fahrenheit_celsius, DoubleType())
    df = df_filtered_chartevents.withColumn("VALUENUM", udf_conversion("ITEMID","VALUENUM"))
    return df

#Filter feature values 
def values_filter (df_filtered_chartevents):

    #TODO : check for additional conditions
    def value_conditions(itemname, itemid, value):
   
        if (itemname == 'TEMP') and (itemid in [678,223761]) and value > 70 and value < 120:
            return (float(value)-32) * (5 /9)
        elif (itemname == 'TEMP') and (itemid in [678,223761]) and value <= 70 and value >= 120:
            return None 
        elif (itemname == 'TEMP') and (itemid in [676,223762]) and value < 10 and value > 50: 
            return None
        elif (itemname == 'Albumin') and (value >10) :
            return None
        elif (itemname == 'AG') and (value >10000) :
            return None
        elif (itemname == 'BANDS') and value >100 and value <0 :
            return None
        elif (itemname == 'Bicarbonate') and value >10000:
            return None        
        elif (itemname == 'Cl') and value >10000:
            return None
        elif (itemname == 'Creatinine') and value >150:
            return None
        elif (itemname == 'Glucose') and value >10000:
            return None
        elif (itemname == 'Ht') and value >100:
            return None
        elif (itemname == 'Hg') and value >50:
            return None
        elif (itemname == 'Platelets') and value >10000:
            return None
        elif (itemname == 'K') and value >30:
            return None        
        elif (itemname == 'PTT') and value >150:
            return None 
        elif (itemname == 'PT') and value >150:
            return None 
        elif (itemname == 'INR') and value >50:
            return None 
        elif (itemname == 'Na') and value >200:
            return None 
        elif (itemname == 'BUN') and value >300:
            return None 
        elif (itemname == 'WBC') and value >1000:
            return None
        elif (itemname == 'HEART_RATE') and value <=0 and value >= 300:
            return None
        elif (itemname == 'SBP') and value <=0 and value >= 400:
            return None
        elif (itemname == 'SBP_Line') and value <=0:
            return None
        elif (itemname == 'SBP_Limb') and value <=0:
            return None
        elif (itemname == 'DBP') and value <=0 and value >= 300:
            return None
        elif (itemname == 'DBP_Line') and value <=0:
            return None
        elif (itemname == 'DBP_Limb') and value <=0:
            return None
        elif (itemname == 'MBP') and value <=0 and value >= 300:
            return None
        elif (itemname == 'MBP_Line') and value <=0:
            return None
        elif (itemname == 'MBP_Limb') and value <=0:
            return None
        elif (itemname == 'RR') and value >= 70:
            return None
        elif (itemname == 'SPO2') and value <= 0 and value >100:
            return None
        elif (itemname == 'Glucose') and value <= 0 :
            return None
        elif (itemname == 'Lactate') and value > 50 :
            return None
        else:
            return value
        
    udf_conversion  = udf(value_conditions, DoubleType())
    df = df_filtered_chartevents.withColumn("VALUENUM", udf_conversion("ITEMNAME", "ITEMID","VALUENUM"))
    return df.na.drop(subset=["VALUENUM"])


def aggregate_temporal_features_hourly(filtered_chartevents_path):
    num_hours = 48
    df_filtered_chartevents = spark.read.csv(filtered_chartevents_path, header=True, inferSchema=False)
    df_filtered_chartevents = df_filtered_chartevents.na.drop(subset=["VALUENUM"]).withColumn("VALUENUM", df_filtered_chartevents["VALUENUM"].cast(DoubleType()))
    df_filtered_chartevents = values_filter (df_filtered_chartevents)
    #df_filtered_chartevents = df_filtered_chartevents.withColumn("VALUENUM_INT", df_filtered_chartevents["VALUENUM"].cast(IntegerType()))
    #df_filtered_chartevents = df_filtered_chartevents.drop(df_filtered_chartevents.VALUENUM).withColumnRenamed('VALUENUM_INT', 'VALUENUM')
    df_standardized_chartevents = standardize_features_max (df_filtered_chartevents)


    #test only
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

    #itemnames = list(set(get_event_key_ids().values()))

    item_avgs = dict(df_hadm_hourly_averages_filled.groupby("ITEMNAME").agg(avg(col("VALUE")).alias("avg")).collect())
    itemnames = list(item_avgs.keys())
    print("Items dict: ", item_avgs)
    all_hours = range(num_hours)

    df_hadm_hourly_averages_filled_agged = df_hadm_hourly_averages_filled.groupBy("HADM_ID").agg(collect_list(create_map(concat(col('ITEMNAME'), lit('_'), col("HOUR")),col('VALUE'))).alias('item_hour_toValues'))

    #df_hadm_hourly_averages_filled_agged.show(10)
    def mapFn(row):
        #print(row)
        list_of_single_entry_dicts_for_each_hr = flatten(row.item_hour_toValues)
        dict_hour_to_feature_row = {k: v for d in list_of_single_entry_dicts_for_each_hr for k, v in d.items()}
        sequences = [[None] * len(itemnames)] * num_hours
        for i in range(len(itemnames)):
            itemname = itemnames[i]

            value = item_avgs.get(itemname)
            if value is None:
                raise Exception("Averages should never have a None value!!  There is a bug!")

            for hour in all_hours:
                if itemname + "_0" in dict_hour_to_feature_row:
                    key = itemname + "_" + str(hour)
                    if key in dict_hour_to_feature_row:
                        found_value = dict_hour_to_feature_row[key]
                        if found_value is not None:
                            value = found_value
                    else:
                        raise Exception("There should be entries for ALL hours for an item in the dict if there is an entry for ANY hour of that item b/c of ffil/ bfill")

                sequences[hour][i] = value

                if sequences[hour][i] is None:
                    raise Exception("None of these should be set as a None!")

        return (row.HADM_ID, sequences)

    rdd_hadm_individual_metrics_hadm_to_sequences = df_hadm_hourly_averages_filled_agged.rdd.map(mapFn)

    return rdd_hadm_individual_metrics_hadm_to_sequences



def get_icd9_features(sparkSQLContext):

    top25_icd9_codes = ['4019', '4280', '42731', '41401', '5849', '25000', '2724', '51881', '5990', '53081', '2720', '2859', '486', '2449', '2851', '2762', '496', '99592', '5070', '389', '5859', '40390', '311', '2875', '3051'] #from supp doc here: https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/jamiaopen/1/1/10.1093_jamiaopen_ooy011/9/ooy011_supplemental_materials.docx?Expires=1555865667&Signature=qKYo3JhhWHA6AxWaf007UiK9Yp8thafIVKYoO6lVETi3nW4h7KzoD4XumvO8tU7aBTl6PM0zIOTvzfPFvJqqVkCzpj-KvwwfKH5lz5lpkPlzFkEhl7hd-VDGW-DxC1TiYAnGFom1u2U6pndj9JcBJLMWTwQ4wSxPEg3ZoO3Wpa9HLz71xyy1Q1sjCfx99onRsmMmF2Rz2dcol6tU-YyqQoGxJ5QWpuqYaYH0BuEYhfRXedXhUxAV0TsR3mjb9xeMjXs6OlJoZY3pO6gcSRj98Nb8u5N9SrbsuRTeEtW31gwE1ENZWvqOTguL6fpicWu9zrmeFczCpt5lUF3br7qf5A__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
    idxs_top_25_codes = range(len(top25_icd9_codes))
    df_diagnoses = sparkSQLContext.read.csv(os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'DIAGNOSES_ICD.csv'), header=True, inferSchema="false")
    df_diagnoses = df_diagnoses.filter(df_diagnoses.ICD9_CODE.isin(top25_icd9_codes))     #filter only top 25 icd9codes
    df_admissions = sparkSQLContext.read.csv(os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv'), header=True, inferSchema="false")

    df_diagnoses = df_diagnoses.join(df_admissions.select(['HADM_ID', 'ADMITTIME']), ['HADM_ID'])

    df_subj_icd9codes_by_admittime = df_diagnoses.groupby('SUBJECT_ID').agg(collect_list(create_map(col("ADMITTIME"),col('ICD9_CODE'))).alias('ICD9_CODEs_occuring_for_hadm_after_admittimes'))

    admissions_to_all_past_preasent_future_diags = df_admissions.select(['HADM_ID', 'ADMITTIME', 'SUBJECT_ID']).join(df_subj_icd9codes_by_admittime, ['SUBJECT_ID'])

    def mapFnRowToKnownDiagnostics(row):
        this_hadm_admittime = row.ADMITTIME
        list_of_single_entry_dicts_admittime_to_icd9 = flatten(row.ICD9_CODEs_occuring_for_hadm_after_admittimes)
        icd9_set = set([])
        for single_entry_dict in list_of_single_entry_dicts_admittime_to_icd9:
             for timestampstr in single_entry_dict.keys():
                  if timestampstr <= this_hadm_admittime:              #'<' means all icd9's from prior admissions; '<=' means prior and current admission (which may include foward looking diagnoses from later in this admission - allowing data leakage)
                       icd9_set.add(single_entry_dict[timestampstr])
        feat_array = [1.0 if x in icd9_set else 0.0 for x in top25_icd9_codes]
        return (row.HADM_ID, feat_array)
    hadmid_to_icd9_feats = admissions_to_all_past_preasent_future_diags.rdd.map(mapFnRowToKnownDiagnostics)
    #print(hadmid_to_icd9_feats.take(100))

    return hadmid_to_icd9_feats


def get_static_features(spark):
    # TODO merge icd9 feats with other static feats, demographics ...??

    df_admissions = spark.read.csv(os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv'), header=True,
                                   inferSchema="false")
    df_patients = spark.read.csv(os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'PATIENTS.csv'), header=True,
                                 inferSchema="false")

    df_merge = df_admissions.join(df_patients, ['SUBJECT_ID'])

    timeFmt = "yyyy-MM-dd' 'HH:mm:ss"  # 2153-09-03 07:15:00
    timeDiff = F.round((F.unix_timestamp('ADMITTIME', format=timeFmt)
                        - F.unix_timestamp('DOB', format=timeFmt)) / (60 * 60 * 24 * 365.242)).cast('integer')
    df_merge = df_merge.withColumn("AGE_ADMISSION", timeDiff)

    df_merge = QuantileDiscretizer(numBuckets=5, inputCol='AGE_ADMISSION', outputCol='QAGE').fit(df_merge).transform(
        df_merge)

    t = {0.0: 'very-young', 1.0: 'young', 2.0: 'normal', 3.0: 'old', 4.0: 'very-old'}
    udf_age = udf(lambda x: t[x], StringType())
    df_merge = df_merge.withColumn('AGE', udf_age('QAGE'))
    df_merge = reduce(DataFrame.drop, ['SUBJECT_ID', 'ROW_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE',
                                       'ADMISSION_LOCATION', 'LANGUAGE', 'RELIGION', 'EDREGTIME', 'EDOUTTIME',
                                       'DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA', 'GENDER', 'DOB',
                                       'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG', 'AGE_ADMISSION', 'QAGE'], df_merge)

    df_merge = df_merge.fillna({'MARITAL_STATUS': 'UNKNOWN_MARITAL'})
    
    categories = list(set(flatten([list(df_merge.select(c).distinct().collect()) for c in df_merge.columns if c not in ['HADM_ID'] ]) ))

    categories_dict = {}
    for i in range(len(categories)):
      categories_dict[categories[i]] = float(i)

    def mapFnLabels(row):
      one = categories_dict[row.AGE]
      two = categories_dict[row.ETHNICITY]
      three = categories_dict[row.DISCHARGE_LOCATION]
      four = categories_dict[row.MARITAL_STATUS]
      five = categories_dict[row.INSURANCE]
      feat_array = [one, two, three, four, five]
      return (row.HADM_ID, feat_array)

    hadmid_to_static_feats = df_merge.rdd.map(mapFnLabels)

    return hadmid_to_static_feats


def merge_temporal_sequences_and_static_features(temporal_features_rdd, static_features_rdd):
    hadm_temporal_and_static_rdd = temporal_features_rdd.join(static_features_rdd)


    def mapFn(hadm_temporal_seqs_static_feats):
        hadm = hadm_temporal_seqs_static_feats[0]
        seqs = hadm_temporal_seqs_static_feats[1][0]
        static_feats = hadm_temporal_seqs_static_feats[1][1]
        seqs_with_static_feats = [x + static_feats for x in seqs]
        return (hadm, seqs_with_static_feats)

    hadm_sequences_of_temporal_and_static_feats_rdd = hadm_temporal_and_static_rdd.map(mapFn)
    #print(hadm_sequences_of_temporal_and_static_feats_rdd.take(5))

    return hadm_sequences_of_temporal_and_static_feats_rdd


def create_and_write_dataset(spark, sequences, label_name):
    schema = StructType([StructField("HADMID", StringType(), True), StructField("SEQUENCES", ArrayType(ArrayType(FloatType()), containsNull=True), True)])
    hadm_sequences = spark.createDataFrame(sequences, schema=schema)




    #TODO: AFTER BUGS HAVE BEEN RESOLVED.  THIS BLOCK CAN BE REMOVED.
    def array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'
    array_to_string_udf = udf(array_to_string, StringType())
    hadm_sequences_mod= hadm_sequences.withColumn('SEQUENCES_STR', array_to_string_udf(hadm_sequences["SEQUENCES"]))
    hadm_sequences_mod = hadm_sequences_mod.drop("SEQUENCES")
    output_ = os.path.join(PATH_OUTPUT, 'hadm_sequences')  #must be absolute path
    hadm_sequences_mod.toPandas().to_csv(os.path.join(PATH_OUTPUT, 'hadm_sequences'))



    hadm_ids, labels, seqs = create_dataset(spark, admissions_csv_path, hadm_sequences)

    pickle.dump(labels, open(os.path.join(PATH_OUTPUT, label_name + ".hadm.labels"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(seqs, open(os.path.join(PATH_OUTPUT, label_name + ".hadm.seqs"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(hadm_ids, open(os.path.join(PATH_OUTPUT, label_name + ".hadm.ids"), 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    conf = SparkConf().setMaster("local[7]").setAppName("My App") #\
        #.set("spark.driver.memory", "15g") \
        #.set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)
    filtered_chart_events_path = os.path.join(PATH_OUTPUT, 'FILTERED_CHARTEVENTS.csv')

    admissions_csv_path = os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'ADMISSIONS.csv')
    filter_chart_events(spark, os.path.join(PATH_MIMIC_ORIGINAL_CSV_FILES, 'CHARTEVENTS.csv'), admissions_csv_path, filtered_chart_events_path)

    rdd_hadm_temporal_sequences_only = aggregate_temporal_features_hourly(filtered_chart_events_path)

    create_and_write_dataset(spark, rdd_hadm_temporal_sequences_only, "temporal_only")

    rdd_icd9_features = get_icd9_features(spark)

    rdd_hadmid_to_sequences_temporal_and_static_feats = merge_temporal_sequences_and_static_features(rdd_hadm_temporal_sequences_only, rdd_icd9_features)

    rdd_static_features = get_static_features(spark)

    rdd_hadmid_to_sequences_temporal_and_static_feats = merge_temporal_sequences_and_static_features(rdd_hadmid_to_sequences_temporal_and_static_feats, rdd_static_features)

    create_and_write_dataset(spark, rdd_hadmid_to_sequences_temporal_and_static_feats,"temporal_and_static")
