# omscs_bigdata_project

Instructions for running the experiment.

1. Gather the correct Spark and PySpark version. We used the Spark and Pyspark versions 2.4.1 to pair with Hadoop version 2.7 to run this experiment. Make sure these services are running.

2. For all the required dependencies, use the environment.yml by running this command "conda env create -f environment.yml" and set the appropriate exports. Example (Replace with your own relative paths):

export JAVA_HOME="/home/ec2-user/jdk1.8.0_201/jre"
export PATH="/home/ec2-user/jdk1.8.0_201/jre/bin":$PATH
export SPARK_HOME="/home/ec2-user/spark-2.4.1-bin-hadoop2.7/"
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPATH
PATH=~/anaconda3/bin:$PATH
~/anaconda3/bin/conda env create -f /home/ec2-user/environment.yml
source activate mortalityicu

3. Gather all the required CSV files. This includes ADMISSIONS.csv, CHARTEVENTS.csv, DIAGNOSES_ICD.csv, ICUSTAYS.csv, and PATIENTS.csv files.

4. Create the necessary folders in HDFS for the CSV files. For instance if you have your variable "PATH_MIMIC_ORIGINAL_CSV_FILES" set to "data/mimic3" in the python file "local_configuration.py", then you would need to run this command to move over the CSV file into HDFS "hdfs dfs -put ADMISSIONS.csv CHARTEVENTS.csv DIAGNOSES_ICD.csv ICUSTAYS.csv PATIENTS.csv data/mimic3".

5. Run the python file "create_hourly_feats.py" to start pre-processing the data.

6. The first part should generate the FILTERED_CHARTEVENTS.csv file. This file can be obtained by using a sample or the whole data depending the boolean variable "use_sample_subset_lines" in "filter_chart_events" function. Move the files over to HDFS by running "hdfs dfs -put FILTERED_CHARTEVENTS.csv data/processed".

7. Run the python file "create_hourly_feats.py" again. This will create sequences and labels.

8. In the python file "train_variable_rnn.py", you can alter the file to use only the temporal sequences or use the temporal plus static sequences. Run the python file "train_variable_rnn.py".

9. This will give you all the graphed results of data, such as on accuracy and AUC.
