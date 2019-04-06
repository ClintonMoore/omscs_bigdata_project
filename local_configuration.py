

#Configure to the paths for your local machine and don't commit it.  I'm subsequently adding to .gitignore.

#CHARTEVENTS.CSV is large (33GB) you may need to put it on another drive please don't commit large data files (would prefer to use the raw mimic files that we all download individually from physionet)


PATH_TRAIN = "data/train/"
PATH_VALIDATION = "data/validation/"
PATH_TEST = "data/test/"
PATH_OUTPUT = "data/processed/"
PATH_MIMIC_ORIGINAL_CSV_FILES = "/media/postgres/ssd_no_os/gt-omscs/big_data/project/mimic3" #"data/mimic3/"