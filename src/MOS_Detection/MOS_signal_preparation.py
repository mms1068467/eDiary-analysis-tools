import datetime
import sqlite3
import numpy as np
import pandas as pd

from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import data_loader as dl
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check

#filename = "C:/Users/MM/Desktop/Uni Salzburg/P.hD/ZGis/Human Sensing/MOS_Detection/MOS_algo_Martin/Data_MOS/KIT_Fon_06_Freddy_2022-06-30T0858.sqlite"

def MOS_detection_signal_preparation(filename):

    print("Empatica E4 Check:", sensor_check.E4_used(filename))

    if sensor_check.E4_used(filename) == True:

        #### GSR
        GSR_cluster, GSR_raw = dl.get_ediary_data(filename = filename, phys_signal = "GSR")

        # all in one
        GSR = pps.GSR_preprocessing(GSR_cluster = GSR_cluster,
                                    GSR_raw = GSR_raw,
                                    phys_signal = "GSR")



        #### ST
        ST_cluster, ST_raw = dl.get_ediary_data(filename = filename, phys_signal = "ST")

        ST = pps.ST_preprocessing(ST_cluster = ST_cluster,
                                  ST_raw = ST_raw,
                                  phys_signal = "ST")

    else:
        print("Make sure to check if Empatica E4 sensor was connected properly.")

    print("BioHarness Check:", sensor_check.BioHarness_used(filename))

    if sensor_check.BioHarness_used(filename):
        #### IBI
        #try:
        IBI_raw = dl.get_ediary_data(filename = filename, phys_signal = "IBI")
            #print("try IBI raw import", IBI_raw)
        #print(IBI_raw)
        #IBI_raw_formatted = "" #format IBI raw
        #except ValueError:
        #    IBI_raw = pd.DataFrame()

        #print("Imported raw IBI data: ", IBI_raw)

        #print("Formatted: ", pps.format_raw_IBI(IBI_raw))
        IBI_raw['IBI'] = pps.format_raw_IBI(IBI_raw)

        if IBI_raw is not None:
            IBI = pps.IBI_preprocessing(IBI_raw)
            #print("PREPROCESSED IBI", IBI)
        else:
            IBI = IBI_raw

        #if IBI is not None:
            #print("IBI prep successful")

        #print(IBI)

        #### HRV ---> get HRV from preprocessed IBIs
        # TODO - this is the old version (just IBI differences)
        if IBI is not None:
            HRV = pps.HRV_preprocessing(IBI)
        else:
            HRV = None

        #if HRV is not None:
            #print("HRV prep successful", HRV)


    # load & preprocess ECG data
    #ECG_raw = dl.get_ediary_data(filename=filename, phys_signal="ECG")
    #ECG = pps.ECG_resampling(ECG_raw)

    else:
        IBI = None

    if IBI is None:
        merged_data = pps.merge_signals(GSR, ST, merge_col = 'time_iso')

    else:
        merged_data = pps.merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')


    #merged_data = pps.merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')
    #print("Final preprocessed and merged dataset: \n", merged_data.head(30))

    merged_data['TimeNum'] = utilities.iso_to_unix(merged_data, 'time_iso')
    merged_data['time_iso'] = pd.to_datetime(merged_data['time_iso'])

    return merged_data


# TODO - write docs for functionality
def get_raw_ediary_signals(sqlite_file_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_file_path)
    count_query = f'SELECT COUNT(*) from sensordata'
    result = pd.read_sql_query(count_query, conn)

    if result['COUNT(*)'][0] == 0:
        print(sqlite_file_path, "has no recordings")

    else:

        GSR_clus, GSR_raw = get_raw_GSR(sqlite_file_path)
        ST_clus, ST_raw = get_raw_ST(sqlite_file_path)
        IBI_raw, HRV = get_raw_IBI_and_HRV(sqlite_file_path)

        return GSR_clus, GSR_raw, ST_clus, ST_raw, IBI_raw, HRV


#print(MOS_detection_signal_preparation(filename))

def get_raw_GSR(filename):

    if sensor_check.E4_used(filename) == True:

        #### GSR
        GSR_cluster, GSR_raw = dl.get_ediary_data(filename = filename, phys_signal = "GSR")
        # TODO - had to uncomment this due to an issue with interpolation - .interpolate cannot handle datetime64[ns] data type
        #GSR_raw['time_iso'] = pd.to_datetime(GSR_raw['time_iso'])
        GSR_raw.columns = ['time_millis', 'time_iso', 'GSR_raw']

        return GSR_cluster, GSR_raw

def get_raw_ST(filename):

    if sensor_check.E4_used(filename) == True:

        #### GSR
        ST_cluster, ST_raw = dl.get_ediary_data(filename = filename, phys_signal = "ST")
        # TODO - check if this is an interpolation issue (same as in GSR)
        #ST_raw['time_iso'] = pd.to_datetime(ST_raw['time_iso'])
        ST_raw.columns = ['time_millis', 'time_iso', 'ST_raw']

        return ST_cluster, ST_raw


def get_raw_IBI_and_HRV(filename):

    if sensor_check.BioHarness_used(filename):
        #### IBI
        IBI_raw = dl.get_ediary_data(filename = filename, phys_signal = "IBI")

        IBI_raw['IBI'] = pps.format_raw_IBI(IBI_raw)

        #### HRV ---> get HRV from preprocessed IBIs
        # TODO - this is the old version (just IBI differences)
        if IBI_raw is not None:
            HRV = pps.extract_HRV_from_IBI(IBI_raw)
        else:
            HRV = None

        return IBI_raw, HRV
