import math
import os
import sqlite3
import zipfile
from scipy.signal import butter, lfilter

import numpy as np
import pandas as pd

#import src.HumanSensing_Preprocessing.utilities as utilities
from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import data_loader as dl
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check

################### Helpers ###################

def check_and_unzip_folder(rootdir: str, logfile: str = None):
    try:
        with zipfile.ZipFile(rootdir,'r') as zip_ref:
            outdir = os.path.join(os.path.dirname(rootdir),os.path.basename(rootdir).split(".zip")[0])
            zip_ref.extractall(outdir)
        rootdir = outdir
        print("Specified path is a zipfile, files extracted to {}".format(rootdir))
    except Exception as e:
        print(e)
    if os.path.isdir(rootdir):
        print("Directory to search for SQLite files: {}".format(rootdir))
    else:
        raise Exception("Not a directory: {}".format(rootdir))
    ##check if a logfile was given
    if os.path.isfile(logfile) == False:
        print("Caution: No path for logfile detected!")


def find_all_sqlite_files(folder_path: str) -> list:
    # find all sqlite files in SQLITE_PATH
    sqlite_files = []
    # check if specified path is already an .sqlite file
    if os.path.isfile(folder_path):
        sqlite_files.append(folder_path)

    else:
        for root, dirs, files in os.walk(folder_path):
            # print("Root: \n", root)
            # print("Directory: \n", dirs)
            # print('Files: \n', files)
            for file in files:
                if file.endswith('.sqlite') or file.endswith('.sqlite3'):
                    sqlite_files.append(os.path.join(root, file))

    print(f"Found {len(sqlite_files)} sqlite ediary files. \n")
    return sqlite_files

def find_and_store_sqlite_file_paths(folder_path):

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    return all_sqlite_files


################### Signal Preparation for MOS Detection ###################

def MOS_detection_signal_preparation(filename, starttime = None):

    print("Empatica E4 Check:", sensor_check.E4_used(filename))

    if sensor_check.E4_used(filename) == True:

        #### GSR
        GSR_cluster, GSR_raw = dl.get_ediary_data(filename = filename, phys_signal = "GSR")

        # all in one
        GSR = pps.GSR_preprocessing(GSR_cluster = GSR_cluster,
                                    GSR_raw = GSR_raw,
                                    phys_signal = "GSR")

        #print(f"GSR Signal after GSR_preprocessing() {GSR.head()}")



        #### ST
        ST_cluster, ST_raw = dl.get_ediary_data(filename = filename, phys_signal = "ST")

        ST = pps.ST_preprocessing(ST_cluster = ST_cluster,
                                  ST_raw = ST_raw,
                                  phys_signal = "ST")

        #print(f"ST Signal after ST_preprocessing() {ST.head()}")


    else:
        print("Make sure to check if Empatica E4 sensor was connected properly.")

    print("BioHarness Check:", sensor_check.BioHarness_used(filename))

    if sensor_check.BioHarness_used(filename):
        #### IBI
        #try:
        IBI_raw = dl.get_ediary_data(filename = filename, phys_signal = "IBI")

        IBI_raw['IBI'] = pps.format_raw_IBI(IBI_raw)

        if IBI_raw is not None:
            IBI = pps.IBI_preprocessing(IBI_raw)
        else:
            IBI = IBI_raw

        #### HRV ---> get HRV from preprocessed IBIs
        # TODO - this is the old version (just IBI differences)
        if IBI is not None:
            HRV = HRV_preprocessing(IBI)
        else:
            HRV = None

        #if HRV is not None:
            #print("HRV prep successful", HRV)


    # TODO load & preprocess ECG data
    #ECG_raw = dl.get_ediary_data(filename=filename, phys_signal="ECG")
    #ECG = pps.ECG_resampling(ECG_raw)


    # TODO - include IBI from E4 here -- as Max did in LabTest class
    else:
        IBI = None

    if IBI is None:
        merged_data = pps.merge_signals(GSR, ST, merge_col = 'time_iso')

    else:
        merged_data = pps.merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')

    # fix timestamp with given start-time
    if starttime is not None:
        # print(merged_data.time_iso[0])
        hour_diff = starttime.hour - merged_data.time_iso[0].hour
        merged_data.time_iso = merged_data.time_iso + pd.Timedelta(hour_diff, unit='hour')

    merged_data.fillna(method="ffill", inplace=True)
    merged_data.fillna(0, inplace=True)

    merged_data['TimeNum'] = utilities.iso_to_unix(merged_data, 'time_iso')
    merged_data['time_iso'] = pd.to_datetime(merged_data['time_iso'])

    return merged_data


def HRV_preprocessing(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Heart Rate Variability (HRV) from formatted, filtered Inter-Beat Intervals (IBIs)
    Returns resampled (1 Hz) HRV values
    :param IBI_data: DF containing formatted, filtered Inter-Beat Intervals
    :return: DataFrame containing extracted and interpolated HRV signal (based on clean IBI data)
    """
    tmp = IBI_data.copy()
    tmp["HRV"] = np.abs(IBI_data.IBI.diff(1).fillna(0))

    return tmp[["time_iso", "HRV"]]



def derive_GSR_and_ST_features(data: pd.DataFrame):
    # standardize features
    data = pps.standardize_filtered_signal(data, "GSR", "GSR")
    data = pps.standardize_filtered_signal(data, "ST", "ST")
    # output: preprocessed dataset with columns: {TimeNum, GSR, ST, time, stress, GSR_standardized, ST_standardized}
    if 'time' in data.columns:
        data.rename(columns={'time': 'time_iso'}, inplace=True)

    # TODO - check if 'time_millis' rename to 'TimeNum' might also be required

    data['time_iso'] = pd.to_datetime(data['time_iso'])

    # calculate first derivative of GSR and ST w.r.t. time and store values in list
    first_derivative_GSR = calculate_first_derivative_wrt_time(data, signal_column_name="GSR",
                                                                   time_column_name="TimeNum")
    data["GSR_1st_der"] = first_derivative_GSR
    first_derivative_ST = calculate_first_derivative_wrt_time(data, signal_column_name="ST",
                                                                  time_column_name="TimeNum")

    data["ST_1st_der"] = first_derivative_ST

    GSR_increase_indicator = calculate_binary_increase_indicator_GSR(first_derivative_GSR)
    data["GSR_increase"] = GSR_increase_indicator
    ST_decrease_indicator = calculate_binary_decrease_indicator_ST(first_derivative_ST)
    data["ST_decrease"] = ST_decrease_indicator

    data["consecutive_GSR_inc"] = get_consecutive_GSR_increase_duration(data)
    data["consecutive_ST_dec"] = get_consecutive_ST_decrease_duration(data)

    # set "GSR Onset" and "GSR Peak" Variables here
    data["GSR_onset"] = np.where((data['consecutive_GSR_inc'].shift(-1) == 1), 1, 0)
    data["GSR_peak"] = np.where((data['consecutive_GSR_inc'] >= 1) & (data['GSR_increase'].shift(-1) == 0), 1,
                                   0)
    return data

def derive_IBI_and_HRV_features(data: pd.DataFrame) -> pd.DataFrame:

    new_df = pd.DataFrame()
    new_df = pd.concat([new_df, data])

    b, a = butter(2, 0.15, "low")
    tmp = lfilter(b, a, new_df.IBI.values)

    b, a = butter(2, 0.05, "high")
    new_df["filtered_IBI"] = lfilter(b, a, tmp)

    new_df["hrv_filtered"] = np.abs(new_df["IBI"].diff(1))
    new_df["sdnn"] = new_df["hrv_filtered"].rolling(2).std().rolling(5).mean().fillna(0)
    new_df["rmsnn"] = new_df["hrv_filtered"].rolling(2).var().rolling(5).mean().agg(math.sqrt).fillna(0)

    return new_df


def calc_sdnn_from_ibi(dataframe: pd.DataFrame, std_window=2, mean_window=5):
    "calc the mean of standard deviations from two Interbeat Intervall differences r_n+1 - r_n"
    #dataframe["hrv"] = np.abs(dataframe["IBI"].diff(1)).fillna(0)
    return dataframe["hrv"].rolling(std_window).std().fillna(0).rolling(mean_window).mean()


def calculate_signal_baseline(df, phys_signal: str = "GSR", mins: int = 5):

    data = df.copy()

    phys_signals_available = ["GSR_std", "ST_std", "GSR", "ST", "IBI", "HRV", "BVP"]

    baseline_calc_duration = pd.to_timedelta(mins, unit = "m")
    start_time = data["time_iso"].min()
    #print(start_time)
    data.set_index("time_iso", inplace=True)
    end_time = start_time + baseline_calc_duration
    #print(end_time)
    baseline_data = data[start_time:end_time]

    data.reset_index(inplace=True)

    if phys_signal in phys_signals_available:
        baseline_mean = baseline_data[phys_signal].mean()

    return baseline_mean


def calculate_first_derivative_wrt_time(df: pd.DataFrame, signal_column_name: str, time_column_name: str) -> pd.DataFrame:
    """
    Takes a DataFrame and two column names as arguments, and calculates the differences physiological signals and recorded timestamps
    (approximates first derivative of 'column_name' w.r.t. 'time_column_name')
    :return: numpy array containing first derivative of physiological signals w.r.t. time
    """

    #calculate first derivative of GSR or ST w.r.t. time
    first_derivative = (np.diff(df[signal_column_name].tolist())) / (np.diff(df[time_column_name].tolist()))
    first_derivative = np.insert(first_derivative, 0, 0)
    return first_derivative

def ST_GSR_derivative_calculation(data: pd.DataFrame):
    # calculate first derivative of GSR and ST w.r.t. time and store values in list
    first_derivative_GSR = calculate_first_derivative_wrt_time(data, signal_column_name="GSR",
                                                                   time_column_name="TimeNum")
    first_derivative_ST = calculate_first_derivative_wrt_time(data, signal_column_name="ST",
                                                                  time_column_name="TimeNum")

    return first_derivative_GSR, first_derivative_ST

def calculate_binary_increase_indicator_GSR(first_derivative_GSR: np.ndarray) -> list:

    positive_GSR_indicator = []

    for derivative in first_derivative_GSR:
        # NOTE: if np.isnan is redundant because np.nan > 0 --> False
        if derivative >= 0:
            positive_GSR_indicator.append(1)
        else:
            positive_GSR_indicator.append(0)

    return positive_GSR_indicator

def calculate_binary_decrease_indicator_ST(first_derivative_ST: np.ndarray) -> list:

    negative_ST_indicator = []

    for derivative in first_derivative_ST:
        # TODO - np.isnan check is redundant
        if derivative <= 0:
            negative_ST_indicator.append(1)
        else:
            negative_ST_indicator.append(0)

    return negative_ST_indicator



def get_consecutive_GSR_increase_duration(df, min: int = 1, max: int = 20):
    ds = df.copy()

    increase = [None] * len(ds)
    idx = 0
    tmp_gsr_value = 1
    for gsr_value in ds["GSR_increase"]:
        if gsr_value == 1 and tmp_gsr_value != 0:
            tmp_gsr_value += 1
        elif gsr_value == 1 and tmp_gsr_value == 0:
            tmp_gsr_value = 1
        else:
            tmp_gsr_value = 0
        increase[idx] = tmp_gsr_value
        idx += 1

    df["increase"] = increase


    ds['GSR_increase'].replace(0, np.nan, inplace = True)

    for i in range(min, max + 1):
        ds[f"GSR_inc_{i}sec"] = ds['GSR_increase'].rolling(window = i, min_periods=i).sum()

    ds['GSR_inc_2sec'].fillna(ds['GSR_increase'], inplace = True)

    for j in range(min + 1, max + 1):
        ds[f'GSR_inc_{j}sec'].fillna(ds[f'GSR_inc_{j-1}sec'], inplace = True)

    return ds[f'GSR_inc_{max}sec']



def get_consecutive_ST_decrease_duration(df, min: int = 2, max: int = 90):
    ds = df.copy()
    ds['ST_decrease'].replace(0, np.nan, inplace = True)

    for i in range(min, max + 1):
        ds[f"ST_dec_{i}sec"] = ds['ST_decrease'].rolling(window = i, min_periods=i).sum()

    ds['ST_dec_2sec'].fillna(ds['ST_decrease'], inplace = True)

    for j in range(min + 1, max + 1):
        ds[f'ST_dec_{j}sec'].fillna(ds[f'ST_dec_{j-1}sec'], inplace = True)

    #ds.rename(columns = {'GSR_inc_6sec': 'GSR_increase_duration'})

    return ds[f'ST_dec_{max}sec']

#TODO
def check_decrease_sum_ST_based_on_GSR_onset(df):

    data = df.copy()

    for i in data.index:
        if data["GSR_onset"] == 1:
            window = data[i:i+10]
            if window["ST_decrease"].sum() <= 5:
                # remove GSR precondition
                pass


    pass