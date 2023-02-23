import numpy as np
import pandas as pd
import math
import re
import itertools
import warnings

#import plotly.express as px
import plotly.graph_objects as go

from MOS_Detection import MOS_signal_preparation as msp
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import utilities

#from MOS_Detection.HRV import hrv_rules


# FIXME - try to fix warnings
warnings.filterwarnings("ignore")



################ Main Algorithm ################
def MOS_main_filepath(filename,
             min_GSR_inc_secs: int = 2, max_GSR_inc_secs: int = 8, weight_rule1: int = 25,
             weight_rule2: int = 25, weight_rule3: int = 25, weight_rule4: int = 25,
             MOSpercentage: int = 75, latency: int = 10, print_number_of_time_rules_are_met: int = False):


    MOS_data_prep = msp.MOS_detection_signal_preparation(filename)

    # Check if 'time_iso' is in datetime64[ns] format
    #print(MOS_data_prep.dtypes)

    if "time" in MOS_data_prep.columns:
        MOS_data_prep.rename(columns={"time": "time_iso"}, inplace=True)

    data = rule_preparation(MOS_data_prep)
    first_derivative_GSR, first_derivative_ST = ST_GSR_derivative_calculation(data)

    GSR_increase_indicator = calculate_binary_increase_indicator_GSR(first_derivative_GSR)
    ST_decrease_indicator = calculate_binary_decrease_indicator_ST(first_derivative_ST)

    # create full dataframe for MOS rule check
    data_r = pd.DataFrame(list(zip(data['time_iso'].values,
                                   data['TimeNum'].values, data['GSR'].values, data['GSR_standardized'],
                                   GSR_increase_indicator, first_derivative_GSR, data['ST'].values,
                                   data['ST_standardized'], ST_decrease_indicator, first_derivative_ST)),
                          columns=['time_iso', 'TimeNum', 'GSR_filtered', 'GSR_std', 'GSR_increase', 'GSR_1st_der',
                                   'ST_filtered', 'ST_std', 'ST_decrease', 'ST_1st_der'])

    #### Rule 1 - GSR Increase Duration ### (2-5 seconds), (5-8 seconds) according to Paper
    data_r1 = R1_GSR_Amplitude_Increase(data_r, min_GSR_inc_secs=min_GSR_inc_secs,
                                              max_GSR_inc_secs=max_GSR_inc_secs)

    data_r1["R1"] = data_r1["R1"] * weight_rule1

    # set "GSR Onset" and "GSR Peak" Variables here
    data_r1["GSR_onset"] = np.where((data_r1['consecutive_GSR_inc'].shift(-1) == 1), 1, 0)
    data_r1["GSR_peak"] = np.where((data_r1['consecutive_GSR_inc'] >= 1) & (data_r1['GSR_increase'].shift(-1) == 0), 1, 0)

    ## Create lag for consecutive ST decrease to compare to GSR onset
    data_r1['ST_3_s_after_GSR'] = np.where((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-3) >= 3),
                                           1, 0)
    # TODO - remove "ST_precondition" because it is redundant
    #data_r1["ST_precondition"] = np.where(data_r1["consecutive_ST_dec"] >= 3, 1, 0)
    data_r1['ST_2-6_s_after_GSR'] = np.where(
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-2) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-4) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-5) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-6) >= 3)), 1, 0)

    #### Rule 2 - ST Decrease of at least 3 seconds 3 seconds or 2-6 seconds after GSR onset (according to paper) ###
    data_r2 = R2_ST_Decrease_after_GSR_Peak_new(data_r1)

    #print("Columns R2", data_r2.columns)

    data_r2["R2"] = data_r2["R2"] * weight_rule2

    # Rule 3 & Rule 4 Preparation
    r3, r4, rel_GSR_slope, GSR_slope = R3_4_mos_angle_and_duration(data_r2)

    data_r3_4 = data_r2.copy()

    #### Rule 3 - GSR increase time between local GSR min and local GSR max #### (1-5 seconds), (5-15 seconds) according to Paper
    data_r3_4['R3'] = r3
    data_r3_4['R4'] = r4
    data_r3_4['R3'] = data_r3_4['R3'] * weight_rule3
    data_r3_4['R4'] = data_r3_4['R4'] * weight_rule4
    data_r3_4['rel_GSR_slope'] = rel_GSR_slope
    data_r3_4['GSR_slope'] = GSR_slope

    data_r1_r2_r3_r4 = shift_rules_to_same_position(data_r3_4)

    if print_number_of_time_rules_are_met:

        print(f"Number of times Rule 1 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R1shifted'] > 0])} \n"
              f"\t (GSR Increase between 2 and 5 seconds (1) or 5 and 8 seconds (0.5)) \n")

        print(f"Number of times Rule 2  was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R2'] > 0])} \n"
              f"\t (ST Decrease of at least 3 seconds, exactly 3 seconds after GSR Onset (1) or 2 and 6 seconds "
              f" after GSR Onset (0.5)) \n")

        print(f"Number of times Rule 3 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R3'] > 0])} \n"
              f"\t (GSR Increase between 1 and 5 seconds (1) or 5 and 15 seconds (0.5)) \n")

        print(f"Number of times Rule 4 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R4'] > 0])} \n"
              f"\t (GSR Increase Slope between local GSR minimum and local GSR maximum"
              f" was greater than 0.1 (1) or greater than 0.08 (0.5)) \n")

        print(f"Number of times all rules were met: "
              f"{len(data_r1_r2_r3_r4[(data_r1_r2_r3_r4['R1shifted'] > 0) & (data_r1_r2_r3_r4['R2'] > 0) & (data_r1_r2_r3_r4['R3'] > 0) & (data_r1_r2_r3_r4['R4'] > 0)])} \n")


    # TODO - continue with Rule 5
    ##### Rule Aggregation #####
    data_r1_r2_r3_r4['MOS_score'] = data_r1_r2_r3_r4[['R1shifted', 'R2', 'R3', 'R4']].sum(axis = 1)


    #### Rule 5: 2 consecutive MOS not within 10 seconds ####
    detected_MOS = data_r1_r2_r3_r4[data_r1_r2_r3_r4['MOS_score'] > MOSpercentage]

    df_r_1_2_3_4_5_met = utilities.check_timestamp_gaps(detected_MOS, duration = latency,
                                                        col_name = "MOS_not_within_10_seconds")

    mos_identified = df_r_1_2_3_4_5_met[df_r_1_2_3_4_5_met['MOS_not_within_10_seconds']==True]

    print("Number of MOS detected based on ST & GSR Rules: ", len(mos_identified))

    final_MOS_output = pd.merge(data_r,
                                mos_identified[['time_iso', 'MOS_score']],
                                on = 'time_iso', how = 'left')

    return final_MOS_output, data_r1_r2_r3_r4


def MOS_main_df(df,
             min_GSR_inc_secs: int = 2, max_GSR_inc_secs: int = 8, weight_rule1: int = 25,
             weight_rule2: int = 25, weight_rule3: int = 25, weight_rule4: int = 25,
             MOSpercentage: int = 75, latency: int = 10, print_number_of_time_rules_are_met: int = False):
    # Check if 'time_iso' is in datetime64[ns] format
    # print(MOS_data_prep.dtypes)

    if "time" in df.columns:
        df.rename(columns={"time": "time_iso"}, inplace=True)

    data = rule_preparation(df)
    first_derivative_GSR, first_derivative_ST = ST_GSR_derivative_calculation(data)

    GSR_increase_indicator = calculate_binary_increase_indicator_GSR(first_derivative_GSR)
    ST_decrease_indicator = calculate_binary_decrease_indicator_ST(first_derivative_ST)

    # create full dataframe for MOS rule check
    data_r = pd.DataFrame(list(zip(data['time_iso'].values,
                                   data['TimeNum'].values, data['GSR'].values, data['GSR_standardized'],
                                   GSR_increase_indicator, first_derivative_GSR, data['ST'].values,
                                   data['ST_standardized'], ST_decrease_indicator, first_derivative_ST)),
                          columns=['time_iso', 'TimeNum', 'GSR_filtered', 'GSR_std', 'GSR_increase', 'GSR_1st_der',
                                   'ST_filtered', 'ST_std', 'ST_decrease', 'ST_1st_der'])

    #### Rule 1 - GSR Increase Duration ### (2-5 seconds), (5-8 seconds) according to Paper
    data_r1 = R1_GSR_Amplitude_Increase(data_r, min_GSR_inc_secs=min_GSR_inc_secs,
                                        max_GSR_inc_secs=max_GSR_inc_secs)

    data_r1["R1"] = data_r1["R1"] * weight_rule1

    # set "GSR Onset" and "GSR Peak" Variables here
    data_r1["GSR_onset"] = np.where((data_r1['consecutive_GSR_inc'].shift(-1) == 1), 1, 0)
    data_r1["GSR_peak"] = np.where((data_r1['consecutive_GSR_inc'] >= 1) & (data_r1['GSR_increase'].shift(-1) == 0), 1,
                                   0)

    ## Create lag for consecutive ST decrease to compare to GSR onset
    data_r1['ST_3_s_after_GSR'] = np.where((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-3) >= 3),
                                           1, 0)
    # TODO - remove "ST_precondition" because it is redundant
    # data_r1["ST_precondition"] = np.where(data_r1["consecutive_ST_dec"] >= 3, 1, 0)
    data_r1['ST_2-6_s_after_GSR'] = np.where(
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-2) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-4) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-5) >= 3)) |
        ((data_r1['GSR_onset'] == 1) & (data_r1['consecutive_ST_dec'].shift(-6) >= 3)), 1, 0)

    #### Rule 2 - ST Decrease of at least 3 seconds 3 seconds or 2-6 seconds after GSR onset (according to paper) ###
    data_r2 = R2_ST_Decrease_after_GSR_Peak_new(data_r1)

    # print("Columns R2", data_r2.columns)

    data_r2["R2"] = data_r2["R2"] * weight_rule2

    # Rule 3 & Rule 4 Preparation
    r3, r4, rel_GSR_slope, GSR_slope = R3_4_mos_angle_and_duration(data_r2)

    data_r3_4 = data_r2.copy()

    #### Rule 3 - GSR increase time between local GSR min and local GSR max #### (1-5 seconds), (5-15 seconds) according to Paper
    data_r3_4['R3'] = r3
    data_r3_4['R4'] = r4
    data_r3_4['R3'] = data_r3_4['R3'] * weight_rule3
    data_r3_4['R4'] = data_r3_4['R4'] * weight_rule4
    data_r3_4['rel_GSR_slope'] = rel_GSR_slope
    data_r3_4['GSR_slope'] = GSR_slope

    data_r1_r2_r3_r4 = shift_rules_to_same_position(data_r3_4)

    if print_number_of_time_rules_are_met:
        print(f"Number of times Rule 1 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R1shifted'] > 0])} \n"
              f"\t (GSR Increase between 2 and 5 seconds (1) or 5 and 8 seconds (0.5)) \n")

        print(f"Number of times Rule 2  was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R2'] > 0])} \n"
              f"\t (ST Decrease of at least 3 seconds, exactly 3 seconds after GSR Onset (1) or 2 and 6 seconds "
              f" after GSR Onset (0.5)) \n")

        print(f"Number of times Rule 3 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R3'] > 0])} \n"
              f"\t (GSR Increase between 1 and 5 seconds (1) or 5 and 15 seconds (0.5)) \n")

        print(f"Number of times Rule 4 was met: "
              f"{len(data_r1_r2_r3_r4[data_r1_r2_r3_r4['R4'] > 0])} \n"
              f"\t (GSR Increase Slope between local GSR minimum and local GSR maximum"
              f" was greater than 0.1 (1) or greater than 0.08 (0.5)) \n")

        print(f"Number of times all rules were met: "
              f"{len(data_r1_r2_r3_r4[(data_r1_r2_r3_r4['R1shifted'] > 0) & (data_r1_r2_r3_r4['R2'] > 0) & (data_r1_r2_r3_r4['R3'] > 0) & (data_r1_r2_r3_r4['R4'] > 0)])} \n")

    # TODO - continue with Rule 5
    ##### Rule Aggregation #####
    data_r1_r2_r3_r4['MOS_score'] = data_r1_r2_r3_r4[['R1shifted', 'R2', 'R3', 'R4']].sum(axis=1)

    #### Rule 5: 2 consecutive MOS not within 10 seconds ####
    detected_MOS = data_r1_r2_r3_r4[data_r1_r2_r3_r4['MOS_score'] > MOSpercentage]

    df_r_1_2_3_4_5_met = utilities.check_timestamp_gaps(detected_MOS, duration=latency,
                                                        col_name="MOS_not_within_10_seconds")

    mos_identified = df_r_1_2_3_4_5_met[df_r_1_2_3_4_5_met['MOS_not_within_10_seconds'] == True]

    print("Number of MOS detected based on ST & GSR Rules: ", len(mos_identified))

    final_MOS_output = pd.merge(data_r,
                                mos_identified[['time_iso', 'MOS_score']],
                                on='time_iso', how='left')

    return final_MOS_output, data_r1_r2_r3_r4


################ Rules ################



def R1_GSR_Amplitude_Increase(data: pd.DataFrame, min_GSR_inc_secs: int,
                              max_GSR_inc_secs: int):

    data_r = data.copy()

    # adding new column which counts the consecutive GSR increases / ST decreases
    data_r['consecutive_GSR_inc'] = get_consecutive_GSR_increase_duration(data_r)
    data_r['consecutive_ST_dec'] = get_consecutive_ST_decrease_duration(data_r)

    data_r['R1'] = np.where( (data_r['consecutive_GSR_inc'] >= min_GSR_inc_secs) & (data_r['consecutive_GSR_inc'] <= 5), 1, 0 ) # >= 2 & <= 5 in paper
    data_r['R1'] = np.where( (data_r['consecutive_GSR_inc'] > 5) & (data_r['consecutive_GSR_inc'] <= max_GSR_inc_secs), 0.5, data_r['R1'] ) # >= 5 & <= 8 in paper

    #data_r['R1'] = data_r['R1'] * weight_rule1

    return data_r

def R2_ST_Decrease_after_GSR_Peak_new(data: pd.DataFrame):

    data_r2 = data.copy()

    data_r2['R2'] = np.where( (data_r2['ST_2-6_s_after_GSR'] == 1), 0.5, 0)
    data_r2['R2'] = np.where( (data_r2['ST_3_s_after_GSR'] == 1), 1, data_r2['R2'])
    #data_r2['R2'] = data_r2['R2'] * weight_rule2

    return data_r2


def R3_4_mos_angle_and_duration(data, min_min_slope_angle: float = 0.08, min_slope_angle: float = 0.1,
                 show_calculations: bool = False):

    data_r = data[["GSR_onset", "GSR_peak", "consecutive_GSR_inc", "GSR_filtered"]]

    rule3 = [0]*len(data)
    rule4 = [0]*len(data)
    rel_GSR_slope = [0]*len(data)
    GSR_slope = [0]*len(data)

    for i in range(0, len(data_r)):
        j = i
        counter = 0
        #rule3 = [0]*len(data)
        #rule4 = [0]*len(data)
        #GSR_slope = [0]*len(data)
        p = np.array([0.0, 0.0])
        q = np.array([0.0, 0.0])
        v = np.array([0.0, 0.0])
        if data["GSR_onset"][i] == 1:
            p[1] = data["GSR_filtered"][i]
            v[1] = p[1]
            #v1[0] = v2[0]

            while data["GSR_peak"][j] != 1 and j != len(data_r) - 1:
                j += 1
                counter += 1
                q[1] = data["GSR_filtered"][j]

            q[0] = counter
            v[0] = counter

            #print(p, q, v)

            v1 = v-p
            v1 = v1/np.linalg.norm(v1)
            v2 = q-p
            v2 = v2/np.linalg.norm(v2)


            ## rule 3 counter
            if counter >= 1 and counter <= 5:
                rule3[i] = 1
            elif counter > 5 and counter <= 15:
                rule3[i] = 0.5

            rel_change = angle(v1, v2)
            rel_GSR_slope[i] = rel_change
            slope = math.degrees(rel_change)
            GSR_slope[i] = slope

            ## rule 4 calc angle from vectors

            if show_calculations:
                print(i,j, counter,  math.degrees(rel_change))


            if slope >= min_slope_angle:
                rule4[i] = 1
            elif (slope >= min_min_slope_angle and slope < min_slope_angle):
                rule4[i] = 0.5

    return rule3, rule4, rel_GSR_slope, GSR_slope


def shift_rules_to_same_position(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'R1shifted' column to DataFrame where all rules that are met are position-corrected for summation of rules

    Parameters
    ----------
    data

    Returns
    -------

    """
    data["R1shifted"] = [0] * len(data)
    r1shift = [0] * len(data)

    for i in data.index:
        if data["R1"][i] > 0:
            secs = int(data["consecutive_GSR_inc"][i])
            r1shift[i - secs] = data["R1"][i]

    data["R1shifted"] = r1shift
    return data



################ Helpers ################


def rule_preparation(data: pd.DataFrame):
    # standardize features
    data = pps.standardize_filtered_signal(data, "GSR", "GSR")
    data = pps.standardize_filtered_signal(data, "ST", "ST")
    # output: preprocessed dataset with columns: {TimeNum, GSR, ST, time, stress, GSR_standardized, ST_standardized}
    if 'time' in data.columns:
        data.rename(columns={'time': 'time_iso'}, inplace=True)

    # TODO - check if 'time_millis' rename to 'TimeNum' might also be required

    data['time_iso'] = pd.to_datetime(data['time_iso'])

    return data

def ST_GSR_derivative_calculation(data: pd.DataFrame):
    # calculate first derivative of GSR and ST w.r.t. time and store values in list
    first_derivative_GSR = calculate_first_derivative_wrt_time(data, signal_column_name="GSR",
                                                                   time_column_name="TimeNum")
    first_derivative_ST = calculate_first_derivative_wrt_time(data, signal_column_name="ST",
                                                                  time_column_name="TimeNum")

    return first_derivative_GSR, first_derivative_ST



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

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def angle(v1, v2):
    return math.acos((np.dot(v1,v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))




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

def get_consecutive_ST_decrease_duration_old(df):
    ds = df.copy()
    ds['ST_decrease'].replace(0, np.nan, inplace = True)

    ds['ST_dec_2sec'] = ds['ST_decrease'].rolling(window = 2, min_periods=2).sum()
    ds['ST_dec_3sec'] = ds['ST_decrease'].rolling(window = 3, min_periods=3).sum()
    ds['ST_dec_4sec'] = ds['ST_decrease'].rolling(window = 4, min_periods=4).sum()
    ds['ST_dec_5sec'] = ds['ST_decrease'].rolling(window = 5, min_periods=5).sum()
    ds['ST_dec_6sec'] = ds['ST_decrease'].rolling(window = 6, min_periods=6).sum()
    ds['ST_dec_7sec'] = ds['ST_decrease'].rolling(window = 7, min_periods=7).sum()
    ds['ST_dec_8sec'] = ds['ST_decrease'].rolling(window = 8, min_periods=8).sum()
    ds['ST_dec_9sec'] = ds['ST_decrease'].rolling(window = 9, min_periods=9).sum()
    ds['ST_dec_10sec'] = ds['ST_decrease'].rolling(window = 10, min_periods=10).sum()
    ds['ST_dec_11sec'] = ds['ST_decrease'].rolling(window = 11, min_periods=11).sum()
    ds['ST_dec_12sec'] = ds['ST_decrease'].rolling(window = 12, min_periods=12).sum()
    ds['ST_dec_13sec'] = ds['ST_decrease'].rolling(window = 13, min_periods=13).sum()

    ds['ST_dec_2sec'].fillna(ds['ST_decrease'], inplace = True)
    ds['ST_dec_3sec'].fillna(ds['ST_dec_2sec'], inplace = True)
    ds['ST_dec_4sec'].fillna(ds['ST_dec_3sec'], inplace = True)
    ds['ST_dec_5sec'].fillna(ds['ST_dec_4sec'], inplace = True)
    ds['ST_dec_6sec'].fillna(ds['ST_dec_5sec'], inplace = True)
    ds['ST_dec_7sec'].fillna(ds['ST_dec_6sec'], inplace = True)
    ds['ST_dec_8sec'].fillna(ds['ST_dec_7sec'], inplace = True)
    ds['ST_dec_9sec'].fillna(ds['ST_dec_8sec'], inplace = True)
    ds['ST_dec_10sec'].fillna(ds['ST_dec_9sec'], inplace = True)
    ds['ST_dec_11sec'].fillna(ds['ST_dec_10sec'], inplace = True)
    ds['ST_dec_12sec'].fillna(ds['ST_dec_11sec'], inplace = True)
    ds['ST_dec_13sec'].fillna(ds['ST_dec_12sec'], inplace = True)

    #ds.rename(columns = {'GSR_inc_6sec': 'GSR_increase_duration'})

    return ds['ST_dec_13sec']
