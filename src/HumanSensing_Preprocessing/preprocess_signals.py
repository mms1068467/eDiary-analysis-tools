"""
Methods for pre-processing physiological data recorded by eDiary app
(Inter-Beat-Interval (IBI), Heart Rate Variability (HRV),
Galvanic Skin Respone (GSR), Skin Temperature (ST), tbc... )
"""

import math
import pandas as pd
import numpy as np
import datetime
import typing
import sqlite3
from scipy import signal  # use scipy.signal.lfilter() function instead of scipy.singal.filtfilt() function --> fixes drops in GSR
from scipy.signal import butter #
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing.lookup_tables import freq_dic, cluster_size_dic, interval_dic, butter_filter_order_dic, lowpass_cutoff_freq_dic, highpass_cutoff_freq_dic
import HumanSensing_Preprocessing.data_loader as dl
from HumanSensing_Preprocessing import sensor_check


#### general (signal-independent)

# downsampling filtered data from 4Hz to 1Hz




def resampling(dataframe: pd.DataFrame, columnName: str, interpolationMethod="linear") -> pd.DataFrame:
    """
    Resample (downsample) DataFrame of 'IBI', 'GSR', 'ST' or 'HRV' measurements to 1 Hz Frequency (from 4 Hz) using
    mean() as aggregation method and linear interpolation of NaN values

    :param dataframe: DataFrame containing preprocessed physiological signals ('IBI', 'GSR', 'ST', or 'HRV')
    :param columnName: String abbreviation of physiological signal ('IBI', 'GSR', 'ST', or 'HRV')
    :param interpolationMethod: interpolation method to use (defaults to 'linear')
    :return: Downsampled DataFrame (from 4 Hz to 1 Hz) containing physiological measurements in 1 Hz

    """
    if dataframe is None:
        return None

    selecteddata = dataframe[["time_millis", "value_real"]]
    selecteddata = selecteddata.set_index('time_millis')

    upsampled = selecteddata.resample("1S").mean()
    interpolated = upsampled.interpolate(method=interpolationMethod, order=1)
    export_dataframe = interpolated.reset_index()
    export_dataframe.rename(columns={'value_real': columnName, "time_millis": "time_iso"}, inplace=True)
    export_dataframe = export_dataframe[["time_iso", columnName]]

    return export_dataframe


# TODO - Merge Data on common time_iso (check if there is a loss of information because of upsampling)
def merge_signals(GSR: pd.DataFrame = None, ST: pd.DataFrame = None,
                  IBI: pd.DataFrame = None, HRV: pd.DataFrame = None,
                  merge_col='time_iso') -> pd.DataFrame:
    """
    Merge individual DataFrames containing physiological signals to one DataFrame

    :param GSR: preprocessed DF containing GSR measurements
    :param ST: preprocessed DF containing GSR measurements
    :param IBI: preprocessed DF containing GSR measurements
    :param HRV: preprocessed DF containing GSR measurements
    :param merge_col: common column to merge DFs on (default = 'time_iso')

    :return: DataFrame with measurements from eDiary app
    """

    ediary_data = pd.merge(GSR, ST, on=merge_col)
    if IBI is not None:
        ediary_data = pd.merge(ediary_data, IBI, on=merge_col)
        ediary_data = pd.merge(ediary_data, HRV, on=merge_col)

    return ediary_data

# TODO - Merge Data on common time_iso (check if there is a loss of information because of upsampling)
def merge_raw_signals(GSR: pd.DataFrame = None, ST: pd.DataFrame = None,
                  IBI: pd.DataFrame = None, HRV: pd.DataFrame = None,
                  merge_col='time_iso') -> pd.DataFrame:
    """
    Merge individual DataFrames containing physiological signals to one DataFrame

    :param GSR: preprocessed DF containing GSR measurements
    :param ST: preprocessed DF containing GSR measurements
    :param IBI: preprocessed DF containing GSR measurements
    :param HRV: preprocessed DF containing GSR measurements
    :param merge_col: common column to merge DFs on (default = 'time_iso')

    :return: DataFrame with measurements from eDiary app
    """

    ediary_data = pd.merge(GSR, ST, on=merge_col)
    if IBI is not None:
        ediary_data = pd.merge(ediary_data, IBI, on=merge_col)
        ediary_data = pd.merge(ediary_data, HRV, on=merge_col)

    return ediary_data


def label_stressmoments(data: pd.DataFrame, starttime: datetime.datetime = None,
                        stresstimes: typing.Tuple[int, ...] = None) -> pd.DataFrame:
    """
    Create new column representing the labeled stress moments (based on seconds after starting time)
    (labels / ground-truth for stress detection)

    :param data: DataFrame containing physiological data from eDiary App
    :param starttime: starting time of the recording (datetime format)
    :param stresstimes: tuple of integers indicating after how many seconds stress moments were induced

    :return: DataFrame with ground-truth labels of stress moments
    """

    if stresstimes is None:
        raise Exception("""stresstimes argument must have at least one value(s)
         for labeling times when stress moments occured (in Seconds from beginning) """)

    data['stress'] = 0

    data["time_iso"] = pd.to_datetime(data["time_iso"])

    for i in data.index:
        # loop through tuples of stress times (given in seconds)
        for sec in stresstimes:
            # set boolean variable for stress if stress time ground-truth occurs at specific timestamp
            if data.time_iso[i] == starttime + datetime.timedelta(seconds=sec):
                data["stress"][i] = 1
                break

    return data


#### IBI
# TODO - changed / added split_by argument --> check if this works
def format_raw_IBI(raw_IBI: pd.DataFrame, split_by=';') -> pd.DataFrame:
    """
    Takes raw IBI signal (format 65018;65018;65018;65018;65018;65018;65018;6501...),
     adjusts timestamp (wrongly recorded by eDiary app),
      and formats IBI values to be in one Pandas DF column

    :param raw_IBI: DataFrame containing IBI signal in format 65018;65018;65018;65018;65018;65018;65018;6501 (split by ;)
    :return: Filtered IBI values DataFrame
    """

    data = raw_IBI.copy()

    data = utilities.adjust_IBI_timestamp(data, unix_time_col='time_millis', iso_time_col='time_iso')

    # split measurements by ";" or any other delimiter specified in 'split_by' argument and stores them in Series
    data_formatted = utilities.split_measurements(data, values_col='value_text', new_col_name='value_real',
                                                  split_by=split_by)

    return data_formatted


# TODO: start time could be extra function
def filter_IBI(raw_IBI: pd.DataFrame,
               starttime: datetime.datetime = None) -> pd.DataFrame:
    """
    Takes raw, unformatted IBI signals (format 65018;65018;65018;65018;65018;65018;65018;6501...)
     and an optional starting time (starttime parameter) and returns filtered IBI signal
     (IBI within a range of 260 to 1500 milliseconds) with adjusted timestamp
    (Heart Beat Assumption - minimum HR = 40, maximum HR = 230 ---> 260 - 1500 ms IBIs)

    :param raw_IBI:
    :return: Filtered IBI values DataFrame
    """

    data = raw_IBI.copy()

    # convert time_millis (in UNIX format) to ISO datetime format (milliseconds)
    # data['time_millis'] = pd.to_datetime(data['time_millis'], unit = 'ms')
    data['time_millis'] = utilities.unix_to_iso_ms(data=data, col_name='time_millis')

    # setting time with time_iso
    if starttime is None:
        data['time_millis'] = data['time_millis'] + (pd.to_datetime(data['time_iso'][0]) - data['time_millis'][0])

    # set the right time given starttime
    else:
        time_diff = starttime.hour - data['time_millis'][0].hour
        data['time_millis'] = data['time_millis'] + pd.Timedelta(time_diff, unit='hour')

    # range of valid IBIs (260 - 1500 ms) - based on Heart Rate Assumption
    # Assumption is that minimum HR = 40 and maximum HR = 230
    ibi_values = []
    data['value_real'] = utilities.split_and_find_IBI(data)

    return data


def interpolate_IBI(filtered_IBI: pd.DataFrame,
                    interpolation_method: str = "linear",
                    interpolation_order: int = 1) -> pd.DataFrame:
    """
    Takes filtered IBI signals and interpolates missing values based on timestamp
    (linear interpolation)

    :param filtered_IBI: DF containing filtered IBIs (within a range of 260 - 1500 ms)
    :param interpolation_method: Interpolation Method to use (default = "linear")
    :param interpolation_order: Interpolation Order to use (default = 1)
    :return: DataFrame containing filtered and interpolated IBI signals

    """

    filtered_IBI = filtered_IBI.set_index('time_millis')
    filtered_IBI = filtered_IBI.interpolate(method=interpolation_method,
                                            order=interpolation_order)

    filtered_interpolated_IBI = filtered_IBI.reset_index()

    filtered_interpolated_IBI.drop(['value_text'], axis=1, inplace=True)

    return filtered_interpolated_IBI


def IBI_preprocessing(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to do all IBI preprocessing steps (filtering, interpolating, resampling)
     in one go

    :return: DataFrame containing preprocessed IBI signal
    """

    #TODO - add check for IBI

    if "IBI" in IBI_data:
        filtered_IBI_df = filter_IBI(raw_IBI = IBI_data)
        prepared_IBI = interpolate_IBI(filtered_IBI_df)
        #final_IBI = interpolate_IBI(filtered_IBI_df)
        final_IBI = resampling(prepared_IBI, "IBI")

        return final_IBI

    else:
        return IBI_data

def IBI_preprocessing_no_IBI_check(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to do all IBI preprocessing steps (filtering, interpolating, resampling)
     in one go

    :return: DataFrame containing preprocessed IBI signal
    """

    #TODO - add check for IBI

    filtered_IBI_df = filter_IBI(raw_IBI = IBI_data)
    prepared_IBI = interpolate_IBI(filtered_IBI_df)
    #final_IBI = interpolate_IBI(filtered_IBI_df)
    final_IBI = resampling(prepared_IBI, "IBI")

    return final_IBI



#### HRV

def get_HRV_from_IBI(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HRV from IBIs and return DataFrame holding HRV values
    (HRV = difference between IBIs = difference between to successive heart beats)

    Heart beats are identified by 2 successive R peaks (also called R-R Interval)

    :param ibi_data: DataFrame containing Inter-Beat-Intervals (IBIs)
    :return: DataFrame containing Heart Rate Variability (HRV) measurements
    """

    if IBI_data is None:
        return None

    data = IBI_data.copy()

    # drop first row
    data.drop(index=data.index[0], axis=0, inplace=True)
    HRV_data = data.reset_index()
    HRV_data.drop(['index'], axis=1, inplace=True)

    # HRV = difference between IBIs (= R-R Interval) --> difference between 1st and 2nd IBI value
    # first HRV = set to second timestamp (index = 1) from original IBI data
    for i in range(0, len(IBI_data) - 1):
        HRV = abs(IBI_data.IBI[i] - IBI_data.IBI[i + 1])
        HRV_data.loc[i, 'value_real'] = HRV if HRV > 0 else np.nan

    # TODO - split HRV interpolation up into own function
    # just to have consistent format for resampling method
    HRV_data.rename(columns={"time_iso": "time_millis"}, inplace=True)
    HRV_data = HRV_data.set_index('time_millis')
    # HRV_data = HRV_data.set_index('time_iso')

    HRV_data = HRV_data.interpolate(method='linear', order=1)
    export_HRV_data = HRV_data.reset_index()

    return export_HRV_data

def extract_HRV_from_IBI(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HRV from IBIs and return DataFrame holding HRV values
    (HRV = difference between IBIs = difference between to successive heart beats)

    Heart beats are identified by 2 successive R peaks (also called R-R Interval)

    :param ibi_data: DataFrame containing Inter-Beat-Intervals (IBIs)
    :return: DataFrame containing Heart Rate Variability (HRV) measurements
    """

    if IBI_data is None:
        return None

    data = IBI_data.copy()

    # drop first row
    data.drop(index=data.index[0], axis=0, inplace=True)
    HRV_data = data.reset_index()
    HRV_data.drop(['index'], axis=1, inplace=True)

    # HRV = difference between IBIs (= R-R Interval) --> difference between 1st and 2nd IBI value
    # first HRV = set to second timestamp (index = 1) from original IBI data
    for i in range(0, len(IBI_data) - 1):
        HRV = abs(IBI_data.IBI[i] - IBI_data.IBI[i + 1])
        HRV_data.loc[i, 'value_real'] = HRV if HRV > 0 else np.nan

    # TODO - split HRV interpolation up into own function
    # just to have consistent format for resampling method
    #HRV_data.rename(columns={"time_iso": "time_millis"}, inplace=True)
    HRV_data = HRV_data.set_index('time_millis')
    # HRV_data = HRV_data.set_index('time_iso')

    HRV_data = HRV_data.interpolate(method='linear', order=1)
    export_HRV_data = HRV_data.reset_index()
    export_HRV_data.rename(columns={"value_real": "HRV"}, inplace = True)

    return export_HRV_data


def HRV_preprocessing(IBI_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Heart Rate Variability (HRV) from formatted, filtered Inter-Beat Intervals (IBIs)
    Returns resampled (1 Hz) HRV values
    :param IBI_data: DF containing formatted, filtered Inter-Beat Intervals
    :return: DataFrame containing extracted and interpolated HRV signal (based on clean IBI data)
    """
    HRV = get_HRV_from_IBI(IBI_data)
    final_HRV = resampling(HRV, "HRV")

    return final_HRV

#### GSR

def GSR_preprocessing(GSR_cluster: pd.DataFrame, GSR_raw: pd.DataFrame, phys_signal: str = "GSR",
                      starting_time=None) -> pd.DataFrame:
    """
    Function to do all GSR preprocessing steps (filtering, interpolating, resampling)
     in one go
    :param GSR_cluster: DataFrame holding all clustered timestamps of GSR signal
    :param GSR_raw: DataFrame holding raw GSR measurements
    :param phys_signal: physiological signal (default = "GSR")
    :param starting_time: optional starting time
    :return: DataFrame containing preprocessed GSR signal
    """

    # preprocess GSR signal -> fix timestamp issue (clusters of timestamps)
    GSR_preprocessed = preprocess_GSR_ST(GSR_cluster, GSR_raw, phys_signal=phys_signal,
                                         starttime=starting_time)
    GSR_filtered = filter_GSR_ST(GSR_preprocessed, phys_signal=phys_signal)
    final_GSR = resampling(GSR_filtered, columnName=phys_signal)

    return final_GSR


#### ST

def ST_preprocessing(ST_cluster: pd.DataFrame, ST_raw: pd.DataFrame, phys_signal="ST",
                     starting_time=None) -> pd.DataFrame:
    """
    Function to do all IBI preprocessing steps (filtering, interpolating, resampling)
     in one go
    :param ST_raw: DataFrame holding raw GSR measurements
    :param ST_cluster: DataFrame holding all clustered timestamps of GSR signal
    :param phys_signal: physiological signal (default = "GSR")
    :param starting_time: optional starting time
    :return: DataFrame containing preprocessed ST signal
    """

    # preprocess ST signal -> fix timestamp issue (clusters of timestamps)
    ST_preprocessed = preprocess_GSR_ST(ST_cluster, ST_raw, phys_signal=phys_signal,
                                        starttime=starting_time)
    ST_filtered = filter_GSR_ST(ST_preprocessed, phys_signal=phys_signal)
    final_ST = resampling(ST_filtered, columnName=phys_signal)

    return final_ST


# TODO: start time could be extra function
def preprocess_GSR_ST(clustered_data: pd.DataFrame, raw_data: pd.DataFrame,
                      phys_signal: str, starttime: datetime.datetime = None) -> pd.DataFrame:
    """
    Resampling (Up-sampling data) to be in 4 Hz, where measurement values coming in with equal timestamps are referred to as clusters
    of measurements and will be spread out over the specific timeframe that is missing (determined by previous timestamp, next timestamp,
    and the number of measurements falling into the cluster)
    Resulting NaN values will be interpolated linearly

    :param clustered_data: cluster refers to the "clusters" of measurements that have the same timestamp (these vary in size)
    :param raw_data: raw GSR and ST measurement
    :param phys_signal: string representing one of the signals ("GSR", "ST")
    :param starttime: optional starting timestamp

    :return: Upsampled and interpolated DataFrame containing GSR and ST measurements (4Hz)
    """

    # variable to hold number of buckets since the last hole was detected
    number_of_buckets_since_hole = -1

    # saves the number of values which are stored with a delayed timestamp and wrong values
    # (values are supplied by sensor and should be missing values from the whole before)
    number_of_values_for_hole = -1

    # numberOfClusters = len(clustered_data.cluster_time_millis)
    numberOfClusters = len(clustered_data.time_millis)

    # TODO - counter for??
    # number of values in a bucket
    j = 1

    # loop through all timestamp clusters (measurements with timestamps that are equal)
    for i in range(numberOfClusters - 1):

        # -------------------------------------------------------------------------------------------------------------
        # ----------- DELETE WRONG VALUES FROM A HOLE -----------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        # add 1 or 0 to determine how many timestamps fall into cluster after a hole
        if number_of_buckets_since_hole >= 0:
            number_of_buckets_since_hole += 1
        else:
            number_of_buckets_since_hole += 0

        if number_of_buckets_since_hole > 2:
            number_of_buckets_since_hole = -1
            number_of_values_for_hole = -1

        # if delayed values, which are missing in the hole 1 or 2 buckets before will be deleted (wrong values)
        elif 0 <= number_of_buckets_since_hole <= 2:
            # TODO -  number_of_values_for_hole is initialized with -1
            #  --> check how it is incremented
            if clustered_data.cluster_size[i] >= number_of_values_for_hole:
                # TODO - check labels argument and the specified range --> counter j is used
                raw_data = raw_data.drop(labels=range(j - 1,
                                                      j - 1 + clustered_data.cluster_size[i] - 6),
                                         axis=0)
                raw_data = raw_data.reset_index(drop=True)

                # GSR is normally in 1500 ms interval
                time_interval = clustered_data['time_millis'][i + 1] - clustered_data['time_millis'][i]
                # normally, every 250 ms one value for GSR
                value_interval = time_interval / 6

                # TODO - check what for_end is
                for_end = j + 5
                for j in range(j, for_end):
                    raw_data.loc[j, 'time_millis'] = raw_data['time_millis'][j - 1] + value_interval

                # jump to second value of a bucket
                j += 2
                continue

        # -------------------------------------------------------------------------------------------------------------
        # ----------- CLUSTER SIZE & INTERVAL CALCULATIONS ------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        # TODO - determine cluster interval ???
        # GSR = 1500 ms
        ist_cluster_interval = clustered_data['time_millis'][i + 1] - clustered_data['time_millis'][i]
        # GSR > 1250 ms
        soll_cluster_interval_min = (1000 / freq_dic[phys_signal]) * (cluster_size_dic[phys_signal] - 1)
        # GSR < 1750 ms
        soll_cluster_interval_max = (1000 / freq_dic[phys_signal]) * (cluster_size_dic[phys_signal] + 1)

        # normal GSR time interval = 1500 ms
        time_interval = clustered_data['time_millis'][i + 1] - clustered_data['time_millis'][i]
        # GSR measurements usually arrive every 250 ms (one value every 250 ms)
        value_interval = time_interval / clustered_data.cluster_size[i]
        # value_interval = time_interval / clustered_data['cluster_size'][i]

        # -------------------------------------------------------------------------------------------------------------
        # ----------- STANDARD CLUSTER SIZE----------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        # Standard cluster size - GSR: 1250 < ist_cluster_interval < 1750
        if soll_cluster_interval_min < ist_cluster_interval < soll_cluster_interval_max:
            # GSR: bucket size = 6
            if (clustered_data['cluster_size'][i] >= cluster_size_dic[phys_signal]):
                # set new time_millis for every value with value_interval
                for_end = j + clustered_data['cluster_size'][i] - 1

                for j in range(j, for_end):
                    raw_data.loc[j, 'time_millis'] = raw_data['time_millis'][j - 1] + value_interval

                j += 2

            # currently not necessary to deal with this case:
            # if bucket_size < 6 -->  vorhandene daten in 250ms anpassen und den rest mit np.nan erweitern bis man 6 Werte pro Intervall hat
            else:
                print(
                    f'This Case is not implemented yet: bucket size is less than 6 in an interval of {int(interval_dic[phys_signal]) / freq_dic[phys_signal] * cluster_size_dic[phys_signal]}')
                print(
                    f'right interval -> bucket size is {clustered_data.cluster_size[i]}, but should {cluster_size_dic[phys_signal]}!')
                print(f'This bucket begins at the time: {clustered_data.time_iso[i]} at index {j}')
                print(i)
                print(clustered_data[i - 2:i + 3])
                j += clustered_data.cluster_size[i]


        # -------------------------------------------------------------------------------------------------------------
        # ----------- BIGGER CLUSTER SIZE-----------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        # GSR: > 1750ms  ->  vorhandene daten in 250ms schritten anpassen und dann nochmal die dif zum nächsten bucket nehmen und dann in 250ms np.nan hineinschieben
        elif ist_cluster_interval > soll_cluster_interval_max:
            soll_interval_min = soll_cluster_interval_min / cluster_size_dic[phys_signal]
            soll_interval_max = soll_cluster_interval_max / cluster_size_dic[phys_signal]

            if soll_interval_min < value_interval < soll_interval_max:
                for_end = j + clustered_data.cluster_size[i] - 1

                for j in range(j, for_end):
                    raw_data.loc[j, 'time_millis'] = raw_data.time_millis[j - 1] + value_interval

                # jump to the second value of a bucket
                j += 2

            # adapt time of given values in standard interval (i.e. GSR: 250ms) and interpolate or fill in missing values
            else:
                soll_interval = 1000 / freq_dic[phys_signal]

                for_end = j + clustered_data.cluster_size[i] - 1

                for j in range(j, for_end):
                    raw_data.loc[j, 'time_millis'] = raw_data.time_millis[j - 1] + soll_interval

                time_to_next_cluster = clustered_data.time_millis[i + 1] - raw_data.time_millis[j]
                number_of_new_values = int(time_to_next_cluster / (1000 / freq_dic[phys_signal]))

                j += 1

                if number_of_new_values > 0:
                    number_of_buckets_since_hole = 0
                    number_of_values_for_hole = number_of_new_values

                    interval_for_new_values = time_to_next_cluster / number_of_new_values

                    old_index = j

                    first_part_data = raw_data[:j]
                    for_end = j + number_of_new_values - 1

                    for j in range(j, for_end):
                        d = {'time_millis': [first_part_data.time_millis[j - 1] + interval_for_new_values],
                             'time_iso': [first_part_data.time_iso[j - 1]],
                             'value_real': [np.nan]}

                        #first_part_data = first_part_data.append(pd.DataFrame(data=d), ignore_index=True)
                        first_part_data = pd.concat([first_part_data, pd.DataFrame(d)], ignore_index=True)

                    # jump to the second value of a bucket
                    j += 2

                    raw_data = pd.concat([first_part_data, raw_data[old_index:]], ignore_index=True)


        # -------------------------------------------------------------------------------------------------------------
        # ----------- SMALLER CLUSTER SIZE--------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        # GSR < 1250 ms
        else:
            # if a regular cluster is accidentally splitted up in to cluster by the sql-query
            if clustered_data.cluster_size[i] + clustered_data.cluster_size[i + 1] == cluster_size_dic[phys_signal]:
                clustered_data.loc[i + 1, 'cluster_size'] = cluster_size_dic[phys_signal]
                clustered_data.loc[i + 1, 'time_millis'] = clustered_data.time_millis[i]
                continue

            upcoming_value_interval = ist_cluster_interval / clustered_data.cluster_size[i]

            # GSR: < 250ms
            if upcoming_value_interval <= (1000 / freq_dic[phys_signal]):
                for_end = j + clustered_data.cluster_size[i] - 1

                for j in range(j, for_end):
                    raw_data.loc[j, 'time_millis'] = raw_data.time_millis[j - 1] + upcoming_value_interval
                # jump to the second value of a bucket
                j += 2

            # Currently not necessary to deal with this case:
            # upcoming_value_interval is bigger than (250ms + 250/6) -> have to add np.nan in an interval of 250ms
            else:
                print(f'This Case is not implemented yet: upcoming value interval is to big')
                print(
                    f'interval is less than 1250ms -> value interval is {upcoming_value_interval} but should be between'
                    f'{(1000 / freq_dic[phys_signal]) - ((1000 / freq_dic[phys_signal]) / cluster_size_dic[phys_signal])}ms and'
                    f'{(1000 / freq_dic[phys_signal]) - ((1000 / freq_dic[phys_signal]) / cluster_size_dic[phys_signal])}ms!')
                j += clustered_data.cluster_size[i]

    # adapt last 6 values in list
    end_for = j + clustered_data.cluster_size[numberOfClusters - 1] - 1

    for j in range(j, end_for):
        raw_data.loc[j, 'time_millis'] = raw_data.time_millis[j - 1] + (1000 / freq_dic[phys_signal])

    # convert time_millis into datetime-format
    raw_data.time_millis = pd.to_datetime(raw_data.time_millis, unit='ms')

    # set right time with time_iso or given starttime
    if starttime is None:
        time_diff = (int(interval_dic[phys_signal]) * cluster_size_dic[phys_signal] / freq_dic[phys_signal])
        raw_data.time_millis = raw_data.time_millis + (
                pd.to_datetime(raw_data.time_iso[0]) - raw_data.time_millis[0] - pd.Timedelta(time_diff,
                                                                                              unit='milliseconds'))
    # use user-specified 'starttime' (if given)
    else:
        hour_diff = starttime.hour - raw_data.time_millis[0].hour
        time_diff = (int(interval_dic[phys_signal]) * cluster_size_dic[phys_signal] / freq_dic[phys_signal])
        raw_data.time_millis = raw_data.time_millis + pd.Timedelta(hour_diff, unit='hour') - pd.Timedelta(time_diff,
                                                                                                          unit='milliseconds')

    # Interpolation with time_millis which has already the right timestamp
    data = raw_data.set_index('time_millis')


    # TODO - interpolate 'linear' has an issue with datetime64[ns] objects
    data = data.interpolate(method="linear", order=1)
    export_dataframe = data.reset_index()

    return export_dataframe


# apply low-pass and high-pass butterworth filters to GSR and ST signals
def filter_GSR_ST(data: pd.DataFrame, phys_signal: str) -> pd.DataFrame:
    """
    Takes pandas DataFrame containing GSR or ST measurements,
     applies high-pass and low-pass filters to GSR and ST signals
     and return filtered pandas DataFrame containing physiological signals
    :param data: pandas DF containing raw physiological data
    :param phys_signal: string representing the signal to apply the filters on ("GSR", "ST", etc.)

    :return: DataFrame containing filtered physiological signals
    """

    data = data.copy()
    order = butter_filter_order_dic[phys_signal]

    b, a = butter(order, lowpass_cutoff_freq_dic[phys_signal], 'low', analog=False)
    c, d = butter(order, highpass_cutoff_freq_dic[phys_signal], "high", analog=False)

    # new version -- replace signal.filtfilt() with signal.lfilter()
    # old filter (causes drops in GSR signal before MOS) --> filters signal twice
    # z = signal.filtfilt(b, a, data.value_real)
    # filteredGSR = signal.filtfilt(c, d, z)

    # new filter --> filters signal only once
    z = signal.lfilter(b, a, data.value_real)
    filteredGSR = signal.lfilter(c, d, z)

    data['value_real'] = filteredGSR

    return data

# apply low-pass and high-pass butterworth filters to GSR for Streamlit implementation
def preprocess_GSR(data,
                   order: int = 1,
                   lowpass_cutoff_frequency: float = 1 / (4 / 2),
                   highpass_cutoff_frequency: float = 0.05 / (4 / 2)):

    data = data.copy()

    b, a = signal.butter(order, lowpass_cutoff_frequency, 'low', analog = False)
    c, d = signal.butter(order, highpass_cutoff_frequency, 'high', analog = False)

    z = signal.lfilter(b, a, data.GSR)
    filteredEDA = signal.lfilter(c, d, z)

    data['GSR'] = filteredEDA

    return data

# apply low-pass and high-pass butterworth filters to ST for Streamlit implementation
def preprocess_ST(data, order: int = 2,
                   lowpass_cutoff_frequency: float = 0.07 / (4 / 2),
                   highpass_cutoff_frequency: float = 0.005 / (4 / 2)):

    data = data.copy()

    b, a = signal.butter(order, lowpass_cutoff_frequency, 'low', analog = False)
    c, d = signal.butter(order, highpass_cutoff_frequency, 'high', analog = False)

    z = signal.lfilter(b, a, data.ST)
    filteredST = signal.lfilter(c, d, z)

    data['ST'] = filteredST

    return data

#separate function for loading and preprocessing ECG signal
def load_ecg_data_from_file(sqlite_file, starttime=None):
    conn = sqlite3.connect(sqlite_file)

    query_raw_data = f'SELECT time_millis, time_iso, value_text FROM sensordata WHERE platform_id = 2 and sensor_id = 15'

    raw_data = pd.read_sql_query(query_raw_data, conn)

    ecg_data = ecg_resampling(raw_data, starttime)

    return ecg_data


#separate function for preprocessing ECG signal
def ecg_resampling(raw_data, starttime):
    data = raw_data.copy()
    time_millis_list = [] # stores the calculated time_millis in 250hz sampling rate
    ecg_data_list = [] # splits the value_text with ';' up and stores it in a list
    time_millis_diff_list = [] #for mean time_step calculation for the last index values

    for i in range(len(data.value_text.values)-1):
        # splitting up the ecg values by ';'
        ecg_values = data.value_text.values[i].split(";")
        ecg_data_list.extend(ecg_values)

        # differenc between to "clusters" which are normally 250ms with 63 values (250hz sampling rate -> 250 values per second)
        time_millis_diff_list.append(data.time_millis[i+1] - data.time_millis[i])

        # the 63 values equally distributed in the 250ms interval
        for i in np.linspace(data.time_millis[i], data.time_millis[i+1], num=len(ecg_values), endpoint=False):
            time_millis_list.append(i)

    # do the same as in the for above for the last entry in the file
    #print(data.iloc[-1].time_millis, "-", data.iloc[-1].value_text)
    ecg_values = data.iloc[-1].value_text.split(";")
    ecg_data_list.extend(ecg_values)

    last_time_millis = data.iloc[-1].time_millis + np.mean(time_millis_diff_list)
    #print(last_time_millis, " - ", int(last_time_millis))
    for i in np.linspace(data.iloc[-1].time_millis, int(last_time_millis), num=len(ecg_values), endpoint=False):
            time_millis_list.append(i)

    # merge both lists to a dataframe
    ecg_data = pd.DataFrame(list(zip(time_millis_list, ecg_data_list)),
               columns =['time_millis', 'ecg_values'])


    # convert time_millis into datetime-format
    ecg_data.time_millis = pd.to_datetime(ecg_data.time_millis, unit='ms')
    # set right time with time_iso
    if starttime is None:
        ecg_data.time_millis = ecg_data.time_millis + (pd.to_datetime(data.time_iso[0]) - ecg_data.time_millis[0])
    # set right time with given starttime
    else:
        time_diff = starttime.hour - ecg_data.time_millis[0].hour
        ecg_data.time_millis = ecg_data.time_millis + pd.Timedelta(time_diff, unit='hour')

    return ecg_data


def preprocess_for_MOS(file_path: str, starttime: datetime.datetime = None) -> pd.DataFrame:
    """
    Gets Data from eDiary App and applies preprocessing pipeline to return DataFrame containing pre-processed
    physiological data measurements in 1 Hz

    :param file_path: file path to data
    :param starttime: known starttime of the given data to fix timestamp issues
    :return: DataFrame containing pre-processed physioloigcal data (GSR, ST, IBI, HRV)
    """
    # load & preprocess GSR data
    GSR_cluster, GSR_raw = dl.get_ediary_data(filename=file_path, phys_signal="GSR")
    GSR = GSR_preprocessing(GSR_cluster=GSR_cluster,
                            GSR_raw=GSR_raw,
                            phys_signal="GSR")
    # load & preprocess ST data
    ST_cluster, ST_raw = dl.get_ediary_data(filename=file_path, phys_signal="ST")
    ST = ST_preprocessing(ST_cluster=ST_cluster,
                          ST_raw=ST_raw,
                          phys_signal="ST")

    # load & preprocess IBI data
    IBI_raw = dl.get_ediary_data(filename=file_path, phys_signal="IBI")
    IBI_raw_formatted = ""  # format IBI raw
    IBI = IBI_preprocessing(IBI_raw)

    #### HRV ---> get HRV from preprocessed IBIs (so far, this is only the difference between consecutive IBIs)
    HRV = HRV_preprocessing(IBI)

    # merge GSR, ST, IBI, and HRV measurements based on common timestamp ('time_iso')
    merged_data = merge_signals(GSR, ST, IBI, HRV, merge_col='time_iso')

    # fix timestamp with given start-time
    if starttime is not None:
        # print(merged_data.time_iso[0])
        hour_diff = starttime.hour - merged_data.time_iso[0].hour
        merged_data.time_iso = merged_data.time_iso + pd.Timedelta(hour_diff, unit='hour')
        # print(merged_data.time_iso[0])

    # print("Final preprocessed and merged dataset: \n", merged_data.head(30))
    return merged_data


def MOS_detection_signal_preparation(filename: str, starttime: datetime.datetime = None):

    print("Empatica E4 Check:", sensor_check.E4_used(filename))

    if sensor_check.E4_used(filename) == True:

        #### GSR
        GSR_cluster, GSR_raw = dl.get_ediary_data(filename = filename, phys_signal = "GSR")

        # all in one
        GSR = GSR_preprocessing(GSR_cluster = GSR_cluster,
                                    GSR_raw = GSR_raw,
                                    phys_signal = "GSR")



        #### ST
        ST_cluster, ST_raw = dl.get_ediary_data(filename = filename, phys_signal = "ST")

        ST = ST_preprocessing(ST_cluster = ST_cluster,
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
        IBI_raw['IBI'] = format_raw_IBI(IBI_raw)

        if IBI_raw is not None:
            IBI = IBI_preprocessing(IBI_raw)
            #print("PREPROCESSED IBI", IBI)
        else:
            IBI = IBI_raw

        #if IBI is not None:
            #print("IBI prep successful")

        #print(IBI)

        #### HRV ---> get HRV from preprocessed IBIs
        # TODO - this is the old version (just IBI differences)
        if IBI is not None:
            HRV = HRV_preprocessing(IBI)
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
        merged_data = merge_signals(GSR, ST, merge_col = 'time_iso')

    else:
        merged_data = merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')


    #merged_data = pps.merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')
    #print("Final preprocessed and merged dataset: \n", merged_data.head(30))

    # fix timestamp with given start-time
    if starttime is not None:
        # print(merged_data.time_iso[0])
        hour_diff = starttime.hour - merged_data.time_iso[0].hour
        merged_data.time_iso = merged_data.time_iso + pd.Timedelta(hour_diff, unit='hour')

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

        GSR_clus, GSR_raw = dl.get_raw_GSR(sqlite_file_path)
        ST_clus, ST_raw = dl.get_raw_ST(sqlite_file_path)
        IBI_raw, HRV = dl.get_raw_IBI_and_HRV(sqlite_file_path)

        return GSR_clus, GSR_raw, ST_clus, ST_raw, IBI_raw, HRV


def geolocate(signal_data: pd.DataFrame, location_data: pd.DataFrame) -> pd.DataFrame:
    """
    Geolocates sensor measurements by merging measurements with location-based smartphone data
    :param signal_data: DataFrame containing preprocessed Signals
    :param location_data: DataFrame containing data from smartphones (location (lat, long), speed, etc.)
    :return: Georeferenced sensor measurements with NaN values at timestamps where there is no phone measurements
    """

    location_data['ts_rounded'] = pd.to_datetime(location_data['time_iso']).dt.round(freq='S')

    merged_geolocated_data = fix_timestamp_issue_ediary(signal_data=signal_data, location_data=location_data)
    # print(merged_geolocated_data[~merged_geolocated_data['latitude'].isna()])

    merged_geolocated_data = merged_geolocated_data[
        ['time_iso_x', 'GSR', 'ST', 'IBI', 'HRV', 'latitude', 'longitude', 'speed']]
    merged_geolocated_data.columns = ['time_iso', 'GSR', 'ST', 'IBI', 'HRV', 'Lat', 'Lon', 'speed']
    return merged_geolocated_data


def fix_timestamp_issue_ediary(signal_data: pd.DataFrame, location_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes eDiary bug where timestamps are recorded in wrong timezone (not UTC) by merging on matching timestamps
    (increases timestamp until it matches the correct timestamps (from phone))

    :param signal_data: DataFrame containing preprocessed Signals
    :param location_data: DataFrame containing data from smartphones (location (lat, long), speed, etc.)
    :return: merged DataFrame holding sensor measurements including phone's location recordings
    """

    time_diff = utilities.calculate_time_difference(sensor_data = signal_data, location_data = location_data)
    signal_data['time_iso'] = signal_data['time_iso'] + datetime.timedelta(hours = time_diff)
    # TODO - note - this was simply merged = signal_data.merge(location_data, left_on = 'time_iso', right_on = 'ts_rounded', how = 'left') before
    # TODO - no if or elif statement
    if 'ts_rounded' in location_data.columns:
        merged = signal_data.merge(location_data, left_on = 'time_iso', right_on = 'ts_rounded', how = 'left')
    elif 'time_iso' in location_data.columns:
        merged = signal_data.merge(location_data, left_on = 'time_iso', right_on = 'time_iso', how = 'left')

    return merged

def normalize_filtered_signal_min_max(df: pd.DataFrame, signal: str, signal_col: str):

    if signal == "GSR":
        GSR_values = df[signal_col].values
        GSR_values = GSR_values.reshape((len(GSR_values), 1))
        GSR_scaler = MinMaxScaler(feature_range=(0, 1))
        GSR_scaler = GSR_scaler.fit(GSR_values)
        GSR_normalized = GSR_scaler.transform(GSR_values)
        df['GSR_normalized'] = GSR_normalized

        return df

    if signal_col == "ST":
        ST_values = df[signal_col].values
        ST_values = ST_values.reshape((len(ST_values), 1))
        ST_scaler = MinMaxScaler(feature_range=(0, 1))
        ST_scaler = ST_scaler.fit(ST_values)
        ST_normalized = ST_scaler.transform(ST_values)
        df['ST_normalized'] = ST_normalized

        return df


def standardize_filtered_signal(df: pd.DataFrame, signal: str, signal_col: str):

    if signal == "GSR":
        GSR_values = df[signal_col].values
        GSR_values = GSR_values.reshape((len(GSR_values), 1))
        GSR_scaler = StandardScaler()
        GSR_scaler = GSR_scaler.fit(GSR_values)
        GSR_standardized = GSR_scaler.transform(GSR_values)
        df['GSR_standardized'] = GSR_standardized

        return df

    if signal_col == "ST":
        ST_values = df[signal_col].values
        ST_values = ST_values.reshape((len(ST_values), 1))
        ST_scaler = StandardScaler()
        ST_scaler = ST_scaler.fit(ST_values)
        ST_standardized = ST_scaler.transform(ST_values)
        df['ST_standardized'] = ST_standardized

        return df






# redundant method (old method from Marius' R Script)
def test_timezone_location(sensor_data, location_data):
    # ====================================================================================
    #                   Test timezone approximation, assuming location is in UTC
    #                   For now, mostly for testing purposes
    # ====================================================================================

    location_sorted = location_data
    sensordata_sorted = sensor_data
    avg_ts_first_locations = round(location_sorted.head(25)['TimeNum'].mean())
    avg_ts_latest_locations = round(location_sorted.tail(25)['TimeNum'].mean())
    avg_ts_all_locations = round(location_sorted['TimeNum'].mean())
    ts_first_location = round(location_sorted['TimeNum'].min())
    ts_last_location = round(location_sorted['TimeNum'].max())

    avg_ts_first_sensordata = round(sensordata_sorted.head(25)['TimeNum'].mean())
    avg_ts_latest_sensordata = round(sensordata_sorted.tail(25)['TimeNum'].mean())
    avg_ts_all_sensordata = round(sensordata_sorted['TimeNum'].mean())
    ts_first_sensordata = round(sensordata_sorted['TimeNum'].min())
    ts_last_sensordata = round(sensordata_sorted['TimeNum'].max())

    diff_ts_latest = (avg_ts_latest_locations - avg_ts_latest_sensordata)
    diff_ts_first = (avg_ts_first_locations - avg_ts_first_sensordata)
    diff_ts_all = (avg_ts_all_locations - avg_ts_all_sensordata)

    count_location = len(location_sorted)
    count_sensordata = len(sensordata_sorted)

    length_of_run_location = (ts_last_location - ts_first_location)
    length_of_run_sensordata = (ts_last_sensordata - ts_first_sensordata)

    try:
        freq_location = round(count_location / length_of_run_location, 2)
    finally:
        pass

    try:
        freq_sensordata = round(count_sensordata / length_of_run_sensordata, 2)
    finally:
        pass

    # Calculate rounded deltatime in hours, reassign them to delta seconds
    diff_h = round(diff_ts_all / 3600)
    deltatime_s = 3600 * diff_h

    sensor_data["TimeNum"] = sensor_data["TimeNum"] + deltatime_s

    return sensor_data
