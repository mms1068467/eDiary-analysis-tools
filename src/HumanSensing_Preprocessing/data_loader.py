"""
Methods for loading in physiological and location-based data recorded by
eDiary app multiple file formats:
 sqlite database, .csv, .xlsx, etc.

"""

import sqlite3
import math
import pandas as pd
import numpy as np
import datetime
import typing

from HumanSensing_Preprocessing import utilities

from HumanSensing_Preprocessing.lookup_tables import interval_dic, platform_id_dic, sensor_id_dic, freq_dic ,cluster_size_dic
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check


# TODO - fix SQL injections --> use '?'
# https://realpython.com/prevent-python-sql-injection/

# TODO - add cases for other file formats
def get_ediary_data(filename: str, phys_signal: str, encoding: str = None,
             delim: str = ',', ignore_extension: bool = False) -> pd.DataFrame:
    """
    Load physiological data from a file and store it in pandas DataFrame
    Supported files: .sqlite, .sqlite3, ....

    :param filename: Absolute or relative path to the file object to read
    :param phys_signal: String representing the type of physiological signal to query from the sqlite database ('IBI', 'GSR', or 'ST')
    :param encoding: Optional encoding
    :param delim: Optional delimiter (for CSV files)
    :param ignore_extension: if True, the extension for the file is not tested

    :return: Pandas DataFrame containing the data in the specified file
    """

    # get file extension
    file_extension = filename.split('.')[-1]

    if file_extension == 'sqlite' or file_extension == 'sqlite3': # or file_extension == 'sql':

        # with sqlite3.connect(filename) as connection: --> Note - this does not close connections
        connection = sqlite3.connect(filename)

        # counting milliseconds
        time_millis = 'time_millis/' + interval_dic[phys_signal]
        count_time_millis = f'count({time_millis})'

        # read in IBI data
        # TODO - create on "IBI_E4" and one "IBI_BioHarness" input for phys_signal in lookup tables
        if phys_signal == "IBI":
            query_raw_data = f'SELECT time_millis, time_iso, value_text FROM sensordata ' \
                             f'WHERE platform_id = {platform_id_dic[phys_signal]} AND ' \
                             f'sensor_id = {sensor_id_dic[phys_signal]}'

            raw_data_IBI = pd.read_sql_query(query_raw_data, connection)
            # TODO - filtering and interpolating IBI Signal should be in a different function
            #resampled_data = IBIreampling(raw_data, starttime)

            #TODO - check if connection need to be closed before each return statement or
            # if it is enough to close it at the end of the function
            # or does it close automatically because connection is only within the scope of
            # this specific function

            #connection.close()

            # TODO - check if this works correctly
            if len(raw_data_IBI) == 0:
                print(f'Error at file {filename}: IBI data from the sensordata at '
                      f'platform_id = {platform_id_dic[phys_signal]} and sensor_id = {sensor_id_dic[phys_signal]} is empty!'
                      f'\nData will be processed without IBI and HRV data due to missing database values!')

                return None


            return raw_data_IBI

        if phys_signal == "GSR":
            # getting cluster size of tuples with same timestamp
            query_get_cluster = f'SELECT time_millis, time_iso, {time_millis} AS cluster_time_millis,' \
                                f'{count_time_millis} AS cluster_size, AVG(value_real) AS avg_value_real ' \
                                f'FROM sensordata ' \
                                f'WHERE sensor_id = {sensor_id_dic[phys_signal]} AND ' \
                                f'platform_id = {platform_id_dic[phys_signal]} ' \
                                f'GROUP BY {time_millis}'


            query_raw_data = f'SELECT time_millis, time_iso, value_real FROM sensordata ' \
                             f'WHERE platform_id = {platform_id_dic[phys_signal]} AND ' \
                             f'sensor_id = {sensor_id_dic[phys_signal]}'


            # cluster_data stores "clusters" of equal timestamps
            # --> will be distributed with separate function (get4Hz() function)
            cluster_data_GSR = pd.read_sql_query(query_get_cluster, connection)
            raw_data_GSR = pd.read_sql_query(query_raw_data, connection)

            return cluster_data_GSR, raw_data_GSR

        if phys_signal == "ST":
            query_get_cluster = f'SELECT time_millis, time_iso, {time_millis} AS cluster_time_millis,' \
                                f'{count_time_millis} AS cluster_size, AVG(value_real) AS avg_value_real ' \
                                f'FROM sensordata ' \
                                f'WHERE sensor_id = {sensor_id_dic[phys_signal]} AND ' \
                                f'platform_id = {platform_id_dic[phys_signal]} ' \
                                f'GROUP BY {time_millis}'

            query_raw_data = f'SELECT time_millis, time_iso, value_real FROM sensordata ' \
                             f'WHERE platform_id = {platform_id_dic[phys_signal]} AND ' \
                             f'sensor_id = {sensor_id_dic[phys_signal]}'

            cluster_data_ST = pd.read_sql_query(query_get_cluster, connection)
            raw_data_ST = pd.read_sql_query(query_raw_data, connection)

            return cluster_data_ST, raw_data_ST

        if phys_signal == "ECG":
            
            query_raw_data = f'SELECT time_millis, time_iso, value_text FROM sensordata ' \
                             f'WHERE platform_id = {platform_id_dic[phys_signal]} AND ' \
                             f'sensor_id = {sensor_id_dic[phys_signal]}'

            raw_data_ECG = pd.read_sql_query(query_raw_data, connection)

            return raw_data_ECG

        # TODO -> check if connection is still open

    # TODO
    if file_extension == 'csv':
        #ediary_data = pd.read_csv(filename, delimiter = delim)
        pass

    # TODO
    if file_extension == 'xlsx':
        #ediary_data = pd.read_excel(filename)
        pass



##### Loading in individual signals (raw data)

def get_raw_GSR(filename):

    if sensor_check.E4_used(filename) == True:

        #### GSR
        GSR_cluster, GSR_raw = get_ediary_data(filename = filename, phys_signal = "GSR")
        GSR_raw['time_iso'] = pd.to_datetime(GSR_raw['time_iso'])
        # TODO - specify column namings -- either 'time_millis' or 'TimeNum'
        GSR_raw.columns = ['TimeNum', 'time_iso', 'GSR_raw']

        return GSR_cluster, GSR_raw


def get_raw_ST(filename):

    if sensor_check.E4_used(filename) == True:

        #### GSR
        ST_cluster, ST_raw = get_ediary_data(filename = filename, phys_signal = "ST")
        ST_raw['time_iso'] = pd.to_datetime(ST_raw['time_iso'])
        # TODO - specify column namings -- either 'time_millis' or 'TimeNum'
        ST_raw.columns = ['TimeNum', 'time_iso', 'ST_raw']

        return ST_cluster, ST_raw


def get_raw_IBI(filename):

    if sensor_check.BioHarness_used(filename):
        #### IBI
        IBI_raw = get_ediary_data(filename = filename, phys_signal = "IBI")

        IBI_raw['IBI'] = pps.format_raw_IBI(IBI_raw)

        return IBI_raw


def get_raw_IBI_and_HRV(filename):

    if sensor_check.BioHarness_used(filename):
        #### IBI
        IBI_raw = get_ediary_data(filename = filename, phys_signal = "IBI")

        IBI_raw['IBI'] = pps.format_raw_IBI(IBI_raw)

        #### HRV ---> get HRV from preprocessed IBIs
        # TODO - this is the old version (just IBI differences)
        if IBI_raw is not None:
            HRV = pps.extract_HRV_from_IBI(IBI_raw)
        else:
            HRV = None

        return IBI_raw, HRV



# TODO -- has not been changed
def get_location_data(filename: str, start_date = 0, end_date = float("inf"),
                      delim: str = ',', ignore_extension: bool = False) -> pd.DataFrame:
    """
    Queries phone data including geo-references and further measurements from sqlite database file
    and stores results in DataFrame

    :param sqlite_file: path to the sqlite db file
    :param start_date: optional starting timestamp (default = 0)
    :param end_date: optional ending timestamp (default = inf)
    :return: DataFrame filtered for specific timestamp range containing:
     ("id", "time_millis", "time_iso", "latitude", "longitude", "provider", "horizontal_accuracy",
     "altitude", "vertical_accuracy", "bearing", "bearing_accuracy", "speed", "speed_accuracy")
    """
    if not isinstance(start_date, (int, float)):
        return False
    if start_date > 0:
        while len(str(start_date)) < 13:
            start_date = start_date * 10
    if not isinstance(end_date, (int, float)):
        return False
    if math.isfinite(end_date):
        if end_date == 0:
            end_date = 1
        while len(str(end_date)) < 13:
            end_date = end_date * 10

    file_extension = filename.split('.')[-1]

    if file_extension == "sqlite" or file_extension == "sqlite3":

        if math.isfinite(end_date):
            #command = "SELECT * FROM location WHERE location.time_millis >= {} AND location.time_millis <= {};".format(
            #    start_date, end_date)
            # TODO - test if this new version using f string works as the old one
            command = f"SELECT * FROM location WHERE location.time_millis >= {start_date} AND location.time_millis <= {end_date};"
        else:
            #command = "SELECT * FROM location WHERE location.time_millis >= {};".format(start_date)
            command = f"SELECT * FROM location WHERE location.time_millis >= {start_date};"

        connection = sqlite3.connect(filename)

        # TODO - check if pd.read_sql_query would be easier
        cur = connection.cursor()
        query = cur.execute(command)
        location_data = query.fetchall()
        location_data = pd.DataFrame(location_data,
                                     columns=["id", "time_millis", "time_iso", "latitude", "longitude", "provider",
                                              "horizontal_accuracy", "altitude", "vertical_accuracy", "bearing",
                                              "bearing_accuracy", "speed", "speed_accuracy"])
        return location_data

    if file_extension == 'csv':
        location_data = pd.read_csv(filename, delimiter = delim)
        return location_data

    if file_extension == 'xlsx':
        location_data = pd.read_excel(filename)
        return location_data



# TODO -- has not been changed
def geolocate_data(location_data, input_filtered_data):
    """
    Merge sensor measurements with phone data based on common timestamps (unix) to
    geo-reference recordings, adjust timestamps to be in appropriate time zone and
    return DataFrame for MOS Detection

    :param location_data: DataFrame containing phone data with locations
    :param input_filtered_data: DataFrame containing

    :return: combined DataFrame of measurements (filtered GSR & ST, IBI, HRV) including Lat/Lon reference
    """

    # ====================================================================================
    #                   1. Get x & y data
    # ====================================================================================
    xy_data = location_data[['time_millis', 'latitude', 'longitude']]
    xy_data.columns = ['TimeNum', 'Lat', 'Lon']

    ## Divide by 1000 to convert to sec from milisecond
    i = 0
    while i < len(xy_data.index):
        xy_data.at[i, "TimeNum"] = xy_data.at[i, "TimeNum"] / 1000
        i += 1

    # TODO -  uncomment and see if same results
    #xy_data['TimeNum'].unix_ms_to_s(data = xy_data, col_name = 'TimeNum')

    xy_data = xy_data.drop_duplicates(subset='TimeNum',keep="first")

    # ====================================================================================
    #                   Test timezone approximation, assuming location is in UTC
    #                   For now, mostly for testing purposes
    # ====================================================================================
    location_sorted = xy_data
    sensordata_sorted = input_filtered_data
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
    diff_h = round(diff_ts_all/3600)
    deltatime_s = 3600*diff_h

    # ====================================================================================
    #                   2. Change timezone of inpFilter Data
    # ====================================================================================
    # The physiological measurements have a different timezone as a result it is not
    # possible to geolocate the detected MOS using the timestamp. Check if inpFilterData
    # and xy_data have different timezones. If so, add 3600 sec (1 hour) to timestamp
    # of filtered physiological measurements to convert to UTC as location timestamp
    # uses as timezone.

    ## Adjust time value of the filtered sensordata to fit it to the location data
    ## In R script +7200 is used, in Python the diff between location and sensordata is estimated

    i = 0
    while i < len(input_filtered_data.index):
        #input_filtered_data.at[i, "TimeNum"] = input_filtered_data.at[i, "TimeNum"] + 3600
        input_filtered_data.at[i, "TimeNum"] = input_filtered_data.at[i, "TimeNum"] + deltatime_s
        i += 1

    # TODO -  uncomment and see if same results
    #input_filtered_data['TimeNum'] = utilities.adjust_sensor_times_to_location(data=input_filtered_data,
    #                                                                           increase_secs = deltatime_s,
    #                                                                           col_name = "TimeNum")

    # ====================================================================================
    #                   3. Concatenate filtered data with location
    # ====================================================================================
    XYfilterData = pd.merge(input_filtered_data, xy_data, how='left', on=['TimeNum'])

    return XYfilterData
