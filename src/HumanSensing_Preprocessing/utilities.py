import numpy as np
import pandas as pd
import typing

# TODO - have one format for each function (pd.Series to pd.Series or pd.DataFrame to pd.DataFrame)

# convert unix timestamp to iso format (used in plotting)
def unix_to_iso_s(data: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Takes pandas DF column and converts it to timestamp in ISO format (seconds)

    :param data: pandas DataFrame
    :param col_name: column name
    :return: Pandas Series containing converted timestamp column (seconds in iso-format)
    """

    return pd.to_datetime(data[col_name], unit = 's')

def unix_to_iso_ms(data: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Takes pandas DF column and converts it to timestamp in ISO format (milliseconds)

    :param data: pandas DataFrame
    :param col_name: column name
    :return: Pandas Series containing converted timestamp column (milliseconds in iso-format)
    """

    return pd.to_datetime(data[col_name], unit='ms')


# convert timestamp from milliseconds to seconds
def unix_ms_to_s(data: pd.DataFrame, col_name: str = 'TimeNum') -> pd.Series:
    """
    Changes unix timestamp from milliseconds to seconds and
    returns a Pandas Series which should be assigned to a DF column

    :param data: Pandas DataFrame which contains desired timestamp column
    :param col_name: column name of timestamp column
    :return: Pandas Series with unix timestamp in seconds (should be assigned to a column in a DF)
    """

    return data[col_name] / 1000

def iso_to_unix(data: pd.DataFrame, col_name: str = 'TimeNum') -> pd.Series:
    """
    Changes iso timestamp to unix timestamp (number of seconds passed since Janurary 1, 1970)
        returns a Pandas Series which should be assigned to a DF column

    :param data: Pandas DataFrame which contains desired timestamp column
    :param col_name: column name of timestamp column
    :return: Pandas Series with unix timestamp in seconds (should be assigned to a column in a DF)
    """

    return data[col_name].map(pd.Timestamp.timestamp)



# fix timestamp issue with phone
# adjust timestamps of sensor measurements to be conform with timestamps from phone (location data)
    # due to a bug in the eDiary app (timezone / daylight savings time) is not considered


def calculate_time_difference(sensor_data: pd.DataFrame, location_data: pd.DataFrame) -> int:
    """
    Calculates time difference between minimum timestamps of sensor recordings and phone recordings
    (to fix eDiary bug)
    :param sensor_data: DataFrame containing sensor measurements
    :param location_data: DataFrame containing location data (phone measurements)
    :return: Integer describing the hours of time difference between sensor and phone recordings
    """
    first_ts_loc = pd.to_datetime(location_data['time_iso']).dt.round(freq='S').min()

    first_ts_sens = sensor_data['time_iso'].min()

    time_diff = first_ts_loc - first_ts_sens

    return time_diff.seconds // 3600



def adjust_sensor_times_to_location(data: pd.DataFrame, increase_secs: int, col_name: str = 'TimeNum') -> pd.Series:
    """
    Adjust sensor time based on increase in seconds and
    returns a Pandas Series which should be assigned to a DF column

    :param data: Pandas DataFrame which contains desired column
    :param increase_secs: increase in seconds to adjust sensor times
    :param col_name: column name of the column which should be adjusted
    :return: Pandas Series containing adjusted sensor time (should be assigned to a column in a DF)
    """

    return data[col_name] + increase_secs


def adjust_IBI_timestamp(data, unix_time_col: str = 'time_millis',
                         iso_time_col: str = 'time_iso') -> pd.DataFrame:
    """
    Takes in raw, formatted IBI signal and fixes timestamp (iso format in milliseconds)
    NOTE:
    This is an error originating from the eDiary app as the timestamps are written wrong
    --> will be redundant once this bug in the eDiary app is fixed

    :param data: DataFrame containing IBI measurements
    :param unix_time_col: column containing unix timestamp
    :param iso_time_col: column containing timestamp in iso format
    :return:
    """

    data[unix_time_col] = unix_to_iso_ms(data = data, col_name = unix_time_col)
    data[unix_time_col] = data[unix_time_col] + (pd.to_datetime(data[iso_time_col][0]) - data[unix_time_col][0])

    return data


def split_measurements(data, values_col: str, new_col_name: str = 'value_real',
                       split_by: str = ';') -> pd.DataFrame:
    """
    Takes a pandas DataFrame containing measurement values in format (5018;65018;65018;65018;65018;...)
    that are stored in one column ('values_col'), splits the values by a certain delimiter (';')
    and stores values in specified column name ('new_col_name').
    Returns the clean measurements in form of a pandas DataFrame

    :param data: eDiary DataFrame loaded by 'get_ediary_data()' function
    :param values_col: column name containing measuremement values to split
    :param new_col_name: column name where new values should be stored
    :param split_by: delimiter to split the string by (';', ',', '.') --> ';' is default
    """

    # split measurements by ";"
    return [np.unique(pd.to_numeric(i.split(split_by)))[0] for i in data[values_col].values]


def split_and_find_IBI(data) -> pd.DataFrame:
    """
    Takes a pandas DataFrame containing measurement values in format (5018;65018;65018;65018;65018;...)
    and returns a list of correct IBI recordings (in range 260 to 1500)

    :param data: eDiary DataFrame loaded by 'get_ediary_data()' function
    :param values_col: column name containing measuremement values to split
    :param new_col_name: column name where new values should be stored
    :param split_by: delimiter to split the string by (';', ',', '.') --> ';' is default
    """

    ibi_values = []

    # split measurements by ";"
    for i in range(len(data.value_text.values)):
        text_values = np.unique(pd.to_numeric(data.value_text.values[i].split(";")))

        # iterate over all values from one timestamp (i.e. 62345;62345;700;700;3425)
        # because the real, true number could be somewhere in the text and not only at the beginning
        for ibi in text_values:
            if 260 <= ibi <= 1500:
                ibi_values.append(ibi)
                break

        if len(ibi_values) == i:
            ibi_values.append(np.nan)


    return ibi_values

def check_timestamp_gaps(df: pd.DataFrame, duration: int = 1, col_name: str = 'time_gap_over_1_sec') -> pd.DataFrame:

    try:
        # Create True/False for each new time gap
        df[col_name] = (df['time_iso'].diff()).dt.seconds > duration

        # set 1st value of 'time_gap_over_1_sec' to True
        df[col_name].iloc[0] = True
    except IndexError:
        pass

    return df