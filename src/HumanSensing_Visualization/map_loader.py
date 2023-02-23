import folium
from folium import plugins
import geopandas
from streamlit_folium import st_folium
from geopy import distance
import pandas as pd

import math
import sqlite3
import numpy as np


# multiple functions wrapped to filter out outliers based on threshold distance in m between each consecutive record and on horizontal accuracy threshold
def filter_location_data(dataframe):
    dataframe = remove_outliers(dataframe, 10)
    dataframe = geopandas.GeoDataFrame(dataframe,
                                       geometry=geopandas.points_from_xy(dataframe.longitude, dataframe.latitude),
                                       crs=4326)
    dataframe = horizontal_accuracy_filter(dataframe)
    return dataframe


def create_map_with_track_and_MOS(geo_df, add_MOS: bool = False) -> folium.Map:
    """
    Generates a folium.Map from geopandas.GeoDataFrame
    :param geo_df: geopandas.GeoDataFrame with lat/lon data
    :return: folium.Map
    """

    map = folium.Map(location=[geo_df["latitude"].mean(), geo_df["longitude"].mean()], zoom_start=16, height='90%',
                     prefer_canvas=True)
    plugins.PolyLineOffset(geo_df[["latitude", "longitude"]], color="blue", weight=3, opacity=0.8).add_to(map)

    # Add Data and Style Map

    # MarkerCluster feature in Folium to aggregate nearby markers together when the map is zoomed out.

    # for loop below configures a popup, styles the Plaque marker, and adds the feature to the MarkerCluster layer.

    # Create a marker for each plaque location. Format popup

    for index, row in geo_df.iterrows():

        html = f"""<strong>Time:</strong> {row['time_iso']}<br>

            <br>

            <strong>Location:</strong> ({row['latitude'], row['longitude']})<br>

            <br>

            <strong>Speed:</strong> {row['speed']}<br>

            <br>

            <strong>Altitude:</strong> {row['altitude']}<br>

                """

        iframe = folium.IFrame(html,

                               width=200,

                               height=200)

        popup = folium.Popup(iframe,

                             max_width=400)

        # TODO - replace this with MOS_score

        ###

        if add_MOS and row["MOS_score"] >= 75:
            folium.Circle(location=[row["latitude"], row["longitude"]],

                          radius=6,

                          color="red",

                          fill=True,

                          fill_color="red",

                          popup=popup).add_to(map)

    return map


def create_map_with_zipped_MOS(geo_df, add_MOS: bool = False) -> folium.Map:
    """
    Generates a folium.Map from geopandas.GeoDataFrame
    :param geo_df: geopandas.GeoDataFrame with lat/lon data
    :return: folium.Map
    """
#
    map = folium.Map(location=[geo_df["Lat"].mean(), geo_df["Lon"].mean()], zoom_start=14, height='90%',
                     prefer_canvas=True)
    #plugins.PolyLineOffset(geo_df[["Lat", "Lon"]], color="blue", weight=3, opacity=0.8).add_to(map)

    # Add Data and Style Map

    # MarkerCluster feature in Folium to aggregate nearby markers together when the map is zoomed out.

    # for loop below configures a popup, styles the Plaque marker, and adds the feature to the MarkerCluster layer.

    # Create a marker for each plaque location. Format popup

    for index, row in geo_df.iterrows():

        html = f"""<strong>Time:</strong> {row['time_iso']}<br>

            <br>

            <strong>Location:</strong> ({row['Lat'], row['Lon']})<br>

            <br>

                """

        iframe = folium.IFrame(html,

                               width=200,

                               height=200)

        popup = folium.Popup(iframe,

                             max_width=400)

        # TODO - replace this with MOS_score

        ###

        if add_MOS and row["MOS_score"] >= 75:
            folium.Circle(location=[row["Lat"], row["Lon"]],

                          radius=6,

                          color="red",

                          fill=True,

                          fill_color="red",

                          popup=popup).add_to(map)

    return map


#### DataFrame preprocessing and calculating

def remove_outliers(df: pd.DataFrame, limit_in_meters=10) -> pd.DataFrame:
    """
    Filters out the outliers based on time and distance
    difference between each consecutive row in a given DataFrame

    :param df: pd.DataFrame with location data to be filtered
    :param limit_in_meters: threshold for distance filtering
    :return: DataFrame containing filtered horizontal accuracy values
    """

    # sort by time frame and set new index
    df = df.sort_values(by="time_iso", ascending=True)
    df = df.reset_index()

    # apply time difference between each consecutive record and filter out time differences greater than 3 seconds
    df['time_diff'] = df['time_iso'].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
    count_residuals = 0
    list_residuals = []
    for i in df.index:
        # still creates vast gap between deleted [i-1] and [i+1] record, maybe conveying a greater time_diff threshold would work
        if df['time_diff'][i] > 3:
            # print(df['time_diff'][i])
            list_residuals.append(df['time_diff'][i])
            df = df.drop([i - 1])
            count_residuals += 1

    # print log
    print("remove_outliers() log: ",
          "\n Residuals removed:",
          count_residuals,
          "\n Residual list(sec): ",
          list_residuals)

    # filter out records between which horizontal distance is greater than given parameter
    distances = [0.0]  # index 0 is skipped, so this default takes its place
    i_max = df.shape[0]  # number of columns
    i = 1
    while i < i_max:
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        dist = get_distance(prev, cur)
        distances.append(dist)
        i += 1
    df["distance"] = distances
    return df[df["distance"] <= limit_in_meters]


def get_distance(dfrow1, dfrow2):
    crd1 = (dfrow1.latitude, dfrow1.longitude)
    crd2 = (dfrow2.latitude, dfrow2.longitude)
    return distance.distance(crd1, crd2).meters


def horizontal_accuracy_filter(dataframe: pd.DataFrame, horizontal_accuracy: int = 100) -> pd.DataFrame:
    """
    Filters out the horizontal_accuracy column in the given dataframe

    :param dataframe: DataFrame with a horizontal_accuracy column
    :param horizontal_accuracy: column holding the horizontal accuracy certainity measurements
    :return: DataFrame containing filtered horizontal accuracy values
    """
    unfiltered_dataframe = dataframe
    dataframe = dataframe.loc[dataframe['horizontal_accuracy'] <= horizontal_accuracy]
    # dataframe = dataframe.loc[dataframe['speed'] > 0]
    # dataframe = dataframe.sort_values(by= 'time_iso')
    count = len(unfiltered_dataframe) - len(dataframe)
    print("Filtered out by horizontal accuracy: {} out of {}".format(count, len(unfiltered_dataframe)))
    return dataframe


### all good till here
#### Track base stats

def get_location_based_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates location-based stats (distance of track, time taken for track, time spent at standstill and average pace)
    Prior to using the function, run the dataframe through remove_outliers()

    :param dataframe: pandas.GeoDataFrame with a horizontal_accuracy column
    :return: pd.DataFrame containing location-based stats as object data types
    """

    # get total distance
    # trackdf['distance_from_previous'] = trackdf.geometry.distance(trackdf.geometry.shift())
    total_distance = dataframe['distance'].sum()
    # total_distance = int(total_distance)
    total_distance = f'{total_distance:.2f}'
    # get track time lapsed
    startTime = dataframe['time_iso'].min()
    endTime = dataframe['time_iso'].max()
    track_time = (endTime - startTime) / pd.Timedelta(minutes=1)
    # track_time = round(track_time, 2)
    track_time = f'{track_time:.2f}'

    standstill_time = stand_still(dataframe)
    average_pace, average_pace_standstill = calculate_average_speed(dataframe)
    # average_pace_standstill = calculate_average_speed(dataframe)[0]
    # new pd.Dataframe column to list out the base stats
    general_movement_stats = {'track_distance (m)': [total_distance],
                              'track_time (min)': [track_time],
                              'standstill_time (min)': [standstill_time],
                              'average_pace (km/h)': [average_pace],
                              'average_pace (no standstill) (km/h)': [average_pace_standstill]}

    general_movement_stats_df = pd.DataFrame(general_movement_stats, index=["track"])

    # print(general_movement_stats_df.dtypes) - all object types

    return general_movement_stats_df


#location based segmentation
def get_location_based_stats_segmentation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates location-based stats (distance of track, time taken for track, time spent at standstill and average pace)
    Prior to using the function, run the dataframe through remove_outliers()

    :param dataframe: pandas.GeoDataFrame with a horizontal_accuracy column
    :return: pd.DataFrame containing location-based stats as object data types
    """
    #new dataframe to append derivates of each category to be iterated
    concatenate_tracks = pd.DataFrame()

    #list of existing categories within the given dataframe
    category_unique = pd.unique(dataframe['category'])
    
    #iterate through each unique category value and append the calculations to a new dataframe
    for category in category_unique:
        df_new = dataframe[dataframe['category'] == category]
        
        #increment category number by one to use as a track number in given dataframe
        category_no = category + 1
        
        #total track distance
        total_distance = df_new['distance'].sum()
        total_distance = f'{total_distance:.2f}'
        
        # get track time lapsed
        startTime = df_new['time_iso'].min()
        endTime = df_new['time_iso'].max()
        track_time = (endTime - startTime) / pd.Timedelta(minutes=1)
        track_time = f'{track_time:.2f}'

        standstill_time = stand_still(df_new)
        average_pace, average_pace_standstill = calculate_average_speed(df_new)
        
        # new pd.Dataframe column to list out the base stats
        general_movement_stats = {
                                'track_distance (m)': [total_distance],
                                'track_time (min)': [track_time],
                                'standstill_time (min)': [standstill_time],
                                'average_pace (km/h)': [average_pace],
                                'average_pace (no standstill) (km/h)': [average_pace_standstill]
                                }

        
        track_index = "track" + str(category_no)
        general_movement_stats_df = pd.DataFrame(general_movement_stats, index=[track_index]) 
        #print(general_movement_stats_df)
        concatenate_tracks = concatenate_tracks.append(general_movement_stats_df)
        
    #print(general_movement_stats_df.dtypes) - all object types
    #print(concatenate_tracks)
    return concatenate_tracks

def segment_track_df(df: pd.DataFrame, category):

    df_new = df[df['category'] == category]

    return df_new





def stand_still(dataframe: pd.DataFrame):
    """
    Calculates the total time with no movement (dataframe.speed = 0)
    return: total standstill as object data type
    """

    count_stay = 0
    count_move = 0
    for i in dataframe.index:
        # filter by time_diff to avoid enormous residuals
        if dataframe['speed'][i] == 0 and dataframe['time_diff'][i] <= 3:
            count_stay += dataframe['time_diff'][i]
        else:
            count_move += 1

    count_stay = count_stay / 60
    return f'{count_stay:.2f}'


def calculate_average_speed(dataframe):
    count_speed = 0
    count_seconds = 0

    startTime = dataframe['time_iso'].min()
    endTime = dataframe['time_iso'].max()
    timeDiff = float((endTime - startTime) / pd.Timedelta(minutes=1)) * 60

    for i in dataframe.index:
        if dataframe['speed'][i] > 0:
            count_speed += dataframe['speed'][i]

        else:

            if dataframe['time_diff'][i] <= 3:
                # maybe subract index by 1 since time_diff result counts for the previous record
                # produces error
                count_seconds += dataframe['time_diff'][i]
    # print("Count seconds: ", count_seconds)
    timeDiff_no_standstill = timeDiff - count_seconds
    # 1 for overall, 1 with filtered out
    avg_speed = count_speed / timeDiff_no_standstill
    avg_speed_kmh = (avg_speed * 3600) / 1000

    count_speed_standstill = 0
    # count_seconds_standstill = 0
    for i in dataframe.index:
        count_speed_standstill += dataframe['speed'][i]
        # count_seconds_standstill += dataframe['time_diff'][i]

    # print("Time difference loc: {}".format(timeDiff))

    avg_speed_standstill = count_speed_standstill / timeDiff
    avg_speed_kmh_standstill = (avg_speed_standstill * 3600) / 1000
    return f'{avg_speed_kmh:.2f}', f'{avg_speed_kmh_standstill:.2f}'


#### Segment track section

#cummulative sum of all the elements
def distance_starting_point(df: pd.DataFrame):
    df['distance_from_start'] = df['distance'].cumsum()

    return df


def segment_track(df:pd.DataFrame, number_segments):
    total_distance = df['distance'].sum()
    distance_per_segment = total_distance / number_segments
    segment_distance_list = []
    count = 0
    give_distance = 0
    while count < number_segments:
        give_distance = give_distance + distance_per_segment
        give_distance_round = round(give_distance, 4)
        segment_distance_list.append(give_distance_round)
        count += 1
    #print("\nSegment list:", segment_distance_list)
    #df['category'] = 1
    count_category = 0
    
    print("\nTotal distance: ", total_distance,
        "\nDistance each segment: ", distance_per_segment,
                "\nSegment list:", segment_distance_list,
                "\nList length: ", len(segment_distance_list))

    segment_distance_list.reverse()
    category_increment = len(segment_distance_list)

    df['category'] = None
    for segment_distance in segment_distance_list:
        segment_distance = segment_distance + 0.0001
        category_increment -= 1
        for record in df.index:
            if df['distance_from_start'][record] <= segment_distance:
                df['category'][record] = category_increment

    return df



"""
def not_used(notused):
    distances = [0.0]  # index 0 is skipped, so this default takes its place
    i_max = df.shape[0] # number of columns
    i = 1
    while i < i_max:
        prev = df.iloc[i-1]
        cur = df.iloc[i]
        dist = get_distance(prev, cur)
        distances.append(dist)
        i += 1
    df["distance"] = distances
    return df[df["distance"] <= limit_in_meters]

"""

def get_distance(dfrow1, dfrow2):
    crd1 = (dfrow1.latitude, dfrow1.longitude)
    crd2 = (dfrow2.latitude, dfrow2.longitude)
    return distance.distance(crd1, crd2).meters