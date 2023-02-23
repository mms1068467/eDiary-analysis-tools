import streamlit as st
import pathlib
import sqlite3
import pandas as pd
import math
import datetime
import os
from io import BytesIO
from streamlit_folium import st_folium

from HumanSensing_Visualization import map_loader as ml

@st.cache_data
def load_data(filename):
    command = f'SELECT * FROM sensordata'

    connection = sqlite3.connect(filename)
    # cur = con.cursor()
    # query = cur.execute(command)
    raw_data = pd.read_sql_query(command, connection)

    return raw_data

@st.cache_data
def load_and_preprocess_data(file_path: str):
    """
    Loads data from .sqlite file in specified file path and applies preprocessing
    :param file_path: path to the .sqlite or .sqlite3 file
    :return: pd.DataFrame containing preproceessed physiological data
    """
    signal_data = msp.MOS_detection_signal_preparation(file_path)

    print("Preprocessed Signals: \n", signal_data, "\n --------------------------------")
    return signal_data

#Generates and returns the dataframe containing GPS records of the given .sqlite file
@st.cache_data
def get_location_data(sqlite_file, start_date = 0, end_date = float("inf")):
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
    if math.isfinite(end_date):
        command = "SELECT * FROM location WHERE location.time_millis >= {} AND location.time_millis <= {};".format(start_date, end_date)
    else:
        command = "SELECT * FROM location WHERE location.time_millis >= {};".format(start_date)
    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()
    query = cur.execute(command)
    location_data = query.fetchall()
    location_data = pd.DataFrame(location_data,
                                columns=["id", "time_millis", "time_iso", "latitude", "longitude", "provider",
                                        "horizontal_accuracy", "altitude", "vertical_accuracy", "bearing", "bearing_accuracy", "speed", "speed_accuracy"])
    return location_data


@st.cache_data
def filter_location_data(dataframe):
    """
    Calls an outter filter_location_data() and caches it - caching solution
    """
    dataframe = ml.filter_location_data(dataframe)
    return dataframe

@st.cache_data
def get_survey_data(file_path: str):

    command = f'SELECT * FROM survey'
    #connection = sqlite3.connect(file_path)
    #survey_data = pd.read_sql_query(command, connection)

    con = sqlite3.connect(file_path)
    cur = con.cursor()
    query = cur.execute(command)
    survey_data = query.fetchall()

    return survey_data

@st.cache_data
def get_marker_data(file_path: str):

    command = f'SELECT * FROM marker'
    connection = sqlite3.connect(file_path)
    marker_data = pd.read_sql_query(command, connection)

    return marker_data

# if time differences between raw signal data and locations data exist, align them
@st.cache_data
def align_data_location_time_diff(dataframe, locations):

    dataframe_hour = dataframe['time_iso'].max().hour
    locations_hour = locations['time_iso'].max().hour
    # print("\nDataframe hour ", dataframe_hour)
    # print("Locations hour ", locations_hour)
    if dataframe_hour < locations_hour:
        time_diff = (locations['time_iso'].max() - dataframe['time_iso'].max()).total_seconds() // 3600
        dataframe['time_iso'] = pd.to_datetime(dataframe['time_iso']) + datetime.timedelta(
            hours=time_diff)
        print("\n\n Locations hour is higher.. \nSignal Dataframe hour: {} \nLocations hour: {}".format(
            dataframe_hour, locations_hour))
    elif dataframe_hour > locations_hour:
        time_diff = (dataframe['time_iso'].max() - locations['time_iso'].max()).total_seconds() // 3600
        dataframe['time_iso'] = pd.to_datetime(dataframe['time_iso']) - datetime.timedelta(
            hours=time_diff)
        print("\n\n Signal dataframe hour is higher.. \nSignal Dataframe hour: {} \nLocations hour: {}".format(
            dataframe_hour, locations_hour))

    else:
        print("\nNo time differences between raw dataframe and locations dataframe exist")

    print(
        "\nFinal output: \nSensor data end time: {} \nLocations data end time: {}".format(dataframe['time_iso'].max(),
                                                                                       locations['time_iso'].max()))
    print("--------------------")
    return dataframe

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

def convert_df_to_gpkg(dataframe):
    shp = BytesIO()
    dataframe.to_file(shp, driver='GPKG')
    return shp

def save_geojson_with_bytesio(dataframe):
    # Function to return bytesIO of the geojson
    shp = BytesIO()
    dataframe.to_file(shp, driver='GeoJSON')
    return shp

def save_uploadedfile(uploaded_file, path: str):
    with open(os.path.join(path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved file: {} to {}".format(uploaded_file.name, path))

st.header("Select an .sqlite database file generated by the eDiary app and start analyzing your data:")

path = pathlib.Path(__file__).parent.resolve()
st.markdown("---")

######## File uploader ########

uploaded_sqlite_file = st.file_uploader("Drag and drop your .sqlite file here...", type=["sqlite", 'sqlite3'])
st.info("Upload .sqlite files")

if uploaded_sqlite_file is not None:

    #### Map and location-based stats
    st.markdown("---")
    st.sidebar.title("Location Data, Statistics & Maps")

    try:
        save_uploadedfile(uploaded_file=uploaded_sqlite_file, path=path)
        st.write(f"Saving file {uploaded_sqlite_file} to: {path}")

        ######### Optional eDiary Tables Display #########
        st.sidebar.title("Display table information")
        #preped_data = st.sidebar.checkbox("Show/Hide preprocessed data:")
        location_data_display = st.sidebar.checkbox("Show/Hide Information contained in Locations Table:")
        sensor_data_display = st.sidebar.checkbox("Show/Hide Information contained in Sensordata Table:")

        preprocessed_data = load_and_preprocess_data(os.path.join(path, uploaded_sqlite_file.name))
        #st.write("Preprocessed Data: ", preprocessed_data)
        #st.write("with Data types:", preprocessed_data.dtypes)

        sensordata = load_data(os.path.join(path, uploaded_sqlite_file.name))
        if sensor_data_display:
            st.write("Raw sensordata table Data: ", sensordata)

        location_data = get_location_data(os.path.join(path, uploaded_sqlite_file.name))
        if location_data_display:
            st.write("Raw locations table Data: ", location_data)

        #survey_data = get_survey_data(os.path_join(path, uploaded_sqlite_file.name))
        #st.write("eDiary Survey table Data: ", survey_data)

        #st.write(len(survey_data))
        preprocessed_data['time_iso'] = pd.to_datetime(preprocessed_data['time_iso'])
        location_data['time_iso'] = pd.to_datetime(location_data['time_iso'])

        data_aligned = align_data_location_time_diff(preprocessed_data, location_data)
        #st.write("Aligned Data ", data_aligned)

    except:
        pass

    st.sidebar.title("Location Data")

    if st.sidebar.checkbox("Show Location Data"):
        try:
            trackdf = filter_location_data(location_data)
            download_geoJSON = st.sidebar.download_button("Download track as GeoJSON",
                                                          data=save_geojson_with_bytesio(trackdf),
                                                          file_name=uploaded_sqlite_file.name.split('.')[0] + ".geojson",
                                                          mime="application/geo+json")

            st.header("Map and location-based Statistics")
            display_track_stats = st.checkbox("Display Track Statistics:")
        except (ValueError, RuntimeError, TypeError, NameError):
            print("Error in Show Location Data section")

        if display_track_stats:

            try:

                # generates a pd.DataFrame with base distance and time stats (avg. speed, track time etc.)
                get_base_stats = ml.get_location_based_stats(trackdf)

                st.dataframe(get_base_stats)


            except (ValueError, RuntimeError, TypeError, NameError):
                print("Error in Display_track_stats section")

        display_track = st.checkbox('Display Track Map:')

        if display_track:
            try:
                add_MOS = st.checkbox("Upload your MOS Output File")
                if add_MOS:

                    MOS_file = st.file_uploader()

                    #final_MOS_output2 = final_MOS_output.dropna()
                    #trackdf_MOS_merge = trackdf.merge(final_MOS_output2, on="time_iso", how="left")

                    display_track_mos = st.checkbox('Display generated MOS')

                    if display_track_mos:
                        #map = ml.create_map_with_track_and_MOS(trackdf_MOS_merge, add_MOS=True)
                        #st_map = st_folium(map, width=1000)
                        st.write("TO BE IMPLEMENTED")
                    else:
                        map = ml.create_map_with_track_and_MOS(trackdf)
                        st_map = st_folium(map, width=1000)
                else:
                    map = ml.create_map_with_track_and_MOS(trackdf)
                    st_map = st_folium(map, width=1000)
            except (ValueError, RuntimeError, TypeError, NameError):
                print("Error in Display_track section")

            try:
                # segmenting the tracks section
                segment_track_number = st.number_input("Number of track segments: ", min_value=2)
                apply_segment = st.checkbox("Apply")

                # clean the dataframe before the segmentation
                locations_segment = ml.remove_outliers(location_data)
                locations_segment = ml.horizontal_accuracy_filter(locations_segment)
                locations_segment = ml.distance_starting_point(locations_segment)

                if apply_segment:
                    # segments the track by adding a new 'category' column
                    locations_segment = ml.segment_track(locations_segment, segment_track_number)
                    # st.write(locations)

                    # st.write("Categorized points")
                    # st.write(locations_segment)

                    # iterates through each track based on 'category' column and returns a new DataFrame
                    get_grouped_locations_stats = ml.get_location_based_stats_segmentation(locations_segment)
                    st.table(get_grouped_locations_stats)

            except (ValueError, RuntimeError, TypeError, NameError):
                print("Error in Segment_track section")