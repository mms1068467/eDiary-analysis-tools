"""
Human Sensing Streamlit -- Version 2.0

New functionalities:
- Updated Signal Visualization based on user selections & option to download
- Track Visualization (leaflet) including track statistics and track export option as .gpkg
- Updated MOS Algorithm (in progress / needs further testing)

"""

import sys
import datetime
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import streamlit as st
import math
import pandas as pd
import numpy as np
import typing
import openpyxl
import os

#map libaries
import folium
from folium import plugins
import geopandas
from streamlit_folium import st_folium
from geopy import distance
from scipy import signal

from io import BytesIO
import xlsxwriter
import random

import plotly.express as px
import plotly.graph_objects as go

from io import StringIO # for hashing the .sqlite file

from MOS_Detection import MOS_signal_preparation as msp
# from MOS_Detection import MOS_rules as rules
# from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check
# from MOS_Detection import MOS_parameters as mp
# from MOS_Detection import MOS_main as mm
from MOS_Detection import MOS_rules_paper_verified as mrp
from HumanSensing_Preprocessing import data_loader as dl
from HumanSensing_Visualization import map_loader as ml

##### Functions

## SQLite (eDiary and locations data) load and preprocess data

def save_uploadedfile(uploaded_file, path: str):
    with open(os.path.join(path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved file: {} to {}".format(uploaded_file.name, path))


@st.cache
def load_data(filename):

    command = f'SELECT * FROM sensordata'

    connection = sqlite3.connect(filename)
    #cur = con.cursor()
    #query = cur.execute(command)
    raw_data = pd.read_sql_query(command, connection)

    return raw_data


@st.cache(
    allow_output_mutation=True)  # sometimes cache of this function casuses errors, the allow_output_mutation fixes for now
def load_and_preprocess_data(file_path: str):
    """
    Loads data from .sqlite file in specified file path and applies preprocessing
    :param file_path: path to the .sqlite or .sqlite3 file
    :return: pd.DataFrame containing preproceessed physiological data
    """
    signal_data = msp.MOS_detection_signal_preparation(file_path)

    print("Preprocessed Signals: \n", signal_data, "\n --------------------------------")
    return signal_data


@st.cache
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

#Generates and returns the dataframe containing GPS records of the given .sqlite file
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


@st.cache(allow_output_mutation=True)
def filter_location_data(dataframe):
    """
    Calls an outter filter_location_data() and caches it - caching solution
    """
    dataframe = ml.filter_location_data(dataframe)
    return dataframe


#### Page setup

st.set_page_config(
    page_title="ESSEM eDiary Data Visualization",
    page_icon=None,
    layout="wide"
)

st.title("ESSEM Data Visualization Dashboard")

expander_project_description = st.expander("Click Here to learn more about the ESSEM project")

    #Vorlage fuer ESSEM Projekt Dashboard: https://github.com/czubert/SERSitiVIS

with expander_project_description:
    st.info("""
        ESSEM stands for "Emotion Sensing für (E-)Fahrradsicherheit und Mobilitätskomfort" \n
        -------------------------------------------------------------------------------------

        Leveraging physiological data collected through non-invasive, wearable sensor technology in combination with other
        contextual data sources relating to perceived feeling of safety when cycling, this project aims to provide insights 
        into how an improved Cycling Infrastructure and (E-)Bike components can contribute
         to an optimized and safer traffic situation.

        """)

st.header("Select an .sqlite database file generated by the eDiary app and start analyzing your data:")

# If neccessary, add an users local path in order to use and store the files from the app
# path = st.text_input("Please enter the path where you want to store the project data INCLUDING a '/' at the end and press Enter (Example: C:/Users/projects/data/)" )
path = Path(__file__).parent.resolve()
st.markdown("---")

######## File uploader

st.header("Drag and drop")
# TODO - file uploader for a single and multiple files: (accept_multiple_files = True flag)
uploaded_data_files = st.file_uploader("Drag and drop your .sqlite file(s) here...", type=["sqlite", 'sqlite3'])
st.info("Upload .sqlite files")
# st.info(data_file.type)
# st.write(type(uploaded_data_files))


# the main, branching part of the application
if uploaded_data_files is not None:
    try:
        save_uploadedfile(uploaded_file=uploaded_data_files, path=path)
        # alread preprocessed, should actually replace with raw data first?
        data = load_and_preprocess_data(os.path.join(path, uploaded_data_files.name))
        raw_data = load_data(os.path.join(path, uploaded_data_files.name))
        ## TODO - this location import is new
        locations = get_location_data(os.path.join(path, uploaded_data_files.name))
        data['time_iso'] = pd.to_datetime(data['time_iso'])
        locations['time_iso'] = pd.to_datetime(locations['time_iso'])

        data_original = data.copy()
        # signal_data  = geolocate(signal_data=data, locations = locations)
        # st.dataframe(signal_data)

        # TODO - import raw data for visualization if rule(s) are met
        # _, raw_GSR = dl.get_raw_GSR(os.path.join(path, uploaded_data_files.name))
        # _, raw_ST = dl.get_raw_ST(os.path.join(path, uploaded_data_files.name))
        # raw_IBI = dl.get_raw_IBI(os.path.join(path, uploaded_data_files.name))

        # st.write(data_combined)
        st.markdown("---")
        st.header("Raw data and automatic preprocessing")
        st.caption("In this section you can visualize your raw and preprocessed dataframes")

        display_raw_data = st.checkbox("Show/hide tables")

        st.sidebar.title("Preprocessed Data Download")

        if display_raw_data:
            try:
                raw_data_col1, raw_data_col2 = st.columns(2, gap="medium")

                with raw_data_col1:
                    st.subheader("Raw Data")
                    st.caption("Table showing the raw data stored in the .sqlite database file generated by the eDiary App:")
                    st.dataframe(raw_data)

                with raw_data_col2:
                    st.subheader("Preprocessed Data")
                    st.caption("Table showing the preprocessed data resampled to 1 Hz sampling frequency:")
                    st.dataframe(data)

                    if st.sidebar.checkbox("Download preprocessed data"):

                        # download preprocessed data as .xslx
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            data.to_excel(writer, sheet_name='Sheet1')

                        download_excel = st.sidebar.download_button(label="Download table as Excel", data=buffer,
                                                            file_name=uploaded_data_files.name.split('.')[0] + ".xlsx",
                                                            mime="application/vnd.ms-excel")

                        # download preprocessed data as .csv
                        preprocessed_data_to_csv = convert_df_to_csv(data)
                        download_csv = st.sidebar.download_button("Download table as CSV", preprocessed_data_to_csv,
                                                          uploaded_data_files.name.split('.')[0] + ".csv", key=1)


            except (ValueError, RuntimeError, TypeError, NameError):
                print("Something wrong in visualize_raw_data checkbox section")

        ######### Raw Data Visualization #########

        st.sidebar.title("Raw Data Visualization")

        visualize_raw_data = st.sidebar.checkbox("Visualize raw Data")
        if visualize_raw_data:

            filtering_raw_data_col1, filtering_raw_data_col2 = st.columns([1, 2])

            with filtering_raw_data_col1:
                if sensor_check.E4_used(
                        os.path.join(path, uploaded_data_files.name)) == True and sensor_check.BioHarness_used(
                    os.path.join(path, uploaded_data_files.name)) == True:

                    signal_options_raw = st.multiselect('Select *one* of the following raw signals to visualize:',
                                                    ['GSR', 'ST', 'IBI', 'ECG', 'HRV'])
                elif sensor_check.E4_used(
                        os.path.join(path, uploaded_data_files.name)) == True and sensor_check.BioHarness_used(
                    os.path.join(path, uploaded_data_files.name)) == False:
                    signal_options_raw = st.multiselect('Select *one* of the following raw signals to visualize:',
                                                    ['GSR', 'ST'])

                else:
                    print("Nope")

            with filtering_raw_data_col2:
                st.warning(f"This feature is not implemented yet - will be included soon.")


                #if len(signal_options_raw) > 0:

                    #try:

                        #st.subheader("Filtering visualization")

                        # Signal filtering and line chart creation

                        #if str(signal_options_raw[0]) == 'IBI':

                            #IBI_raw_data = raw_data[ (raw_data["platform_id"] == 2) & (raw_data["sensor_id"] == 16)]

                            #raw_IBI = pps.format_raw_IBI(raw_data)
                            #st.write(raw_IBI)

                            #st.caption("Raw Inter Beat Interval (IBI)")

                                #fig = go.Figure(data=go.Scatter(x=raw_data["time_iso"], y=raw_data["IBI"],
                                #                                line=dict(color="#00CC96")))

                                #describe_data = raw_data["IBI"].describe(include='all').fillna("").astype("str")

                                #st.dataframe(raw_data[['time_iso', 'IBI']])


                            #elif str(signal_options_raw[0]) == "GSR":
                            #    st.caption("Galvanic Skin Response(GSR)")

                            #    fig = go.Figure(data=go.Scatter(x=filtered_data["time_iso"], y=filtered_data["GSR"],
                            #                                    line=dict(color="#636EFA")))

                            #    describe_data = filtered_data["GSR"].describe(include='all').fillna("").astype("str")

                            #    st.dataframe(filtered_data[['time_iso', 'GSR']])

                    #except(ValueError, RuntimeError, TypeError, NameError):
                    #    print("Rng {}: Something wrong in filtering_raw_data_col2".format(random.randint(0, 50)))

        ######### Preprocessed Data Visualization #########

        #st.sidebar.title("Preprocessed Data Visualization")

        #if st.sidebar.checkbox("Hide Prepocessed Signal Visualization & optional Filtering"):

        st.markdown("---")
        st.header("Prepocessed Signal Visualization & optional Filtering")


        filtering_data_col1, filtering_data_col2 = st.columns([1, 2])

        with filtering_data_col1:
            if sensor_check.E4_used(
                    os.path.join(path, uploaded_data_files.name)) == True and sensor_check.BioHarness_used(
                    os.path.join(path, uploaded_data_files.name)) == True:
                signal_options = st.multiselect('Select *one* of the following signals to filter and visualize:',
                                                ['GSR', 'ST', 'IBI', 'ECG', 'HRV'])

            elif sensor_check.E4_used(
                    os.path.join(path, uploaded_data_files.name)) == True and sensor_check.BioHarness_used(
                    os.path.join(path, uploaded_data_files.name)) == False:
                signal_options = st.multiselect('Select *one* of the following signals to filter and visualize:',
                                                ['GSR', 'ST'])

            else:
                print("Nope")

        with filtering_data_col2:
            try:
                duration = pd.to_datetime(data['time_iso'].max()) - pd.to_datetime(data['time_iso'].min())
                st.write("Duration of recording: \t ", duration.seconds, "seconds \t => \t", time.strftime("%H:%M:%S", time.gmtime(duration.seconds)))

                double_ended_seconds_slider_filter = st.slider("Seconds after start of recording to display",
                                                               value=[duration.seconds - duration.seconds,
                                                                      duration.seconds])
                # double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display", min_value = start_date, value= end_date, max_value= end_date, format=format)
                start_time = data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[0])
                end_time = data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[1])
                filtered_data = data[(data['time_iso'] >= start_time) & (data['time_iso'] <= end_time)]

                # if GSR or ST selected, apply Butterworth filter with provided values
                if len(signal_options) == 0:
                    pass

                elif str(signal_options[0]) == "GSR":
                    apply_butterworth = st.checkbox("Apply / Change butterworth filter")
                    if apply_butterworth:
                        butterworth_pass_filter = st.number_input('Butter filter order', value=1)
                        low_pass_filter = st.number_input('Low-pass cutoff frequency', value=0.50, step=0.01,
                                                          format="%.2f")
                        high_pass_filter = st.number_input('High-pass cutoff frequency', value=0.03, step=0.01,
                                                           format="%.2f")

                        filtered_data = pps.preprocess_GSR(data=filtered_data, order=butterworth_pass_filter,
                                                           lowpass_cutoff_frequency=low_pass_filter,
                                                           highpass_cutoff_frequency=high_pass_filter)

                elif str(signal_options[0]) == "ST":
                    apply_butterworth = st.checkbox("Apply / Change butterworth filter")
                    if apply_butterworth:
                        st.write('Butterworth filter')
                        butterworth_pass_filter = st.number_input('Butter filter order', value=2)
                        low_pass_filter = st.number_input('Low-pass cutoff frequency', value=0.07 / (4 / 2), step=0.001,
                                                          format="%.3f")
                        high_pass_filter = st.number_input('High-pass cutoff frequency', value=0.025, step=0.001,
                                                           format="%.3f")

                        filtered_data = pps.preprocess_ST(data=filtered_data, order=butterworth_pass_filter,
                                                          lowpass_cutoff_frequency=low_pass_filter,
                                                          highpass_cutoff_frequency=high_pass_filter)


            except(ValueError, RuntimeError, TypeError, NameError):
                print("Rng {}: Something wrong in filtering_data_col2".format(random.randint(0, 50)))

        if len(signal_options) > 0:

            try:

                st.markdown("---")

                filtered_data_col1, filtered_data_col2, filtered_data_col3 = st.columns([1, 1, 2])

                with filtered_data_col1:

                    st.subheader("Filtering visualization")

                    # Signal filtering and line chart creation

                    if str(signal_options[0]) == 'IBI':

                        st.caption("Inter Beat Interval (IBI)")

                        fig = go.Figure(data=go.Scatter(x=filtered_data["time_iso"], y=filtered_data["IBI"],
                                                        line=dict(color="#00CC96")))

                        describe_data = filtered_data["IBI"].describe(include='all').fillna("").astype("str")

                        st.dataframe(filtered_data[['time_iso', 'IBI']])


                    elif str(signal_options[0]) == "GSR":
                        st.caption("Galvanic Skin Response(GSR)")

                        fig = go.Figure(data=go.Scatter(x=filtered_data["time_iso"], y=filtered_data["GSR"],
                                                        line=dict(color="#636EFA")))

                        describe_data = filtered_data["GSR"].describe(include='all').fillna("").astype("str")

                        st.dataframe(filtered_data[['time_iso', 'GSR']])


                    elif str(signal_options[0]) == "ST":
                        st.caption("Skin Temperature (ST)")

                        fig = go.Figure(data=go.Scatter(x=filtered_data["time_iso"], y=filtered_data["ST"],
                                                        line=dict(color="#EF553B")))

                        describe_data = filtered_data["ST"].describe(include='all').fillna("").astype("str")

                        st.dataframe(filtered_data[['time_iso', 'ST']])


                    elif str(signal_options[0]) == "HRV":
                        st.caption("Heart Rate Variability (HRV)")

                        fig = go.Figure(data=go.Scatter(x=filtered_data["time_iso"], y=filtered_data["HRV"],
                                                        line=dict(color="#AB63FA")))

                        describe_data = filtered_data["HRV"].describe(include='all').fillna("").astype("str")

                        st.dataframe(filtered_data[['time_iso', 'HRV']])


                    elif str(signal_options[0]) == "ECG":
                        filtered_data = pps.load_ecg_data_from_file(os.path.join(path, uploaded_data_files.name))

                        plot_data_form = pd.melt(filtered_data[15000:16500], id_vars=['time_millis'],
                                                 value_vars=['ecg_values'],
                                                 value_name='Amplitude [mV]')

                        Signal_subset = plot_data_form.query("variable == 'ecg_values'")

                        fig = px.line(data_frame=Signal_subset, x='time_millis', y='Amplitude [mV]', color='variable',
                                      title="ECG Diagram")

                        describe_data = filtered_data["ecg_values"].describe(include='all').fillna("").astype("str")

                        st.dataframe(filtered_data)

                    else:
                        st.write("Please select one of the signals listed: GSR, ST, IBI, ECG, HRV")

                # Second column for base descriptive statistics of the data from the first column
                with filtered_data_col2:
                    try:
                        st.subheader("Descriptive statistics")
                        st.write(describe_data)
                    except (ValueError, RuntimeError, TypeError, NameError):
                        print("Error in filtered_data_col2")
                # Third column for visualizing the data from the first column
                with filtered_data_col3:
                    try:
                        st.subheader("Graphical visualization")
                        st.plotly_chart(fig)
                    except (ValueError, RuntimeError, TypeError, NameError):
                        print("Error in filtered_data_col3")


            except (ValueError, RuntimeError, TypeError, NameError):
                print("Something wrong in the visualization section !")

        #else:

        st.markdown("---")

        fig = px.line(filtered_data, x='time_iso', y=['GSR', 'ST', 'IBI', 'HRV'], title="Preprocessed signals plot")

        fig.update_layout(width=1500, height=600)

        st.plotly_chart(fig)

        #### MOS Detection

        st.sidebar.title("MOS Detection")

        if st.sidebar.checkbox("Generate MOS"):
            final_MOS_output, extended_MOS_output = mrp.MOS_main_df(df=filtered_data)
            # final_MOS_output = mm.MOS_algorithm(signal_data = data)

            st.subheader("MOS Detection Output")
            st.write("\n **Number of MOS Detected:** \n", len(final_MOS_output[~final_MOS_output['MOS_score'].isna()]),
                     '\n')
            st.write(final_MOS_output[~final_MOS_output['MOS_score'].isna()])

            convert_df_to_csv(final_MOS_output)

            # convert_df_to_csv(path = path, file_name = uploaded_data_files.name + "MOS", df = final_MOS_output)
            final_MOS_output.to_csv("MOS_output_" + uploaded_data_files.name + ".csv")

            # TODO - plain MOS without coordinates
            if st.sidebar.checkbox("Download MOS as .csv"):
                csv_to_download = convert_df_to_csv(final_MOS_output)
                st.sidebar.download_button("Start Download", csv_to_download, uploaded_data_files.name + ".csv",
                                   key='download-csv')
                st.sidebar.write("Save file as", uploaded_data_files.name + ".csv")

            # TODO - geo-referenced MOS
            if st.sidebar.checkbox("Download geo-coded MOS as .csv"):
                final_MOS_output_geo = final_MOS_output.copy()
                final_MOS_output_geo['time_iso'] = pd.to_datetime(final_MOS_output_geo['time_iso'])
                locations['time_iso'] = pd.to_datetime(locations['time_iso'])

                time_diff = (locations['time_iso'].max() - final_MOS_output_geo['time_iso'].max()).total_seconds() // 3600
                # TODO - check if this works for all files
                #  --> potentially use absolute value of time_diff if it is negative
                if time_diff < 0:
                    time_diff = 0
                #st.write("locations", locations['time_iso'].max())
                #st.write("MOS identified", final_MOS_output['time_iso'].max())
                #st.write("Time Difference:", time_diff)

                final_MOS_output_geo['time_iso'] = pd.to_datetime(final_MOS_output_geo['time_iso']) + datetime.timedelta(
                    hours=time_diff)

                if 'time_iso' in locations.columns:
                    merged = final_MOS_output_geo.merge(locations[['time_iso', 'latitude', 'longitude', 'altitude', 'speed']],
                                                left_on='time_iso', right_on='time_iso', how='left')
                else:
                    print("Merge problem - timestamp compatibility issue")

                csv_to_download = convert_df_to_csv(merged)
                st.sidebar.download_button("Start Download", csv_to_download, uploaded_data_files.name + ".csv",
                                   key='download-csv')
                st.sidebar.write("Save file as", uploaded_data_files.name + ".csv")

        if st.sidebar.checkbox("Visualize MOS"):
            #MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")
            #MOS_detected = MOS_csv_file[~MOS_csv_file['MOS_score'].isna()]

            MOS_detected = final_MOS_output[~final_MOS_output['MOS_score'].isna()]

            #figMOS = px.line(filtered_data, x='time_iso', y=['GSR', 'ST', 'IBI', 'HRV'], title="Preprocessed signals plot")

            #figMOS.update_layout(width=1500, height=600)


            # adding detected MOS (stress moments)
            for i in MOS_detected.index:
                if MOS_detected.MOS_score[i] > 75:
                    fig.add_vline(MOS_detected['time_iso'][i], line_dash = 'dash',line_color='black')

                # adding detected stress moments
                # for i in gt.index:
                #    if gt.stress[i] == 1:
                #        # fig.add_vline(lab2_5_1['time_iso'][i], line_color = 'red')
                #        fig.add_trace(go.Scatter(x=[filename['time_iso'][i]],
                #                                 y=[filename['HRV'][i]],
                #                                 mode='markers',
                #                                 marker=dict(color='red', size=10),
                #                                 showlegend=False))

                #st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig)

        #### Map and location-based stats
        st.markdown("---")
        st.sidebar.title("Location Data, Statistics & Maps")

        if st.sidebar.checkbox("Show Location Data"):
            st.header("Map and location-based Statistics")
            display_track_stats = st.checkbox("Display Track Statistics:")
            display_track = st.checkbox('Display Track:')
            # st.write(st.session_state.key)

            # filters out the locations data by removing outliers based on distance(default: 10m) and on horizontal accuracy (default: >=100m)
            trackdf = filter_location_data(locations)

            if display_track_stats:

                try:

                    # generates a pd.DataFrame with base distance and time stats (avg. speed, track time etc.)
                    get_base_stats = ml.get_location_based_stats(trackdf)

                    st.dataframe(get_base_stats)

                    st.markdown("---")
                    st.markdown("---")

                except (ValueError, RuntimeError, TypeError, NameError):
                    print("Error in Track stats section")

            if display_track:

                # generate a map and plot it with specified width
                map = ml.create_map_with_track_and_MOS(trackdf)
                st_map = st_folium(map, width=1000)



            st.markdown("---")
            st.markdown("---")

    except (ValueError, RuntimeError, TypeError, NameError):
        print("\nRng{}: Unable to process your request ! Something wrong in the main section".format(
            random.randint(0, 50)))

    finally:
        os.remove(os.path.join(path, uploaded_data_files.name))


