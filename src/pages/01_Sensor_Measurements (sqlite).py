import streamlit as st
import sqlite3
import pathlib
import os
import pandas as pd
import math
import datetime
from io import BytesIO
import random
import time

import plotly.express as px
import plotly.graph_objects as go

from HumanSensing_Preprocessing import preprocess_signals as pps
from MOS_Detection import MOS_signal_preparation as msp
from HumanSensing_Preprocessing import sensor_check


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

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

# the main, branching part of the application
if uploaded_sqlite_file is not None:
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

        st.markdown("---")
        st.header("Raw Data vs. Preprocessed (filtered) Data")

        #display_raw_data = st.checkbox("Show/Hide tables")

        #st.sidebar.title("Preprocessed Data Download")




        #if display_raw_data:
        try:
            raw_data_col1, raw_data_col2 = st.columns(2, gap="medium")

            with raw_data_col1:
                st.subheader("Raw Data")
                st.caption(
                    "Table showing the raw data stored in the .sqlite database file generated by the eDiary App:")
                st.dataframe(sensordata)

                # download preprocessed data as .xslx
                #buffer = BytesIO()
                #with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                #    sensordata.to_excel(writer, sheet_name='Sheet1')

                #download_excel = st.download_button(label="Download raw data as Excel", data=buffer,
                #                                            file_name=uploaded_sqlite_file.name.split('.')[
                #                                                          0] + "raw_ediary_data" + ".xlsx",
                #                                            mime="application/vnd.ms-excel")

            with raw_data_col2:
                st.subheader("Preprocessed Data")
                st.caption("Table showing the preprocessed data resampled to 1 Hz sampling frequency:")
                st.dataframe(preprocessed_data)

                # download preprocessed data as .xslx
                #buffer = BytesIO()
                #with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                #    preprocessed_data.to_excel(writer, sheet_name='Sheet1')

                #download_excel = st.download_button(label="Download preprocessed data as Excel", data=buffer,
                #                                            file_name=uploaded_sqlite_file.name.split('.')[
                #                                                          0] + "preprocessed_ediary_data" + ".xlsx",
                #                                            mime="application/vnd.ms-excel")

        except (ValueError, RuntimeError, TypeError, NameError):
            print("Something wrong in 'Preprocessed Data' Section")

        finally:
            pass

    except:
        pass


######### Raw Data Visualization #########
st.sidebar.title("Raw Data Visualization")

visualize_raw_data = st.sidebar.checkbox("Visualize raw Data (to be implemented...)")

if visualize_raw_data:

    filtering_raw_data_col1, filtering_raw_data_col2 = st.columns([1, 2])

    with filtering_raw_data_col1:
        if sensor_check.E4_used(
                os.path.join(path, uploaded_sqlite_file.name)) == True and sensor_check.BioHarness_used(
            os.path.join(path, uploaded_sqlite_file.name)) == True:

            signal_options_raw = st.multiselect('Select *one* of the following raw signals to visualize:',
                                                ['GSR', 'ST', 'IBI', 'ECG', 'HRV'])
        elif sensor_check.E4_used(
                os.path.join(path, uploaded_sqlite_file.name)) == True and sensor_check.BioHarness_used(
            os.path.join(path, uploaded_sqlite_file.name)) == False:
            signal_options_raw = st.multiselect('Select *one* of the following raw signals to visualize:',
                                                ['GSR', 'ST'])

        else:
            print("Nope")

    with filtering_raw_data_col2:
        st.warning(f"This feature is not implemented yet - will be included soon.")

st.markdown("---")
######### Raw Data Visualization #########
st.sidebar.title("Preprocessed Data Visualization")

st.header("Prepocessed Signal Visualization & optional Filtering")

filtering_data_col1, filtering_data_col2 = st.columns([1, 2])

try:

    with filtering_data_col1:
        if sensor_check.E4_used(
                os.path.join(path, uploaded_sqlite_file.name)) == True and sensor_check.BioHarness_used(
            os.path.join(path, uploaded_sqlite_file.name)) == True:
            signal_options = st.multiselect('Select *one* of the following signals to filter and visualize:',
                                            ['GSR', 'ST', 'IBI', 'ECG', 'HRV'])

        elif sensor_check.E4_used(
                os.path.join(path, uploaded_sqlite_file.name)) == True and sensor_check.BioHarness_used(
            os.path.join(path, uploaded_sqlite_file.name)) == False:
            signal_options = st.multiselect('Select *one* of the following signals to filter and visualize:',
                                            ['GSR', 'ST'])

        else:
            print("Nope")

    with filtering_data_col2:
        try:
            duration = pd.to_datetime(preprocessed_data['time_iso'].max()) - pd.to_datetime(preprocessed_data['time_iso'].min())
            st.sidebar.subheader("Time Filter:")
            st.sidebar.write("Recording duration: \t ", duration.seconds, "seconds \t => \t",
                     time.strftime("%H:%M:%S", time.gmtime(duration.seconds)))

            double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display",
                                                           value=[duration.seconds - duration.seconds,
                                                                  duration.seconds])
            # double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display", min_value = start_date, value= end_date, max_value= end_date, format=format)
            start_time = preprocessed_data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[0])
            end_time = preprocessed_data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[1])
            filtered_data = preprocessed_data[(preprocessed_data['time_iso'] >= start_time) & (preprocessed_data['time_iso'] <= end_time)]
            st.dataframe(filtered_data)

            # if GSR or ST selected, apply Butterworth filter with provided values

            st.sidebar.subheader("Frequency Filter: ")
            st.sidebar.info("Note: You need to select 'ST' or 'GSR' in 'Prepocessed Signal Visualization & optional Filtering' Section")

            if len(signal_options) == 0:
                pass

            elif str(signal_options[0]) == "GSR":

                apply_butterworth = st.sidebar.checkbox("Apply / Change butterworth filter")
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
                apply_butterworth = st.sidebar.checkbox("Apply / Change butterworth filter")
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
                    filtered_data = pps.load_ecg_data_from_file(os.path.join(path, uploaded_sqlite_file.name))

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

    st.markdown("---")

    if 'IBI' in filtered_data:

        fig = px.line(filtered_data, x='time_iso', y=['GSR', 'ST', 'IBI', 'HRV'], title="Preprocessed signals plot")

        fig.update_layout(width=1500, height=600)

        st.plotly_chart(fig)

    else:
        fig = px.line(filtered_data, x='time_iso', y=['GSR', 'ST'], title="Preprocessed signals plot")

        fig.update_layout(width=1500, height=600)

        st.plotly_chart(fig)


    st.sidebar.title("Preprocessed Data Download")

    preprocessed_data_download_checkbox = st.sidebar.checkbox("Download preprocessed data")

    filtered_data_download_checkbox = st.sidebar.checkbox("Download time & frequency filtered data")


    if preprocessed_data_download_checkbox:
        st.subheader("Prepocessed Data (1 Hz sampling Frequency)")
        st.write(preprocessed_data)

        if st.checkbox("Download preprocessed data as CSV"):
            csv_to_download1 = convert_df_to_csv(preprocessed_data)
            st.download_button("Start Download", csv_to_download1, uploaded_sqlite_file.name + ".csv",
                               key='download-csv')
            st.write("Saved file as", uploaded_sqlite_file.name + ".csv")

    if filtered_data_download_checkbox:
        st.subheader("Time & Frequency filtered Data (1 Hz sampling Frequency)")
        st.write(filtered_data)

        if st.checkbox("Download preprocessed and time-filtered data as CSV"):
            csv_to_download1 = convert_df_to_csv(filtered_data)
            st.download_button("Start Download", csv_to_download1, uploaded_sqlite_file.name + ".csv",
                               key='download-csv')
            st.write("Saved file as", uploaded_sqlite_file.name + ".csv")



except:
    pass

#"""
#    finally:
#        # another try statement that wraps up this one and finally that deletes files there
#        store_directory = os.getcwd()
#        files_wd_csv = [file for file in os.listdir(store_directory) if
#                        file.endswith((".csv", ".xlsx", ".sqlite", ".sqlite3"))]
#
#        # st.info(f"All csv files in the WD: {files_wd_csv}")
#
#        # remove temporarily stored files (.csv, .xlsx, etc.)
#        for file in files_wd_csv:
#            path_to_file = os.path.join(store_directory, file)
#            os.remove(path_to_file)
#
#        # os.remove(os.path.join(path, uploaded_data_files.name))
#
#        files_wd = [f for f in os.listdir(store_directory) if os.path.isfile(f)]
#        # st.info(f"All files in the WD after removing temporarily stored files: {files_wd}")
#
#"""
