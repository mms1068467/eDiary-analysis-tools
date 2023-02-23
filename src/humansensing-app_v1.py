import datetime
import sys

#sys.path.append(r"\humansensing-webapp")

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import streamlit as st
import math
import pandas as pd
import numpy as np
import typing


import os

from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go

from io import StringIO # for hashing the .sqlite file

from MOS_Detection import MOS_signal_preparation as msp
# from MOS_Detection import MOS_rules as rules
# from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import preprocess_signals as pps
# from MOS_Detection import MOS_parameters as mp
# from MOS_Detection import MOS_main as mm
from MOS_Detection import MOS_rules_paper_verified as mrp
from HumanSensing_Preprocessing import data_loader as dl

#  set path where the humansensing-app_v1.py file is located
#path = "C:/Users/MM/Desktop/Uni Salzburg/P.hD/ZGis/Human Sensing/MOS_Detection/MOS_algo_Martin/"


def save_uploadedfile(uploaded_file, path: str):
    with open(os.path.join(path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved file: {} to {}".format(uploaded_file.name, path))

@st.cache
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')


# TODO - avoid the upload and processing of the file every time when visualizing different things --> use @st.cache decorator but avoid error
#@st.cache
@st.experimental_memo
def load_and_preprocess_data(file_path: str):
    """
    Loads data from .sqlite file in specified file path and applies preprocessing
    :param file_path: path to the .sqlite or .sqlite3 file
    :return: pd.DataFrame containing preproceessed physiological data
    """
    signal_data = msp.MOS_detection_signal_preparation(file_path)

    print("Preprocessed Signals: \n", signal_data, "\n --------------------------------")
    return signal_data

@contextmanager
def sqlite_connect(db_bytes):
    fp = Path(str(uuid4()))
    fp.write_bytes(db_bytes.getvalue())
    conn = sqlite3.connect(str(fp))

    try:
        yield conn
    finally:
        conn.close()
        fp.unlink()


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

def geolocate(signal_data: pd.DataFrame, location_data: pd.DataFrame) -> pd.DataFrame:
    """
    Geolocates sensor measurements by merging measurements with location-based smartphone data
    :param signal_data: DataFrame containing preprocessed Signals
    :param location_data: DataFrame containing data from smartphones (location (lat, long), speed, etc.)
    :return: Georeferenced sensor measurements with NaN values at timestamps where there is no phone measurements
    """

    location_data['ts_rounded'] = pd.to_datetime(location_data['time_iso']).dt.round(freq='S')

    merged_geolocated_data = pps.fix_timestamp_issue_ediary(signal_data = signal_data, location_data = location_data)
    #print(merged_geolocated_data[~merged_geolocated_data['latitude'].isna()])

    if "IBI" in merged_geolocated_data.columns:

        merged_geolocated_data = merged_geolocated_data[['time_iso_x', 'GSR', 'ST', 'IBI', 'HRV', 'latitude', 'longitude', 'speed', 'TimeNum']]
        merged_geolocated_data.columns = ['time_iso', 'GSR', 'ST', 'IBI', 'HRV', 'Lat', 'Lon', 'speed', 'TimeNum']

    else:
        merged_geolocated_data = merged_geolocated_data[['time_iso_x', 'GSR', 'ST', 'latitude', 'longitude', 'speed', 'TimeNum']]
        merged_geolocated_data.columns = ['time_iso', 'GSR', 'ST', 'Lat', 'Lon', 'speed', 'TimeNum']

    return merged_geolocated_data


def calender_filter(GSR_ST_data_long):
    GSR_ST_data_long['time_iso'] = pd.to_datetime(GSR_ST_data_long['time_iso'])
    start_dt = st.sidebar.date_input('Start date', value=GSR_ST_data_long['time_iso'].min())
    end_dt = st.sidebar.date_input('End date', value=GSR_ST_data_long['time_iso'].max())
    if start_dt <= end_dt:
        df = GSR_ST_data_long[GSR_ST_data_long['time_iso'] > datetime(start_dt.year, start_dt.month, start_dt.day)]
        df = GSR_ST_data_long[GSR_ST_data_long['time_iso'] < datetime(end_dt.year, end_dt.month, end_dt.day)]
        st.write(GSR_ST_data_long)
    else:
        st.error('Start date must be > End date')



st.set_page_config(
    page_title="ESSEM-physiological-data-Visualization",
    page_icon=None,
    layout="wide"
)

st.title("ESSEM Data Visualization")

expander_project_description = st.expander("Click Here to learn more about the ESSEM project")
with expander_project_description:
    st.info("""
    ESSEM stands for .... \n
    The goal of the project is to enhance traffic safety through the use of physiological data collected 
    through sensors and a centralized app named eDiary
    """)

st.subheader("Visualizing sensor data from the eDiary app")


# TODO - path to root directory -- redundant -- "" is sufficient
#st.info("Enter path below and press Enter --- Example: \n C:/Users/projects/data/ ")
#path = st.text_input("Please enter the path where you stored this project INCLUDING a / at the end" )
# path to root directory
#path = Path(__file__).parent.resolve()
#st.write(path)

path = ""

st.text("Vorlage fuer ESSEM Projekt Dashboard: https://github.com/czubert/SERSitiVIS")

st.markdown("---")

container_app_description = st.container()
with container_app_description:
    st.markdown("With this app, you can drop your .sqlite file that was generated by the eDiary app and it will be analyzed")


######## File uploader


# TODO - file uploader for a single and multiple files: (accept_multiple_files = True flag)
uploaded_data_files = st.file_uploader("Drag and drop your .sqlite file(s) here...", type = ["sqlite", 'sqlite3'])
st.info("Upload .sqlite files")
#st.info(data_file.type)
#st.write(type(uploaded_data_files))

if uploaded_data_files is not None:
    try:
        save_uploadedfile(uploaded_file=uploaded_data_files, path = path)
        data = load_and_preprocess_data(os.path.join(path, uploaded_data_files.name))#
        ## TODO - this location import is new
        locations = get_location_data(os.path.join(path, uploaded_data_files.name))
        data['time_iso'] = pd.to_datetime(data['time_iso'])
        locations['time_iso'] = pd.to_datetime(locations['time_iso'])
        # signal_data  = geolocate(signal_data=data, locations = locations)
        # st.dataframe(signal_data)

        # TODO - import raw data for visualization if rule(s) are met
        # _, raw_GSR = dl.get_raw_GSR(os.path.join(path, uploaded_data_files.name))
        # _, raw_ST = dl.get_raw_ST(os.path.join(path, uploaded_data_files.name))
        # raw_IBI = dl.get_raw_IBI(os.path.join(path, uploaded_data_files.name))

        #st.write(data_combined)

        st.dataframe(data)
        st.dataframe(locations)

    except (ValueError, RuntimeError, TypeError, NameError):
        print("Unable to process your request dude!!")
    # TODO - this "else" statement is redundant

    finally:
        os.remove(os.path.join(path, uploaded_data_files.name))


#TODO - visualize raw signal data
#TODO - visualize track of location recordings


st.sidebar.title("Selections & Filters:")

if st.sidebar.checkbox("Show number of recordings per minute"):

    st.subheader("GSR and ST recordings by Minute")
    hist_values = np.histogram(
        data['time_iso'].dt.minute, bins = 60
    )[0]

    st.bar_chart(hist_values)

# TODO - Data Sanity Check
# if st.sidebar.checkbox("Visualize raw data"):
#     st.subheader("Raw Data with vakyes out of range")
#     mos_data = data[['time_iso', 'GSR', 'ST', 'MOS_score']]
#     st.write(mos_data[~mos_data['MOS_score'].isnull()])
      #fig_GSR = px.line(data_frame = mos_data, x = 'time_iso', y = 'GSR', title = 'GSR with red boxes to indicate out of range values')
      #fig_ST.add_hrect(y0 = -1, y1 = -10, line_width = 0, fillcolor = 'red', opacity = 0.2)
      #fig_ST.add_hrect(y0 = 1, y1 = 10, line_width = 0, fillcolor = 'red', opacity = 0.2)
      #fig_ST = px.line(data_frame = mos_data, x = 'time_iso', y = 'ST', title = 'ST with red boxes to indicate out of range values')
      #fig_GSR.add_hrect(y0 = -0.4, y1 = -10, line_width = 0, fillcolor = 'red', opacity = 0.2)
      #fig_GSR.add_hrect(y0 = 0.4, y1 = 10, line_width = 0, fillcolor = 'red', opacity = 0.2)

        
if st.sidebar.button("Generate MOS"):

    final_MOS_output, extended_MOS_output = mrp.MOS_main_df(df = data)
    #final_MOS_output = mm.MOS_algorithm(signal_data = data)

    st.subheader("MOS Detection Output")
    st.write("\n **Number of MOS Detected:** \n", len(final_MOS_output[~final_MOS_output['MOS_score'].isna()]), '\n')
    st.write(final_MOS_output[~final_MOS_output['MOS_score'].isna()])

    convert_df_to_csv(final_MOS_output)

    #convert_df_to_csv(path = path, file_name = uploaded_data_files.name + "MOS", df = final_MOS_output)
    final_MOS_output.to_csv("MOS_output_" + uploaded_data_files.name + ".csv")

if st.sidebar.checkbox("Visualize MOS"):
    MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")
    MOS_detected = MOS_csv_file[~MOS_csv_file['MOS_score'].isna()]

    GSR_ST_data = data[['time_iso', 'GSR', 'ST']]

    GSR_ST_data_long = pd.melt(GSR_ST_data, id_vars=['time_iso'], value_vars=['GSR', 'ST'],
                               var_name='signal', value_name='value')
    #print(GSR_ST_data_long)

    st.subheader("Combined Visualization")

    fig = px.line(data_frame=GSR_ST_data_long, x='time_iso', y='value', color='signal',
                  title="Interactive Visualization of physiological data gathered (highlight area to zoom in/out")

    #fig.add_trace(go.Scatter(raw_GSR['time_iso'], raw_GSR['GSR_raw']))
    #fig.add_trace(go.Scatter(raw_ST['time_iso'], raw_ST['ST_raw']))

    # adding detected MOS (stress moments)
    for i in MOS_detected.index:
        if MOS_detected.MOS_score[i] > 75:
            fig.add_vline(MOS_detected['time_iso'][i], line_color='black')

    # adding detected stress moments
    #for i in gt.index:
    #    if gt.stress[i] == 1:
    #        # fig.add_vline(lab2_5_1['time_iso'][i], line_color = 'red')
    #        fig.add_trace(go.Scatter(x=[filename['time_iso'][i]],
    #                                 y=[filename['HRV'][i]],
    #                                 mode='markers',
    #                                 marker=dict(color='red', size=10),
    #                                 showlegend=False))


    st.plotly_chart(fig, use_container_width=True)


# TODO - plain MOS without coordinates
if st.button("Download MOS as .csv"):

    MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")

    csv_to_download = convert_df_to_csv(MOS_csv_file)
    st.download_button("Start Download", csv_to_download, uploaded_data_files.name + ".csv", key = 'download-csv')
    st.write("Save file as", uploaded_data_files.name + ".csv")

# TODO - geo-referenced MOS
if st.button("Download geo-coded MOS as .csv"):
    MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")
    MOS_csv_file['time_iso'] = pd.to_datetime(MOS_csv_file['time_iso'])
    locations['time_iso'] = pd.to_datetime(locations['time_iso'])

    time_diff = (locations['time_iso'].max() - MOS_csv_file['time_iso'].max()).total_seconds() // 3600
    # TODO - check if this works for all files
    #  --> potentially use absolute value of time_diff if it is negative
    if time_diff < 0:
        time_diff = 0
    st.write("locations", locations['time_iso'].max())
    st.write("MOS identified", MOS_csv_file['time_iso'].max())
    st.write("Time Difference:", time_diff)

    MOS_csv_file['time_iso'] = pd.to_datetime(MOS_csv_file['time_iso']) + datetime.timedelta(hours = time_diff)

    if 'time_iso' in locations.columns:
        merged = MOS_csv_file.merge(locations[['time_iso', 'latitude', 'longitude', 'altitude', 'speed']], left_on = 'time_iso', right_on = 'time_iso', how = 'left')
    else:
        print("Merge problem - timestamp compatibility issue")

    csv_to_download = convert_df_to_csv(merged)
    st.download_button("Start Download", csv_to_download, uploaded_data_files.name + ".csv", key = 'download-csv')
    st.write("Save file as", uploaded_data_files.name + ".csv")


# TODO - put this in "try:" clause of data file input

try:
    GSR_data = data[['time_iso', 'GSR']]
    ST_data = data[['time_iso', 'ST']]
except NameError:
    print("An eDiary .sqlite / .sqlite3 file needs to be uploaded for further processing")


signal_col1, signal_col2 = st.columns(2)
with signal_col1:
    st.subheader("Filtered GSR")
    st.text("(Scroll to zoom in/out)")
    try:
        st.line_chart(GSR_data.set_index('time_iso'), width = 800)
    except NameError:
        print("An eDiary .sqlite / .sqlite3 file needs to be uploaded for further processing")
with signal_col2:
    st.subheader("Filtered ST")
    st.text("(Scroll to zoom in/out)")
    try:
        st.line_chart(ST_data.set_index('time_iso'), width = 800)
    except NameError:
        print("An eDiary .sqlite / .sqlite3 file needs to be uploaded for further processing")


# TODO - plotly interactive chart

try:
    GSR_ST_data = data[['time_iso', 'GSR', 'ST']]

    GSR_ST_data_long = pd.melt(GSR_ST_data, id_vars=['time_iso'], value_vars=['GSR', 'ST'], var_name='signal', value_name='value')
    #print(GSR_ST_data_long)

    st.subheader("Combined Visualization")

    fig = px.line(data_frame=GSR_ST_data_long, x='time_iso', y='value', color='signal',
                  title="Interactive Visualization of physiological data gathered (highlight area to zoom in/out")

    st.plotly_chart(fig, use_container_width=True)
except NameError:
    print("An eDiary .sqlite / .sqlite3 file needs to be uploaded for further processing")

#minute_to_filter = st.slider('minute', pd.Timestamp(ST_data['time_iso'].min()), pd.Timestamp(ST_data['time_iso'].max()))
#start_time = st.slider("When do you want to start visualizing the signal??",
#                       min_value = pd.to_datetime(GSR_ST_data_long['time_iso'].min()))

#TODO - slider to select specific times
# TODO - create duration variable pd.to_datetime(GSR_ST_data_long['time_iso'].max()), pd.to_datetime(GSR_ST_data_long['time_iso'].min()) in seconds
# TODO - specify how many seconds after the start to slide

try:

    duration = pd.to_datetime(GSR_ST_data_long['time_iso'].max()) - pd.to_datetime(GSR_ST_data_long['time_iso'].min())
    st.write("Duration of recording: \t ", duration.seconds, "seconds")


    st.sidebar.header("Filter Options")

    st.header("Time-based Filtering of Signals")


    #################### Sliders ###############################
    # TODO - single ended slider
    #end_seconds_slider_filter = st.slider("Seconds after start of recording to display", min_value=0, max_value=duration.seconds)

    # TODO - double ended slider - could be extended to have date times
    double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display", value=[duration.seconds - duration.seconds , duration.seconds])

    # TODO - convert to datetime
    GSR_ST_data_long['time_iso'] = pd.to_datetime(GSR_ST_data_long['time_iso'])

    #st.write("Start", GSR_ST_data_long['time_iso'].min())
    #st.write("Start", GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = double_ended_seconds_slider_filter[0]))
    #start_time = GSR_ST_data_long['time_iso'].min()
    start_time = GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = double_ended_seconds_slider_filter[0])

    #st.write("End", GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = double_ended_seconds_slider_filter[1]))
    end_time = GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = double_ended_seconds_slider_filter[1])
    #st.write("End", GSR_ST_data_long['time_iso'] + datetime.timedelta(seconds = double_ended_seconds_slider_filter[1]))
    #end_time = GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = end_seconds_slider_filter)



    st.sidebar.markdown(
        f"""
        * **Start Time:** \t {start_time} \n
        * **End Time:**  \t {end_time}
        """
    )

    if st.sidebar.checkbox("Display filtered Data?"):
        filtered_data = GSR_ST_data_long[ ( GSR_ST_data_long['time_iso'] >= start_time ) & ( GSR_ST_data_long['time_iso'] <= end_time )] #GSR_ST_data_long['time_iso'].min() + datetime.timedelta(seconds = seconds_slider_filter) ]
        # Prepare plot
        table, figure = st.columns(2)
        with table:
            st.write("Filtered Data", filtered_data)
        with figure:
            st.write('Filtered Signals')
            fig2 = px.line(data_frame= filtered_data, x='time_iso', y='value', color='signal', title= "Data GSR and ST")
            st.plotly_chart(fig2, width = 800, height = 700)

except NameError:
    st.warning("An eDiary .sqlite / .sqlite3 file needs to be uploaded for further processing")







# TODO - map MOS detected (check if NAs can be handled)
#st.subheader("MOS on Map")
#st.map(mos_data[['latitude', 'longitude', 'mos_score']])




# TODO - receiving user input:
# text input with st.text_input or st.text_area("Text")
# numeric input with st.number_input("Text", min_value, max_value, step) --> integer/float
# checkboxes with st.checkbox() --> True/False
# sliders with st.slider("Text", min_value, max_value, value (default), step)
# list selection from a set of options with st.selectbox("Text", ("Option1", "Option2", "Option3"))


# TODO - import modules and use functionalities (maybe change folder structure)
#file_name = "../Data_MOS/LabData1-3/lab3/raw_data/Session3/2021-04-21T0941_5/zgis_phone_5_2021-04-21T0941.sqlite"
#signal_data = msp.MOS_detection_signal_preparation(filename = file_name)

#print(signal_data)


