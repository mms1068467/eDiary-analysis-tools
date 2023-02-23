import streamlit as st
import sqlite3
import pathlib
import shutil
import pandas as pd


from HumanSensing_Visualization import map_loader as ml
from HumanSensing_Preprocessing import process_zip_files as pzf

@st.cache_data
def MOS_analysis_fieldstudy_folder(read_zip_files):
    output_noNa, combined_output = pzf.MOS_analysis_fieldstudy_folder(read_zip_files)

    return output_noNa, combined_output

@st.cache_data
def NEW_MOS_analysis_fieldstudy_folder(read_zip_files, MOS_threshold, start_time_trim, end_time_trim):
    output_noNa, combined_output = pzf.NEW_MOS_analysis_fieldstudy_folder(read_zip_files, MOS_threshold, start_time_trim, end_time_trim)

    return output_noNa, combined_output

@st.cache_data
def sensor_measurements_fieldstudy_folder(read_zip_files):

    combined_sensor_data = pzf.sensor_measurements_fieldstudy_folder(read_zip_files)

    return combined_sensor_data

@st.cache_data
def location_data_fieldstudy_folder(read_zip_files):

    combined_location_data = pzf.location_data_fieldstudy_folder(read_zip_files)

    return combined_location_data

@st.cache_data
def survey_data_fieldstudy_folder(read_zip_files):

    combined_survey_data = pzf.survey_data_fieldstudy_folder(read_zip_files)

    return combined_survey_data


path = pathlib.Path(__file__).parent.resolve()
st.markdown("---")

st.header("Select a .zip folder with field study .sqlite database files collected by the eDiary app:")

uploaded_zip_files = st.file_uploader("Drag and drop your zip folder(s) here...", type=["zip", "7z"],
                                      accept_multiple_files=False)
st.info("Upload zip folder")

if uploaded_zip_files is not None:

    try:
        # reads zip files and stores in temporary directory
        read_zip_files = pzf.open_and_extract_zip(path, uploaded_zip_files)

        st.sidebar.title("Combined Field Study Data")

        st.sidebar.subheader("Combined Sensor Measurements")

        show_combined_sensor_data = st.sidebar.checkbox("Show combined sensor data:")

        # TODO - include time filtering option, where users can exclude a certain amount of seconds from start & end
        # functionality is already implemented in 02_Stress_Analysis(sqlite).py, but needs to be adapted for zip file case

        #st.sidebar.subheader("Time-based Filtering")

        if show_combined_sensor_data:

            combined_sensor_data = sensor_measurements_fieldstudy_folder(read_zip_files)
            st.subheader("Sensor Data for all .sqlite files in .zip folder")
            st.write(combined_sensor_data)

        st.sidebar.subheader("Raw Sensor Measurements")
        # TODO - use the following script to get the raw measurements
        #https: // git.sbg.ac.at / geo - social - analytics / human_sensing / human - sensing / - / blob / MLvsRuleMOSdetection / src / Automation / raw_data_to_csv.py

        st.sidebar.subheader("Combined Location (Phone) Measurements")

        show_combined_location_data = st.sidebar.checkbox("Show combined location data:")

        if show_combined_location_data:

            combined_location_data = location_data_fieldstudy_folder(read_zip_files)
            st.subheader("Location Data for all .sqlite files in .zip folder")
            st.write(combined_location_data)

        st.sidebar.subheader("Combined Survey (eDiary) Entries")

        show_combined_survey_data = st.sidebar.checkbox("Show combined survey data:")

        if show_combined_survey_data:

            combined_survey_data = survey_data_fieldstudy_folder(read_zip_files)
            st.subheader("Survey Data for all .sqlite files in .zip folder")
            st.write(combined_survey_data)


        st.sidebar.title("Stress Analysis")

        st.sidebar.subheader("Selection Stress algorithm to apply:")

        kyriakou_2019 = st.sidebar.checkbox("MOS Detection - Kyriakou et al. (2019)")
        moser_2023 = st.sidebar.checkbox("MOS Detection - Moser et al. (2023)")


        if kyriakou_2019:

            st.subheader("Stress Detection based on Kyriakou et al. (2023)")
            st.info("https://pubmed.ncbi.nlm.nih.gov/31484366/")

            # reads zip files, converges them into a single dataframe
            MOS_analysis_fieldstudy_noNA, MOS_analysis_fieldstudy = MOS_analysis_fieldstudy_folder(read_zip_files)

            st.write(MOS_analysis_fieldstudy_noNA)

            number_of_mos_detected = len(MOS_analysis_fieldstudy_noNA[MOS_analysis_fieldstudy_noNA['MOS_score'] >= 75])

            st.write("Detected Number of MOS (overall): ", number_of_mos_detected)

        if moser_2023:

            st.subheader("Stress Detection based on Moser et al. (2023)")

            ###### Time Trim which is excluded from Baseline Calculation(s)
            start_time_trim = st.number_input(
                "Number of seconds to trim from start (transient phase) - excluded from the baseline calculation (default = 3 minutes)",
                value=180)
            end_time_trim = st.number_input(
                "Number of seconds to trim from end (when person took off sensor) - excluded from the baseline calculation (default = 30 seconds)",
                value=30)

            ###### MOS Threshold
            MOS_thresh = st.number_input("Please enter the desired MOS threshold: ", value=0.75)

            MOS_analysis_fieldstudy_noNA, MOS_analysis_fieldstudy = NEW_MOS_analysis_fieldstudy_folder(read_zip_files, MOS_threshold = MOS_thresh,
                                               start_time_trim = start_time_trim,
                                               end_time_trim = end_time_trim)

            st.write(MOS_analysis_fieldstudy_noNA)

            number_of_mos_detected = len(MOS_analysis_fieldstudy_noNA[MOS_analysis_fieldstudy_noNA['detectedMOS'] == 1])

            st.write("Detected Number of MOS (overall): ", number_of_mos_detected)

        # removes the temporary directory
        shutil.rmtree(read_zip_files)


    except:
        pass
