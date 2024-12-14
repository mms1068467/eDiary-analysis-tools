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

from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check
# MOS Detection Kyriakou et al. 2019
from MOS_Detection import MOS_rules_paper_verified as mrp
from MOS_Detection import MOS_signal_preparation as msp
# MOS Detection Moser et al. 2023
from MOS_Detection import MOS_signal_preparation_verified as msp_new
from MOS_Detection import MOS_rules_NEW as MOS_paper_new
#import MOS_Detection.MOS_rules_NEW as MOS_paper_new


def check_same_hour(df1: pd.DataFrame, df2: pd.DataFrame, datetime_column: str) -> pd.Series:
    """
    Compares the 'datetime' columns in two DataFrames and checks if they have the same hour.

    :param df1: First pandas DataFrame.
    :param df2: Second pandas DataFrame.
    :param datetime_column: The name of the datetime column in both DataFrames.
    :return: A pandas Series of boolean values indicating whether the hours match.
    """
    
    # Ensure datetime columns are in pandas datetime format
    df1[datetime_column] = pd.to_datetime(df1[datetime_column])
    df2[datetime_column] = pd.to_datetime(df2[datetime_column])
    
    # Extract the hour from both datetime columns
    df1['hour'] = df1[datetime_column].dt.hour
    df2['hour'] = df2[datetime_column].dt.hour
    
    # Compare the hours
    same_hour = df1['hour'] - df2['hour']
    
    return same_hour

def merge_on_matching_seconds(df1: pd.DataFrame, df2: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Merges two DataFrames based on matching seconds in the datetime column.
    
    :param df1: First pandas DataFrame.
    :param df2: Second pandas DataFrame.
    :param datetime_column: The name of the datetime column in both DataFrames.
    :return: A merged pandas DataFrame based on matching seconds.
    """
    
    # Ensure datetime columns are in pandas datetime format
    df1[datetime_column] = pd.to_datetime(df1[datetime_column])
    df2[datetime_column] = pd.to_datetime(df2[datetime_column])
    
    # Round the datetime values to the nearest second by using .dt.floor('S')
    df1[datetime_column] = df1[datetime_column].dt.floor('S')
    df2[datetime_column] = df2[datetime_column].dt.floor('S')
    
    # Merge the dataframes on the datetime column
    merged_df = pd.merge(df1, df2, on=datetime_column, how='left')
    
    return merged_df

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


st.header("Select an .sqlite database file generated by the eDiary app to analyze Stress:")


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

        st.sidebar.title("Filtering:")

        st.sidebar.subheader("Time-based Filtering")

        duration = pd.to_datetime(preprocessed_data['time_iso'].max()) - pd.to_datetime(preprocessed_data['time_iso'].min())
        st.sidebar.write("Recording Duration: \t ", duration.seconds, "seconds \t => \t",
                 time.strftime("%H:%M:%S", time.gmtime(duration.seconds)))

        double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display",
                                                       value=[duration.seconds - duration.seconds,
                                                              duration.seconds])
        # double_ended_seconds_slider_filter = st.sidebar.slider("Seconds after start of recording to display", min_value = start_date, value= end_date, max_value= end_date, format=format)
        start_time = preprocessed_data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[0])
        end_time = preprocessed_data['time_iso'].min() + datetime.timedelta(seconds=double_ended_seconds_slider_filter[1])
        filtered_data = preprocessed_data[(preprocessed_data['time_iso'] >= start_time) & (preprocessed_data['time_iso'] <= end_time)]

        st.subheader("Time-filtered Data:")
        st.dataframe(filtered_data)

        st.sidebar.subheader("Frequency-based Filtering (optional - not recommended):")

        change_gsr_butterworth = st.sidebar.checkbox("Change GSR butterworth filter")

        if change_gsr_butterworth:
            st.subheader('Butterworth filter for GSR Signal')
            butterworth_pass_filter = st.number_input('Butter filter order', value=1)
            low_pass_filter = st.number_input('Low-pass cutoff frequency', value=0.50, step=0.01,
                                              format="%.2f")
            high_pass_filter = st.number_input('High-pass cutoff frequency', value=0.03, step=0.01,
                                               format="%.2f")

            filtered_data = pps.preprocess_GSR(data=filtered_data, order=butterworth_pass_filter,
                                               lowpass_cutoff_frequency=low_pass_filter,
                                               highpass_cutoff_frequency=high_pass_filter)

        apply_st_butterworth = st.sidebar.checkbox("Change ST butterworth filter")
        if apply_st_butterworth:
            st.subheader('Butterworth filter for ST Signal')
            butterworth_pass_filter = st.number_input('Butter filter order', value=2)
            low_pass_filter = st.number_input('Low-pass cutoff frequency', value=0.07 / (4 / 2), step=0.001,
                                              format="%.3f")
            high_pass_filter = st.number_input('High-pass cutoff frequency', value=0.025, step=0.001,
                                               format="%.3f")

            filtered_data = pps.preprocess_ST(data=filtered_data, order=butterworth_pass_filter,
                                              lowpass_cutoff_frequency=low_pass_filter,
                                              highpass_cutoff_frequency=high_pass_filter)


        ######### Stress Detection #########
        st.sidebar.title("Stress Detection:")
        #preped_data = st.sidebar.checkbox("Show/Hide preprocessed data:")
        kyriakou_2019 = st.sidebar.checkbox("MOS Detection - Kyriakou et al. (2019)")
        moser_2023 = st.sidebar.checkbox("MOS Detection - Moser et al. (2023)")
        st.sidebar.info("Note that Moser et al. (2023) works only on the full data as of right now (no time filtering)...")

        st.header("Stress Detection Output")

        if kyriakou_2019:

            filtered_data_kyriakou = filtered_data.copy()
            final_MOS_output, extended_MOS_output = mrp.MOS_main_df(df=filtered_data_kyriakou)

            st.subheader("MOS Detection Output")
            st.write("\n **Number of MOS Detected:** \n", len(final_MOS_output[~final_MOS_output['MOS_score'].isna()]),
                     '\n')
            st.write(final_MOS_output[~final_MOS_output['MOS_score'].isna()])

            convert_df_to_csv(final_MOS_output)

            # convert_df_to_csv(path = path, file_name = uploaded_data_files.name + "MOS", df = final_MOS_output)
            final_MOS_output.to_csv("MOS_output_" + uploaded_sqlite_file.name + ".csv")

            kyriakou_2019_visualize_mos = st.checkbox("Visualize MOS based on Kyriakou et al. (2019)")

            if kyriakou_2019_visualize_mos:
                # MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")
                # MOS_detected = MOS_csv_file[~MOS_csv_file['MOS_score'].isna()]

                MOS_detected = final_MOS_output[~final_MOS_output['MOS_score'].isna()]

                if 'IBI' in filtered_data.columns:
                    fig = px.line(filtered_data, x='time_iso', y=['GSR', 'ST', 'IBI', 'HRV'], title="Preprocessed signals plot")
                else:
                    fig = px.line(filtered_data, x='time_iso', y=['GSR', 'ST'], title="Preprocessed signals plot")
                    
                # figMOS.update_layout(width=1500, height=600)

                # adding detected MOS (stress moments)
                for i in MOS_detected.index:
                    if MOS_detected.MOS_score[i] > 75:
                        fig.add_vline(MOS_detected['time_iso'][i], line_dash='dash', line_color='black')
                        fig.update_layout(title="Preprocessed signals plot with MOS")

                    # adding detected stress moments
                    # for i in gt.index:
                    #    if gt.stress[i] == 1:
                    #        # fig.add_vline(lab2_5_1['time_iso'][i], line_color = 'red')
                    #        fig.add_trace(go.Scatter(x=[filename['time_iso'][i]],
                    #                                 y=[filename['HRV'][i]],
                    #                                 mode='markers',
                    #                                 marker=dict(color='red', size=10),
                    #                                 showlegend=False))

                    # st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(fig)

            st.subheader("Please select one of the Download Options:")

            # TODO - plain MOS without coordinates
            if st.checkbox("Download MOS as CSV"):
                csv_to_download1 = convert_df_to_csv(final_MOS_output)
                st.download_button("Start Download", csv_to_download1, uploaded_sqlite_file.name + ".csv",
                                           key='download-csv')
                st.write("Saved file as", uploaded_sqlite_file.name + ".csv")

            # TODO - geo-referenced MOS
            if st.checkbox("Download geo-coded MOS as CSV"):
                final_MOS_output_geo = final_MOS_output.copy()
                final_MOS_output_geo['time_iso'] = pd.to_datetime(final_MOS_output_geo['time_iso'])
                location_data['time_iso'] = pd.to_datetime(location_data['time_iso'])

                st.write(location_data)

                st.write(final_MOS_output_geo)

                hour_diff_location_data = check_same_hour(location_data, final_MOS_output_geo, datetime_column='time_iso')
                
                if np.sum(hour_diff_location_data) > 0:
                    location_data["time_iso"] = location_data["time_iso"] - pd.Timedelta(hours=1)
                else:
                    location_data["time_iso"] = location_data["time_iso"] + pd.Timedelta(hours=1)

                # Merge the dataframes based on matching seconds
                if 'time_iso' in location_data.columns:
                    merged_df = merge_on_matching_seconds(final_MOS_output_geo, location_data, datetime_column='iso_time')
                    st.write(merged_df)
                else:
                    st.write("Merge problem - timestamp compatibility issue")

                #time_diff = (location_data['time_iso'].max() - final_MOS_output_geo[
                #    'time_iso'].max()).total_seconds() // 3600
                # TODO - check if this works for all files
                #  --> potentially use absolute value of time_diff if it is negative
                #if time_diff < 0:
                #    time_diff = 0
                #print("\nlocations", location_data['time_iso'].max())
                #print("MOS identified", final_MOS_output['time_iso'].max())
                #print("Time Difference:", time_diff)
                #final_MOS_output_geo['time_iso'] = pd.to_datetime(
                #    final_MOS_output_geo['time_iso']) + datetime.timedelta(
                #    hours=time_diff)

                #if 'time_iso' in location_data.columns:
                #    merged = final_MOS_output_geo.merge(
                #        location_data[['time_iso', 'latitude', 'longitude', 'altitude', 'speed']],
                #        left_on='time_iso', right_on='time_iso', how='left')
                #else:
                #    print("Merge problem - timestamp compatibility issue")

                csv_to_download2 = convert_df_to_csv(merged)
                st.write(merged)
                st.download_button("Start Download", csv_to_download2, uploaded_sqlite_file.name + ".csv",
                                           key='download-csv')
                st.write("Save file as", uploaded_sqlite_file.name + ".csv")


        if moser_2023:

            #MOS_data_prep = msp.MOS_detection_signal_preparation(file)

            st.subheader("Hyperparameters for MOS Detection - Moser et al. (2023)")

            filtered_data_moser = filtered_data.copy()

            initial_start_time = filtered_data_moser['time_iso'].min()
            #st.write(initial_start_time)
            initial_end_time = filtered_data_moser['time_iso'].max().round('1s')
            #st.write(initial_end_time)

            ###### Time Trim which is excluded from Baseline Calculation(s)
            start_time_trim = st.number_input(
                "Number of seconds to trim from start (transient phase) - excluded from the baseline calculation (default = 3 minutes)",
                value=180)
            end_time_trim = st.number_input(
                "Number of seconds to trim from end (when person took off sensor) - excluded from the baseline calculation (default = 30 seconds)",
                value=30)

            ###### MOS Threshold
            MOS_thresh = st.number_input("Please enter the desired MOS threshold: ", value=0.75)

            start_time_base = initial_start_time + pd.to_timedelta(start_time_trim, "s")
            #st.write(start_time_base)
            end_time_base = initial_end_time - pd.to_timedelta(end_time_trim, "s")
            #st.write(end_time_base)


            #MOS_analysis_fieldstudy_noNA, MOS_analysis_fieldstudy = MOS_paper_new.MOS_main_df(filtered_data_moser, MOS_thresh = MOS_thresh,
            #                                   start_time_trim = start_time_base,
            #                                   end_time_trim = end_time_base)




            #st.write(MOS_analysis_fieldstudy_noNA)

            filtered_data1 = msp_new.derive_GSR_and_ST_features(filtered_data)

            #st.write(filtered_data1)

            if "IBI" in filtered_data1:
                data_ready_f1 = msp_new.derive_IBI_and_HRV_features(filtered_data1)
                #st.write(data_ready_f1)
            else:
                data_ready_f1 = filtered_data1.copy()

            # FIXME - Here's the bug when time filtering is used

            # add features for GSR rules
            data_ready_f11 = MOS_paper_new.GSR_amplitude_duration_slope(data_ready_f1)
            #st.write(data_ready_f11)

            # TODO - add start_time_trim and end_time_trim values for trimmed baseline calculation as input
            threshold_data = data_ready_f11.set_index("time_iso")[start_time_base:end_time_base]

            amplitude_mean = MOS_paper_new.calculate_GSR_ampltiude_baseline(threshold_data)
            amplitude_std = MOS_paper_new.calculate_GSR_ampltiude_spread(threshold_data)
            # st.write(amplitude_mean, amplitude_std)
            duration_mean = MOS_paper_new.calculate_GRS_duration_baseline(threshold_data)
            duration_std = MOS_paper_new.calculate_GRS_duration_spread(threshold_data)
            # st.write(duration_mean, duration_std)
            slope_mean = MOS_paper_new.calculate_GSR_Slope_baseline(threshold_data)
            slope_std = MOS_paper_new.calculate_GSR_Slope_spread(threshold_data)

            MOS_out_martin = MOS_paper_new.MOS_rules_apply_n(data_ready_f11,
                                                             amplitude_mean=amplitude_mean,
                                                             amplitude_std=amplitude_std,
                                                             slope_mean=slope_mean,
                                                             slope_std=slope_std,
                                                             MOSpercentage=MOS_thresh)

            detected_MOS = MOS_out_martin[MOS_out_martin['MOS_Score'] > MOS_thresh]
            df_GSR_rules_met = utilities.check_timestamp_gaps(detected_MOS, duration=10,
                                                              col_name="MOS_not_within_10_seconds")

            mos_identified = df_GSR_rules_met[
                df_GSR_rules_met['MOS_not_within_10_seconds'] == True]

            MOS_gsr_and_st_clean = pd.merge(
                MOS_out_martin[MOS_out_martin.columns.difference(["detectedMOS"])],
                mos_identified[['time_iso', "detectedMOS"]],
                on='time_iso', how='left')

            MOS_gsr_and_st_clean["detectedMOS"] = MOS_gsr_and_st_clean["detectedMOS"].fillna(0)

            number_of_mos = len(MOS_gsr_and_st_clean[MOS_gsr_and_st_clean['detectedMOS'] > 0])

            st.write("Length of data", len(MOS_gsr_and_st_clean))

            st.write("Number of MOS Detected: ", number_of_mos)

            #MOS_output_ordered = MOS_gsr_and_st_clean[["time_iso", "TimeNum", "GSR", "GSR_standardized", "ST", "ST_standardized",
            #                                           "HRV", "hrv_filtered", "rmsnn", "sdnn", "IBI", "filtered_IBI", "MOS_Score", "detectedMOS"]]
            #st.write(MOS_gsr_and_st_clean)
            
            # new version
            MOS_detected = MOS_gsr_and_st_clean[MOS_gsr_and_st_clean['detectedMOS'] > 0]
            st.write(MOS_detected)

            moser_2023_visualize_mos = st.checkbox("Visualize MOS based on Moser et al. (2023)")

            if moser_2023_visualize_mos:
                # MOS_csv_file = pd.read_csv(path + "MOS_output_" + uploaded_data_files.name + ".csv")
                # MOS_detected = MOS_csv_file[~MOS_csv_file['MOS_score'].isna()]

                # older version
                #MOS_detected = MOS_output_ordered[~MOS_output_ordered['MOS_Score'].isna()]


                if 'IBI' in filtered_data.columns:
                    figM = px.line(filtered_data_moser, x='time_iso', y=['GSR', 'ST', 'IBI', 'HRV'], title="Preprocessed signals plot")
                else:
                    figM = px.line(filtered_data_moser, x='time_iso', y=['GSR', 'ST'], title="Preprocessed signals plot")
                 
                # figMOS.update_layout(width=1500, height=600)

                # adding detected MOS (stress moments)
                for i in MOS_detected.index:
                    if MOS_detected.detectedMOS[i] > 0:
                        figM.add_vline(MOS_detected['time_iso'][i], line_dash='dash', line_color='black')
                        figM.update_layout(title="Preprocessed signals plot with MOS")

                    # adding detected stress moments
                    # for i in gt.index:
                    #    if gt.stress[i] == 1:
                    #        # fig.add_vline(lab2_5_1['time_iso'][i], line_color = 'red')
                    #        fig.add_trace(go.Scatter(x=[filename['time_iso'][i]],
                    #                                 y=[filename['HRV'][i]],
                    #                                 mode='markers',
                    #                                 marker=dict(color='red', size=10),
                    #                                 showlegend=False))

                    # st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(figM)

            st.subheader("Please select one of the Download Options:")

            # TODO - plain MOS without coordinates
            if st.checkbox("Download MOS as CSV"):
                csv_to_download3 = convert_df_to_csv(MOS_gsr_and_st_clean)
                st.download_button("Start Download", csv_to_download3, uploaded_sqlite_file.name + ".csv",
                                           key='download-csv')
                st.write("Saved file as", uploaded_sqlite_file.name + ".csv")


            # TODO - geo-referenced MOS
            if st.checkbox("Download geo-coded MOS as CSV"):
                final_MOS_output_geo = MOS_gsr_and_st_clean.copy()
                final_MOS_output_geo['time_iso'] = pd.to_datetime(final_MOS_output_geo['time_iso'])
                location_data['time_iso'] = pd.to_datetime(location_data['time_iso'])

                time_diff = (location_data['time_iso'].max() - final_MOS_output_geo[
                    'time_iso'].max()).total_seconds() // 3600
                # TODO - check if this works for all files
                #  --> potentially use absolute value of time_diff if it is negative
                if time_diff < 0:
                    time_diff = 0
                print("\nlocations", location_data['time_iso'].max())
                print("MOS identified", MOS_output_ordered['time_iso'].max())
                print("Time Difference:", time_diff)
                final_MOS_output_geo['time_iso'] = pd.to_datetime(
                    final_MOS_output_geo['time_iso']) + datetime.timedelta(
                    hours=time_diff)

                if 'time_iso' in location_data.columns:
                    merged = final_MOS_output_geo.merge(
                        location_data[['time_iso', 'latitude', 'longitude', 'altitude', 'speed']],
                        left_on='time_iso', right_on='time_iso', how='left')
                else:
                    print("Merge problem - timestamp compatibility issue")

                csv_to_download4 = convert_df_to_csv(merged)
                st.write(merged)
                st.download_button("Start Download", csv_to_download4, uploaded_sqlite_file.name + ".csv",
                                           key='download-csv')
                st.write("Save file as", uploaded_sqlite_file.name + ".csv")


        st.sidebar.title("Stress Data Download:")




    except:
        pass
