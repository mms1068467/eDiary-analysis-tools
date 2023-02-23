
"""
Human Sensing Streamlit -- Version 3.0

New functionalities:
- Updated Signal Visualization based on user selections & option to download
- Track Visualization (leaflet) including track statistics and track export option as .gpkg
- Updated MOS Algorithm
- Set Visualize MOS sidebar checkbox as child element of Generate MOS sidebar checkbox
- Added Display generated MOS checkbox in Map and location-based Statistics section to plot MOS on the same map
- Added the segmentation of tracks functionality in Map and location-based Statistics section which returns a DataFrame with location-based statistics for each track
- Added the 'uploaded_zip_files' file uploader:
    - To present the merged MOS dataframe on a MAP
    - Removed the input path widget and set files to save locally in the default download folder
    - Added the separate download button for the data points containing MOS values
    - Removes temporarily stored files


"""


import streamlit as st

from io import BytesIO

from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import sensor_check

from MOS_Detection import MOS_signal_preparation as msp

from MOS_Detection import MOS_rules_paper_verified as mrp

from HumanSensing_Visualization import map_loader as ml
from HumanSensing_Preprocessing import process_zip_files as pzf

st.header("A Web Application for analyzing eDiary data:")

st.title("eDiary Data Analysis Dashboard")

expander_project_description = st.expander("Click Here to learn more about the eDiary App project")

with expander_project_description:
    st.info("""
        The eDiary smartphone app is an Android application,which enables sensor connections with two wearables (Empatica E4, Zephyr BioHarness) \n
        -------------------------------------------------------------------------------------

        Leveraging physiological data collected through non-invasive, wearable sensor technology has plenty of potential use cases
         ranging from health and traffic safety enhancement to city planning.
         
         Learn more about the eDiary App in the following paper by Petutschnig et al. (2022):
         
         https://www.mdpi.com/1424-8220/22/16/6120

        """)

