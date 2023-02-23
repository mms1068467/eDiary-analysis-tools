import os
from zipfile import ZipFile
from pathlib import Path
import importlib
import geopandas
import sqlite3
import pandas as pd
from io import BytesIO
from shapely.geometry import Point, LineString

from HumanSensing_Preprocessing import sensor_check
from HumanSensing_Preprocessing import data_loader as dl

# MOS Detection - Kyriakou et al. (2019)
from MOS_Detection import MOS_signal_preparation as msp
from MOS_Detection import MOS_rules_paper_verified as MOS_paper

# MOS Detection - Moser et al. (2023)
from MOS_Detection import MOS_signal_preparation_verified as msp_new
from MOS_Detection import MOS_rules_NEW as MOS_paper_new

#happ = importlib.import_module("humansensing-app")

#path = Path(__file__).parent.resolve()

#finds all sqlite files in specified directory and extracts as a list for further use
def find_all_sqlite_files(folder_path: str) -> list:
    
    sqlite_files = []

    if os.path.isfile(folder_path):
        sqlite_files.append(folder_path)

    else:
        for root, dirs, files in os.walk(folder_path):
            # print("Root: \n", root)
            # print("Directory: \n", dirs)
            # print('Files: \n', files)
            for file in files:
                if file.endswith('.sqlite') or file.endswith('.sqlite3'):
                    sqlite_files.append(os.path.join(root, file))

    print(f"Found {len(sqlite_files)} sqlite files. \n")
    return sqlite_files



def find_all_csv_xslx_files(folder_path: str) -> list:
    
    csv_xslx_files = []

    if os.path.isfile(folder_path):
        csv_xslx_files.append(folder_path)
        print("Yep")
    else:
        for root, dirs, files in os.walk(folder_path):
            # print("Root: \n", root)
            # print("Directory: \n", dirs)
            # print('Files: \n', files)
            for file in files:
                if file.endswith('.csv') or file.endswith('.xlsx'):
                    csv_xslx_files.append(os.path.join(root, file))

    print(f"Found {len(csv_xslx_files)} .csv and .xslx files. \n")
    return csv_xslx_files




#opens a specified zip and extracts it to folder in provided path with same name as .zip
def open_and_extract_zip(path, zip_folder: str) -> list:

    path_str = str(path)
    zip_folder_name = zip_folder.name.split(".")[0]

    folder_location = path_str + zip_folder_name
    print("File location: ", folder_location)
    if not os.path.isdir(folder_location):
        os.makedirs(folder_location)

    z = ZipFile(zip_folder)
    for name in z.namelist():
        print(name)
        z.extract(name, folder_location)

    return folder_location


def open_and_extract_zip_pat(path, zip_folder: str) -> list:

    path_str = str(path)
    zip_folder_name = zip_folder.name.split(".")[0] + zip_folder.name.split(".")[1]

    folder_location = path_str + zip_folder_name
    print("Folder location: ", folder_location)
    if not os.path.isdir(folder_location):
        os.makedirs(folder_location)

    z = ZipFile(zip_folder)
    for name in z.namelist():
        #print(name)
        z.extract(name, folder_location)

    return folder_location


def process_sqlite_files(sqlite_files):
    print("h")
    #for file in sqlite_files:
        #data = happ.load_and_preprocess_data(file)



#takes in the location dataframe, converts to and returns the GeoDataFrame and saves it in the same directory
def create_gpkg_track(sqlitefile):
    file_name = sqlitefile.split(".")[0]
    conn = sqlite3.connect(sqlitefile, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
    trackdf = pd.read_sql_query("SELECT * FROM location", conn)
    gdf = geopandas.GeoDataFrame(trackdf, geometry=geopandas.points_from_xy(trackdf.longitude, trackdf.latitude), crs=4326)

    gdf.to_file(f"{os.path.abspath(file_name)}_track.gpkg", driver = "GPKG")

    return gdf





"""def MOS_analysis_fieldstudy(sqlite_files,
                            write_output_to_csv: bool = False) -> pd.DataFrame:

    mos_dfs = []
    files_to_check = []
    files_to_check_sensordataTable = []
    files_to_check_locationTable = []
    number_of_MOS_detected = {}

    file_counter = 0
    for file in sqlite_files:
        print(f'file {file_counter + 1} of {len(sqlite_files)}')
        filename = file.split('\\')[-1].split('.')[0]
        file_path = file.split('.')[0]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("File path", file_path)

        try:
            # sensor data from 'sensordata' table
            sensordata_table_existence = sensor_check.check_sensordata_data(file)
            if sensordata_table_existence.empty:
                print("----------------------- \n No sensor recordings found.. \n -----------------------")
        except (sqlite3.DatabaseError, pd.errors.DatabaseError) as e:
            files_to_check_sensordataTable.append(filename)
            logging.exception(f"Error due to malformed database disk image in file {filename} \n")

        else:
            try:
                MOS_output, extended_MOS_output, MOS_count = MOS_paper.MOS_main_filepath(file)
                number_of_MOS_detected[filename] = MOS_count


            except (UnboundLocalError, ZeroDivisionError, sqlite3.DatabaseError, pd.errors.DatabaseError) as e:
                files_to_check_sensordataTable.append(filename)
                logging.exception(f"Error occurred in file {filename} \n"
                                  f"Seems like there is no table named 'sensordata'.")

        try:
            location_table_existence = sensor_check.check_location_data(file)
            # location data from 'location' table
            if location_table_existence.empty:
                print("----------------------- \n No location recordings found.. \n -----------------------")

        except (sqlite3.DatabaseError, pd.errors.DatabaseError) as e:
            files_to_check.append(filename)
            logging.exception(f"Error due to malformed database disk image in file {filename} \n")

        else:

            try:
                locations = dl.get_location_data(file)
                # print("Locations table", locations)
                merged_geolocated_mos = dl.geolocate_data(location_data=locations,
                                                          input_filtered_data=MOS_output)

                merged_geolocated_mos['run_id'] = filename

                mos_dfs.append(merged_geolocated_mos)


            except (UnboundLocalError, ZeroDivisionError, ValueError,
                    sqlite3.DatabaseError, pd.errors.DatabaseError) as e:
                files_to_check_locationTable.append(filename)
                logging.exception(f"Error occurred in file {filename} \n"
                                  f"Seems like there is no table named 'location'.")

        file_counter += 1

        if write_output_to_csv:
            try:
                relevant_mos_output = merged_geolocated_mos[["time_iso", "TimeNum", "MOS_score", "Lat", "Lon"]]
                relevant_mos_output.rename(columns = {"time_iso": "iso_time", "TimeNum": "unix_time", "MOS_score": "mos_score",
                                                      "Lat": "latitude", "Lon": "longitude"}, inplace=True)

                # TODO - check if this is necessary to write into same directory where file(s) were stored 'os.path.abspath(file)'
                #store_path = os.path.join(file_path + file_identifier)
                #print("Storing MOS detection output in path: ", store_path)
                #relevant_mos_output.to_csv(f"{os.path.join(file + file_identifier)}.csv", index_label=False)
                #relevant_mos_output.to_csv(f"{file_path}.csv", index_label=False)
                relevant_mos_output.to_csv(f"{os.path.abspath(file)}_MOS.csv", index_label=False) # writes into same directory where file was stored

            except UnboundLocalError as e:
                logging.exception(f"MOS Analysis did not work for file {filename}")

    try:
        if len(mos_dfs) != 0:
            fieldstudy_mos_combined = pd.concat(mos_dfs)
            return fieldstudy_mos_combined, files_to_check_sensordataTable, files_to_check_locationTable, files_to_check, number_of_MOS_detected

    except ValueError as e:
        logging.exception(f"Cannot concatenate DataFrames where MOS analysis cannot be performed")
        files_to_check.append(filename)
        fieldstudy_mos_combined = pd.DataFrame()

    return fieldstudy_mos_combined, files_to_check_sensordataTable, files_to_check_locationTable, files_to_check, number_of_MOS_detected

    #fieldstudy_mos.to_csv(os.path.join(path, "labtest-data-preprocessed_combined.csv"), index=False)

"""
#takes a list of sqlite files, calculates MOS, aggregates and returns a single dataframe (Kyriakou et al. 2019)
def MOS_analysis_fieldstudy_folder(folder_path: str, write_output_to_csv: bool = False, 
                            write_output_to_xlsx: bool = False,
                            write_output_to_geopackage: bool = False) -> pd.DataFrame:

    mos_dfs = []

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    for file in all_sqlite_files:
        filename = file.split('\\')[-1]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("Name of that file: ", file_identifier)

        # sensor data from 'sensordata' table
        sensordata_table_existence = sensor_check.check_sensordata_data(file)
        if sensordata_table_existence.empty:
            print("----------------------- \n No sensor recordings found.. \n -----------------------")

        else:
            MOS_output, extended_MOS_output = MOS_paper.MOS_main_filepath(file)


        # location data from 'location' table
        location_table_existence = sensor_check.check_location_data(file)

        if location_table_existence.empty:
            print("----------------------- \n No location recordings found.. \n -----------------------")
        else:
            locations = dl.get_location_data(file)
            # print("Locations table", locations)
            merged_geolocated_mos = dl.geolocate_data(location_data=locations,
                                                      input_filtered_data=MOS_output)

            merged_geolocated_mos['run_id'] = filename

            mos_dfs.append(merged_geolocated_mos)

        relevant_mos_output = merged_geolocated_mos[["time_iso", "TimeNum", "MOS_score", "Lat", "Lon"]]
        relevant_mos_output.rename(columns = {"time_iso": "iso_time", "TimeNum": "unix_time", "MOS_score": "mos_score",
                                              "Lat": "latitude", "Lon": "longitude"})
        
    
    fieldstudy_mos_combined = pd.concat(mos_dfs)
    field_study_mos_combined_dropna = fieldstudy_mos_combined[fieldstudy_mos_combined['Lat'].notna()]
    if write_output_to_csv:
        
        # TODO - check if this is necessary to write into same directory where file(s) were stored 'os.path.abspath(file)'
        #store_at = os.path.join(os.path.abspath(file) + file_identifier)
        fieldstudy_mos_combined.to_csv(f"{os.path.join(os.path.abspath(file) + file_identifier)}.csv", index_label=False)

    if write_output_to_xlsx:
        """
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            relevant_mos_output.to_excel(writer, sheet_name='Sheet1')
#       
        """
        out_path = f"{os.path.join(os.path.abspath(file) + file_identifier)}"
        
        writer = pd.ExcelWriter(out_path + ".xlsx", engine='xlsxwriter')
        fieldstudy_mos_combined.to_excel(writer, sheet_name="Sheet1")
        writer.save()

    if write_output_to_geopackage:
        
        fieldstudy_mos_combined_gdf = geopandas.GeoDataFrame(fieldstudy_mos_combined,
                                                        geometry = geopandas.points_from_xy(fieldstudy_mos_combined.Lon, fieldstudy_mos_combined.Lat),
                                                        crs=4326)
        out_path = f"{os.path.join(os.path.abspath(file) + file_identifier)}"
        
        fieldstudy_mos_combined_gdf.to_file(out_path + ".gpkg", layer='data_example', driver='GPKG')

        #relevant_mos_output_shp = relevant_mos_output_gdf.copy()
        
        #relevant_mos_output_shp2 = relevant_mos_output_shp.apply(lambda x: LineString(x.tolist()))
        #relevant_mos_output_shp2 = geopandas.GeoDataFrame(relevant_mos_output_shp2, geometry="geometry")
        #relevant_mos_output_shp2.to_file(r"C:\Users\b1095832\Desktop\Edah Tasks\01-Signal_integration\human-sensing-git\human-sensing\src\HumanSensing_Preprocessing\LabData\TerrainData\zip_testing\example_geopackage2.gpkg", layer='data_example', driver='GPKG')
        #relevant_mos_output_shp['time_iso'] = relevant_mos_output_shp['time_iso'].dt.strftime("%Y/%m/%d %H:%M:%S")
        #relevant_mos_output_shp.to_file(r"C:\Users\b1095832\Desktop\Edah Tasks\01-Signal_integration\human-sensing-git\human-sensing\src\HumanSensing_Preprocessing\LabData\TerrainData\zip_testing\example_shp.shp", layer='data_example', driver='ESRI Shapefile')
    #fieldstudy_mos.to_csv(os.path.join(path, "labtest-data-preprocessed_combined.csv"), index=False)

    return field_study_mos_combined_dropna, fieldstudy_mos_combined



#takes a list of sqlite files, calculates MOS, aggregates and returns a single dataframe (Kyriakou et al. 2019)
def NEW_MOS_analysis_fieldstudy_folder(folder_path: str, MOS_threshold: float,
                                       start_time_trim: int, end_time_trim: int,
                                       write_output_to_csv: bool = False,
                                       write_output_to_xlsx: bool = False,
                                       write_output_to_geopackage: bool = False) -> pd.DataFrame:

    mos_dfs = []

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    for file in all_sqlite_files:
        filename = file.split('\\')[-1]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("Name of that file: ", file_identifier)

        # sensor data from 'sensordata' table
        sensordata_table_existence = sensor_check.check_sensordata_data(file)
        if sensordata_table_existence.empty:
            print("----------------------- \n No sensor recordings found.. \n -----------------------")

        else:

            MOS_data_prep = msp.MOS_detection_signal_preparation(file)
            initial_start_time = MOS_data_prep['time_iso'].min()
            initial_end_time = MOS_data_prep['time_iso'].max().round('1s')

            start_time_base = initial_start_time + pd.to_timedelta(start_time_trim, "s")
            end_time_base = initial_end_time - pd.to_timedelta(end_time_trim, "s")


            MOS_output, number_of_MOS = MOS_paper_new.MOS_main_df(df = MOS_data_prep,
                                                                  baseline_time_filter_start = start_time_base,
                                                                  baseline_time_filter_end = end_time_base,
                                                                  MOS_thresh = MOS_threshold)

        # location data from 'location' table
        location_table_existence = sensor_check.check_location_data(file)

        if location_table_existence.empty:
            print("----------------------- \n No location recordings found.. \n -----------------------")
        else:
            locations = dl.get_location_data(file)
            # print("Locations table", locations)
            merged_geolocated_mos = dl.geolocate_data(location_data=locations,
                                                      input_filtered_data=MOS_output)

            merged_geolocated_mos['run_id'] = filename

            mos_dfs.append(merged_geolocated_mos)

    fieldstudy_mos_combined = pd.concat(mos_dfs)
    field_study_mos_combined_dropna = fieldstudy_mos_combined[fieldstudy_mos_combined['Lat'].notna()]
    if write_output_to_csv:
        # TODO - check if this is necessary to write into same directory where file(s) were stored 'os.path.abspath(file)'
        # store_at = os.path.join(os.path.abspath(file) + file_identifier)
        fieldstudy_mos_combined.to_csv(f"{os.path.join(os.path.abspath(file) + file_identifier)}.csv",
                                       index_label=False)

    if write_output_to_xlsx:
        """
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            relevant_mos_output.to_excel(writer, sheet_name='Sheet1')
#       
        """
        out_path = f"{os.path.join(os.path.abspath(file) + file_identifier)}"

        writer = pd.ExcelWriter(out_path + ".xlsx", engine='xlsxwriter')
        fieldstudy_mos_combined.to_excel(writer, sheet_name="Sheet1")
        writer.save()

    if write_output_to_geopackage:
        fieldstudy_mos_combined_gdf = geopandas.GeoDataFrame(fieldstudy_mos_combined,
                                                             geometry=geopandas.points_from_xy(
                                                                 fieldstudy_mos_combined.Lon,
                                                                 fieldstudy_mos_combined.Lat),
                                                             crs=4326)
        out_path = f"{os.path.join(os.path.abspath(file) + file_identifier)}"

        fieldstudy_mos_combined_gdf.to_file(out_path + ".gpkg", layer='data_example', driver='GPKG')

        # relevant_mos_output_shp = relevant_mos_output_gdf.copy()

        # relevant_mos_output_shp2 = relevant_mos_output_shp.apply(lambda x: LineString(x.tolist()))
        # relevant_mos_output_shp2 = geopandas.GeoDataFrame(relevant_mos_output_shp2, geometry="geometry")
        # relevant_mos_output_shp2.to_file(r"C:\Users\b1095832\Desktop\Edah Tasks\01-Signal_integration\human-sensing-git\human-sensing\src\HumanSensing_Preprocessing\LabData\TerrainData\zip_testing\example_geopackage2.gpkg", layer='data_example', driver='GPKG')
        # relevant_mos_output_shp['time_iso'] = relevant_mos_output_shp['time_iso'].dt.strftime("%Y/%m/%d %H:%M:%S")
        # relevant_mos_output_shp.to_file(r"C:\Users\b1095832\Desktop\Edah Tasks\01-Signal_integration\human-sensing-git\human-sensing\src\HumanSensing_Preprocessing\LabData\TerrainData\zip_testing\example_shp.shp", layer='data_example', driver='ESRI Shapefile')
    # fieldstudy_mos.to_csv(os.path.join(path, "labtest-data-preprocessed_combined.csv"), index=False)

    return field_study_mos_combined_dropna, fieldstudy_mos_combined


#takes a list of sqlite files, extracts sensor measurements, and returns a single (combined) dataframe
def sensor_measurements_fieldstudy_folder(folder_path: str, write_output_to_csv: bool = False,
                            write_output_to_xlsx: bool = False,
                            write_output_to_geopackage: bool = False) -> pd.DataFrame:

    sensor_dfs = []

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    for file in all_sqlite_files:
        filename = file.split('\\')[-1]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("Name of that file: ", file_identifier)

        # sensor data from 'sensordata' table
        sensordata_table_existence = sensor_check.check_sensordata_data(file)
        if sensordata_table_existence.empty:
            print("----------------------- \n No sensor recordings found.. \n -----------------------")

        else:
            signals_prepared = msp.MOS_detection_signal_preparation(file)

            signals_prepared['run_id'] = filename

            sensor_dfs.append(signals_prepared)

    combined_sensordata = pd.concat(sensor_dfs)

    return combined_sensordata


#takes a list of sqlite files, extracts location data, and returns a single (combined) dataframe
def location_data_fieldstudy_folder(folder_path: str, write_output_to_csv: bool = False,
                            write_output_to_xlsx: bool = False,
                            write_output_to_geopackage: bool = False) -> pd.DataFrame:

    location_dfs = []

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    for file in all_sqlite_files:
        filename = file.split('\\')[-1]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("Name of that file: ", file_identifier)


        # location data from 'location' table
        location_table_existence = sensor_check.check_location_data(file)

        if location_table_existence.empty:
            print("----------------------- \n No location recordings found.. \n -----------------------")
        else:
            locations = dl.get_location_data(file)

            locations['run_id'] = filename

            location_dfs.append(locations)

    combined_locations = pd.concat(location_dfs)

    return combined_locations



def check_single_ediary_info(sqlite_file_path):

    conn = sqlite3.connect(sqlite_file_path)


    count_query = f'SELECT COUNT(*) from survey'


    result = pd.read_sql_query(count_query, conn)

    if result['COUNT(*)'][0] == 0:
        print(sqlite_file_path, "is empty")

    else:
        # store information in one large DataFrame
        query = f'SELECT time_millis, time_iso, latitude, longitude, was_beschreibt_ihr_gefühl_am_besten, ' \
                f'wie_stark_ist_ihr_gefühl, was_hat_ihr_gefühl_ausgelöst, weitere_kommentare_ FROM survey'

        ed_entries = pd.read_sql_query(query, conn)
        # check what pd.read_sql_table() does / if it works for this purpose

        # rename columns
        ed_entries.columns = ['time_millis', 'time_iso', 'lat', 'lon', 'feeling_descr', 'strength', 'cause', 'comments']

        filename = sqlite_file_path.split('.')[0]

        ed_entries['run_id'] = filename
        #ed_entries['location'] = location
        ed_entries['time_iso'] = pd.to_datetime(ed_entries['time_iso'])

        return ed_entries


#takes a list of sqlite files, extracts survey data, and returns a single (combined) dataframe
def survey_data_fieldstudy_folder(folder_path: str, write_output_to_csv: bool = False,
                            write_output_to_xlsx: bool = False,
                            write_output_to_geopackage: bool = False) -> pd.DataFrame:

    survey_dfs = []

    all_sqlite_files = find_all_sqlite_files(folder_path)
    print(f"Number of of sqlite files found: \t {len(all_sqlite_files)}")

    for file in all_sqlite_files:
        filename = file.split('\\')[-1]
        file_identifier = filename.split(".")[0]
        print("Analyzing file", file, "\n")
        print("Name of that file: ", file_identifier)

        conn = sqlite3.connect(file)

        count_query = f'SELECT COUNT(*) from survey'

        result = pd.read_sql_query(count_query, conn)

        if result['COUNT(*)'][0] == 0:
            print(file, "is empty")

        else:
            # store information in one large DataFrame
            query = f'SELECT time_millis, time_iso, latitude, longitude, was_beschreibt_ihr_gefühl_am_besten, ' \
                    f'wie_stark_ist_ihr_gefühl, was_hat_ihr_gefühl_ausgelöst, weitere_kommentare_ FROM survey'

            ed_entries = pd.read_sql_query(query, conn)
            # check what pd.read_sql_table() does / if it works for this purpose

            # rename columns
            ed_entries.columns = ['time_millis', 'time_iso', 'lat', 'lon', 'feeling_descr', 'strength', 'cause',
                                  'comments']

            ed_entries['run_id'] = filename

            survey_dfs.append(ed_entries)

        # TODO - implement check for survey data table existence in sensor_check
        #location_table_existence = sensor_check.check_location_data(file)

        #if location_table_existence.empty:
        #    print("----------------------- \n No location recordings found.. \n -----------------------")

    combined_surveys = pd.concat(survey_dfs)

    return combined_surveys





    