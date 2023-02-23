import os
import sqlite3
import pandas as pd
import os


def store_sqlite_file_paths(folder_path):
    # find all sqlite files in SQLITE_PATH
    sqlite_files = []
    # check if specified path is already an .sqlite file
    if os.path.isfile(folder_path):
        sqlite_files.append(folder_path)

    else:
        for root, dirs, files in os.walk(folder_path):
            #print("Root: \n", root)
            #print("Directory: \n", dirs)
            #print('Files: \n', files)
            for file in files:
                if file.endswith('.sqlite'):
                    sqlite_files.append(os.path.join(root, file))

    print(f"Found {len(sqlite_files)} sqlite ediary files. \n")
    if len(sqlite_files) != 0:
        print(f"Successfully stored file paths into list")
    return sqlite_files


### Check Sensor Usage

def check_if_E4_used(sqlite_file_path, show_df_head = False):
    conn = sqlite3.connect(sqlite_file_path)
    E4_query = f'SELECT * FROM sensordata WHERE platform_id = 3'
    result = pd.read_sql_query(E4_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    if len(result) == 0:
        print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n --------------------------------- \n Was the Empatica E4 wristband used in this study? \n"
                                                                                                               "\t --> If YES, check connection of the device. \n")
    else:
        print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n --> Successful recording \n")
        if show_df_head == True:
            print("File Name:", file_name, "\n",  result.head())

def E4_used(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    E4_query = f'SELECT * FROM sensordata WHERE platform_id = 3'
    result = pd.read_sql_query(E4_query, conn)
    if len(result) == 0:
        return False
    else:
        return True


def check_if_BioHarness_used(sqlite_file_path, show_df_head = False):
    conn = sqlite3.connect(sqlite_file_path)
    BioHarness_query = f'SELECT * FROM sensordata WHERE platform_id = 2'
    result = pd.read_sql_query(BioHarness_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    if len(result) == 0:
        print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n Was the Zephyr BioHarness chest strap used in this study? \n"
                                                                                                               "\t --> If YES, check connection of the device. \n")
    else:
        print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n --> Successful recording \n")
        if show_df_head == True:
            print("File Name:", file_name, "\n",  result.head())

def BioHarness_used(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    BioHarness_query = f'SELECT * FROM sensordata WHERE platform_id = 2'
    result = pd.read_sql_query(BioHarness_query, conn)
    #print(result)
    if len(result) == 0:
        return False
    else:
        return True

### Sensor Recordings Check

def GSR_recording_check(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    GSR_query = f'SELECT * from sensordata where sensor_id = 3 and platform_id = 3'
    result = pd.read_sql_query(GSR_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n", "Number of GSR recordings:", len(result), "\n Timestamp range from ", pd.Timestamp(result['time_iso'].min()), "to", pd.Timestamp(result['time_iso'].max()), "-->", (pd.Timestamp(result['time_iso'].max()) - pd.Timestamp(result['time_iso'].min())), "\n")

    if len(result) == 0:
        print(sqlite_file_path, "is empty")

def ST_recording_check(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    ST_query = f'SELECT * from sensordata where sensor_id = 7 and platform_id = 3'
    result = pd.read_sql_query(ST_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n", "Number of GSR recordings:", len(result), "\n Timestamp range from ", pd.Timestamp(result['time_iso'].min()), "to", pd.Timestamp(result['time_iso'].max()), "-->", (pd.Timestamp(result['time_iso'].max()) - pd.Timestamp(result['time_iso'].min())), "\n")

    if len(result) == 0:
        print(sqlite_file_path, "is empty")

def IBI_recording_check_E4(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    count_query = f'SELECT * from sensordata where sensor_id = 22 and platform_id = 3'
    result = pd.read_sql_query(count_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n", "Number of IBI recordings (E 4):", len(result), "\n Timestamp range from ", pd.Timestamp(result['time_iso'].min()), "to", pd.Timestamp(result['time_iso'].max()), "-->", (pd.Timestamp(result['time_iso'].max()) - pd.Timestamp(result['time_iso'].min())), "\n")

    if len(result) == 0:
        print(sqlite_file_path, "is empty")

def IBI_recording_check_BioHarness(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    count_query = f'SELECT * from sensordata where sensor_id = 16 and platform_id = 2'
    result = pd.read_sql_query(count_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n", "Number of IBI recordings (BioHarness):", len(result), "\n Timestamp range from ", pd.Timestamp(result['time_iso'].min()), "to", pd.Timestamp(result['time_iso'].max()), "-->", (pd.Timestamp(result['time_iso'].max()) - pd.Timestamp(result['time_iso'].min())), "\n")

    if len(result) == 0:
        print(sqlite_file_path, "is empty")

### Location Recordings Check

# count number of location recorded:
def count_locations_recorded(sqlite_file_path):
    conn = sqlite3.connect(sqlite_file_path)
    count_query = f'SELECT *, COUNT(*) from location'
    result = pd.read_sql_query(count_query, conn)
    file_name = sqlite_file_path.split('/')[-1]
    print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n Number of location recordings: \n", result['COUNT(*)'][0], "\n")

    if result['COUNT(*)'][0] == 0:
        print("File Path:", sqlite_file_path, "\n --------------------------------- \n File Name:", file_name, "\n --> empty")

def check_location_data(sqlite_file_path: str) -> bool:

    conn = sqlite3.connect(sqlite_file_path)
    check_location_table_existence_query = f'SELECT name FROM sqlite_master WHERE type = "table" AND name = "location";'
    result = pd.read_sql_query(check_location_table_existence_query, conn)
    file_name = sqlite_file_path.split('\\')[-1]
    return result

def check_sensordata_data(sqlite_file_path: str) -> bool:

    conn = sqlite3.connect(sqlite_file_path)
    check_sensordata_table_existence_query = f'SELECT name FROM sqlite_master WHERE type = "table" AND name = "sensordata";'
    result = pd.read_sql_query(check_sensordata_table_existence_query, conn)
    file_name = sqlite_file_path.split('\\')[-1]
    return result

