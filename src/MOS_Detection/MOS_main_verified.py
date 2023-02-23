import MOS_rules_paper_verified as MOS_paper
from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import lookup_tables
from HumanSensing_Visualization import visualize_MOS_rules_met as visual
import pandas as pd

labkey = "lab2/session2"

# labtest 1 session 1
filename = "C:/Users/b1081018/Desktop/projects/test_HRV/human-sensing/src/Data/usable_IBI_data/lab1/session1/20180627_084235_phone_1.sqlite3"
# labtest 2 session 2
filename2 = "C:/Users/b1081018/Desktop/projects/test_HRV/human-sensing/src/Data/usable_IBI_data/lab2/session2/20190213_103500.sqlite3"
filename3 = "C:/Users/b1081018/Desktop/projects/test_HRV/human-sensing/src/Data/usable_IBI_data/lab2/session2/20190213_103509.sqlite3"


MOS_output, MOS_output_rules = MOS_paper.MOS_main_filepath(filename = filename2,
                                                  print_number_of_time_rules_are_met=True)

time_diff = lookup_tables.lab_session_starttime_dic[labkey].hour - MOS_output_rules['time_iso'][0].hour

MOS_output_rules['time_iso'] = MOS_output_rules['time_iso'] + pd.Timedelta(time_diff, unit='hour')


MOS_out_gt = pps.label_stressmoments(MOS_output_rules, starttime=lookup_tables.lab_session_starttime_dic[labkey],
                                     stresstimes=lookup_tables.lab_session_stresstimes_dic[labkey])


#visual.visualize_MOS(MOS_out_gt, MOSpercentage=75)
visual.visualize_MOS(MOS_out_gt, MOSpercentage=100, add_labtest_groundtruth=True)
