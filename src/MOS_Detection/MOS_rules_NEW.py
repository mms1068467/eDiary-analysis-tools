import math

import numpy as np
import pandas as pd

from HumanSensing_Preprocessing import preprocess_signals as pps
from HumanSensing_Preprocessing import utilities
from HumanSensing_Preprocessing import lookup_tables
from MOS_Detection import MOS_signal_preparation as msp
from MOS_Detection import MOS_signal_preparation_verified as msp_new

########### rules ###########

def MOS_main_filepath(filepath,
                      baseline_time_filter_start = 180, baseline_time_filter_end = 60,
                      MOS_thresh = 0.75):

    MOS_data_prep = msp.MOS_detection_signal_preparation(filepath)

    filtered_data = MOS_data_prep.copy()

    MOS_data_prep_GSR_ST_features = msp_new.derive_GSR_and_ST_features(filtered_data)

    if "IBI" in MOS_data_prep_GSR_ST_features:
        MOS_data_prep_GSR_ST_IBI_HRV_features = msp_new.derive_IBI_and_HRV_features(MOS_data_prep_GSR_ST_features)
    else:
        MOS_data_prep_GSR_ST_IBI_HRV_features = MOS_data_prep_GSR_ST_features.copy()

    ## Add additional GSR features for new rules:
    MOS_data_prep_GSR_ST_IBI_HRV_features_complete = GSR_amplitude_duration_slope(MOS_data_prep_GSR_ST_IBI_HRV_features)

    # TODO - add start_time_trim and end_time_trim values for trimmed baseline calculation as input
    threshold_data = MOS_data_prep_GSR_ST_IBI_HRV_features_complete.set_index("time_iso")[baseline_time_filter_start:baseline_time_filter_end]

    amplitude_mean = calculate_GSR_ampltiude_baseline(threshold_data)
    amplitude_std = calculate_GSR_ampltiude_spread(threshold_data)
    # st.write(amplitude_mean, amplitude_std)
    duration_mean = calculate_GRS_duration_baseline(threshold_data)
    duration_std = calculate_GRS_duration_spread(threshold_data)
    # st.write(duration_mean, duration_std)
    slope_mean = calculate_GSR_Slope_baseline(threshold_data)
    slope_std = calculate_GSR_Slope_spread(threshold_data)

    MOS_out_martin = MOS_rules_apply_n(MOS_data_prep_GSR_ST_IBI_HRV_features_complete,
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

    MOS_output_ordered = MOS_gsr_and_st_clean[
        ["time_iso", "TimeNum", "GSR", "GSR_standardized", "ST", "ST_standardized",
         "HRV", "hrv_filtered", "rmsnn", "sdnn", "IBI", "filtered_IBI", "MOS_Score", "detectedMOS"]]

    return MOS_output_ordered, number_of_mos

def MOS_main_df(df,
                baseline_time_filter_start, baseline_time_filter_end,
                MOS_thresh = 0.75):

    filtered_data = df.copy()

    MOS_data_prep_GSR_ST_features = msp_new.derive_GSR_and_ST_features(filtered_data)

    if "IBI" in MOS_data_prep_GSR_ST_features:
        MOS_data_prep_GSR_ST_IBI_HRV_features = msp_new.derive_IBI_and_HRV_features(MOS_data_prep_GSR_ST_features)
    else:
        MOS_data_prep_GSR_ST_IBI_HRV_features = MOS_data_prep_GSR_ST_features.copy()

    ## Add additional GSR features for new rules:
    MOS_data_prep_GSR_ST_IBI_HRV_features_complete = GSR_amplitude_duration_slope(MOS_data_prep_GSR_ST_IBI_HRV_features)

    # TODO - add start_time_trim and end_time_trim values for trimmed baseline calculation as input
    threshold_data = MOS_data_prep_GSR_ST_IBI_HRV_features_complete.set_index("time_iso")[baseline_time_filter_start:baseline_time_filter_end]

    amplitude_mean = calculate_GSR_ampltiude_baseline(threshold_data)
    amplitude_std = calculate_GSR_ampltiude_spread(threshold_data)
    # st.write(amplitude_mean, amplitude_std)
    duration_mean = calculate_GRS_duration_baseline(threshold_data)
    duration_std = calculate_GRS_duration_spread(threshold_data)
    # st.write(duration_mean, duration_std)
    slope_mean = calculate_GSR_Slope_baseline(threshold_data)
    slope_std = calculate_GSR_Slope_spread(threshold_data)

    MOS_out_martin = MOS_rules_apply_n(MOS_data_prep_GSR_ST_IBI_HRV_features_complete,
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

    MOS_output_ordered = MOS_gsr_and_st_clean[
        ["time_iso", "TimeNum", "GSR", "GSR_standardized", "ST", "ST_standardized",
         "HRV", "hrv_filtered", "rmsnn", "sdnn", "IBI", "filtered_IBI", "MOS_Score", "detectedMOS"]]

    return MOS_output_ordered, number_of_mos




########### rules ###########

# TODO - add weigths (default weights)
def MOS_rules_apply_n(df: pd.DataFrame,
                    amplitude_mean, amplitude_std, slope_mean, slope_std,
                    apply_GSR_rules: bool = True,
                    n_GSRampl_min: float = 0.0, n_GSRampl_mid: float = 0.1, n_GSRampl_max: float = 0.4,
                    weight_ampl_min: float = 0.0, weight_ampl_mid: float = 0.1, weight_ampl_max: float = 0.4,
                    n_GSRslop_min: float = 0.0, n_GSRslop_mid: float = 0.1, n_GSRslop_max: float = 0.4,
                    weight_slop_min: float = 0.0, weight_slop_mid: float = 0.1, weight_slop_max: float = 0.4,
                    GSRincDur_min: int = 1, GSRincDur_max: int = 6,
                    apply_ST_rules: bool = True,
                    STdecDur_min: int = 2, STdecDur_max: int = 13, STminThresh: float = 0.7,
                    MOSpercentage: float = 0.75):

    if apply_GSR_rules:
        df_gsr1, GSR_Amp_Inc = GSR_Amplitude_Increase_n(df, amplitude_mean = amplitude_mean, amplitude_std=amplitude_std,
                                                                      n_GSRampl_min = n_GSRampl_min, n_GSRampl_mid = n_GSRampl_mid, n_GSRampl_max = n_GSRampl_max,
                                                                      weight_ampl_min = weight_ampl_min, weight_ampl_mid = weight_ampl_mid, weight_ampl_max = weight_ampl_max)
        #print(f"Rule 1 (GSR Amplitude Increase) with mean + {n_GSRampl} standard deviations was met: {sum(GSR_Amp_Inc)} times")

        df_gsr3, GSR_Dur_Inc = GSR_Duration_Increase(df_gsr1, minimum_duration=GSRincDur_min, maximum_duration=GSRincDur_max)
        #print(f"Rule 3 (GSR Increase Time) between {GSRincDur_min} and {GSRincDur_max} was met: {sum(GSR_Dur_Inc)} times")

        df_gsr4, GSR_Slope_Deg = GSR_Slope_Degrees_n(df_gsr3, slope_mean = slope_mean, slope_std = slope_std,
                                                                 n_min = n_GSRslop_min, n_mid = n_GSRslop_mid, n_max = n_GSRslop_max,
                                                                weight_slop_min = weight_slop_min, weight_slop_mid = weight_slop_mid, weight_slop_max = weight_slop_max)
        #print(f"Rule 4 (GSR Slope Angle) with mean + {n_GSRslop} standard deviations was met: {sum(GSR_Slope_Deg)} times")

        df_GSR_Score_met, number_of_MOS = check_if_GSR_score_met(df_gsr4, MOSpercentage=MOSpercentage)

        print(f"GSR rules Score was met {number_of_MOS} times")

        if apply_ST_rules:

            df_st = df_GSR_Score_met.copy()

            df_st_gsr_all, ST_Dur_Dec = ST_Decrease_X_seconds_after_GSR_onset(data = df_st,
                                                            minimum_duration=STdecDur_min,
                                                            maximum_duration=STdecDur_max,
                                                            min_threshold=STminThresh)
            print(f"Rule 2 (ST Decrease after GSR Onset) between {STdecDur_min} and {STdecDur_max} with a threshold of {STminThresh} was met: {sum(ST_Dur_Dec)} times")

            return df_st_gsr_all

        else:

            return df_GSR_Score_met

#### rule 1 - GSR Amplitude Increase Value between local Minimum and local Maximum ####

def GSR_Amplitude_Increase(data: pd.DataFrame, amplitude_mean, amplitude_std, n):

    GSR_Ampl_Inc = [0]*len(data)

    data_GSR_amplitude = data.copy()

    rule1_amplitude = data[data["GSR_amplitude"] > (amplitude_mean + (n * amplitude_std))]

    for index in rule1_amplitude.index:
        GSR_Ampl_Inc[index] = 1

    data_GSR_amplitude["Rule1_Amp_met"] = GSR_Ampl_Inc

    return data_GSR_amplitude, GSR_Ampl_Inc

def GSR_Amplitude_Increase_n(data: pd.DataFrame, amplitude_mean, amplitude_std, n_GSRampl_min =0.0, n_GSRampl_mid = 0.2, n_GSRampl_max = 0.5,
                             weight_ampl_min: float = 0.25, weight_ampl_mid: float = 0.5, weight_ampl_max: float = 0.1):

    GSR_Ampl_Score = [0]*len(data)

    data_GSR_amplitude = data.copy()

    for index in data.index:
        if (data["GSR_amplitude"][index] > (amplitude_mean + (n_GSRampl_min * amplitude_std))) & (data["GSR_amplitude"][index] <= (amplitude_mean + (n_GSRampl_mid * amplitude_std))):
            GSR_Ampl_Score[index] = weight_ampl_min
        elif (data["GSR_amplitude"][index] > (amplitude_mean + (n_GSRampl_mid * amplitude_std))) & (data["GSR_amplitude"][index] <= (amplitude_mean + (n_GSRampl_max * amplitude_std))):
            GSR_Ampl_Score[index] = weight_ampl_mid
        elif data["GSR_amplitude"][index] > (amplitude_mean + (n_GSRampl_max * amplitude_std)):
            GSR_Ampl_Score[index] = weight_ampl_max

    data_GSR_amplitude["Rule1_Amp_Score"] = GSR_Ampl_Score

    return data_GSR_amplitude, GSR_Ampl_Score


#### rule 2 - ST Decrease (not monotonic) in a X-Y second window after GSR onset (e.g. 7 s Decrease out of 10 s) ####

def ST_Decrease_X_seconds_after_GSR_onset(data: pd.DataFrame, minimum_duration, maximum_duration, min_threshold):

    ST_Dur_Dec = [0]*len(data)

    data_ST = data.copy()

    for index in data.index:
        if data.loc[index, "GSR_onset"] == 1:
            ST_dec_sum = sum(data.loc[index+minimum_duration : index+maximum_duration, "ST_decrease"])
            ST_dec_per = ST_dec_sum / (maximum_duration - minimum_duration)

            if ST_dec_per > min_threshold:
                ST_Dur_Dec[index] = 1

    data_ST["Rule2_STdec_met"] = ST_Dur_Dec

    return data_ST, ST_Dur_Dec



#### rule 3 - GSR Increase Time ####

def GSR_Duration_Increase(data: pd.DataFrame, minimum_duration = 1, maximum_duration = 7):

    GSR_Dur_Inc = [0]*len(data)

    data_GSR_IncDuration = data.copy()

    rule3_duration = data[(data["GSR_increase_durations"] >= minimum_duration) & (data["GSR_increase_durations"] <= maximum_duration)]
    #print(len(rule3_duration))

    for index in rule3_duration.index:
        GSR_Dur_Inc[index] = 1

    data_GSR_IncDuration["Rule3_GSRincTime_met"] = GSR_Dur_Inc

    return data_GSR_IncDuration, GSR_Dur_Inc



#### rule 4 - GSR Slope Value between local Minimum and local Maximum ####

def GSR_Slope_Degrees(data: pd.DataFrame, slope_mean, slope_std, n):

    GSR_Slope_Deg = [0]*len(data)

    data_GSR_slope = data.copy()

    rule4_slope = data[data["GSR_slope"] > (slope_mean + (n * slope_std))]

    for index in rule4_slope.index:
        GSR_Slope_Deg[index] = 1

    data_GSR_slope["Rule4_Slope_met"] = GSR_Slope_Deg

    return data_GSR_slope, GSR_Slope_Deg


def GSR_Slope_Degrees_n(data: pd.DataFrame, slope_mean, slope_std, n_min=0.0, n_mid=0.2, n_max=0.5,
                        weight_slop_min: float = 0.0, weight_slop_mid: float = 0.1, weight_slop_max: float = 0.4):

    GSR_Slope_Deg_Score = [0] * len(data)

    data_GSR_slope = data.copy()

    for index in data.index:
        if (data["GSR_slope"][index] > (slope_mean + (n_min * slope_std))) & (data["GSR_slope"][index] <= (slope_mean + (n_mid * slope_std))):
            GSR_Slope_Deg_Score[index] = weight_slop_min
        elif (data["GSR_slope"][index] > (slope_mean + (n_mid * slope_std))) & (data["GSR_slope"][index] <= (slope_mean + (n_max * slope_std))):
            GSR_Slope_Deg_Score[index] = weight_slop_mid
        elif data["GSR_slope"][index] > (slope_mean + (n_max * slope_std)):
            GSR_Slope_Deg_Score[index] = weight_slop_max

    data_GSR_slope["Rule4_Slope_Score"] = GSR_Slope_Deg_Score

    return data_GSR_slope, GSR_Slope_Deg_Score

def check_if_GSR_score_met(data, MOSpercentage: float = 0.75):

    MOS_Score = [0]*len(data)

    data_to_check = data.copy()

    data_to_check = data_to_check[data_to_check["Rule3_GSRincTime_met"] == 1]

    data_to_check["MOS_Score"] = data_to_check[["Rule1_Amp_Score", "Rule4_Slope_Score"]].sum(axis = 1)

    for index in data_to_check.index:
        MOS_Score[index] = data_to_check["MOS_Score"][index]

    data["MOS_Score"] = MOS_Score

    data["detectedMOS"] = np.where(data["MOS_Score"] >= MOSpercentage, 1, 0)
    number_of_MOS = sum(data["detectedMOS"])

    return data, number_of_MOS


def check_if_GSR_rules_met(data):

    GSR_all_rules = [0]*len(data)
    GSR_two_rules = [0]*len(data)

    data_GSR = data.copy()

    GSR_all_rules_met = data[(data["Rule1_Amp_met"] == 1) & (data['Rule3_GSRincTime_met'] == 1) & (data['Rule4_Slope_met'] == 1)]

    GSR_two_rules_met = data[ ( (data["Rule1_Amp_met"] == 1) & (data['Rule3_GSRincTime_met'] == 1) ) |
                              ( (data["Rule1_Amp_met"] == 1) & (data['Rule4_Slope_met'] == 1) ) |
                              ( (data["Rule3_GSRincTime_met"] == 1) & (data['Rule4_Slope_met'] == 1) ) ]

    for index in GSR_two_rules_met.index:
        GSR_two_rules[index] = 1
    for index in GSR_all_rules_met.index:
        GSR_all_rules[index] = 1

    data_GSR["GSR_all_rules"] = GSR_all_rules
    data_GSR["GSR_two_rules"] = GSR_two_rules

    return data_GSR



#### rule 5 - no two consecutive MOS within 10 seconds #### --> leave this rule for now !


########### rule preparation ###########


def prepare_signals_for_GS(sqlite_file, key: str = None):


    data_f1 = msp_new.MOS_detection_signal_preparation(sqlite_file,
                                                       starttime=lookup_tables.lab_session_starttime_dic2[
                                                           key])

    data_prep_f1 = msp_new.derive_GSR_and_ST_features(data_f1)

    if "IBI" in data_prep_f1:
        data_ready_f1 = msp_new.derive_IBI_and_HRV_features(data_prep_f1)
    # print(data_ready_f1.head())

    # add features for GSR rules
    data_ready_f11 = GSR_amplitude_duration_slope(data_ready_f1)

    data_ready_f11_labeled = pps.label_stressmoments(data=data_ready_f11,
                                                     starttime=lookup_tables.lab_session_starttime_dic2[key],
                                                     stresstimes=lookup_tables.lab_session_stresstimes_dic2[key])

    print(f"Data labeled {data_ready_f11_labeled} with columns: \n"
          f"{data_ready_f11_labeled}")

    print(f" \n Number of Stress Moments labled: {len(data_ready_f11_labeled[data_ready_f11_labeled['stress'] > 0])} ")

    return data_ready_f11_labeled


#from src.MOS_Detection.MOS_NEW import MOS_rules_NEW as MOS_paper_new

def calc_GSR_rule_statistics(data, trim_time = False, start_mins = 0, end_mins = 0):
    # calculate mean and standard deviation for GSR Increase amplitude, duration and steepness proportional to time

    data_ready_f11 = data.copy()

    #print(data.dtypes)

    if trim_time:
        start_time = data_ready_f11["time_iso"].min()
        end_time = data_ready_f11["time_iso"].max()
        #print(start_time)
        #print(end_time)
        data_ready_f11.set_index("time_iso", inplace=True)

        start_trim = pd.to_timedelta(start_mins, unit="m")
        new_start_time = start_time + start_trim
        end_trim = pd.to_timedelta(end_mins, unit="m")
        new_end_time = end_time - end_trim
        #print(new_start_time)
        #print(new_end_time)
        data_ready_f11 = data_ready_f11[new_start_time:new_end_time]

        data_ready_f11.reset_index(inplace=True)

        amplitude_mean = calculate_GSR_ampltiude_baseline(data_ready_f11)
        amplitude_std = calculate_GSR_ampltiude_spread(data_ready_f11)
        duration_mean = calculate_GRS_duration_baseline(data_ready_f11)
        duration_std = calculate_GRS_duration_spread(data_ready_f11)
        slope_mean = calculate_GSR_Slope_baseline(data_ready_f11)
        slope_std = calculate_GSR_Slope_spread(data_ready_f11)
        #relSlope_mean = calculate_GSR_relSlope_baseline(data_ready_f11)
        #relSlope_std = calculate_GSR_relSlope_spread(data_ready_f11)
        #steepness_mean = calculate_GSR_steepness_baseline(data_ready_f11)
        #steepness_std = calculate_GSR_steepness_spread(data_ready_f11)
        #print(amplitude_mean, amplitude_std, duration_mean, duration_std, slope_mean, slope_std, relSlope_mean, relSlope_std, steepness_mean, steepness_std)

        data_ready_f11.reset_index(inplace = True)

    else:
        amplitude_mean = calculate_GSR_ampltiude_baseline(data_ready_f11)
        amplitude_std = calculate_GSR_ampltiude_spread(data_ready_f11)
        duration_mean = calculate_GRS_duration_baseline(data_ready_f11)
        duration_std = calculate_GRS_duration_spread(data_ready_f11)
        slope_mean = calculate_GSR_Slope_baseline(data_ready_f11)
        slope_std = calculate_GSR_Slope_spread(data_ready_f11)
        relSlope_mean = calculate_GSR_relSlope_baseline(data_ready_f11)
        relSlope_std = calculate_GSR_relSlope_spread(data_ready_f11)
        steepness_mean = calculate_GSR_steepness_baseline(data_ready_f11)
        steepness_std = calculate_GSR_steepness_spread(data_ready_f11)
        #print(f"GSR Amplitude Mean {amplitude_mean} -- Amplitude Standard Deviation{amplitude_std} \n"
        #      f"GSR Increase Duration Mean {duration_mean} -- Increase Duration Standard Deviation {duration_std} \n "
        #      f"GSR Slope Mean {slope_mean}  -- Slope Standard Deviation {slope_std} \n ")


    return amplitude_mean, amplitude_std, duration_mean, duration_std, slope_mean, slope_std


def calc_GSR_rule_statistics_trimmed(data):
    # calculate mean and standard deviation for GSR Increase amplitude, duration and steepness proportional to time

    data_ready_f11 = data.copy()

    #print(data.dtypes)

    start_time = data_ready_f11["time_iso"].min()
    end_time = data_ready_f11["time_iso"].max()
    #print(start_time)
    #print(end_time)
    data_ready_f11.set_index("time_iso", inplace=True)

    data_ready_f11 = data_ready_f11[start_time:end_time]

    data_ready_f11.reset_index(inplace=True)

    amplitude_mean = calculate_GSR_ampltiude_baseline(data_ready_f11)
    amplitude_std = calculate_GSR_ampltiude_spread(data_ready_f11)
    duration_mean = calculate_GRS_duration_baseline(data_ready_f11)
    duration_std = calculate_GRS_duration_spread(data_ready_f11)
    slope_mean = calculate_GSR_Slope_baseline(data_ready_f11)
    slope_std = calculate_GSR_Slope_spread(data_ready_f11)
    #relSlope_mean = calculate_GSR_relSlope_baseline(data_ready_f11)
    #relSlope_std = calculate_GSR_relSlope_spread(data_ready_f11)
    #steepness_mean = calculate_GSR_steepness_baseline(data_ready_f11)
    #steepness_std = calculate_GSR_steepness_spread(data_ready_f11)
    #print(amplitude_mean, amplitude_std, duration_mean, duration_std, slope_mean, slope_std, relSlope_mean, relSlope_std, steepness_mean, steepness_std)

    data_ready_f11.reset_index(inplace = True)

    return amplitude_mean, amplitude_std, duration_mean, duration_std, slope_mean, slope_std


def angle(v1, v2):
    return math.acos((np.dot(v1,v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def GSR_amplitude_duration_slope(data: pd.DataFrame,
                                 show_calculations: bool = False):

    data_r = data[["GSR_onset", "GSR_peak", "consecutive_GSR_inc", "GSR"]]

    #print(data_r)
    #print(data)

    GSR_increase_durations = [0]*len(data)
    GSR_amplitude = [0]*len(data)
    rel_GSR_slope = [0]*len(data)
    GSR_slope = [0]*len(data)
    rel_change_proportional_to_time = [0]*len(data)


    for i in range(0, len(data_r)):
        j = i
        counter = 0

        p = np.array([0.0, 0.0])
        q = np.array([0.0, 0.0])
        v = np.array([0.0, 0.0])
        if data["GSR_onset"][i] == 1:
            local_GSR_onset = data["GSR"][i]
            p[1] = data["GSR"][i]
            v[1] = p[1]
            #v1[0] = v2[0]

            while data["GSR_peak"][j] != 1 and j != len(data_r) - 1:
                j += 1
                counter += 1
                local_GSR_peak = data["GSR"][j]
                q[1] = data["GSR"][j]

            # rule 1 prep - GSR Amplitude Increase between local GSR minimum and local GSR maximum
            amplitude = local_GSR_peak - local_GSR_onset

            GSR_amplitude[i] = amplitude

            # rule 2 prep - GSR Increase Duration between local GSR minimum and local GSR maximum
            GSR_increase_durations[i] = counter

            q[0] = counter
            v[0] = counter

            #print(p, q, v)

            v1 = v-p
            v1 = v1/np.linalg.norm(v1)
            v2 = q-p
            v2 = v2/np.linalg.norm(v2)

            rel_change = angle(v1, v2)
            rel_change_prop_to_time = rel_change / counter
            rel_change_proportional_to_time[i] = rel_change_prop_to_time
            rel_GSR_slope[i] = rel_change
            slope = math.degrees(rel_change)
            GSR_slope[i] = slope

    data["GSR_amplitude"] = GSR_amplitude
    data["GSR_increase_durations"] = GSR_increase_durations
    data["GSR_slope"] = GSR_slope
    data["GSR_relative_slope"] = rel_GSR_slope
    data["GSR_steepnessWrtToTime"] = rel_change_proportional_to_time

    #return GSR_amplitude, GSR_increase_durations, GSR_slope, rel_GSR_slope, rel_change_proportional_to_time
    return data


def calculate_GSR_ampltiude_baseline(data: pd.DataFrame, mode: str = "mean"):

    data_GSR = data[["GSR", "GSR_amplitude"]]

    GSR_amplitudes = data_GSR[data_GSR["GSR_amplitude"] > 0]

    if mode == "mean":
        amplitude_mean = np.mean(GSR_amplitudes["GSR_amplitude"])
        return amplitude_mean
    if mode == "median":
        amplitude_median = np.median(GSR_amplitudes["GSR_amplitude"])
        return amplitude_median


def calculate_GSR_ampltiude_spread(data: pd.DataFrame, mode: str = "std"):

    data_GSR = data[["GSR", "GSR_amplitude"]]

    GSR_amplitudes = data_GSR[data_GSR["GSR_amplitude"] > 0]

    if mode == "std":
        amplitude_std = np.std(GSR_amplitudes["GSR_amplitude"])
        return amplitude_std
    if mode == "var":
        amplitude_var = np.var(GSR_amplitudes["GSR_amplitude"])
        return amplitude_var

def calculate_GSR_steepness_baseline(data: pd.DataFrame, mode: str = "mean"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_steepnessWrtToTime"]]

    GSR_steepnessWrtToTime = data_GSR[data_GSR["GSR_steepnessWrtToTime"] > 0]

    if mode == "mean":
        steepnessWrtToTime_mean = np.mean(GSR_steepnessWrtToTime["GSR_steepnessWrtToTime"])
        return steepnessWrtToTime_mean
    if mode == "median":
        steepnessWrtToTime_median = np.median(GSR_steepnessWrtToTime["GSR_steepnessWrtToTime"])
        return steepnessWrtToTime_median

def calculate_GSR_steepness_spread(data: pd.DataFrame, mode: str = "std"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_steepnessWrtToTime"]]

    GSR_steepnessWrtToTime = data_GSR[data_GSR["GSR_steepnessWrtToTime"] > 0]

    if mode == "std":
        steepnessWrtToTime_std = np.std(GSR_steepnessWrtToTime["GSR_steepnessWrtToTime"])
        return steepnessWrtToTime_std
    if mode == "var":
        steepnessWrtToTime_var = np.var(GSR_steepnessWrtToTime["GSR_steepnessWrtToTime"])
        return steepnessWrtToTime_var

def calculate_GSR_Slope_baseline(data: pd.DataFrame, mode: str = "mean"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_slope"]]

    GSR_slope = data_GSR[data_GSR["GSR_slope"] > 0]

    if mode == "mean":
        slope_mean = np.mean(GSR_slope["GSR_slope"])
        return slope_mean
    if mode == "median":
        slope_median = np.median(GSR_slope["GSR_slope"])
        return slope_median

def calculate_GSR_Slope_spread(data: pd.DataFrame, mode: str = "std"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_slope"]]

    GSR_slope = data_GSR[data_GSR["GSR_slope"] > 0]

    if mode == "std":
        slope_std = np.std(GSR_slope["GSR_slope"])
        return slope_std
    if mode == "var":
        slope_var = np.var(GSR_slope["GSR_slope"])
        return slope_var

def calculate_GSR_relSlope_baseline(data: pd.DataFrame, mode: str = "mean"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_relative_slope"]]

    GSR_relSlope = data_GSR[data_GSR["GSR_relative_slope"] > 0]

    if mode == "mean":
        relSlope_mean = np.mean(GSR_relSlope["GSR_relative_slope"])
        return relSlope_mean
    if mode == "median":
        relSlope_median = np.median(GSR_relSlope["GSR_relative_slope"])
        return relSlope_median

def calculate_GSR_relSlope_spread(data: pd.DataFrame, mode: str = "std"):

    #rel_change_proportional_to_time
    data_GSR = data[["GSR", "GSR_relative_slope"]]

    GSR_relSlope = data_GSR[data_GSR["GSR_relative_slope"] > 0]

    if mode == "std":
        relSlope_std = np.std(GSR_relSlope["GSR_relative_slope"])
        return relSlope_std
    if mode == "var":
        relSlope_var = np.var(GSR_relSlope["GSR_relative_slope"])
        return relSlope_var

def calculate_GRS_duration_baseline(data: pd.DataFrame, mode: str = "mean"):

    data_GSR = data[["GSR", "GSR_increase_durations"]]

    GSR_durations = data_GSR[data_GSR["GSR_increase_durations"] > 0]

    if mode == "mean":
        durations_mean = np.mean(GSR_durations["GSR_increase_durations"])
        return durations_mean
    if mode == "median":
        durations_median = np.median(GSR_durations["GSR_increase_durations"])
        return durations_median


def calculate_GRS_duration_spread(data: pd.DataFrame, mode: str = "std"):

    data_GSR = data[["GSR", "GSR_increase_durations"]]

    GSR_durations = data_GSR[data_GSR["GSR_increase_durations"] > 0]

    if mode == "std":
        durations_std = np.std(GSR_durations["GSR_increase_durations"])
        return durations_std
    if mode == "var":
        durations_var = np.var(GSR_durations["GSR_increase_durations"])
        return durations_var


######### Skin Temperature

def calculate_ST_baseline(data: pd.DataFrame, mode: str = "mean"):

    data_ST = data[["ST"]]

    if mode == "mean":
        st_mean = np.mean(data_ST["ST"])
        return st_mean
    if mode == "median":
        st_median = np.median(data_ST["ST"])
        return st_median


def calculate_ST_spread(data: pd.DataFrame, mode: str = "std"):

    data_ST = data[["ST"]]

    if mode == "std":
        st_std = np.std(data_ST["ST"])
        return st_std
    if mode == "var":
        st_var = np.var(data_ST["ST"])
        return st_var


def calc_ST_rule_statistics(data, trim_time=False, start_mins=0, end_mins=0):
    # calculate mean and standard deviation for GSR Increase amplitude, duration and steepness proportional to time

    data_ready_f11 = data.copy()

    if trim_time:
        start_time = data_ready_f11["time_iso"].min()
        end_time = data_ready_f11["time_iso"].max()
        # print(start_time)
        # print(end_time)
        data_ready_f11.set_index("time_iso", inplace=True)

        start_trim = pd.to_timedelta(start_mins, unit="m")
        new_start_time = start_time + start_trim
        end_trim = pd.to_timedelta(end_mins, unit="m")
        new_end_time = end_time - end_trim
        # print(new_start_time)
        # print(new_end_time)
        data_ready_f11 = data_ready_f11[new_start_time:new_end_time]

        data_ready_f11.reset_index(inplace=True)

        ST_mean = calculate_ST_baseline(data_ready_f11, mode = "mean")
        ST_median = calculate_ST_baseline(data_ready_f11, mode = "median")
        ST_variance = calculate_ST_spread(data_ready_f11, mode = "var")
        ST_std = calculate_ST_spread(data_ready_f11, mode = "std")

        print(ST_mean, ST_median, ST_variance, ST_std)

        data_ready_f11.reset_index(inplace=True)

    else:
        ST_mean = calculate_ST_baseline(data_ready_f11, mode = "mean")
        ST_median = calculate_ST_baseline(data_ready_f11, mode = "median")
        ST_variance = calculate_ST_spread(data_ready_f11, mode = "var")
        ST_std = calculate_ST_spread(data_ready_f11, mode = "std")
        # print(f"ST Mean {ST_mean} -- ST Standard Deviation{ST_std} \n"
        #      f"ST Median {ST_median} -- ST Variance {ST_variance} \n ")

    return ST_mean, ST_median, ST_variance, ST_std


def calculate_signal_baseline(df, phys_signal: str = "GSR", mins: int = 5):

    data = df.copy()

    phys_signals_available = ["GSR_std", "ST_std", "IBI", "HRV", "BVP"]

    baseline_calc_duration = pd.to_timedelta(mins, unit = "m")
    start_time = data["time_iso"].min()
    #print(start_time)
    data.set_index("time_iso", inplace=True)
    end_time = start_time + baseline_calc_duration
    #print(end_time)
    baseline_data = data[start_time:end_time]

    data.reset_index(inplace=True)

    if phys_signal in phys_signals_available:
        baseline_mean = baseline_data[phys_signal].mean()


    return baseline_mean







##### old #####


def get_consecutive_GSR_increase_duration(df, min: int = 1, max: int = 20):
    ds = df.copy()

    increase = [None] * len(ds)
    idx = 0
    tmp_gsr_value = 1
    for gsr_value in ds["GSR_increase"]:
        if gsr_value == 1 and tmp_gsr_value != 0:
            tmp_gsr_value += 1
        elif gsr_value == 1 and tmp_gsr_value == 0:
            tmp_gsr_value = 1
        else:
            tmp_gsr_value = 0
        increase[idx] = tmp_gsr_value
        idx += 1

    df["increase"] = increase


    ds['GSR_increase'].replace(0, np.nan, inplace = True)

    for i in range(min, max + 1):
        ds[f"GSR_inc_{i}sec"] = ds['GSR_increase'].rolling(window = i, min_periods=i).sum()

    ds['GSR_inc_2sec'].fillna(ds['GSR_increase'], inplace = True)

    for j in range(min + 1, max + 1):
        ds[f'GSR_inc_{j}sec'].fillna(ds[f'GSR_inc_{j-1}sec'], inplace = True)

    return ds[f'GSR_inc_{max}sec']



def get_consecutive_ST_decrease_duration(df, min: int = 2, max: int = 90):
    ds = df.copy()
    ds['ST_decrease'].replace(0, np.nan, inplace = True)

    for i in range(min, max + 1):
        ds[f"ST_dec_{i}sec"] = ds['ST_decrease'].rolling(window = i, min_periods=i).sum()

    ds['ST_dec_2sec'].fillna(ds['ST_decrease'], inplace = True)

    for j in range(min + 1, max + 1):
        ds[f'ST_dec_{j}sec'].fillna(ds[f'ST_dec_{j-1}sec'], inplace = True)

    #ds.rename(columns = {'GSR_inc_6sec': 'GSR_increase_duration'})

    return ds[f'ST_dec_{max}sec']


def shift_rules_to_same_position(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'R1shifted' column to DataFrame where all rules that are met are position-corrected for summation of rules
    Parameters
    ----------
    data
    Returns
    -------
    """
    data["R1shifted"] = [0] * len(data)
    r1shift = [0] * len(data)

    for i in data.index:
        if data["R1"][i] > 0:
            secs = int(data["consecutive_GSR_inc"][i])
            r1shift[i - secs] = data["R1"][i]

    data["R1shifted"] = r1shift
    return data




####### new ST rule -- for each GSR Onset (or Peak), check a time window before and after the point and check for drastic increase
#  Sum of the first derivative values of Skin Temperature changes (slopes)
#  --> potentially use Absolute Values to account for Increases & Decreases


def ST_window_IncreaseDecrease_value(data_n,
                   start_mins = 0, end_mins = 0,
                   from_window=-5, to_window=5,
                   threshold=0.1, MOSpercentage=0.5,
                   print_stats: bool = False):

    data = data_n.copy()

    ST_average_window_sum = []
    ST_avg_window = [0] * len(data)

    ST_window_rule = [0] * len(data)

    start_time = data["time_iso"].min()
    end_time = data["time_iso"].max()

    data.set_index("time_iso", inplace=True)

    start_trim = pd.to_timedelta(start_mins, unit="m")
    new_start_time = start_time + start_trim
    end_trim = pd.to_timedelta(end_mins, unit="m")
    new_end_time = end_time - end_trim

    data = data[new_start_time:new_end_time]

    data.reset_index(inplace=True)

    print(f"Number of GSR peaks {len(data[data['GSR_peak'] > 0])}")

    for i in data.index:

        # TODO - check what's better -- "GSR_onset" or "GSR_peak"
        if data.loc[i, "GSR_peak"] > 0: # should be equivalent to data.loc[i, "GSR_onset"] == 1

            from_win = i + from_window if i + from_window > 0 else 0
            to_win = i + to_window if i + to_window < len(data) else len(data) - 1

            # TODO - could replace values with abs() value to account for Increases and Decreases
            # TODO - could also calculate variance & standard deviation and add this to avg (similar to new GSR rules)
            ST_1st_derivate_window_sum = sum(data.loc[from_win:to_win + 1].ST_1st_der)
            print(f"Sum of first derivate ST at position {from_win} to position {to_win}: {ST_1st_derivate_window_sum}")
            # TODO - check if this is the correct sum
            #ST_dec_sum = sum(data.loc[index + minimum_duration: index + maximum_duration, "ST_decrease"])

            ST_avg_window[i] = ST_1st_derivate_window_sum

            ST_average_window_sum.append(ST_1st_derivate_window_sum)

    data_n['ST_avg_window'] = ST_avg_window
    ST_avg_window_sum = np.mean(ST_average_window_sum)
    print(f"Average over ST window sums: {ST_avg_window_sum}")
    print(f"Number of GSR peaks {len(ST_average_window_sum)}")

    for i in data_n.index:

        # TODO - check what's better -- "GSR_onset" or "GSR_peak" or "MOS_Score"
        if ((data_n.loc[i, "MOS_Score"] > MOSpercentage) and (data_n.loc[i, "ST_avg_window"] > threshold)) or ((data_n.loc[i, "MOS_Score"] > MOSpercentage) and (data_n.loc[i, "ST_avg_window"] < -threshold)):
            ST_window_rule[i] = 1

    data_n['ST_rule_above_avg_window'] = ST_window_rule

    if print_stats:

        print(f"Average Window Sum over 1st Derivatives of Skin Temperature: {ST_avg_window_sum} \n")

    return data_n, ST_avg_window_sum