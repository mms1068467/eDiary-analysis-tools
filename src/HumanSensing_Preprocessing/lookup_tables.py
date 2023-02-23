from datetime import datetime

interval_dic = {'GSR': '1000', 'ST': '1000', 'IBI': '1000', 'ECG': '1000'}  # sampling interval
platform_id_dic = {'GSR': '3', 'ST': '3', 'IBI': '2', 'ECG': '2'} # signal lookup table for platform_id
sensor_id_dic = {'GSR': '7', 'ST': '3', 'IBI': '16', 'ECG': '15'}  # signal lookup table for sensor_id

freq_dic = {'GSR': 4, 'ST': 4, 'IBI': 1}  # frequency lookup table (in Hz)
cluster_size_dic = {'GSR': 6, 'ST': 8, 'IBI': 1}  # values per cluster

# dictionaries of known filter parameter
butter_filter_order_dic = {'GSR': 1, 'ST': 2}  # low-, and high-pass butterworth filters - have actually the same order

# defining low- and high-pass cutoff frequencies for filtering GSR and ST measurements (based on literature)
# w = cutoff Freq. / (Sampling Freq / 2) -> (Sampling Freq / 2) = Niquist Theorem
lowpass_cutoff_freq_dic = {'GSR': 1 / (4 / 2), 'ST': 0.07 / (4 / 2)}  # GSR 0.5 as second option
highpass_cutoff_freq_dic = {'GSR': 0.05 / (4 / 2), 'ST': 0.005 / (4 / 2)}


### Labtest data

# Corresponding starttimes to labtest sessions
lab_session_starttime_dic = {'lab1/session1': datetime(2018, 6, 27, 10, 48, 16),
                             'lab2/session1': datetime(2019,2,13, 10,51,48),
                             'lab2/session2': datetime(2019,2,13, 11,40,37),
                             'lab2/session3': datetime(2019,2,13, 14,40,52),
                             'lab2/session4': datetime(2019,2,13, 15,43,32),
                             'lab2/session5': datetime(2019,2,13, 16,34,14),
                             'lab3/session1': datetime(2021,4,20, 9,15,34),
                             'lab3/session2': datetime(2021,4,20, 10,7,22),
                             'lab3/session3': datetime(2021,4,21, 11,44,29),
                             'lab3/session4': datetime(2021,4,21, 12,48,51),
                             'lab4/session2': datetime(2022,2,16, 11,14,1),
                             'lab4/session3': datetime(2022,2,16, 12,11,11),
                             }

# Corresponding starttimes to labtest sessions
lab_session_starttime_dic2 = {'lab1\\session1': datetime(2018, 6, 27, 10, 48, 16),
                             'lab2\\session1': datetime(2019,2,13, 10,51,48),
                             'lab2\\session2': datetime(2019,2,13, 11,40,37),
                             'lab2\\session3': datetime(2019,2,13, 14,40,52),
                             'lab2\\session4': datetime(2019,2,13, 15,43,32),
                             'lab2\\session5': datetime(2019,2,13, 16,34,14),
                             'lab3\\session1': datetime(2021,4,20, 9,15,34),
                             'lab3\\session2': datetime(2021,4,20, 10,7,22),
                             'lab3\\session3': datetime(2021,4,21, 11,44,29),
                             'lab3\\session4': datetime(2021,4,21, 12,48,51),
                             'lab4\\session2': datetime(2022,2,16, 11,14,1),
                             'lab4\\session3': datetime(2022,2,16, 12,11,11),
                             }

# Corresponding stresstimes to labtest sessions
lab_session_stresstimes_dic = {'lab1/session1': (56, 136, 223, 321, 410, 513, 600, 698, 789, 866),
                               'lab2/session1': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2/session2': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2/session3': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2/session4': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2/session5': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab3/session1': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3/session2': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3/session3': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3/session4': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab4/session2': (65, 190, 263, 331, 408, 483, 557, 655, 752, 825),
                               'lab4/session3': (65, 190, 263, 331, 408, 483, 557, 655, 752, 825),
                               }

# Corresponding stresstimes to labtest sessions
lab_session_stresstimes_dic2 = {'lab1\\session1': (56, 136, 223, 321, 410, 513, 600, 698, 789, 866),
                               'lab2\\session1': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2\\session2': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2\\session3': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2\\session4': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab2\\session5': (70, 149, 241, 315, 422, 499, 596, 667, 779, 861),
                               'lab3\\session1': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3\\session2': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3\\session3': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab3\\session4': (65, 190, 263, 315, 408, 483, 557, 655, 752, 825),
                               'lab4\\session2': (65, 190, 263, 331, 408, 483, 557, 655, 752, 825),
                               'lab4\\session3': (65, 190, 263, 331, 408, 483, 557, 655, 752, 825),
                               }
