# import necessary libralies  
from pylsl import StreamInfo, StreamOutlet
from pylsl import StreamInlet, resolve_byprop, local_clock
from collections import deque
import numpy as np
import time
import function_pool as fp

# # 40 letters
# freq_line = [[x+8.0 for x in range(8)], 
#               [x+8.2 for x in range(8)], 
#               [x+8.4 for x in range(8)], 
#               [x+8.6 for x in range(8)],
#               [x+8.8 for x in range(8)]]
# freqs = [freq for line in freq_line for freq in line]

# 38 letters
freqs = [240/31, 240/29, 240/27, 240/25, 240/23, 240/22, 240/21, 240/20]


# create Predict stream
info_predict = StreamInfo('MyPredictStream', 'Predicts', 1, 0, 'string', 'PredictID')
outlet_predict = StreamOutlet(info_predict) 

# find EEG stream
print('looking for EEG stream...')
stream_eeg = resolve_byprop('type', 'EEG', timeout=2)
if len(stream_eeg) == 0:
    raise(RuntimeError, "can\'t find EEG stream.")    
else:
    print('success: found EEG stream.')
inlet_eeg = StreamInlet(stream_eeg[0])

info = inlet_eeg.info()
description_eeg = info.desc()
fs = int(info.nominal_srate())
n_channels = info.channel_count()
ch = description_eeg.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, n_channels):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))
      
# select channel of interests
# channel_interest = ['O1','O2','Oz','PO3','PO4','POz','Pz']  # dry electrodes
channel_interest = ['O1', 'O2', 'Pz', 'P3', 'P4']   # wet electrodes
channel_interest = ['O1', 'O2', 'Oz', 'PO3', 'PO4']
ch_id = []
for channel in channel_interest:
    if channel in ch_names:
        ch_id.append(ch_names.index(channel))
n_channels = len(ch_id)

# find Marker stream   
print('looking for Marker stream...')
stream_marker = resolve_byprop('type', 'Markers', timeout=10)
if stream_marker:
    inlet_marker = StreamInlet(stream_marker[0])
    print('find Markers stream')
else:
    inlet_marker = False
    print("can't find Markers stream")
    
# set online decoding parameters       
test_length = 3
eeg_test = np.zeros((n_channels, int(fs * test_length)))

# initialize raw EEG data
EEG_raw = []
maxlength = 5 * fs
for i in range(n_channels):
    EEG_raw.append(deque([], maxlen=maxlength))

ssvep_ = fp.ssvep_cca(list_freqs = freqs, harmonics = 3, fs = fs, dim = n_channels, winSize = int(test_length * fs))
      
start_time = time.time()
while True:
    try:                      
        current_time = time.time()
        
        data, _ = inlet_eeg.pull_sample(timeout=0.0)        
        if data:
            data_interest = [data[i] for i in ch_id]     
            for i in range(n_channels):
                EEG_raw[i].append(data_interest[i])
                
        if inlet_marker:
            marker, _ = inlet_marker.pull_sample(timeout=0.0)            
            if marker:
                print('\ngot Marker: %s' % (marker[0]))
                if marker[0] == '999':
                    break
                elif marker[0] == '1':
                    # pad, filter and notch here 
                    for i in range(n_channels):
                        temp =  np.pad(EEG_raw[i], 10*fs, 'edge')
                        temp_filtered = fp.butter_bandpass_filter(temp, 4, 48, fs, notch=50)[10*fs:-10*fs]           
                        eeg_test[i] = temp_filtered[-int(test_length * fs):]
                        
                    # calc CCA
                    vec_rho = ssvep_.predict(eeg_test)
                    
                    for rho in vec_rho:
                        print('%.2f' %(rho), end=', ')
                    
                    decision = np.argmax(vec_rho)
                        
                    outlet_predict.push_sample([str(decision)], local_clock())    
                                
                    print('\nsend predict: %s, in time interval: %.3f' % (str(decision), current_time - start_time))             
                    start_time = current_time
            
    except KeyboardInterrupt:   
        break
        print('Closing!')
    
    
   
        

