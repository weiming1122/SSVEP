# Import necessary libralies
import numpy as np  
import time
import function_pool as fp
from collections import deque
from scipy.stats import mode
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock  # lab streaming layer

# list_freqs = [240/28, 240/23, 240/21, 240/18]
list_freqs = [240/28, 240/23, 240/21]

Nf = len(list_freqs)

# Create prediction stream
info_predict = StreamInfo('MyPredictStream', 'Predicts', 1, 0, 'string', 'CytonPredictID')
outlet_predict = StreamOutlet(info_predict) 

# Get EEG information
print("looking for EEG streams...")
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise(RuntimeError, "Can\'t find EEG stream.")    
else:
    print("Success: found EEG stream.")
inlet = StreamInlet(streams[0])

info = inlet.info()
description_eeg = info.desc()
fs = int(info.nominal_srate())
Nchan = info.channel_count()
ch = description_eeg.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, Nchan):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))
      
# Select channel of interests
channel_interest = ['O1','O2','Oz','PO3','PO4','POz','Pz'] 
# channel_interest = ['O1','O2', 'P3', 'P4']
ch_id = []
for channel in channel_interest:
    if channel in ch_names:
        ch_id.append(ch_names.index(channel))
N_channel = len(ch_id)

# Set online decoding parameters       
# test_length = 1.5
# decision_window_size = 3
# freshrate = 2.5

test_length = 1.5
decision_window_size = 5
freshrate = 3

# Reference signals
Nh = 3

# Initialize raw EEG data
eeg_test = np.zeros((N_channel, int(fs * test_length)))

EEG_raw = []
maxlength = 5 * fs
for i in range(N_channel):
    EEG_raw.append(deque([], maxlen=maxlength))
    
corr = []
for i in range(Nf):
    corr.append(deque([], maxlen=5))
   
send = -1
decision_count = 0
decision_buffer = []
start_time = time.time()
sample_count = 0

while True:
    try:                      
        current_time = time.time()
        data, timestamp = inlet.pull_sample(timeout=0.0)
        if data:
            sample_count += 1
            data_interest = [data[i] for i in ch_id]   # (samples, channels)      
            for i in range(N_channel):
                EEG_raw[i].append(data_interest[i])
            
    except KeyboardInterrupt:   
        break
        print('Closing!')
    
    # Real time predict 
    if current_time - start_time >= 1/freshrate and len(EEG_raw[0]) > 4 * fs:
        # print(sample_count)
        sample_count = 0
        
        # Pad，filter and notch here 
        for i in range(N_channel):
            temp =  np.pad(EEG_raw[i], 10*fs, 'edge')
            temp_filtered = fp.butter_bandpass_filter(temp, 4, 48, fs, notch=50)[10*fs:-10*fs]           
            eeg_test[i] = temp_filtered[-int(test_length * fs):]
            
        # CCA
        ssvep_ = fp.ssvep_cca(list_freqs = list_freqs, harmonics = Nh, fs = fs, dim = N_channel, winSize = int(test_length * fs))
        vec_rho = ssvep_.predict(eeg_test)
        
        for rho in vec_rho:
            print('%.2f' %(rho), end=', ')
        print('\n')
        
        decision = np.argmax(vec_rho)
            
        decision_buffer.append(decision)
        decision_count += 1
    
        if decision_count == decision_window_size:
            decision_final = mode(decision_buffer)[0][0]
            n_decision_final = mode(decision_buffer)[1][0] 
            print(decision_buffer)
            # if n_decision_final > decision_window_size/2 and decision_final != -1:
            if n_decision_final >= decision_window_size - 1 and decision_final != -1:
                send = np.round(list_freqs[decision_final],3)
            else:
                send = -1
            
            outlet_predict.push_sample([str(send)], local_clock())    
            
            decision_buffer.pop(0) 
            decision_count -= 1
        
        print('\nCommand: %s, at time %.3f\n' % (str(send), current_time - start_time))             
        start_time = current_time
   
        

