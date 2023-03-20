import function_pools as fp
from scipy.io import loadmat
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import os
import mne

# Parser
parser = argparse.ArgumentParser(description = 'Add some integers.')

parser.add_argument('--length', action = 'store', type = float, default = 5,
                    help = 'Length of data to take into account (0,5].')

parser.add_argument('--subjects', action = 'store', type = int, default = 35,
                    help = 'Number of subjects to use [1,35].')

parser.add_argument('--tag', action = 'store', default = '',
                    help = 'Tag to add to the files.')

args = parser.parse_args()
N_sec = args.length
Ns = args.subjects
sTag = args.tag

print("Extended CCA: Tag: " + sTag + ", Subjects: " + str(Ns) + ", Data length: " + str(N_sec))

# Set Working Directory
abspath = os.path.abspath('__file__')
dirname = os.path.dirname(abspath)
os.chdir(dirname)

# dirname = os.getcwd()

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

# Load and prepare data
# mat_locations = np.genfromtxt(os.path.join(dir_data, '64-channel_locations.txt'))
mat_locations = np.genfromtxt(os.path.join(dir_data, '64-channels.loc'), dtype = str)

dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

list_subject_data = fp.load_sub_data(os.path.join(dirname, dir_data), '.mat')        # load all subject data

# Convert to pandas dataframe
# df_location = pd.read_table(os.path.join(dir_data, '64-channel_locations.txt'),
#                             names=['Electrode', 'Degree', 'Radius', 'Label'])
df_location = pd.read_table(os.path.join(dir_data, '64-channels.loc'),
                            names=['Electrode', 'Degree', 'Radius', 'Label'])
df_location['Label'] = df_location['Label'].astype('string').str.strip()
df_location['Electrode'] = df_location['Electrode'].astype(int) 

# channel selection
# list_el = [str('PZ'), str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
#            str('O2')]           # Electrodes to use

# list_el = [str('O1'), str('O2'), str('PO3'), str('PO4')]          # Electrodes to use

list_el = [str('O1'), str('O2')]          # Electrodes to use

vec_ind_el = df_location[df_location['Label'].isin(list_el)].index             # Vector with indexes of electrodes to use
# ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]   # Index of reference electrode 'Cz'
ind_ref_el = df_location[df_location['Label'] == 'Cz'].index[0]

fs = 250                         # sampling frequency in hz
N_channel = len(list_el)         # number of channels
# N_sec = 5
N_pre = int(0.5 * fs)            # pre stim
N_delay = int(0.140 * fs)        # SSVEP delay
N_stim = int(N_sec * fs)         # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

# Create Reference signals
vec_t = np.arange(-0.5, 5.5, 1 / 250)    # time vector
Nh = 5                                   # Number of harmonics
Nf = len(vec_freq)                       # Number of frequencies
Nb = 6                                   # Number of Blocks

mat_Y = np.zeros([Nf, Nh * 2, N_stim])   # [Frequency, Harmonics * 2, Samples]

for k in range(0, Nf):
    for i in range(1, Nh + 1):
        mat_Y[k, i - 1, :] = np.sin(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])
        mat_Y[k, i-1+Nh, :] = np.cos(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])

# Frequency detection using advanced CCA
list_time = []             # list to store the time per trial
list_rho = []
list_result = []           # list to store the subject wise results
list_bool_result = []      # list to store the classification as true/false
list_bool_thresh = []      # list to store the classification with thresholds

list_max = []

num_iter = 0
mat_filtered = np.zeros([Ns, Nb, Nf, N_channel, N_stim])
for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = fp.preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)
            
            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
            mat_filtered[s, b, f, :, :] = mat_filt

for s in range(0, Ns):
    mat_time = np.zeros([Nf, Nb], dtype='object')   # matrix to store time needed
    mat_rho = np.zeros([Nf, Nb])
    mat_ind_max = np.zeros([Nf, Nb])                # index of maximum cca
    mat_bool = np.zeros([Nf, Nb])
    mat_bool_thresh = np.zeros([Nf, Nb])    
    mat_max = np.zeros([Nf, Nb])

    t_start = datetime.now()

    # average within each subject
    for b in range(0, Nb):

        mat_blocks_dropped = np.delete(mat_filtered[s], b, axis=0)
        mat_X_train = np.mean(mat_blocks_dropped, axis=0)

        for f in range(0, Nf):
            t_trial_start = datetime.now()
            
            vec_rho = np.zeros(Nf)
            
            # Apply CCA                        
            for k in range(0, Nf):
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 0:250], mat_Y[k, :, 0:250], mat_X_train[k, :, 0:250])               # accuracy(1s_1): 50.95; 67.19             
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 250:500], mat_Y[k, :, 250:500], mat_X_train[k, :, 250:500])         # accuracy(1s_2): 52.50; 
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 500:750], mat_Y[k, :, 500:750], mat_X_train[k, :, 500:750])         # accuracy(1s_3): 54.60;
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 750:1000], mat_Y[k, :, 750:1000], mat_X_train[k, :, 750:1000])      # accuracy(1s_4): 44.54;
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 1000:1250], mat_Y[k, :, 1000:1250], mat_X_train[k, :, 1000:1250])   # accuracy(1s_5): 49.55; 
                
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 0:500], mat_Y[k, :, 0:500], mat_X_train[k, :, 0:500])               # accuracy(2s_1): 69.19; 83.35
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 250:750], mat_Y[k, :, 250:750], mat_X_train[k, :, 250:750])         # accuracy(2s_2): 70.13
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 500:1000], mat_Y[k, :, 500:1000], mat_X_train[k, :, 500:1000])      # accuracy(2s_3): 70.51
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 750:1250], mat_Y[k, :, 750:1250], mat_X_train[k, :, 750:1250])      # accuracy(2s_4): 66.49

                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 0:750], mat_Y[k, :, 0:750], mat_X_train[k, :, 0:750])               # accuracy(3s_1): 77.51; 88.52
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 250:1000], mat_Y[k, :, 250:1000], mat_X_train[k, :, 250:1000])      # accuracy(3s_2): 79.37
                vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 500:1250], mat_Y[k, :, 500:1250], mat_X_train[k, :, 500:1250])      # accuracy(3s_3): 78.38
                
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 0:1000], mat_Y[k, :, 0:1000], mat_X_train[k, :, 0:1000])            # accuracy(4s_1): 83.74; 92.43
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 250:1250], mat_Y[k, :, 250:1250], mat_X_train[k, :, 250:1250])      # accuracy(4s_2):
                
                # vec_rho[k] = fp.apply_ext_cca(mat_filtered[s, b, f, :, 0:1250], mat_Y[k, :, 0:1250], mat_X_train[k, :, 0:1250])            # accuracy(5s_1): 86.86; 94.37
                 
                    
            t_trial_end = datetime.now()
            mat_time[f, b] = t_trial_end - t_trial_start
            mat_rho[f, b] = np.max(vec_rho)
            mat_ind_max[f, b] = np.argmax(vec_rho)                   # get index of maximum -> frequency -> letter
            
            mat_bool[f, b] = mat_ind_max[f, b].astype(int) == f      # compare if classification is true
            mat_bool_thresh[f, b] = mat_ind_max[f, b].astype(int) == f
            
            # apply threshold
            mat_stand = fp.standardize(mat_filt)
            mat_max[f, b] = np.max(np.abs(mat_stand))
            thresh = 6
            if np.max(np.abs(mat_stand)) > thresh:
                # minus 1 if it is going to be removed
                mat_bool_thresh[f, b] = -1
                
            num_iter = num_iter + 1
    
    list_time.append(mat_time)          # store time for calculating cca per subject
    list_rho.append(mat_rho)
    list_result.append(mat_ind_max)     # store results per subject   
    list_bool_result.append(mat_bool)
    list_bool_thresh.append(mat_bool_thresh)    
    list_max.append(mat_max)

    t_end = datetime.now()
    print("Extended CCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_time = np.concatenate(list_time, axis=1)
mat_rho = np.concatenate(list_rho, axis=1)
mat_result = np.concatenate(list_result, axis=1)
mat_b = np.concatenate(list_bool_result, axis=1)
mat_b_thresh = np.concatenate(list_bool_thresh, axis=1)
mat_max = np.concatenate(list_max, axis=1)

# Analysis
accuracy_all = fp.accuracy(vec_freq, mat_result)
accuracy_drop = fp.acc(mat_b_thresh)

print("Extended CCA: accuracy: " + str(accuracy_all))
print("Extended CCA: accuracy dropped: " + str(accuracy_drop))

sTag = '_' + str(sTag)
sSec = '_' + str(N_sec)
if sTag != "":
    sNs = '_' + str(Ns)

np.save(os.path.join(dir_results, 'cca_mat_time' + sSec + sNs + sTag), mat_time)
np.save(os.path.join(dir_results, 'cca_mat_rho' + sSec + sNs + sTag), mat_rho)
np.save(os.path.join(dir_results, 'cca_mat_result' + sSec + sNs + sTag), mat_result)
np.save(os.path.join(dir_results, 'cca_mat_b' + sSec + sNs + sTag), mat_b)
np.save(os.path.join(dir_results, 'cca_mat_b_thresh' + sSec + sNs + sTag), mat_b_thresh)
np.save(os.path.join(dir_results, 'cca_mat_max' + sSec + sNs + sTag), mat_max)