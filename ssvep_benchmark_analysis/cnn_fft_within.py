import function_pools as fp
from scipy.io import loadmat
import numpy as np
import pandas as pd
import argparse
import os
import mne

from sklearn.model_selection import KFold

# import tensorflow as tf
from tensorflow.keras import optimizers
from keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

def EEGNet_SSVEP(nb_classes = 3, Chans = 7, Samples = 250, 
             dropoutRate = 0.5, kernLength = 256, F1 = 96, 
             D = 1, F2 = 96, dropoutType = 'Dropout'):
    """ SSVEP Variant of EEGNet, as used in [1]. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. 
      D               : number of spatial filters to learn within each temporal
                        convolution.
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
      
      
    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6). 
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

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
list_el = [str('PZ'), str('PO3'), str('POz'), str('PO4'), str('O1'), str('Oz'), str('O2')]           # Electrodes to use

# list_el = [str('O1'), str('O2'), str('PO3'), str('PO4')]          # Electrodes to use

# list_el = [str('O1'), str('O2')]          # Electrodes to use

vec_ind_el = df_location[df_location['Label'].isin(list_el)].index             # Vector with indexes of electrodes to use
# ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]   # Index of reference electrode 'Cz'
ind_ref_el = df_location[df_location['Label'] == 'Cz'].index[0]

fs = 250                         # sampling frequency in hz
N_channel = len(list_el)         # number of channels
Nb = 6                                   # Number of Blocks
N_sec = 5
N_pre = int(0.5 * fs)            # pre stim
N_delay = int(0.140 * fs)        # SSVEP delay
N_stim = int(N_sec * fs)         # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

num_iter = 0

# ind_freq = [0, 4, 7]  # freq: 8Hz, 12Hz, 15Hz
ind_freq = [24, 18, 19]  #freq: 8.6Hz, 10.4Hz, 11.4Hz
Nf = len(ind_freq)
mat_filtered = np.zeros([Ns, Nb, Nf, N_channel, N_stim])
for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = fp.preprocess(list_subject_data[s][:, :, ind_freq[f], b], vec_ind_el, ind_ref_el, N_start, N_stop)
            
            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 0.5, 40, method='fir', phase='zero-double', verbose=False)
            mat_filtered[s, b, f, :, :] = mat_filt
            
mat_filtered_1 = mat_filtered[:,:,:,:,0:250]
mat_filtered_2 = mat_filtered[:,:,:,:,250:500]
mat_filtered_3 = mat_filtered[:,:,:,:,500:750]
mat_filtered_4 = mat_filtered[:,:,:,:,750:1000]
mat_filtered_5 = mat_filtered[:,:,:,:,1000:1250]

# feature 1: time-domain EEG features
mat_filtered = np.concatenate((mat_filtered_1, mat_filtered_2, mat_filtered_3, mat_filtered_4, mat_filtered_5), axis= 1)

# frequency resolution of the FFT is set to 0.2930 Hz
fft_resolution = 0.2930  
NFFT = round(fs/fft_resolution) 
 
# the frequency components between 3Hz and 35Hz are selected
fft_freq_start, fft_freq_end = 3.0, 35.0
fft_index_start = int(round(fft_freq_start/fft_resolution))
fft_index_end = int(round(fft_freq_end/fft_resolution)) + 1

# feature 2: Complex spectrum features
fft_complex_features = np.zeros([Ns, Nb*N_sec, Nf, N_channel, 2*(fft_index_end - fft_index_start)])  # length: 220

# feature 3: Magnitude spectrum features
fft_magnitude_features = np.zeros([Ns, Nb*N_sec, Nf, N_channel, (fft_index_end - fft_index_start)])  # length: 110

for s in range(0, Ns):
    for b in range(0, Nb*N_sec):
        for f in range(0, Nf):
            for c in range(0, N_channel):                
                # temp_FFT = np.fft.fft(mat_filtered[s,b,f,c,:]) / fs
                temp_FFT = np.fft.fft(mat_filtered[s,b,f,c,:], NFFT)/fs
                real_part = np.real(temp_FFT)
                imag_part = np.imag(temp_FFT)
                magnitude_spectrum = 2 * np.abs(temp_FFT)
                fft_complex_features[s,b,f,c,:] = np.concatenate((real_part[fft_index_start:fft_index_end], 
                                                                  imag_part[fft_index_start:fft_index_end]), axis = 0)
                fft_magnitude_features[s,b,f,c,:] = magnitude_spectrum[fft_index_start:fft_index_end] 
                    
all_acc = np.zeros((Ns, 1))
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True)
for sub in range(0, Ns):
    X1 = []
    X2 = []
    X3 = []
    Y = []
    for i in range(0, Nb*N_sec):
        for f in range(0, Nf):            
            X1.append(mat_filtered[sub, i, f, :, :])
            X2.append(fft_complex_features[sub, i, f, :, :])
            X3.append(fft_magnitude_features[sub, i, f, :, :])
            
            Y.append(f)
            
    X = np.asarray(X3)
    Y = np.asarray(Y)
    
    kf.get_n_splits(X)
    cv_acc = np.zeros((num_folds, 1))
    fold = -1
     
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        fold = fold + 1
        print("Subject:", sub+1, "Fold:", fold+1, "Training...")
            
        # convert data to NHWC (trials, channels, samples, kernels) format. Data 
        # contains 7 channels and 250 time-points. Set the number of kernels to 1.
        kernels = 1
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)
        
        # convert labels to one-hot encodings.
        Y_train = np_utils.to_categorical(Y_train)
        Y_test = np_utils.to_categorical(Y_test)
        
        model = EEGNet_SSVEP(nb_classes = Nf, Chans = N_channel, Samples = X_train.shape[2], 
                       dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                       dropoutType = 'Dropout')
        
        # compile the model and set the optimizers
        sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)  # stochastic gradient descent
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])
        
        # fit the model       
        fittedModel = model.fit(X_train, Y_train, batch_size = 32, epochs = 300, verbose = 0)
        
        score = model.evaluate(X_test, Y_test, verbose=0) 
    
        cv_acc[fold, :] = score[1]*100
        
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    
    all_acc[sub] = np.mean(cv_acc)
    print("...................................................")
    print("Subject:", sub+1, " - Accuracy:", all_acc[sub],"%")
    print("...................................................")

print(".....................................................................................")
print("Overall Accuracy Across Subjects:", np.mean(all_acc), "%", "std:", np.std(all_acc), "%")
print(".....................................................................................")

# np.save('acc_all_within_subject_time_domain_01.npy',all_acc)
# np.save('acc_all_within_subject_fft_complex_01.npy',all_acc)
np.save('acc_all_within_subject_fft_magnitude_01.npy',all_acc)

# a1 = np.load('acc_all_within_subject_time_domain_01.npy')  
# print("time_domain:", np.sum(a1 > 90), round(np.mean(a1),2)) 
# print(a1,"\n")

# a2 = np.load('acc_all_within_subject_fft_complex_01.npy') 
# print("fft_complex:", np.sum(a2 > 90), round(np.mean(a2),2)) 
# print(a2,"\n")

# a3 = np.load('acc_all_within_subject_fft_magnitude_01.npy') 
# print("fft_magnitude:", np.sum(a3 > 90), round(np.mean(a3),2)) 
# print(a3,"\n")

    
