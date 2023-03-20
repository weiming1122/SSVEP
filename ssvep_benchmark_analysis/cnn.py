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

ind_freq = [0, 4, 7]
Nf = len(ind_freq)
mat_filtered = np.zeros([Ns, Nb, Nf, N_channel, N_stim])
for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = fp.preprocess(list_subject_data[s][:, :, ind_freq[f], b], vec_ind_el, ind_ref_el, N_start, N_stop)
            
            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
            mat_filtered[s, b, f, :, :] = mat_filt
            
# Nf = len(vec_freq)
# mat_filtered = np.zeros([Ns, Nb, Nf, N_channel, N_stim])
# for s in range(0, Ns):
#     for b in range(0, Nb):
#         for f in range(0, Nf):
#             # Referencing and baseline correction
#             mat_data = fp.preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)
            
#             # Filter data
#             mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
#             mat_filtered[s, b, f, :, :] = mat_filt

mat_filtered_1 = mat_filtered[:,:,:,:,0:250]
mat_filtered_2 = mat_filtered[:,:,:,:,250:500]
mat_filtered_3 = mat_filtered[:,:,:,:,500:750]
mat_filtered_4 = mat_filtered[:,:,:,:,750:1000]
mat_filtered_5 = mat_filtered[:,:,:,:,1000:1250]
mat_filtered = np.concatenate((mat_filtered_1, mat_filtered_2, mat_filtered_3, mat_filtered_4, mat_filtered_5), axis= 1)

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

def EEGNet_SSVEP(nb_classes = 12, Chans = 8, Samples = 256, 
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

acc_all = []
for sub in range(0, Ns):
    X = []
    Y = []
    for i in range(0, Nb*N_sec):
        for f in range(0, Nf):
            X.append(mat_filtered[sub, i, f, :, :])
            Y.append(f)
    X = np.asarray(X)
    Y = np.asarray(Y)
     
    X_train      = X[0:60,]
    Y_train      = Y[0:60]
    X_validate   = X[60:75,]
    Y_validate   = Y[60:75]
    X_test       = X[75:,]
    Y_test       = Y[75:]
    
    
    # convert data to NHWC (trials, channels, samples, kernels) format. Data 
    # contains 7 channels and 250 time-points. Set the number of kernels to 1.
    kernels = 1
    X_train = X_train.reshape(X_train.shape[0], N_channel, fs, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], N_channel, fs, kernels)
    X_test = X_test.reshape(X_test.shape[0], N_channel, fs, kernels)
    
    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train)
    Y_validate = np_utils.to_categorical(Y_validate)
    Y_test = np_utils.to_categorical(Y_test)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    
    model = EEGNet_SSVEP(nb_classes = Nf, Chans = N_channel, Samples = fs, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout')
    
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    # model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    
    # count number of parameters in the model
    numParams  = model.count_params()    
    
    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                   save_best_only=True)
    
    # set class weight
    class_weights = {0:1, 1:1, 2:1}
    
    # fit the model       
    fittedModel = model.fit(X_train, Y_train, batch_size = 4, epochs = 100, 
                            verbose = 2, validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight = class_weights)

    # print(fittedModel.history)
    
    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')
    
    
    # make prediction on test set.
    probs = model.predict(X_test)
    preds = probs.argmax(axis = -1)  
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    print("########################################################")
    print("Subject {} ACC {}".format(sub, acc))
    print("########################################################")
    acc_all.append(acc)
print("Classification accuracy: ", acc_all)
np.save('acc_all_within_subject_batch_4.npy',acc_all)


a1 = np.load('acc_all_within_subject_batch_1.npy') 
print("Batch 1:", np.sum(a1 > 0.9)) 
print(a1,"\n")

a2 = np.load('acc_all_within_subject_batch_2.npy')  
print("Batch 2:", np.sum(a2 > 0.9)) 
print(a2,"\n")

a4 = np.load('acc_all_within_subject_batch_4.npy')  
print("Batch 4:", np.sum(a4 > 0.9)) 
print(a4,"\n")

a8 = np.load('acc_all_within_subject_batch_8.npy')
print("Batch 8:", np.sum(a8 > 0.9))   
print(a8,"\n")

a16 = np.load('acc_all_within_subject_batch_16.npy')  
print("Batch 16:", np.sum(a16 > 0.9)) 
print(a16,"\n")

a32 = np.load('acc_all_within_subject_batch_32.npy') 
print("Batch 32:", np.sum(a32 > 0.9))  
print(a32,"\n")

a64 = np.load('acc_all_within_subject_batch_64.npy') 
print("Batch 64:", np.sum(a64 > 0.9))  
print(a64,"\n")

a128 = np.load('acc_all_within_subject_batch_128.npy') 
print("Batch 128:", np.sum(a128 > 0.9))  
print(a128,"\n")
    
