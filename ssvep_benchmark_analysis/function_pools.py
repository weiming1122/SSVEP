# import necessary libralies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt 
from scipy.io import loadmat
import os
import seaborn as sns
from sklearn.cross_decomposition import CCA         
import warnings
warnings.filterwarnings('ignore')

def load_sub_data(fpath, fname):
    """
    Load all files (unless specify certain file excluded) in the folder and save them into a list
    
    Parameters
    ----------
    fname: string    
    file name to look for
    
    fpath: string
    file path
        
    Return
    -------
    list_data: list
    list of all data in files
    """
    counter = 0
    list_sub_data = list()
    for file in os.listdir(fpath):
        try:
            if file.endswith(fname) and file != "Freq_Phase.mat":
                file_load = os.path.join(fpath, str(file))
                list_sub_data.append(loadmat(str(file_load))['data'])
                counter += 1
        except Exception as e:
            raise e
            print("No files found here!")

    print(".mat Files found:\t" + str(counter))
    return list_sub_data

def preprocess(mat_input, vec_pick_el, i_ref_el, n_start, n_stop):
    """
    Extract and preprocess the data by applying baseline correction, referencing, channel selection, and cropping
    
    Parameters
    ----------
    mat_input: array, shape(n_channel,n_samples)
    Array containing the data for one trial.
    
    vec_pick_el: Int64Index, size(n_channel)
    The indices of the selected electrodes.
    
    i_ref_el: int
    Index for reference electrode.
    
    n_start: int
    Index for start sample.
    
    n_stop: int
    Index for stop sample.
        
    Returns
    -------
    mat_output: array, shape(n_channel,n_sample)
    The preprocessed data.
    """
    # Referencing and baseline correction
    mat_output = mat_input - mat_input[i_ref_el, :]                    # reference
    baseline = np.mean(np.mean(mat_output[:, 0:n_start], axis=0))      # get baseline (DC offset)    
    # mat_output = mat_output - np.mean(mat_output, axis=0)            # common average referencing
    mat_output = mat_output[vec_pick_el, n_start:n_stop] - baseline    # channel selection and apply baseline correction
    return mat_output

def apply_cca(X, Y):
    """
    Computes the maximum canonical correltion via cca
    
    Parameters
    ----------
    X: array, shape (n_channels, n_samples)
    Input data.
    
    Y: array, shape (n_ref_signals, n_samples)
    Reference signal to find correlation with
        
    Returns
    -------
    rho : int
    The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca = CCA(n_components = n_comp)
    cca.fit(X.transpose(), Y.transpose())             # transpose to bring into shape (n_samples, n_features)
    x, y = cca.transform(X.transpose(), Y.transpose())
    rho = np.diag(np.corrcoef(x, y, rowvar=False)[ : n_comp, n_comp: ])
    return rho

def apply_ext_cca(X, Y, X_Train):
    """
    Computes the maximum canonical correltion via cca
    
    Parameters
    ----------
    X: array, shape (n_channels, n_samples)
    Input data.
    
    Y: array, shape (n_ref_signals, n_samples)
    Reference signal to find correlation with
    
    X_Train: array, shape (n_channels, b_n_samples)
    Second Reference data
           
    Returns
    -------
    rho: int
    The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca1 = CCA(n_components = n_comp)
    cca2 = CCA(n_components = n_comp)
    cca3 = CCA(n_components = n_comp)
    cca4 = CCA(n_components = n_comp)

    cca1.fit(X.transpose(), Y.transpose())
    x, y = cca1.transform(X.transpose(), Y.transpose())
    rho_1 = np.diag(np.corrcoef(x, y, rowvar=False)[ : n_comp, n_comp: ])
    
    cca2.fit(X.transpose(), X_Train.transpose())
    w_xxt = cca2.x_weights_
    
    cca3.fit(X.transpose(), Y.transpose())
    w_xy = cca3.x_weights_
    
    cca4.fit(X_Train.transpose(), Y.transpose())
    w_xty = cca4.x_weights_
        
    rho_2 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xxt), np.matmul(X_Train.transpose(), w_xxt), 
                                rowvar=False)[ : n_comp, n_comp: ])
    
    rho_3 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), 
                                rowvar=False)[ : n_comp, n_comp: ])
    
    rho_4 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), 
                                rowvar=False)[ : n_comp, n_comp: ])
    
    # eq 8, Chen 2015,PNAS
    rho = np.sign(rho_1) * rho_1 ** 2 + np.sign(rho_2) * rho_2 ** 2 + np.sign(
          rho_3) * rho_3 ** 2 + np.sign(rho_4) * rho_4 ** 2

    return rho
    
def apply_ext_fbcca(X, Y, X_Train):
    """
    Computes the maximum canonical correltion via cca
    
    Parameters
    ----------
    X: array, shape (n_channels, n_samples)
    Input data.
    
    Y: array, shape (n_ref_signals, n_samples)
    Reference signal to find correlation with
    
    X_Train: array, shape (n_channels, b_n_samples)
    Second Reference data
    
    Returns
    -------
    rho: int
    The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca1 = CCA(n_components=n_comp)
    cca2 = CCA(n_components=n_comp)
    cca3 = CCA(n_components=n_comp)
    cca4 = CCA(n_components=n_comp)

    cca1.fit(X.transpose(), Y.transpose())
    x, y = cca1.transform(X.transpose(), Y.transpose())
    rho_1 = np.diag(np.corrcoef(x, y, rowvar=False)[ : n_comp, n_comp: ])
    
    cca2.fit(X.transpose(), X_Train.transpose())
    w_xxt_x = cca2.x_weights_
    w_xxt_y = cca2.y_weights_
    
    cca3.fit(X.transpose(), Y.transpose())
    w_xy = cca3.x_weights_
    
    cca4.fit(X_Train.transpose(), Y.transpose())
    w_xty = cca4.x_weights_
    
    rho_2 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xxt_x), np.matmul(X_Train.transpose(), w_xxt_x), 
                                rowvar=False)[ : n_comp, n_comp: ])
    rho_3 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), 
                                rowvar=False)[ : n_comp, n_comp: ])
    rho_4 = np.diag(np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), 
                                rowvar=False)[ : n_comp, n_comp: ])
    rho_5 = np.diag(np.corrcoef(np.matmul(X_Train.transpose(), w_xxt_x), np.matmul(X_Train.transpose(), w_xxt_y), 
                                rowvar=False)[ : n_comp, n_comp: ])

    rho = np.sign(rho_1) * rho_1 ** 2 + np.sign(rho_2) * rho_2 ** 2 + np.sign(
          rho_3) * rho_3 ** 2 + np.sign(rho_4) * rho_4 ** 2 + np.sign(rho_5) * rho_5 ** 2

    return rho

def accuracy(freqs, result):
    """
    Computes the accuracy
    
    Parameters
    ----------
    freqs: array, shape (n_freqs, 1)
    Frequencies / labels.
    
    result: array, shape (n_freqs, n_trials)
    The estimated frequencies
        
    Returns
    -------
    accuracy: float
    The accuracy in percent
    """
    n_correct = np.sum(freqs.reshape(40, 1) == freqs[result.astype(int)])
    return 100 * n_correct / (np.size(result))

def weight(n):
    """
    Computes the weight for fbcca coefficients
    
    Follow computation in paper "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based
    brain-computer interface", Chen et al., 2015 J.Neural Engl.
    
    Set a and b to the values they estimated to be the best
    a = 1.25
    b = 0.25
    
    Parameters
    ----------
    n: int
    index of subband
        
    Returns
    -------
    weight: float
    weight
    """
    a = 1.25
    b = 0.25
    return np.power(n, -a) + b     # eq. 7

def make_df(results, freqs, phase, n_freq, n_sub, n_blocks, time=None):
    """
    create dataframe
    
    Parameters
    ----------
    results: array, shape(n_freq, n_subjects * n_blocks)
    results per trial
        
    freqs: array, shape(n_freq,)
    The stimulation frequencies
        
    phase: array, shape(n_freq,)
    The stimulation phases
        
    n_freq: int
    number of frequencies
        
    n_sub: int
    number of subjects
        
    n_blocks: int
    number of blocks
        
    time: array, shape(n_freq, n_subjects * n_blocks)
    time needed per trial in ms
        
    Return
    -------
    df: DataFrame, shape(n_trials, ['Estimation', 'Frequency', 'Phase', 'Subject', 'Block'])
    The DataFrame
    """

    list_col_names = ['Estimation', 'Frequency', 'Phase', 'Subject', 'Block']
    df = pd.DataFrame(columns = list_col_names)

    df['Estimation'] = freqs[results.astype(int)].flatten('F')
    df['Frequency'] = np.concatenate(n_sub * n_blocks * [freqs])
    df['Phase'] = np.concatenate(n_sub * n_blocks * [phase])
    
    if time is not None:
        df['Time'] = (pd.to_timedelta(time.flatten('F'))).astype('timedelta64[ms]')

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq : s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq, s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq : s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)
    df['Error'] = (df['Estimation'] - df['Frequency']).abs()
    df['Compare'] = df['Estimation'] == df['Frequency']
    
    return df

def mk_df(time, rho, results, threshold, max, freqs, n_freq, n_sub, n_blocks):
    """
    create dataframe
    
    Parameters
    ----------    
    time: array, shape(n_freq, n_subjects * n_blocks)
    time needed per trial in ms
    
    rho: array, shape(n_freq, n_subjects * n_blocks)
    The maximum rho per trial
    
    results: array, shape(n_freq, n_subjects * n_blocks)
    classification results per trial
        
    threshold: array, shape(n_freq, n_subjects * n_blocks)
    results with applied thresholds. rejected trials are stored as -1
            
    max: array, shape(n_freq, n_subjects * n_blocks)
    The maximum value per trial    
    
    freqs: array, shape(n_freq,)
    The stimulation frequencies
    
    n_freq: int
    number of frequencies
    
    n_sub: int
    number of subjects
    
    n_blocks: int
    number of blocks
    
    Return
    -------
    df: DataFrame, shape(n_trials, ['Subject', 'Block', 'Frequency', 'Estimation', 'Threshold', 'Max', 'Rho', 'Compare', 'Time'])
    The DataFrame
    """

    list_col_names = ['Subject', 'Block', 'Frequency', 'Estimation', 'Threshold', 'Max', 'Rho', 'Compare', 'Time']
    df = pd.DataFrame(columns = list_col_names)

    df['Estimation'] = freqs[results.astype(int)].flatten('F')
    df['Threshold'] = threshold.flatten('F')
    df['Max'] = max.flatten('F')
    df['Rho'] = rho.flatten('F')
    df['Frequency'] = np.concatenate(n_sub * n_blocks * [freqs])
    df['Time'] = (pd.to_timedelta(time.flatten('F'))).astype('timedelta64[ms]')

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq:s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq, s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq:s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)
    df['Compare'] = df['Estimation'] == df['Frequency']
    
    return df

def set_style(fig, ax=None):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset={'left': 10, 'bottom': 5})

    if ax:
        ax.yaxis.label.set_size(10)
        ax.xaxis.label.set_size(10)
        ax.grid(axis='y', color='C7', linestyle='--', lw=.5)
        ax.tick_params(which='major', direction='out', length=3, width=1, bottom=True, left=True)
        ax.tick_params(which='minor', direction='out', length=2, width=0.5, bottom=True, left=True)
        plt.setp(ax.spines.values(), linewidth=.8)
    return fig, ax

def set_size(fig, a, b):
    fig.set_size_inches(a, b)
    fig.set_tight_layout(True)
    return fig

def itr(df, t):          # the average time for a selection (N_sec + 0.5); 0.5 - gaze shifting time
    m = 40               # the number of classes/frequencies
    p = df / 100         # the accuracy of target identification
    # if p == 1.0:
    #     p = 0.99
    return (np.log2(m) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (m - 1))) * 60 / t

def plot_trial(results):
    """
    plot the passed matrix as heatmap/imshow
    
    Parameters
    ----------
    results: array, shape(n_freq, n_block*n_subject)
    data that contains the results of classification
    
    Return
    -------
    fig, ax
    """
    # n_freq = np.shape(results)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Subjects')
    ax.set_xticks(np.arange(0, 210, 6))
    ax.set_xticklabels(np.arange(1, 36))
    set_size(fig, 8, 2.5)
    fig.tight_layout()
    return fig, ax

# Lambdas
acc = lambda mat: np.sum(mat[mat > 0]) / (np.size(mat) - np.size(mat[mat == -1])) * 100
standardize = lambda mat: (mat - np.mean(mat, axis=1)[:, None]) / np.std(mat, axis=1)[:, None]

# Pool of functions used for online SSVEP-based BCI decoding
def butter_highpass(lowcut, fs, order=5):
    '''
    Create a Butterworth high pass filter

    Parameters
    ==========
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.

    Returns
    ======
    b, a: ndarray, ndarray
    Numerator (b) and denominator (a) polynomials of the IIR filter. 
    '''
    nyq = 0.5 * fs
    low =  lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Create a Butterworth band pass filter

    Parameters
    ==========
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.

    Returns
    ======
    b, a: ndarray, ndarray
    Numerator (b) and denominator (a) polynomials of the IIR filter. 
    '''
    nyq = 0.5 * fs
    low =  lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandstop(lowpass, highpass, fs, order=2):
    '''
    Create a Butterworth band stop filter

    Parameters
    ==========
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.

    Returns
    ======
    b, a: ndarray, ndarray
    Numerator (b) and denominator (a) polynomials of the IIR filter. 
    '''
    nyq = 0.5 * fs
    low =  lowpass / nyq
    high = highpass / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def butter_highpass_filter(data, lowcut, fs, order=5, notch=0):
    '''
    Filter data using a highpass filter.

    Parameters
    ==========
    data: ndarray, data to be filtered.
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.
    notch: int 0 or 1, whether or not applying a notch filter.
    Returns
    ======
    y: ndarray, filtered data. 
    '''
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    if notch > 0:
        b, a = butter_bandstop(notch-1, notch+1, fs)
        y = filtfilt(b, a, y)
        
        notch = notch * 2 #Second-harmonic generation
        b, a = butter_bandstop(notch-1, notch+1, fs)
        y = filtfilt(b, a, y)
        
        notch = notch * 3 #Third-harmonic generation
        b, a = butter_bandstop(notch-1, notch+1, fs)
        y = filtfilt(b, a, y)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, notch=0):
    '''
    Filter data using a bandpass filter.

    Parameters
    ==========
    data: ndarray, data to be filtered.
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.
    notch: int 0 or 1, whether or not applying a notch filter.
    Returns
    ======
    y: ndarray, filtered data. 
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    if notch > 0:
        if notch < fs/2:
            b, a = butter_bandstop(notch-1, notch+1, fs)
            y = filtfilt(b, a, y)
        
        notch_2 = notch *2 #Second-harmonic generation
        if notch_2 < fs/2:
            b, a = butter_bandstop(notch_2-1, notch_2+1, fs)
            y = filtfilt(b, a, y)
        
        notch_3 = notch * 3 #Third-harmonic generation
        if notch_3 < fs/2:
            b, a = butter_bandstop(notch_3-1, notch_3+1, fs)
            y = filtfilt(b, a, y)
    return y


def nextpow2(i):
    """
    Gets the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n

def update_buffer(data, data_new):
    """
    Updates the buffer with new data.    
    """
    buffer = np.concatenate((data, data_new), axis=0)
    buffer = buffer[data_new.shape[0]:, :]
    
    return buffer  
    
def SinCos(list_freqs, harmonics, t_frame):
    cca_Y_FS = []
    for f in list_freqs:
        cca_Y = []
        for i in range(harmonics):
            cca_Y.append( np.sin(2*(i+1)*np.pi*f*t_frame) )
            cca_Y.append( np.cos(2*(i+1)*np.pi*f*t_frame) )
        
        cca_Y_FS.append( np.asarray(cca_Y).T )
        
    return cca_Y_FS

class ssvep_cca(object):
    '''
    canonical correlation analysis (CCA)-based method
    '''
    def __init__(self, list_freqs, harmonics=2, fs=250, dim=2, winSize=250*3):
        self.list_freqs = list_freqs
        self.harmonics = harmonics
        self.fs = fs
        self.dim = dim # O1; O2; 
        self.winSize = winSize
        
        self.t_frame = np.arange(self.winSize) / self.fs
        self.cca_Y_FS = SinCos(self.list_freqs, self.harmonics, self.t_frame)
        
    def correlation(self, Data):
        cca_X = np.asarray(Data).T
        assert cca_X.shape[0] == self.winSize, "winSize not same"
        assert cca_X.shape[1] == self.dim, "EEG Channel Num not same"
        
        r_f = []
        for i in range(len(self.list_freqs)):
            cca_Y = self.cca_Y_FS[i]
            n_components = 1
            cca = CCA(n_components)
            cca.fit(cca_X, cca_Y)
            cc_corr = np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[0, 1]            
            r_f.append(cc_corr)
            
        return r_f
            
    def predict(self, Data):
        correlation = self.correlation(Data)
        return correlation
   
class ssvep_fbcca(object):
    '''
    filter bank canonical correlation analysis (FBCCA)-based method
    '''
    def __init__(self, fbank, list_freqs, harmonics=2, fs=250, dim=2, winSize=250*3):
        self.fbank = fbank
        self.list_freqs = list_freqs
        self.fs = fs
        self.dim = dim    # O1; O2; 
        self.winSize = winSize
        
        self._cca = ssvep_cca(list_freqs=list_freqs, harmonics=harmonics, fs=fs, dim=self.dim, winSize=winSize)
        
        self.subBands_sos = []
        for i in range(len(fbank)):
            sos = signal.cheby1(10, 1, fbank[i], 'bandpass', fs=fs, output='sos')
            self.subBands_sos.append(sos)
        
    def predict(self, Data):
        Data = np.asarray(Data)
        assert Data.shape[0] == self.dim, "EEG Channel Num not same"
        assert Data.shape[1] == self.winSize, "winSize not same"
        
        
        p_N_K = np.zeros([len(self.subBands_sos), len(self.list_freqs)])
        # weight = np.arange(len(self.subBands_sos),0,-1)
        #n^(−a) + b, n ∈ [1 N]
        a = 1.25
        b = 0.25
        weight = np.power(np.arange(len(self.subBands_sos))+1,-a) + b
        weight = weight / np.sum(weight)
        for i in range( len(self.subBands_sos) ):
            sos = self.subBands_sos[i]
            Data_SubBand = []
            for j in range(Data.shape[0]):
                
                filtered = signal.sosfiltfilt(sos, Data[j,:])
                Data_SubBand.append(filtered)
            
            p_N_K[i,:] = weight[i] * np.asarray(self._cca.correlation(Data_SubBand))**2
        
        p_K = np.sum(p_N_K, axis=0)
        # print(p_K)
        # if np.max(p_K) > 0.5:
        #     f_pred = self.list_freqs[np.argmax(p_K)]
        # else:
        #     f_pred = -1
        
        # return f_pred
        return p_K
