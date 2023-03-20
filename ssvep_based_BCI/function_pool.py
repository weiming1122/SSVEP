"""
Pool of functions used for SSVEP-based BCI with CCA algorithm
"""

# Import necessary libralies
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt
import numpy as np            
import warnings
warnings.filterwarnings('ignore')
from scipy import signal

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
    Canonical correlation analysis (CCA)-based method
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
    Filter bank canonical correlation analysis (FBCCA)-based method
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

def apply_ext_cca(X, Y, X_Train):
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