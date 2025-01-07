import numpy as np
from scipy.io import savemat
import scipy.signal as signal

#-----------------------------------------------------------------------------------
# Class type  : Filter design
# Description : Design filter group by the configure vector 
#-----------------------------------------------------------------------------------
class Filter_designer():
    
    def __init__(self, filter_len, F_vector, fs):

        self.filter_len = filter_len
        self.filter_num = len(F_vector)
        self.wc         = np.zeros((self.filter_num, self.filter_len))
        for i in range(self.filter_num):
            self.wc[i,:] = signal.firwin(self.filter_len, F_vector[i], pass_zero='bandpass', window ='hamming',fs=fs) 
        
    
    def __save_mat__(self, FILE_NAME_PATH):
        mdict= {'Wc_v': self.wc}
        savemat(FILE_NAME_PATH, mdict)
#-----------------------------------------------------------------------------------
# Function  :   Broadband Filter design by given freqency bands 
#-----------------------------------------------------------------------------------
def Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename, F_bands, fs):
    Filters = Filter_designer(filter_len=1024, F_vector= F_bands, fs=fs)
    Filters.__save_mat__(MAT_filename)
    print(Filters.filter_num)
    


