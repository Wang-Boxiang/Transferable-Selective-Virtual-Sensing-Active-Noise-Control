import numpy as np
from Fixed_Filter_noise_cancellation import Fxied_filters
from ONED_CNN_PRE import OneD_CNN_Predictor

#-------------------------------------------------------------
# Function: Is multiple length of samples
#-------------------------------------------------------------
def Casting_multiple_time_length_of_primary_noise(primary_noise,fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len]

def Casting_single_time_length_of_training_noise(filter_training_noise,fs):
    assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
    print(filter_training_noise[:,:,:fs].shape)
    return filter_training_noise[:,:,:fs]
    
#-------------------------------------------------------------
# Class :   Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor(OneD_CNN_Predictor):
    
    def __init__(self,MODEL_PATH,device,filter_training_noise,fs):
        
        OneD_CNN_Predictor.__init__(self,MODEL_PATH,device)
        # Checking the length of the training noise 
        assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
        assert filter_training_noise.shape[2]%fs == 0, 'The length of the training noise sample should be 1 second!'
        # Detach the information of the the training nosie 
        self.frequency_charactors_tensor = filter_training_noise 
        self.len_of_class                = filter_training_noise.shape[0]
        self.fs                          = fs 
    
    def predic_ID(self, noise):
        similarity_rato = []
        for ii in range(self.len_of_class):
            similarity_rato.append(self.cosSimilarity_minmax(noise, self.frequency_charactors_tensor[ii]))
        index = np.argmax(similarity_rato)
        return index
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise.
        assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        # Computing how many seconds the primary noise containt.
        Time_len              = int(primary_noise.shape[1]/self.fs) 
        print(Time_len)
        print(f'The primary nosie has {Time_len} seconds !!!')
        # Bulding the matric of the primary noise [times x 1 x fs ]
        primary_noise_vectors = primary_noise.reshape(Time_len,self.fs).unsqueeze(1)
        
        
        # Implementing the noise classification for each frame whose length is 1 second. 
        ID_vector = []
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        
        return ID_vector

def Control_filter_selection(MODEL_PTH_type=0, fs = 16000, Primary_noise=None, Filter_mat_name=None):
    """This function is used to chose the index of the control filter for the primary nosie. 
    """
    Fxied_control_filter = Fxied_filters(MATFILE_PATH=Filter_mat_name, fs=fs)
    
    Charactors=Casting_single_time_length_of_training_noise(Fxied_control_filter.Charactors,fs=fs)
    
    # cnn modle path 
    if MODEL_PTH_type == 0:
        MODEL_PTH = 'CNN.pth'
        
    device    = "cpu"
    
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PTH
                                                                             ,device= device
                                                                             ,filter_training_noise=Charactors
                                                                             ,fs=fs)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise,fs=fs)
    
    Id_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(Primary_noise)
    
    return Id_vector
    