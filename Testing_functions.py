from MyLoader import MyNoiseDataset, minmaxscaler, create_data_loader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from Bcolors import bcolors
import matplotlib.pyplot as plt
import torch 
from CNN_Pre import OneD_CNN_Predictor
import numpy as np
import scipy.io as sio
import scipy.signal as signal

#-----------------------------------------------------------------------------------
# Class type  : Fxied_filters 
# Description : Loading pre-trained filter from .mat file
#-----------------------------------------------------------------------------------
class Fxied_filters():

    def __init__(self, MATFILE_PATH, fs):
        mat_contents    = sio.loadmat(MATFILE_PATH)
        self.Wc_vectors = mat_contents['Wc_v']
        self.len        = len(self.Wc_vectors)
        self.filterlen  = self.Wc_vectors.shape[1]
        self.Charactors = torch.zeros([self.len, 1, fs], dtype=torch.float)
        self.fs         = fs 

        for ii in range(self.len):
            self.Charactors[ii] = self.frequency_charactors_tensor(ii)
    
    def cancellator(self, classID, Fx, Dir):
        Yt = signal.lfilter(self.Wc_vectors[classID,:], 1, Fx)
        Er = Dir - Yt 
        return Er 

    def frequency_charactors_tensor(self, classID):
        fs    = self.fs
        N     = fs + self.filterlen 
        xin   = np.random.randn(N)
        yout  = signal.lfilter(self.Wc_vectors[classID,:],1,xin)
        yout  = yout[self.filterlen:]
        # Standarlize 
        yout = minmaxscaler(yout)
        # return a tensor of [1 x sample rate]
        return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

class Filter_ID_predictor(OneD_CNN_Predictor,Fxied_filters):

    def __init__(self, MODEL_PATH, MATFILE_PATH, fs,device):
        OneD_CNN_Predictor.__init__(self, MODEL_PATH,device)
        Fxied_filters.__init__(self, MATFILE_PATH, fs)
    
    def predic_ID(self, noise_1):
        similarity_rato = []
        for ii in range(self.len):
            similarity_rato.append(self.cosSimilarity(noise_1, self.Charactors[ii]))
        index = np.argmax(similarity_rato)
        return index

#----------------------------------------------------------------
def tst_accuracy_of_model(tst_data_loder, model, return_index =None): 
    if return_index == None :
        accuracy_vec = []
        i            = 0 
        for input, target in tst_data_loder:
            i += 1
            batch_acc = 0
            for signal_1d, target_1d in zip(input, target):
                batch_acc +=(model.predic_ID(signal_1d)==target_1d.numpy())
            acc = batch_acc/len(target)
            print(f"----------------------------------------")
            print(f"The {i}-th iteration's accuracy is {acc}")
            accuracy_vec.append(acc)
        return accuracy_vec, sum(accuracy_vec)/len(accuracy_vec)
    else:
        i             = 0 
        predict_index = []
        target_index  = []
        accuracy_vec  = []
        for input, target in tst_data_loder:
            i += 1
            batch_acc = 0
            for signal_1d, target_1d in zip(input, target):
                pre, tar = model.predic_ID(signal_1d), target_1d.numpy()
                predict_index.append(pre)
                target_index.append(tar)
                batch_acc += (pre == tar)
            acc = batch_acc/len(target)
            print(f"----------------------------------------")
            print(f"The {i}-th iteration's accuracy is {acc}")
            accuracy_vec.append(acc)
        return accuracy_vec, sum(accuracy_vec)/len(accuracy_vec), predict_index, target_index

#----------------------------------------------------------------
def Testing_model_accuracy(MODEL_PATH, MATFILE_PATH, VALIDATTION_FILE, Report=None, Class_Num=5):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    #D
    BATCH_SIZE       = 100
    fs               = 16000
    sheet            = "Index.csv"
    CNN_classfier  = Filter_ID_predictor(MODEL_PATH, MATFILE_PATH, fs, device)
    valid_data     = MyNoiseDataset(VALIDATTION_FILE,sheet)
    valid_dataloader = create_data_loader(valid_data,BATCH_SIZE)
    if Report == None: 
        _, average_acc = tst_accuracy_of_model(valid_dataloader,CNN_classfier)
        print(f"The average accuracy is {average_acc}")
    else:
        _, average_acc, y_pred, y_true = tst_accuracy_of_model(valid_dataloader,CNN_classfier,return_index=True)
        print(bcolors.OKCYAN + f"The average accuracy is {average_acc}" + bcolors.ENDC)
        target_names = []
        for jj in range(Class_Num):
            target_names.append(f'class {jj}')
        #target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4']
        print(bcolors.RED + '<<====================Classification report=========================>>' + bcolors.ENDC)
        print(classification_report(y_true, y_pred, target_names=target_names))
        print(bcolors.RED +'<<==========================End=====================================>>' + bcolors.ENDC)
        cm = confusion_matrix(y_true, y_pred)
        disp  = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
        disp.plot()
        plt.show()
    return average_acc
