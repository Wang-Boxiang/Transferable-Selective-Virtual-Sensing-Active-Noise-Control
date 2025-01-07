import torch 
from torch import nn
from Res_1DCNN import OneD_CNN_res_pre
from MyLoader import minmaxscaler

# Loading the weights to model from pre-trained coefficients 
def load_weigth_for_model(model, pretrained_path, device):
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(pretrained_path,map_location= device)

    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    
    model.load_state_dict(model_dict)

#------------------------------------------------------------------------
class   OneD_CNN_Predictor():

    def __init__(self, MODEL_PATH,device):
        self.cnn = OneD_CNN_res_pre().to(device)
        load_weigth_for_model(self.cnn,MODEL_PATH,device)
        self.cnn.eval()
        self.cos = nn.CosineSimilarity(dim=1).to(device)
        self.device = device
    
    def cosSimilarity(self, signal_1, signal_2):
        signal1, signal2 = signal_1.unsqueeze(0), signal_2.unsqueeze(0)
        similarity = self.cos(self.cnn(signal1.to(self.device)), self.cnn(signal2.to(self.device)))
        return similarity.cpu().item() 
    
    def cosSimilarity_minmax(self, signal_1, signal_2):
        signal1, signal2 = minmaxscaler(signal_1).unsqueeze(0), minmaxscaler(signal_2).unsqueeze(0)
        similarity = self.cos(self.cnn(signal1.to(self.device)), self.cnn(signal2.to(self.device)))
        return similarity.cpu().item() 

