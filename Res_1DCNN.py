import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Adjusting the residual in case of channel mismatch
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class OneD_CNN_res(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer 
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels= 10, 
            kernel_size = 3, 
            stride      = 1 
        )
        
        # Residual Block replacing the second convolutional layer
        self.residual_block = ResidualBlock(10, 20, stride=1)
        
        # Max-pool 
        self.maxpool = nn.MaxPool1d(
            kernel_size = 512,
            stride      = 512 
        )

        # Normalized layer 
        self.normalized = nn.BatchNorm1d(
            num_features = 620
        )

        # Linear layer 1
        self.linear1 = nn.Linear(
            in_features  = 620, 
            out_features = 15
        )

        # Activation function 
        self.active = nn.Tanh() 
        
        # Linear layer 2
        self.linear2 = nn.Linear(
            in_features  = 15, 
            out_features = 15 
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.residual_block(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.normalized(x)
        x = self.linear1(x)
        x = self.active(x)
        x = self.linear2(x)
        return x


class OneD_CNN_res_pre(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer 
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels= 10, 
            kernel_size = 3, 
            stride      = 1 
        )
        
        # Residual Block replacing the second convolutional layer
        self.residual_block = ResidualBlock(10, 20, stride=1)
        
        # Max-pool 
        self.maxpool = nn.MaxPool1d(
            kernel_size = 512,
            stride      = 512 
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.residual_block(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        
        return x