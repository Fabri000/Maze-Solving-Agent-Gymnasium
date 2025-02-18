import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import Conv2d, ConvTranspose2d, ReLU,  MaxPool2d, Sigmoid


class CAE(nn.Module):

    def __init__(self,in_channels:int, h_channels:int):
        """
        Args:
            in_channels (int): channels of input for the first layer
            h_channels (int): channels of hidden layer
        """
        super(CAE,self).__init__()

        self.encoder = nn.Sequential(
            Conv2d(in_channels,h_channels,kernel_size=2,stride=2),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(h_channels, h_channels // 2,kernel_size=3,stride=1,padding=1),
            ReLU(),
        )

        self.decoder = nn.Sequential(
            ConvTranspose2d(h_channels // 2, h_channels,kernel_size=3,stride=2),
            ReLU(),
            ConvTranspose2d(h_channels,in_channels,kernel_size=3,stride=2,padding=1,output_padding=2),
            Sigmoid()
        )


    def forward(self,x):
        """
        Args:
            x (torch.Tensor): input tensor for the network
        """
        feature = self.encoder(x)
        return self.decoder(feature)