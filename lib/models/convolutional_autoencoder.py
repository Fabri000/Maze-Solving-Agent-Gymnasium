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
            Conv2d(in_channels,h_channels,kernel_size=3,stride=1),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(h_channels, h_channels,kernel_size=3,stride=1),
            ReLU(),
        )

        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(h_channels, h_channels, kernel_size=3, stride=1, output_padding=0),  # Inverting Conv2
            nn.ReLU(),
            nn.ConvTranspose2d(h_channels, in_channels, kernel_size=2, stride=2, output_padding=0),  # Inverting MaxPool
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, output_padding=1),  # Inverting Conv1
            nn.ReLU()
        )

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): input tensor for the network
        """
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out 