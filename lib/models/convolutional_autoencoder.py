import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import Conv2d


class CAE(nn.Module):

    def __init__(self,in_channels:int, h_channels:int):
        """
        Args:
            in_channels (int): channels of input for the first layer
            h_channels (int): channels of hidden layer
        """
        super(CAE,self).__init__()

        self.encoder =nn.Sequential(
            Conv2d(in_channels,h_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2,2),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(h_channels, in_channels, kernel_size=2, stride=2, output_padding=1),  # upsample
            nn.Sigmoid()
        )


    def forward(self,x):
        """
        Args:
            x (torch.Tensor): input tensor for the network
        """
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out 