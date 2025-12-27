import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            activation,
            nn.Dropout3d(p=0.1), #0.001 #dropout  layer1 #0.1
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            activation,
            nn.Dropout3d(p=0.1)  #0.001 #dropout layer2  #0.1
        )
    def forward(self, x):
        return self.conv(x)


class AE_dropout(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=2, features=[4, 8, 16], activation=nn.ReLU(inplace=True)): #[2,4]
        super(AE_dropout, self).__init__()
        self.device = device
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, activation))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature, feature, activation))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, activation)
        self.final_conv = nn.Conv3d(features[0], 2, 3, 1, 1, bias=False)
        print('Model initialized: Lightweight AE (with dropout).')
    def forward(self, x, y=0, skip_connections=0):
        if y == 0 or y == 'get_bottleneck':
            for down in self.downs:
                x = down(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            x = self.activation(x)
            if y == 'get_bottleneck':
                return x, skip_connections
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)
            x = self.final_conv(x)
            lower = x[:, 0:1, :, :, :]
            upper = x[:, 1:2, :, :, :]
            out = torch.cat([lower, upper], dim=1)
            return out
        if y == 'get_MD_output':
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)
            x = self.final_conv(x)
            lower = x[:, 0:1, :, :, :]
            upper = x[:, 1:2, :, :, :]
            out = torch.cat([lower, upper], dim=1)
            return out 
