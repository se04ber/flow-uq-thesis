"""model_aleatoric

This script contains the modified PyTorch models that add aleatoric uncertainty
to the original AE-RNN architecture. The AE is modified to output both mean
predictions and uncertainty estimates.

"""
import torch.nn as nn
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DoubleConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


def tensor_FIFO_pipe(tensor, x, device):
    """The tensor_FIFO_pipe function acts as a first-in-first-out updater and
    is required for the hybrid models. It takes a tensor 'tensor' containing
    information from previous timesteps and concatenates new information 'x'
    to the front fo 'tensor'. In a FIFO manner, it returns all but the last
    element of 'tensor'.

    Args:
        tensor:
          Object of PyTorch-type tensor containing information from previous
          timesteps.
        x:
          Object of PyTorch-type tensor containing information from current
          timestep.
    Return:
        result:
          Object of PyTorch-type tensor upated in a FIFO manner.
    """
    result = torch.cat((tensor[1:].to(device), x.to(device)))
    return result


class AE_aleatoric(nn.Module):
    """The AE_aleatoric class is a modified version of the original AE that
    outputs both mean predictions and aleatoric uncertainty estimates.
    
    The model outputs two channels: one for the mean prediction and one for
    the log variance (uncertainty estimate).

    Attributes:
        in_channels:
          Object of integer type describing number of channels in the input data.
        out_channels:
          Object of integer type describing number of channels in the output data.
        features:
          Object of type List containing integers that correspond to the number
          of kernels applied per convolutional
        activation:
          Object of PyTorch type torch.nn containing an activation function
    """

    def __init__(self, device, in_channels=1, out_channels=2, features=[4, 8, 16], activation=nn.ReLU(inplace=True)):
        super(AE_aleatoric, self).__init__()
        self.device = device

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Make helper layers dynamic based on features
        last_feature = features[-1]  # This will be the last feature size
        self.helper_down = nn.Conv3d(
            in_channels=last_feature, out_channels=last_feature, kernel_size=2, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.helper_up_1 = nn.ConvTranspose3d(
            in_channels=last_feature*2, out_channels=last_feature*2, kernel_size=2, stride=1, padding=0, bias=False)
        self.helper_up_2 = nn.Conv3d(
            in_channels=features[0], out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Down part of AE
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, activation))
            in_channels = feature

        # Up part of AE
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature, feature, activation))

        # This is the "deepest" part.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, activation)
        print('Model initialized: Autoencoder with Aleatoric Uncertainty.')

    def forward(self, x, y=0, skip_connections=0):
        """The forward method acts as a quasi-overloaded method in that depending
        on the passed flag 'y', the forward method begins and returns different
        values. This is necessary to later feed time-series latent space predictions
        back into the model.

        Args:
            x:
              Object of PyTorch-type tensor containing the information of a timestep.
            y:
              Object of string type acting as a flag to choose desired forward method.
            skip_connections:
              Object of type list containing objects of PyTorch-type tensor that
              contain the U-Net unique skip_connections for later concatenation.
        Return:
            result:
              Object of PyTorch-type tensor returning the autoencoded result with
              uncertainty estimates.
        """
        if y == 0 or y == 'get_bottleneck':
            # The following for-loop describes the entire (left) contracting side,
            for down in self.downs:
                x = down(x)
                x = self.pool(x)

            # This is the bottleneck
            x = self.helper_down(x)
            x = self.activation(x)
            x = self.bottleneck(x)
            x = self.activation(x)

            if y == 'get_bottleneck':
                return x, skip_connections

            x = self.helper_up_1(x)
            x = self.activation(x)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)

            x = self.helper_up_2(x)

            return x

        if y == 'get_MD_output':
            x = self.helper_up_1(x)
            x = self.activation(x)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)

            x = self.helper_up_2(x)
            return x





class RNN(nn.Module):
    """The RNN class implements a recurrent neural network for temporal
    predictions in the latent space.

    Attributes:
        input_size:
          Object of integer type describing the size of the input.
        hidden_size:
          Object of integer type describing the size of the hidden state.
        seq_size:
          Object of integer type describing the sequence length.
        num_layers:
          Object of integer type describing the number of RNN layers.
        device:
          Object of torch.device type describing the device to run on.
    """

    def __init__(self, input_size, hidden_size, seq_size, num_layers, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Set initial hidden states(for RNN, GRU, LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class Hybrid_MD_RNN_AE_aleatoric(nn.Module):
    """The Hybrid_MD_RNN_AE_aleatoric class implements a hybrid model that
    combines the aleatoric uncertainty-aware autoencoder with RNN for
    temporal predictions.

    Attributes:
        device:
          Object of torch.device type describing the device to run on.
        AE_Model_x:
          Object of AE_aleatoric type for x-component.
        AE_Model_y:
          Object of AE_aleatoric type for y-component.
        AE_Model_z:
          Object of AE_aleatoric type for z-component.
        RNN_Model_x:
          Object of RNN type for x-component.
        RNN_Model_y:
          Object of RNN type for y-component.
        RNN_Model_z:
          Object of RNN type for z-component.
        seq_length:
          Object of integer type describing the sequence length.
    """

    def __init__(self, device, AE_Model_x, AE_Model_y, AE_Model_z, RNN_Model_x, RNN_Model_y, RNN_Model_z, seq_length=15):
        super(Hybrid_MD_RNN_AE_aleatoric, self).__init__()
        self.device = device
        self.AE_Model_x = AE_Model_x
        self.AE_Model_y = AE_Model_y
        self.AE_Model_z = AE_Model_z
        self.RNN_Model_x = RNN_Model_x
        self.RNN_Model_y = RNN_Model_y
        self.RNN_Model_z = RNN_Model_z
        self.seq_length = seq_length

    def forward(self, x):
        # print('Shape [pre AE] u_x: ', x[:, 0, :, :, :].shape)
        # print('Shape [pre AE] u_y: ', x[:, 1, :, :, :].shape)
        # print('Shape [pre AE] u_z: ', x[:, 2, :, :, :].shape)

        # Extract bottleneck representations for each component
        bottleneck_x, skip_connections_x = self.AE_Model_x(x[:, 0, :, :, :], y='get_bottleneck')
        bottleneck_y, skip_connections_y = self.AE_Model_y(x[:, 1, :, :, :], y='get_bottleneck')
        bottleneck_z, skip_connections_z = self.AE_Model_z(x[:, 2, :, :, :], y='get_bottleneck')

        # Flatten bottleneck representations for RNN
        batch_size = bottleneck_x.shape[0]
        bottleneck_x_flat = bottleneck_x.view(batch_size, -1)
        bottleneck_y_flat = bottleneck_y.view(batch_size, -1)
        bottleneck_z_flat = bottleneck_z.view(batch_size, -1)

        # Create sequence for RNN (repeat the same bottleneck for seq_length times)
        seq_x = bottleneck_x_flat.unsqueeze(1).repeat(1, self.seq_length, 1)
        seq_y = bottleneck_y_flat.unsqueeze(1).repeat(1, self.seq_length, 1)
        seq_z = bottleneck_z_flat.unsqueeze(1).repeat(1, self.seq_length, 1)

        # RNN predictions
        rnn_out_x = self.RNN_Model_x(seq_x)
        rnn_out_y = self.RNN_Model_y(seq_y)
        rnn_out_z = self.RNN_Model_z(seq_z)

        # Reshape RNN outputs back to bottleneck shape
        rnn_out_x = rnn_out_x.view(bottleneck_x.shape)
        rnn_out_y = rnn_out_y.view(bottleneck_y.shape)
        rnn_out_z = rnn_out_z.view(bottleneck_z.shape)

        # Decode back to spatial domain with uncertainty
        output_x = self.AE_Model_x(rnn_out_x, y='get_MD_output')
        output_y = self.AE_Model_y(rnn_out_y, y='get_MD_output')
        output_z = self.AE_Model_z(rnn_out_z, y='get_MD_output')

        # Stack outputs along channel dimension
        # Each output has 2 channels: [mean, log_var]
        output = torch.stack([output_x, output_y, output_z], dim=1)
        
        return output


def resetPipeline(model):
    """The resetPipeline function resets the model pipeline by setting all
    gradients to zero and setting the model to evaluation mode.

    Args:
        model:
          Object of PyTorch Module class, i.e. the model to be reset.

    Returns:
        NONE:
          This function does not have a return value.
    """
    model.zero_grad()
    model.eval()


def test_forward_overloading():
    """The test_forward_overloading function tests the forward method
    overloading functionality of the AE model.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Testing forward method overloading...')
    _model = AE_aleatoric(device=device, in_channels=1, out_channels=2, features=[4, 8, 16])
    _x = torch.randn(1, 1, 24, 24, 24)
    _y = _model(_x)
    print('Standard forward pass shape:', _y.shape)
    _bottleneck, _skip = _model(_x, y='get_bottleneck')
    print('Bottleneck shape:', _bottleneck.shape)
    _md_output = _model(_bottleneck, y='get_MD_output')
    print('MD output shape:', _md_output.shape)
    print('Forward method overloading test completed.')


if __name__ == "__main__":
    test_forward_overloading() 
