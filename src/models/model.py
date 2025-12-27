import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


def tensor_FIFO_pipe(tensor, x, device):
    result = torch.cat((tensor[1:].to(device), x.to(device)))
    return result


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            activation,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class AE(nn.Module):
    """The AE class aims at implementing a strictly convolutional autoencoder
    comparable to U-Net (Ronneberger et al., 2015).

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

    def __init__(self, device, in_channels=3, out_channels=3, features=[4, 6, 8, 10], activation=nn.ReLU(inplace=True)):
        super(AE, self).__init__()
        self.device = device

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.helper_down = nn.Conv3d(
            in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.helper_up_1 = nn.ConvTranspose3d(
            in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0, bias=False)
        self.helper_up_2 = nn.Conv3d(
            in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

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
              Object of PyTorch-type tensor returning the autoencoded result.
        """
        if y == 0 or y == 'get_bottleneck':
            # print(x.shape)
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


class AE_u_i(nn.Module):
    """The AE_u_i class aims at implementing a single channel version of the
    aforementioned AE class.

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

    def __init__(self, device, in_channels=1, out_channels=1, features=[4, 6, 8, 10], activation=nn.ReLU(inplace=True)):
        super(AE_u_i, self).__init__()
        self.device = device

        self.activation = nn.ReLU()

        # Generic module placeholders
        self.ups_i = nn.ModuleList()

        # Generic module placeholders
        self.downs_i = nn.ModuleList()

        # Generic 3d maxpool
        self.pool_i = nn.MaxPool3d(kernel_size=2, stride=2)

        self.helper_down_i = nn.Conv3d(in_channels=16, out_channels=16,
                                       kernel_size=2, stride=1, padding=0, bias=False)

        self.helper_up_1_i = nn.ConvTranspose3d(in_channels=32, out_channels=32,
                                                kernel_size=2, stride=1, padding=0, bias=False)

        self.helper_up_2_i = nn.Conv3d(in_channels=4, out_channels=1,
                                       kernel_size=3, stride=1, padding=1, bias=False)
        # Down part of AE
        for feature in features:
            self.downs_i.append(DoubleConv(in_channels, feature, activation))
            in_channels = feature

        # Up part of AE
        for feature in reversed(features):
            self.ups_i.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups_i.append(DoubleConv(feature, feature, activation))

        # This is the "deepest" part.
        self.bottleneck_i = DoubleConv(
            features[-1], features[-1]*2, activation)
        print('Model initialized: Autoencoder.')

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
        Return:
            result:
              Object of PyTorch-type tensor returning the autoencoded result.
        """

        if y == 0 or y == 'get_bottleneck':
            # The following for-loop describes the entire (left) contracting side,
            t, h, d, w = x.shape
            x.to(device)
            u_i = torch.reshape(x, (t, 1, h, d, w)).to(device)
            print('Shape of u_i: ', u_i.shape)
            for down_i in self.downs_i:
                u_i = down_i(u_i)
                print('Double Conv Down: ', u_i.shape)
                u_i = self.pool_i(u_i)
                print('Pooling Down: ', u_i.shape)

            # This is the bottleneck
            u_i = self.helper_down_i(u_i)
            u_i = self.activation(u_i)
            print('Helper Down Shape: ', u_i.shape)
            u_i = self.bottleneck_i(u_i)
            u_i = self.activation(u_i)
            print('Bottleneck Shape: ', u_i.shape)

            if y == 'get_bottleneck':
                return u_i

            u_i = self.helper_up_1_i(u_i)
            u_i = self.activation(u_i)
            print('Helper Up [1] Shape: ', u_i.shape)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups_i), 2):
                u_i = self.ups_i[idx](u_i)
                print('DeConv Shape: ', u_i.shape)
                u_i = self.ups_i[idx+1](u_i)
                print('DoubleConv Up Shape: ', u_i.shape)

            u_i = self.helper_up_2_i(u_i)
            print('Helper Up [2] Shape: ', u_i.shape)

            return u_i

        if y == 'get_MD_output':
            u_i = self.helper_up_1_i(x)
            u_i = self.activation(u_i)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups_i), 2):
                u_i = self.ups_i[idx](u_i)
                u_i = self.ups_i[idx+1](u_i)

            u_i = self.helper_up_2_i(u_i)
            return u_i


class RNN(nn.Module):
    """The RNN class aims at implementing an adapted RNN to be used in the hybrid
    model for time-series prediction of unrolled latent spaces.

    input.shape  = (batch_size, num_seq, input_size)
    output.shape = (batch_size, 1, input_size)

    Attributes:
        input_size:
          Object of integer type describing the number of features in a single
          element of the time-series sequence.
        hidden_size:
          Object of integer type describing the number of features in the hidden
          state, i.e. the number of features in the output.
        seq_size:
          Object of integer type describing the number of elements in the
          time-series sequence.
        num_layers:
          Object of integer type describing the number of stacked RNN units.
    """

    def __init__(self, input_size, hidden_size, seq_size, num_layers, device):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.num_layers = num_layers
        self.device = device
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        print('Model initialized: RNN.')

    def forward(self, x):
        # Set initial hidden states(for RNN, GRU, LSTM)
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.rnn(x, h0)

        # Take only the last timestep output (batch_size, hidden_size)
        out = out[:, -1, :]

        # Apply linear regressor to the last time step
        out = self.fc(out)
        return out


class Hybrid_MD_RNN_AE_u_i(nn.Module):
    """The Hybrid_MD_RNN_AE class aims at implementing the strictly convolutional
    autoencoder and RNN based hybrid model for time-series prediction of MD
    velocity distributions.

    input.shape  = (batch_size = 1, 3, 24, 24, 24)
    output.shape = (batch_size = 1, 3, 24, 24, 24)

    Attributes:
        AE_Model:
          Object of type torch.nn.Module containing the trained AE_u_i model
        RNN_Model:
          Object of type torch.nn.Module containing the trained RNN model
        seq_length:
          Object of integer type describing the number of elements to be
          considered in the time-series sequence of latent spaces.
    """

    def __init__(self, device, AE_Model_x, AE_Model_y, AE_Model_z, RNN_Model_x, RNN_Model_y, RNN_Model_z, seq_length=15):
        super(Hybrid_MD_RNN_AE_u_i, self).__init__()
        self.device = device
        self.AE_x = AE_Model_x.eval().to(self.device)
        self.AE_y = AE_Model_y.eval().to(self.device)
        self.AE_z = AE_Model_z.eval().to(self.device)
        self.rnn_x = RNN_Model_x.eval().to(self.device)
        self.rnn_y = RNN_Model_y.eval().to(self.device)
        self.rnn_z = RNN_Model_z.eval().to(self.device)
        self.seq_length = seq_length
        self.sequence_x = torch.zeros(self.seq_length, 256).to(self.device)
        self.sequence_y = torch.zeros(self.seq_length, 256).to(self.device)
        self.sequence_z = torch.zeros(self.seq_length, 256).to(self.device)
        print('Model initialized: Hybrid_MD_RNN_AE_u_i')

    def forward(self, x):
        # print('Shape [pre AE] u_x: ', x[:, 0, :, :, :].shape)

        u_x = self.AE_x(x[:, 0, :, :, :], y='get_bottleneck').to(self.device)
        u_y = self.AE_y(x[:, 1, :, :, :], y='get_bottleneck').to(self.device)
        u_z = self.AE_z(x[:, 2, :, :, :], y='get_bottleneck').to(self.device)

        u_x_shape = u_x.shape
        # print('Shape [latentspace] u_x: ', u_x.shape)
        ()
        self.sequence_x = tensor_FIFO_pipe(
            tensor=self.sequence_x,
            x=torch.reshape(u_x, (1, 256)),
            device=self.device).to(self.device)
        # print('Shape [sequence_x]: ', self.sequence_x.shape)

        u_y_shape = u_y.shape
        self.sequence_y = tensor_FIFO_pipe(
            tensor=self.sequence_y,
            x=torch.reshape(u_y, (1, 256)),
            device=self.device).to(self.device)

        u_z_shape = u_z.shape
        self.sequence_z = tensor_FIFO_pipe(
            tensor=self.sequence_z,
            x=torch.reshape(u_z, (1, 256)),
            device=self.device).to(self.device)

        interim_x = torch.reshape(
            self.sequence_x, (1, self.seq_length, 256)).to(self.device)
        interim_y = torch.reshape(
            self.sequence_y, (1, self.seq_length, 256)).to(self.device)
        interim_z = torch.reshape(
            self.sequence_z, (1, self.seq_length, 256)).to(self.device)

        u_x = self.rnn_x(interim_x).to(self.device)
        u_y = self.rnn_y(interim_y).to(self.device)
        u_z = self.rnn_z(interim_z).to(self.device)
        # print('Shape [post RNN] u_x: ', u_x.shape)

        u_x = torch.reshape(u_x, u_x_shape).to(self.device)
        u_y = torch.reshape(u_y, u_y_shape).to(self.device)
        u_z = torch.reshape(u_z, u_z_shape).to(self.device)
        # print('Shape [reshaped] u_x: ', u_x.shape)

        u_x = self.AE_x(u_x, y='get_MD_output').to(self.device)
        u_y = self.AE_y(u_y, y='get_MD_output').to(self.device)
        u_z = self.AE_z(u_z, y='get_MD_output').to(self.device)
        # print('Shape [post AE]: ', u_x.shape)

        out = torch.cat((u_x, u_y, u_z), 1).to(device)
        # print('Shape [out]: ', out.shape)
        return out


def resetPipeline(model):
    """The resetPipeline function is required to reset the model.sequence
    attribute that contains the latent spaces from previous timesteps. In
    particular, this is necessary to prepare the model for data from a new data
    distribution.

    Args:
        model:
          Object of PyTorch-type nn.Module containing one of the above defined
          hybrid models
    Return:
        none
    """
    shape = model.sequence.size()
    model.sequence = torch.zeros(shape)
    return


def test_forward_overloading():
    """The test_forward_overloading function aims to check AE's workaround
    forward overloading. This functionality is vital in order to train and save
    the model for hybrid use where a RNN is used to pass a unique bottleneck
    value. This function checks if the functionality yields identical tensors:
    x_check_1 = model(x)
    x_interim, skips = model(x, y=get_interim)
    x_check_2 = model(x_interim,y=get_output, skips=skips)

    Args:
        none
    Return:
        none
    """

    model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )

    _x_in = torch.ones(1000, 3, 24, 24, 24)
    _x_out_1 = model(x=_x_in)

    _x_bottleneck, _skips = model(x=_x_in, y='get_bottleneck')

    _x_out_2 = model(x=_x_bottleneck, y='get_MD_output',
                     skip_connections=_skips)

    if torch.equal(_x_out_1, _x_out_2):
        print('Tensors are equal.')
    else:
        print('Tensors are not equal.')


def calculateFLOPS_ConvAE(c, d, h, w, features):
    """The calculateFLOPS_ConvAE function is used to calculate the number of
    FLOPs and MACCs required to perform a predicition. In particular it considers
    the following relationships:
    - Convolutional Layers
      MACCs: number of kernels x kernel shape x output shape
      FLOPs: 2 * MACCs
    - Pooling Layers
      FLOPs: channels x (depth/stride) x (height/stride) x (width/stride)
    - Activations
      FLOPs: channels x depth x height x width

    Args:
        c:
          Number of channels in the input volume
        d:
          Depth of the input volume
        h:
          Height of the input volume
        w:
          Width of the input volume
        features:

    Returns:
        NONE:
          This function returns the number of MACCs and FLOPs required to
          perform a single prediction.
    """
    _FLOPs = 0
    _MACCs = 0
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    def get_MACCs_Conv(i_channels, n_kernels, k_shape, o_shape):
        k_vol = k_shape**3
        o_vol = k_shape**3
        return n_kernels * i_channels * k_vol * o_vol

    def get_FLOPs_Pool(c, d, h, w):
        s = 2
        return c * d/s * h/s * w/s

    def get_FLOPs_Activ(c, d, h, w):
        return c * d * h * w

    for down in features:
        #DoubleConvs
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=down, k_shape=3, o_shape=h)
        c = down
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _FLOPs += get_FLOPs_Pool(c, d, h, w)
        d = d/2
        h = h/2
        w = w/2
        print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_down
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=c, k_shape=2, o_shape=h)
    d = d - 1
    h = h - 1
    w = w - 1
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # Bottleneck DoubleConv
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=2*c, k_shape=3, o_shape=h)
    c = 2 * c
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    _MACCs += get_MACCs_Conv(i_channels=c,
                             n_kernels=down, k_shape=3, o_shape=h)
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_up_1
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=c, k_shape=2, o_shape=h)
    d = d + 1
    h = h + 1
    w = w + 1
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    for up in reversed(features):
        # First Up-Convolution
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=up, k_shape=2, o_shape=h*2)
        c = up
        d = d*2
        h = h*2
        w = w*2
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_up_2
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=3, k_shape=3, o_shape=h)
    c = 3
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    print(f'Total number of  MACCs: {_MACCs}')
    print(f'Total number of  FLOPs: {_FLOPs + 2*_MACCs}')

    print(f'Total number of GFLOPs: {(_FLOPs + 2*_MACCs)/(10**9)}')


if __name__ == "__main__":
    pass
