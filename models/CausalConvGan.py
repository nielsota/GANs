import torch
from torch import nn
import torch.nn.functional as F
from Data import get_noise
from torch.nn.utils import spectral_norm


################################################################################
########################## Causal Conv GAN  ####################################
################################################################################


class AddDimension(nn.Module):
    """
    Turn [B, T] -> [B, 1, T]
    """

    def forward(self, x):
        return x.view(len(x), 1, -1)


class SqeezeDimension(nn.Module):
    """
    Turn [B, C, T] -> [B, C*T]
    """

    def forward(self, x):
        return x.view(len(x), -1)


# noinspection PyTypeChecker
class CausalConv1d(torch.nn.Conv1d):
    """
    Overwrite 1d convolution to ensure output at time t cannot see time after t
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


# noinspection PyTypeChecker
def get_conv_block(kernel_size: int = 3,
                   dilation: int = 1,
                   in_channels: int = 64,
                   out_channels: int = 64,
                   hidden_channels: int = 64,
                   last: bool = False):
    """

    :param kernel_size: Kernel Size
    :param dilation: Dilation
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param hidden_channels: Number of hidden channels
    :param last: Boolean for last layer
    :return: Sequential object containing CC, LR, CC, LR, CC layers
    """

    # Instantiate first part
    block = nn.Sequential(CausalConv1d(in_channels,
                                       hidden_channels,
                                       kernel_size=kernel_size,
                                       dilation=dilation),

                          nn.LeakyReLU(0.2, inplace=True),

                          CausalConv1d(hidden_channels,
                                       hidden_channels,
                                       kernel_size=kernel_size,
                                       dilation=dilation))

    # Instantiate second part if not last layer
    if not last:
        block = nn.Sequential(block,

                              nn.LeakyReLU(0.2, inplace=True),

                              CausalConv1d(hidden_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           dilation=dilation))

    return block


class CausalConvDiscriminator(nn.Module):
    """
    Critic that uses CausalConvolutions with residual connections
    """

    # noinspection PyTypeChecker
    def __init__(self,
                 num_channels: int = 32,
                 num_labels: int = 6,
                 timeseries_dim: int = 1,
                 in_channels=8):
        super(CausalConvDiscriminator, self).__init__()

        self.leakyrelu = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.addDim = AddDimension()

        self.block1 = get_conv_block(in_channels=in_channels,
                                     hidden_channels=num_channels,
                                     out_channels=num_channels)

        self.block2 = get_conv_block(in_channels=num_channels,
                                     hidden_channels=num_channels,
                                     out_channels=num_channels)

        self.block3 = get_conv_block(in_channels=num_channels,
                                     hidden_channels=num_channels,
                                     out_channels=num_channels,
                                     dilation=2)

        self.block4 = nn.Sequential(
            CausalConv1d(num_channels,
                         num_channels,
                         kernel_size=5,
                         dilation=1),
            nn.LeakyReLU(0.2, inplace=True),

            CausalConv1d(num_channels,
                         out_channels=1,
                         kernel_size=5,
                         dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block5 = nn.Sequential(nn.Linear(100, 1),
                                    SqeezeDimension())

    def forward(self, noise):
        noise = self.addDim(noise)
        out1 = self.block1(noise)
        res1 = self.leakyrelu(out1)

        out2 = self.block2(res1)
        res2 = self.leakyrelu(out2 + res1)

        out3 = self.block3(res2)
        res3 = self.leakyrelu(out3 + res2)

        out4 = self.block4(res3)
        out5 = self.block5(out4)

        return out5


class CausalConvGenerator(nn.Module):
    ''' Generator class
    Input: No input required
    Output: Noise transformed into images. For now use .view() w/ correct dimensions
    to transform (len(noise) contains batch size).
    '''

    def __init__(self,
                 z_dim=100,
                 len_sample=100,
                 hid_dim=128,
                 in_channels=1):
        super(CausalConvGenerator, self).__init__()

        # Add dimension to time series
        self.addDim = AddDimension()

        # Leaky relu
        self.leakyrelu = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))

        # Sample up to 64 channels
        self.first_block = self.get_conv_block(in_channels=in_channels)

        # Blocks for skip connections
        self.block2 = get_conv_block()
        self.block3 = get_conv_block(dilation=2)
        self.block4 = get_conv_block(dilation=2)
        self.block5 = get_conv_block()

        # Downsample up to 64 channels
        self.down_sample_block = self.get_conv_block(out_channels=1, last=True)
        self.linear_block = nn.Sequential(nn.Linear(z_dim, 100))

    def get_conv_block(self, kernel_size: int = 3, dilation: int = 1, in_channels: int = 64,
                       out_channels: int = 64, hidden_channels: int = 64, last: bool = False):

        block = nn.Sequential(CausalConv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                                           dilation=dilation),
                              nn.LeakyReLU(0.2, inplace=True),
                              CausalConv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                                           dilation=dilation),
                              nn.LeakyReLU(0.2, inplace=True),
                              CausalConv1d(hidden_channels, out_channels, kernel_size=kernel_size,
                                           dilation=dilation)
                              )

        if last: block = nn.Sequential(block, SqeezeDimension())

        return block

    def forward(self, noise):
        noise = self.addDim(noise)

        z_new = self.first_block(noise)
        a_new = self.leakyrelu(z_new)
        a_old = a_new

        skip_layers = ['block2', 'block3', 'block4', 'block5']
        for name, layer in self.named_children():

            if name in skip_layers:
                z_new = layer(a_old)
                a_new = self.leakyrelu(z_new + a_old)
                a_old = a_new

            if name == 'down_sample_block':
                a6 = self.down_sample_block(a_old)
                a7 = self.linear_block(a6)

        return a7


class ConvGenerator(nn.Module):
    def __init__(self, kernel_size: int = 5,
                 padding: int = 2,
                 z_dim: int = 100,
                 len_sample: int = 100,
                 in_channels: int = 1):
        super().__init__()
        self.num_filters = 72
        self.len_sample = len_sample
        self.emb_dim = 16
        self.emb = nn.Sequential(nn.Linear(in_channels, self.emb_dim),
                                 nn.LeakyReLU(0.2, inplace=True))

        self.gen = nn.Sequential(#nn.Linear(z_dim, 100),
                                 #nn.LeakyReLU(0.2, inplace=True),
                                 #AddDimension(),
                                 spectral_norm(nn.Conv1d(self.emb_dim, self.num_filters, kernel_size, padding=padding),
                                               n_power_iterations=10),
                                 nn.Upsample(200),

                                 spectral_norm(nn.Conv1d(self.num_filters, 2*self.num_filters, kernel_size, padding=padding), n_power_iterations=10),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Upsample(400),

                                 spectral_norm(nn.Conv1d(2*self.num_filters, self.num_filters, kernel_size, padding=padding), n_power_iterations=10),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Upsample(800),

                                 spectral_norm(nn.Conv1d(self.num_filters, 1, kernel_size, padding=padding), n_power_iterations=10),
                                 nn.LeakyReLU(0.2, inplace=True),

                                 SqeezeDimension(),
                                 nn.Linear(800, len_sample)
                                 )

    def forward(self, x):
        batch_size = len(x)

        # Switch T and C dimension, [B, T, C] -> [B*T, C]
        x = x.transpose(1, 2).contiguous().view(-1, x.shape[1])

        # Embedding: [B*T, C] -> [B*T, emb_dim]
        x = self.emb(x)

        # [B*T, emb_dim]-> [B, T, emb_dim]
        x = x.view(batch_size, self.len_sample, self.emb_dim)

        # [B, emb_dim, T]
        x = x.transpose(1, 2).contiguous()

        return self.gen(x)


class ConvDiscriminator(nn.Module):
    def __init__(self,
                 kernel_size: int = 5,
                 padding: int = 2,
                 z_dim: int = 100,
                 len_sample: int = 100,
                 in_channels: int = 1):
        super().__init__()
        self.disc = nn.Sequential(#AddDimension(),
                                  spectral_norm(nn.Conv1d(in_channels, 48, kernel_size, padding=padding),
                                                n_power_iterations=10),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.MaxPool1d(2),

                                  spectral_norm(nn.Conv1d(48, 48, kernel_size, padding=padding), n_power_iterations=10),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.MaxPool1d(2),

                                  spectral_norm(nn.Conv1d(48, 32, kernel_size, padding=padding), n_power_iterations=10),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Flatten(),

                                  nn.Linear(800, 50),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Linear(50, 15),
                                  nn.LeakyReLU(0.2, inplace=True),

                                  nn.Linear(15, 1)
                                  )

    def forward(self, input):
        return self.disc(input)


################################################################################
################################################################################


if __name__ == '__main__':

    print("Building test generator...")
    test_input = get_noise(n_samples=32, z_dim=64)
    print("input shape: {}".format(test_input.shape))
    try:
        test_generator = ConvGenerator(in_channels=1, z_dim=64, len_sample=100)
        test_output = test_generator(test_input)
        print("output shape: {}".format(test_output.shape))
        print("Convolution Generator functional!\n")
    except Exception as e:
        print(e)

    print("Building test discriminator...")
    test_input = get_noise(n_samples=32, z_dim=100)
    print("input shape: {}".format(test_input.shape))
    try:
        test_discriminator = ConvDiscriminator()
        test_output = test_discriminator(test_input)
        print("output shape: {}".format(test_output.shape))
        print("Convolution discriminator functional!\n")
    except Exception as e:
        print(e)
