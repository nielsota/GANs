from torch import nn
from Data import get_noise

################################################################################
############################### MLP GAN  #######################################
################################################################################


class MLPGenerator(nn.Module):
    """ Generator class
    Input: No input required
    Output: Noise transformed into images. For now use .view() w/ correct dimensions
    to transform (len(noise) contains batch size).
    """

    def __init__(self, z_dim=100, len_sample=100, hid_dim=128, depth=2):
        super(MLPGenerator, self).__init__()
        self.z_dim = z_dim

        mlp_blocks = []
        mlp_blocks.append(self.generator_block(z_dim, hid_dim))
        mlp_blocks.append(self.generator_block(hid_dim, 2*hid_dim))
        mlp_blocks.append(self.generator_block(2 * hid_dim, hid_dim))
        mlp_blocks.append(self.generator_block(hid_dim, len_sample))
        self.generator = nn.Sequential(*mlp_blocks)

    def generator_block(self, in_dimenion, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimenion, out_dimension),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise):
        return self.generator(noise)


class MLPDiscriminator(nn.Module):
    """ Critic class
        Input: Time Series [B, N]
        Output: W-distance is the sup over all f's that are 1-LC. f is parametrized by this
                network. Find f that calculates W-distance by supping the loss.
    """

    def __init__(self, len_sample=100):
        super(MLPDiscriminator, self).__init__()
        self.critic = nn.Sequential(  # AddDimension(),
            nn.Linear(len_sample, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.critic(x)


################################################################################
################################################################################


if __name__ == '__main__':

    print("Building test generator...")
    test_input = get_noise(n_samples=32, z_dim=64)
    print("input shape: {}".format(test_input.shape))
    try:
        test_generator = MLPGenerator(z_dim=64, len_sample=100, hid_dim=128, depth=3)
        test_output = test_generator(test_input)
        print("output shape: {}".format(test_output.shape))
        print("MLP Generator functional!\n")
    except Exception as e: print(e)

    print("Building test discriminator...")
    test_input = get_noise(n_samples=32, z_dim=100)
    print("input shape: {}".format(test_input.shape))
    try:
        test_generator = MLPDiscriminator(len_sample=100)
        test_output = test_generator(test_input)
        print("output shape: {}".format(test_output.shape))
        print("MLP discriminator functional!\n")
    except Exception as e: print(e)
