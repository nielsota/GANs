import torch
from torch import nn
from models.model_utils import *
from torch.nn.utils import spectral_norm
from Data import *
from Losses import *
from tqdm.auto import tqdm
from itertools import zip_longest


class ConvGenerator(nn.Module):

    def __init__(self, kernel_size: int = 5,
                 padding: int = 2,
                 z_dim: int = 100,
                 num_filters: int = 72,
                 len_sample: int = 100,
                 emb_dim: int = 16,
                 in_channels: int = 1):
        # Inherit nn.Module
        super().__init__()

        # Specify number of filter
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding

        # Specify target length
        self.z_dim = z_dim
        self.len_sample = len_sample
        self.in_channels = in_channels

        # Specify embedding dimension
        self.emb_dim = emb_dim

        # Create an embedding layer: map every time step to higher dimensional vec space
        self.emb = nn.Sequential(nn.Linear(self.in_channels, self.emb_dim),
                                 nn.LeakyReLU(0.2, inplace=True))

        # Optional: Add dimension for convolution
        self.add_dim = AddDimension()

        # Create Generator
        self.gen = self._getgenerator()

    def _getgenerator(self):
        # Set number of channels in central
        chan = [self.emb_dim, self.num_filters, 2 * self.num_filters, self.num_filters, 1]

        # Create list with 1D conv objects
        convs = [spectral_norm(nn.Conv1d(chan[i], chan[i + 1], self.kernel_size, padding=self.padding),
                               n_power_iterations=10) for i in range(len(chan) - 1)]

        # Create Upsample dimension
        ups_dim = [i * self.z_dim for i in range(1, len(chan))]

        # Create Upsample layers
        ups = [nn.Upsample(i) for i in ups_dim]

        # Create ReLU layers
        relus = [nn.LeakyReLU(0.2, inplace=True) for _ in chan]

        # Zip together layers
        layers = [func for z in zip(convs, ups, relus) for func in z]

        # Create network
        network = nn.Sequential(*layers,
                                SqeezeDimension(),
                                nn.Linear(ups_dim[-1], self.len_sample)
                                )

        # Return network
        return network

    def forward(self, x):
        batch_size = len(x)

        # add dimension if shape is [B,T]
        if len(x.shape) == 2:
            x = self.add_dim(x)

        # Switch T and C dimension, [B, T, C] -> [B*T, C]
        x = x.transpose(1, 2).contiguous().view(-1, x.shape[1])

        # Embedding: [B*T, C] -> [B*T, emb_dim]
        x = self.emb(x)

        # [B*T, emb_dim]-> [B, T, emb_dim]
        x = x.view(batch_size, self.len_sample, self.emb_dim)

        # [B, emb_dim, T]
        x = x.transpose(1, 2).contiguous()

        return self.gen(x)


class ConvCritic(nn.Module):

    def __init__(self, kernel_size: int = 5,
                 padding: int = 2,
                 z_dim: int = 100,
                 num_filters=None,
                 num_neurons=None,
                 len_sample: int = 100,
                 in_channels: int = 1):

        # Inherit nn Module
        super().__init__()

        # Specify number of filter
        if num_filters is None:
            self.num_filters = [1, 50, 50]
        else:
            self.num_filters = num_filters

        # Specify number of neurons
        if num_filters is None:

            # Calculate number of neurons after flatten
            n_linear_1 = int(self.num_filters[-1] * (100 / ((len(self.num_filters) - 1) * 2)))

            # Set number of neurons linear layers
            self.num_neurons = [n_linear_1, 50, 15, 1]
        else:
            assert num_filters[0] == int(self.num_filters[-1] * (100 / ((len(self.num_filters) - 1) * 2)))
            self.num_filters = num_filters

        # Specify kernel size and padding
        self.kernel_size = kernel_size
        self.padding = padding

        # Specify target length
        self.z_dim = z_dim
        self.len_sample = len_sample
        self.in_channels = in_channels

        # Add dimension for convolution
        self.add_dim = AddDimension()

        # Create Critic
        self.critic = self._getcritic()

    def _getcritic(self):

        # Set number of channels in central, len is num layers
        chan = self.num_filters

        # Create list with 1D conv objects
        convs = [spectral_norm(nn.Conv1d(chan[i], chan[i + 1], self.kernel_size, padding=self.padding),
                               n_power_iterations=10) for i in range(len(chan) - 1)]

        # Maxpool layers
        maxpool = [nn.MaxPool1d(2) for _ in range(len(chan) - 1)]

        # Create ReLU layers
        relus = [nn.LeakyReLU(0.2, inplace=True) for _ in range(len(chan) - 1)]

        # Zip together layers
        conv_layers = [func for z in zip_varlen(convs, maxpool, relus) for func in z]

        # Calculate number of neurons after flatten
        n_linear_1 = int(chan[-1] * (100 / (len(maxpool) * 2)))

        # Set number of neurons each layer
        n_linear = self.num_neurons

        # Create Linear Layers
        linear = [nn.Linear(n_linear[i], n_linear[i + 1]) for i in range(len(n_linear) - 1)]

        # Create ReLUs
        relus = [nn.LeakyReLU(0.2, inplace=True) for _ in range(len(n_linear) - 2)]

        # Zip together linear and relu functions
        linear_layers = [func for z in zip_varlen(linear, relus) for func in z]

        # Create network
        network = nn.Sequential(*conv_layers,
                                nn.Flatten(),
                                *linear_layers
                                )

        # Return network
        return network

    def forward(self, x):

        # add dimension if shape is [B,T]
        if len(x.shape) == 2:
            x = self.add_dim(x)

        return self.critic(x)


class CwganTrainer:

    def __init__(self, gen,
                 critic,
                 gen_opt,
                 crit_opt,
                 dataloader,
                 epochs: int = 50,
                 crit_repeats: int = 5,
                 c_lambda: int = 10,
                 z_dim: int = 100,
                 plot_loss=True,
                 conditional=False,
                 device='cpu'):

        # Store models
        self.critic = critic
        self.generator = gen

        # Store optimizers
        self.gen_opt = gen_opt
        self.crit_opt = crit_opt

        # Store data and noise dimensions
        self.z_dim = z_dim
        self.dataloader = dataloader

        # Store hyperparameters
        self.c_lambda = c_lambda
        self.crit_repeats = crit_repeats
        self.epochs = epochs

        # Store losses for inspection
        self.crit_losses = []
        self.gen_losses = []

        # store if conditional GAN
        self.conditional = conditional

        # Show learning
        self.plot_loss = plot_loss

        # Store device
        self.device = device

    def _trainiteration(self, real, labels):

        # Retrieve batch size, real data, and labels
        cur_batch_size = len(real)
        real = real.to(self.device).float()
        labels = labels.to(self.device).float()

        # Update critic
        mean_iteration_critic_loss = 0
        for _ in range(self.crit_repeats):

            # Zero out gradients
            self.crit_opt.zero_grad()

            # Generate sample and run generated and real through disc
            fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)

            # Combine with labels if conditional GAN
            if self.conditional:
                fake_noise = combine_noise_and_labels(fake_noise, labels)

            # Input noise into generator
            fake = self.generator(fake_noise)

            # Combine fakes with labels if conditional GAN
            if self.conditional:
                fake = combine_noise_and_labels(fake, labels)
                real = combine_noise_and_labels(real, labels)

            # Input fake and real into discriminator
            crit_fake_pred = self.critic(fake.detach())
            crit_real_pred = self.critic(real)

            # Set correct shape for epsilon
            epsilon_shape = [len(real), 1]
            if self.conditional:
                epsilon_shape = [len(real), 1, 1]

            # Generate random epsilon in correct shape
            epsilon = torch.rand(*epsilon_shape, device=self.device, requires_grad=True)

            # Compute gradient wrt mixed images
            gradient = compute_gradient(self.critic, real, fake.detach(), epsilon)

            # Compute gradient penalty
            gp = gradient_penalty(gradient)

            # Reset real to only first channel (contains TS)
            if self.conditional:
                real = real[:, 0, :]

            # Compute loss
            crit_loss = critic_loss(crit_fake_pred, crit_real_pred, gp, self.c_lambda)

            # Update gradients
            crit_loss.backward(retain_graph=True)

            # Update optimizer
            self.crit_opt.step()

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item()

        # Save losses
        crit_iteration_loss = mean_iteration_critic_loss / self.crit_repeats
        # End Update critic

        # Update generator

        # Zero out old gradients
        self.gen_opt.zero_grad()

        # Generate noise sample
        fake_noise_2 = get_noise(cur_batch_size, self.z_dim, device=self.device)

        # combine noise with labels if conditional GAN
        if self.conditional:
            fake_noise_2 = combine_noise_and_labels(fake_noise_2, labels)

        # input noise into generator
        fake_2 = self.generator(fake_noise_2)

        # combine fakes with labels if conditional GAN
        if self.conditional:
            fake_2 = combine_noise_and_labels(fake_2, labels)

        # input fakes to discriminator
        crit_fake_pred = self.critic(fake_2)

        # Compute loss
        gen_loss = generator_loss(crit_fake_pred)

        # Compute gradients using back prop
        gen_loss.backward()

        # Take a step
        self.gen_opt.step()

        # Keep track of the average generator loss
        gen_iteration_loss = gen_loss.item()
        # End Update Generator

        return crit_iteration_loss, gen_iteration_loss

    def _trainloop(self):
        """
        Function for training loop
        """

        # Store epoch loss
        epoch_gen_losses = []
        epoch_crit_losses = []

        for data, labels in tqdm(self.dataloader):
            # Take step
            crit_iteration_loss, gen_iteration_loss = self._trainiteration(data, labels)

            # Store loss
            epoch_gen_losses += [gen_iteration_loss]
            epoch_crit_losses += [crit_iteration_loss]

        # Keep track of the average generator loss
        self.gen_losses += [np.mean(epoch_gen_losses)]
        self.crit_losses += [np.mean(epoch_crit_losses)]
        # End Update

    def fit(self):
        """
        fits model
        """
        for i in range(self.epochs):
            print("\n--------------- starting epoch {} --------------- \n".format(str(i + 1)))
            self._trainloop()
            print("\n--------------- Finished epoch {} --------------- \n".format(str(i + 1)))

            if self.plot_loss and i + 1 % 25 == 0:
                plt.plot(self.gen_losses)
                plt.plot(self.crit_losses)
                plt.grid()
                plt.show()


if __name__ == '__main__':
    # Unit tests Generator
    net = ConvGenerator(in_channels=3)
    noise = torch.randn(32, 3, 100, device='cpu')
    pred = net(noise)

    print(net)

    print('Input shape : {}'.format(noise.shape))
    print('Output shape: {}'.format(pred.shape))

    # Unit tests Critic
    net = ConvCritic(in_channels=1)
    noise = torch.randn(32, 100, device='cpu')
    pred = net(noise)

    print(net)

    print('Input shape : {}'.format(noise.shape))
    print('Output shape: {}'.format(pred.shape))

    # Unit test Trainer
    dl = load_sin_data(batch_size=128, num_samples=10000,
                       len_sample=100)

    g = ConvGenerator(in_channels=1)
    c = ConvCritic(in_channels=1)
    print(c)

    # Create optimizer
    a = torch.optim.Adam(g.parameters(), lr=0.0001, betas=(0.5, 0.999))
    b = torch.optim.Adam(c.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Option needed for debugging
    torch.autograd.set_detect_anomaly(True)

    trainer = CwganTrainer(g, c, a, b, dl)
    trainer.fit()
