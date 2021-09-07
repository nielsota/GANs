from models.Conv_GAN import *
from models.Embedding import *
from collections import OrderedDict
import warnings
import seaborn as sns
warnings.simplefilter('ignore', UserWarning)
sns.set()

# Goal: Create trainer that uses Embeddings as labels for GAN


class OneShotTrainer:

    def __init__(self,
                 generator,
                 critic,
                 embedder,
                 dataloader,
                 gen_opt,
                 crit_opt,
                 emb_opt,
                 emb_criterion,
                 critic_repeats: int = 5,
                 c_lambda: int = 10,
                 z_dim: int = 100,
                 emb_epochs: int = 100,
                 gan_epochs: int = 100,
                 device='cpu'):

        # Define models
        self.generator = generator
        self.critic = critic
        self.embedder = embedder

        # Set up hyperparameters
        self.emb_epochs = emb_epochs
        self.gan_epochs = gan_epochs
        self.device = device
        self.z_dim = z_dim
        self.critic_repeats = critic_repeats
        self.c_lambda = c_lambda

        # Store data loader
        self.dataloader = dataloader

        # placeholder for batch features
        self.embeddings = []
        self.embedder.network.register_forward_hook(self._forward_hook('network'))

        # Store all optimizers
        self.gen_opt, self.crit_opt, self.emb_opt = gen_opt, crit_opt, emb_opt

        # Store criterion embedder network
        self.emb_criterion = emb_criterion

        # Keep track of the average generator loss
        self.emb_losses = []
        self.gen_losses = []
        self.crit_losses = []

    # Create forward hook to store embeddings
    def _forward_hook(self, layer_name):
        def hook(module, input, output):
            self.embeddings = output
        return hook

    def _emb_trainiteration(self, data, labels):
        """
        Function to perform 1 iteration of generator update
        :param data: time series in shape [B, T]
        :param labels: labels in shape [B, C]
        :return: loss for this iteration
        """

        # Retrieve data and labels
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()

        # Create predictions
        pred = self.embedder(data)

        # Compute loss
        loss = self.emb_criterion(pred, labels)

        # Zero out old gradients
        self.emb_opt.zero_grad()

        # Compute gradients using back prop
        loss.backward()

        # Take a step
        self.emb_opt.step()

        return loss.item()

    def _gen_trainiteration(self, data, labels):
        """
        Function to perform 1 iteration of generator update
        :param data: time series in shape [B, T]
        :param labels: labels in shape [B, C]
        :return: loss for this iteration
        """

        # Retrieve batch size, real data, and labels
        cur_batch_size = len(data)
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()

        # forward pass [with feature extraction]
        _ = self.embedder(data)
        emb = self.embeddings

        # Zero out old gradients
        self.gen_opt.zero_grad()

        # Generate noise sample
        fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)

        # Combine noise with embeddings
        fake_noise = combine_noise_and_labels(fake_noise, emb)

        # input noise into generator
        fake = self.generator(fake_noise)

        # combine fakes with REAL labels
        fake = combine_noise_and_labels(fake, labels)

        # input fakes to discriminator
        crit_fake_pred = self.critic(fake)

        # Compute loss
        gen_loss = generator_loss(crit_fake_pred)

        # Compute gradients using back prop
        gen_loss.backward()

        # Take a step
        self.gen_opt.step()

        # Keep track of the average generator loss
        gen_iteration_loss = gen_loss.item()
        # End Update Generator

        return gen_iteration_loss

    def _crit_trainiteration(self, data, labels):
        """
        Function to perform 1 iteration of critic update
        :param data: time series in shape [B, T]
        :param labels: labels in shape [B, C]
        :return: loss for this iteration
        """

        # Retrieve batch size, real data, and labels
        cur_batch_size = len(data)
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()

        # forward pass [with feature extraction]
        _ = self.embedder(data)
        emb = self.embeddings

        # Update critic
        total_iteration_critic_loss = 0

        for _ in range(self.critic_repeats):

            # Zero out gradients
            self.crit_opt.zero_grad()

            # Generate sample and run generated and real through disc
            fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)

            # Combine with labels if conditional GAN
            fake_noise = combine_noise_and_labels(fake_noise, emb)

            # Input noise into generator
            fake = self.generator(fake_noise)

            # Combine fakes with labels if conditional GAN
            fake = combine_noise_and_labels(fake, labels)
            real = combine_noise_and_labels(data, labels)

            # Input fake and real into discriminator
            crit_fake_pred = self.critic(fake.detach())
            crit_real_pred = self.critic(real)

            # Set correct shape for epsilon
            epsilon_shape = [len(data), 1, 1]

            # Generate random epsilon in correct shape
            epsilon = torch.rand(*epsilon_shape, device=self.device, requires_grad=True)

            # Compute gradient wrt mixed images
            gradient = compute_gradient(self.critic, real, fake.detach(), epsilon)

            # Compute gradient penalty
            gp = gradient_penalty(gradient)

            # Compute loss
            crit_loss = critic_loss(crit_fake_pred, crit_real_pred, gp, self.c_lambda)

            # Update gradients
            crit_loss.backward(retain_graph=True)

            # Update optimizer
            self.crit_opt.step()

            # Keep track of the average critic loss in this batch
            total_iteration_critic_loss += crit_loss.item()

            # Save losses
        mean_iteration_critic_loss = total_iteration_critic_loss / self.critic_repeats
        # End Update critic

        return mean_iteration_critic_loss


    def plot_loss(self):
        """
        plot losses
        :param losses_list: list of losses you wish to plot
        :return: plot
        """

        # Check number of functions to plot
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(18.5, 10.5)

        # Plot losses
        axs[0].plot(self.emb_losses)
        axs[0].set_title("embedding loss")
        axs[1].plot(self.gen_losses)
        axs[1].set_title("generator loss")
        axs[2].plot(self.crit_losses)
        axs[2].set_title("critic loss")

        # Show plot
        plt.show()

    def _trainloop(self):
        """
        Function for training loop
        """
        # Store epoch loss
        epoch_emb_losses = []
        epoch_gen_losses = []
        epoch_crit_losses = []

        for data, labels in tqdm(self.dataloader):

            # Take step
            emb_iteration_loss = self._emb_trainiteration(data, labels)
            crit_iteration_loss = self._crit_trainiteration(data, labels)
            gen_iteration_loss = self._gen_trainiteration(data, labels)

            # Store loss
            epoch_emb_losses += [emb_iteration_loss]
            epoch_gen_losses += [gen_iteration_loss]
            epoch_crit_losses += [crit_iteration_loss]

        # Keep track of the average generator loss
        self.emb_losses += [np.mean(epoch_emb_losses)]
        self.gen_losses += [np.mean(epoch_gen_losses)]
        self.crit_losses += [np.mean(epoch_crit_losses)]
        # End Update

    def fit(self):
        """
        fits model
        """
        for i in range(self.emb_epochs):
            print("\n--------------- starting epoch {} --------------- \n".format(str(i + 1)))
            self._trainloop()
            print("\n--------------- Finished epoch {} --------------- \n".format(str(i + 1)))

            if self.plot_loss and (i + 1) % 10 == 0:
                self.plot_loss()

    def save_model(self, path, model_type, data_type):

        # Generate paths
        emb_path = os.path.join(path, model_type + '_' + data_type + '_embedder.pth')
        generator_path = os.path.join(path, model_type + '_' + data_type + '_generator.pth')
        disc_path = os.path.join(path, model_type + '_' + data_type + '_critic.pth')

        # Save models
        torch.save(self.embedder, emb_path)
        torch.save(self.generator, generator_path)
        torch.save(self.critic, disc_path)

        print("--------------- models saved --------------- \n")


def main():

    # Create model
    gen = ConvGenerator(in_channels=17)
    crit = ConvCritic(in_channels=3, num_filters=[3, 50, 50])
    embedder = Embedder()

    # Create dataloader
    dataloader = load_arima_data(batch_size=512, num_samples=4*2560,
                                 len_sample=100, dgp='arma_11_variable')

    # Create optimizers
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.00005, betas=(0.5, 0.999))
    crit_opt = torch.optim.Adam(crit.parameters(), lr=0.00005, betas=(0.5, 0.999))
    emb_opt = torch.optim.Adam(embedder.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # Create loss function
    emb_crit = nn.MSELoss()

    trainer = OneShotTrainer(gen, crit, embedder, dataloader, gen_opt, crit_opt, emb_opt, emb_crit)

    data, labels = next(iter(dataloader))

    trainer._gen_trainiteration(data, labels)
    trainer._crit_trainiteration(data, labels)

    trainer.fit()


if __name__ == '__main__':
    main()