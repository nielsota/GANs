import torch
from torch import nn
from Data import *
from models.model_utils import AddDimension, SqeezeDimension
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class Embedder(nn.Module):

    def __init__(self,
                 len_sample: int = 100,
                 emb_dim: int = 16,
                 kernel_size: int = 5,
                 padding: int = 2,
                 chn=None):
        super().__init__()

        # Dimensions for network
        self.len_sample = len_sample
        self.emb_dim = emb_dim

        # Parameters for convolution
        self.kernel_size = kernel_size
        self.padding = padding

        # Set channels
        if chn is None:
            self.chn = [1, 32, 32, 32, 1]
        else:
            self.chn = chn

        # Build model
        self.network = self.build_network()
        self.final = nn.Sequential(nn.Linear(self.emb_dim, 2),
                                   nn.Tanh())

    def build_network(self):

        # Define number of channels each layer
        chn = self.chn

        # Create list holding 1D-conv objects
        convs = [nn.Conv1d(chn[i], chn[i + 1], kernel_size=self.kernel_size,
                           padding=self.padding) for i in range(len(chn) - 1)]

        # Create lists with relus
        relus = [nn.LeakyReLU(0.2, inplace=True) for _ in range(len(chn) - 1)]

        # Zip together lists
        comb = [func for z in zip(convs, relus) for func in z]

        # Build network
        network = nn.Sequential(AddDimension(),
                                *comb,
                                SqeezeDimension(),
                                nn.Linear(self.len_sample, self.emb_dim))
        # Return the model
        return network

    def forward(self, x):
        """
        Implement forward pass
        :param x: input to model
        :return: prediction
        """
        x = self.network(x)
        x = self.final(x)
        return x


class ClassificationTrainer:
    """
    Class to run training
    """

    def __init__(self, model, opt, dataloader, criterion,
                 plot_loss = True, epochs: int = 50, device='cpu'):

        # Store model, optimizer and dataloader
        self.model = model
        self.opt = opt
        self.dataloader = dataloader

        # Create list for losses
        self.losses = []

        # Create loss
        self.criterion = criterion

        # Set hyperparameters
        self.epochs = epochs

        # Store device
        self.device = device

        # Plot loss
        self.plot_loss = plot_loss

    def _trainiteration(self, data, labels):
        # Retrieve data and labels
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()

        # Create predictions
        pred = self.model(data)

        # Compute loss
        loss = self.criterion(pred, labels)

        # Zero out old gradients
        self.opt.zero_grad()

        # Compute gradients using back prop
        loss.backward()

        # Take a step
        self.opt.step()

        return loss.item()

    def _trainloop(self):
        """
        Function for training loop
        """

        # Store epoch loss
        epoch_losses = []

        for data, labels in tqdm(self.dataloader):

            # Take step
            loss_item = self._trainiteration(data, labels)

            # Store loss
            epoch_losses += [loss_item]

        # Keep track of the average generator loss
        self.losses += [np.mean(epoch_losses)]
        # End Update

    def fit(self):
        """
        fits model
        """
        for i in range(self.epochs):
            print("\n--------------- starting epoch {} --------------- \n".format(str(i + 1)))
            self._trainloop()
            print("\n--------------- Finished epoch {} --------------- \n".format(str(i + 1)))

            if self.plot_loss and (i + 1) % 50 == 0:
                plt.plot(self.losses)
                plt.grid()
                plt.show()

    def save_model(self, path, model_type, data_type):

        # Generate paths
        model_path = os.path.join(path, model_type + '_' + data_type + '_classifier.pth')

        # Save models
        torch.save(self.model, model_path)

        print("--------------- models saved --------------- \n")


if __name__ == '__main__':

    # Create model
    model = Embedder()

    # Create dataloader
    dataloader = load_arima_data(batch_size=512, num_samples=512,
                                 len_sample=100, dgp='arma_11_variable')

    # Create optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # Create loss function
    crit = nn.MSELoss()

    # Create trainer
    trainer = ClassificationTrainer(model, opt, dataloader, crit, epochs=1)

    # Fit model
    trainer.fit()

    # Create save path
    #models_path = '/Users/nielsota/Documents/GitHub/GANs/fitted_models'
    #model_type = '1_D_Conv_v2'
    #data_type = 'arma_11_variable'

    # Save model
    #trainer.save_model(models_path, model_type, data_type)


    # Helper function for building hook
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook


    # Register hook
    model.network.register_forward_hook(get_features('feats'))

    # placeholders
    PREDS = []
    embeddings = []

    # placeholder for batch features
    features = {}

    # forward pass [with feature extraction]
    test_input = get_noise(32, 100)
    preds = model(test_input)

    # add feats and preds to lists
    PREDS.append(preds.detach().cpu().numpy())
    embeddings.append(features['feats'].cpu().numpy())

    print('Time series batch has shape {}'.format(test_input.shape))
    print('Embedded features have shape {}'.format(embeddings[0].shape))

    def _forward_hook(layer_name):
        def hook(module, input, output):
            selected_out[layer_name] = output
        return hook


    from collections import OrderedDict
    selected_out = OrderedDict()
    print(model._modules.keys())
    print(getattr(model, 'network'))
    fhooks = []
    fhooks.append(model.network.register_forward_hook(_forward_hook('network')))
    # forward pass [with feature extraction]
    test_input = get_noise(32, 100)
    preds = model(test_input)
    test_input2 = get_noise(32, 100)
    preds2 = model(test_input2)
    print(model.network)
    print(fhooks)
    print(selected_out['network'].shape)