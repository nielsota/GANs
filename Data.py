import torch
import numpy as np
from utils import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from typing import Sequence


################################################################################
############################ Generate Noise ####################################
################################################################################


# Get random noise
def get_noise(n_samples, z_dim, device='cpu'):
    """

    :param n_samples: number of samples to generate
    :param z_dim:  dimension of each noise sample
    :param device: device to save tensor on
    :return: noise in batch [batch_size, features] = [n_dim, z_dim]
    """
    return torch.randn(n_samples, z_dim, device=device)

################################################################################
################################################################################

################################################################################
######################## FashionMNIST DATA LOADING #############################
################################################################################


def load_FashionMNIST_data(batch_size: int = 128):
    """
    Loads data into dataloader
    Returns data in batches of shape (128, 1, 28, 28)
    """
    print("Starting dataloading...")
    training_data = datasets.FashionMNIST(root='data',
                                          train=True,
                                          download=True,
                                          transform=ToTensor())

    test_data = datasets.FashionMNIST(root='data',
                                      train=False,
                                      download=True,
                                      transform=ToTensor())

    train_DataLoader = DataLoader(training_data, batch_size=batch_size)
    test_DataLoader = DataLoader(test_data, batch_size=batch_size)

    print('Finished dataloading!\n')
    return train_DataLoader, test_DataLoader

################################################################################
################################################################################


################################################################################
############################ Generate ARIMA ####################################
################################################################################


# ARMA Dataset Generator
class BasicARMA(Dataset):
    """ARMA dataset."""

    def __init__(self, num_samples: int = 100, len_sample: int = 100, seed: int = 1, transform=None):
        """
        Class to generate and store a PyTorch Iteratble dataset w/ ARMA time series

        Args:
            num_samples: number of time series
            len_sample: length of time series
            seed: random seed for reproducability
            transform: how to transform data (often toTensor)
        """

        self.num_samples = num_samples
        self.len_sample = len_sample
        self.seed = seed
        self.data, self.labels = self.__make_data__()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx][:]
        label = self.labels[idx][:]

        # Transform if applicable
        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __make_data__(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        AR = 0.5
        MA = 0.5
        clip = int(self.len_sample / 10)

        data = []
        labels = []

        print("Started generating ARMA samples!")
        for i in range(self.num_samples):
            # Draw parameters
            ar = AR
            ma = MA
            labels.append([ar, ma])

            # Draw sample
            ar = np.r_[1, -ar]
            ma = np.r_[1, ma]
            ts = arma_generate_sample(ar=ar, ma=ma, nsample=self.len_sample, burnin=clip)
            data.append(ts)
        print("Finished generating ARMA samples!")

        return np.array(data), np.array(labels)


# ARMA Dataset Generator
class ARMA(Dataset):
    """ARMA dataset."""

    def __init__(self, num_samples: int = 1000, len_sample: int = 100, seed: int = 1, transform=None):
        """
        Class to generate and store a PyTorch Iteratble dataset w/ ARMA time series

        Args:
            num_samples: number of time series
            len_sample: length of time series
            seed: random seed for reproducability
            transform: how to transform data (often toTensor)
        """

        self.num_samples = num_samples
        self.len_sample = len_sample
        self.seed = seed
        self.data, self.labels = self.__make_data__()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx][:]
        label = self.labels[idx][:]

        # Transform if applicable
        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __make_data__(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        params = np.arange(-0.95, 0.95, 0.1)
        clip = int(self.len_sample / 10)

        data = []
        labels = []

        print("Started generating ARMA samples!")
        for i in range(self.num_samples):
            # Draw parameters
            ar = np.random.choice(params)
            ma = np.random.choice(params)
            labels.append([ar, ma])

            # Draw sample
            ar = np.r_[1, -ar]
            ma = np.r_[1, ma]
            ts = arma_generate_sample(ar=ar, ma=ma, nsample=self.len_sample, burnin=clip)
            data.append(ts)
        print("Finished generating ARMA samples!")

        return np.array(data), np.array(labels)


def load_arima_data(batch_size: int = 128, dgp="ARMA11", num_samples=250, len_sample=100, seed=1, transform=None):
    """
    Loads data into dataloader
    Returns data in batches of shape (128, 1, 28, 28)
    """

    if dgp == "ARMA11":
        print("Starting dataloading...")
        data = BasicARMA(num_samples=num_samples, len_sample=len_sample, seed=seed, transform=transform)
        dataloader = DataLoader(data, batch_size=batch_size)
        print('Finished dataloading!\n')
        return dataloader
    else:
        return "Wrong type!"


################################################################################
################################################################################

################################################################################
############################ Generate SIN ######################################
################################################################################

class Sines(Dataset):

    def __init__(self, frequency_range: Sequence[float], amplitude_range: Sequence[float],
                 num_samples: int = 200, len_sample: int = 100, seed: int = None):
        """
        Pytorch Dataset to produce sines.
        y = A * sin(B * x)
        :param frequency_range: range of A
        :param amplitude_range: range of B
        :param num_samples: number of sines in your dataset
        :param len_sample: length of each sample
        :param seed: random seed
        """
        self.num_samples = num_samples
        self.len_sample = len_sample
        self.seed = seed
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.dataset = self._generate_sines()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.dataset[idx], 1

    def _generate_sines(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(start=0, stop=2 * np.pi, num=self.len_sample)
        low_freq, up_freq = self.frequency_range[0], self.frequency_range[1]
        low_amp, up_amp = self.amplitude_range[0], self.amplitude_range[1]

        freq_vector = (up_freq - low_freq) * np.random.rand(self.num_samples, 1) + low_freq
        ampl_vector = (up_amp - low_amp) * np.random.rand(self.num_samples, 1) + low_amp

        return ampl_vector * np.sin(freq_vector * x)


def load_sin_data(batch_size: int = 128, num_samples=250, len_sample=100, seed=1):
    data = Sines(frequency_range=[0, 2 * np.pi], amplitude_range=[0, 2 * np.pi], seed=seed,
                 num_samples=num_samples, len_sample=len_sample)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


################################################################################
################################################################################

if __name__ == '__main__':
    print("Building test dataset...")
    test_dataloader = load_sin_data(batch_size=128)
    sample = next(iter(test_dataloader))
    print("Output shape: {}".format(sample.shape))
    print("Dataloader functional!\n")
    make_timeseries_plots(sample)

