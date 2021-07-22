from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import os
import torch
from Data import *

################################################################################
############################## UTILITY FUNCTIONS ###############################
################################################################################


# Show images
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    # The asterisk (*) can be used in python to unpack a list into its individual elements,
    # thus passing to view the correct form of input arguments it expects.
    # .detach(): cannot call view on a variable that requires grad
    # .cpu() because stores in CUDA

    image_tensor_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_tensor_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    #plt.show()


# Show time series plots
def make_timeseries_plots(time_series_tensor, num_plots: int = 10):
    fig, axs = plt.subplots(num_plots)
    fig.set_size_inches(18.5, num_plots * 2)
    colors = ["blue", "red", "green", "purple"]
    # fig.suptitle('Time series plot', fontsize=12)
    for i in range(num_plots):
        axs[i].plot(time_series_tensor.detach().view(len(time_series_tensor), -1)[i], color=colors[i % len(colors)])
        axs[i].grid()
    return fig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def makedirectory(dir_name):
    parent_dir = os.getcwd()
    directory = dir_name
    models_path = os.path.join(str(parent_dir), directory)
    if not os.path.exists(models_path):
        os.mkdir(models_path)
        print("Directory '% s' created" % directory)
    else:
        pass


def combine_vectors(z, y):
    return torch.cat((z.float(),y.float()), 1)


def combine_noise_and_labels(data, labels):
    """
    Combine [32,100] and [32,C] into [32,C+1,100]
    where each element [32,C] is repeated over entire channel
    """
    # shape -> [32, 6 , 1]
    labels = labels[:, :, None]

    # shape -> [32, 6 , 100]
    repeated_labels = labels.repeat(1, 1, 100)

    # Combine; data[:, None, :] has shape [32, 1, 100]
    data_and_labels = combine_vectors(data[:, None, :], repeated_labels)

    return data_and_labels


################################################################################
################################################################################


if __name__ == '__main__':
    print("Building test dataset...")
    test_dataloader = load_arima_data(batch_size=128, dgp = 'arma_11_variable')
    X, y = next(iter(test_dataloader))
    combined = combine_noise_and_labels(X, y)
    print("Output shape: {}".format(X.shape))
    print("Labels shape: {}".format(y.shape))
    print("Combined shape: {}".format(combined.shape))
    print(combined[:, 0, :].shape)
    print(y.shape[1])
    make_timeseries_plots(combined[:, 0, :])
