from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import os

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
    plt.show()


# Show time series plots
def make_timeseries_plots(time_series_tensor, num_plots: int = 10):
    fig, axs = plt.subplots(num_plots)
    fig.set_size_inches(18.5, num_plots * 2)
    colors = ["blue", "red", "green", "purple"]
    # fig.suptitle('Time series plot', fontsize=12)
    for i in range(num_plots):
        axs[i].plot(time_series_tensor.detach().view(len(time_series_tensor), -1)[i], color=colors[i % len(colors)])
        axs[i].grid()
    plt.show()
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
################################################################################
################################################################################

