from Data import *
from utils import *
import seaborn as sns
import os


def plot_timeseries(disc, gen, args, device='cpu'):
    fake_noise = get_noise(args.batch_size, args.noise_dim, device=device)
    fake = gen(fake_noise)
    make_timeseries_plots(fake)


def parameter_distribution(disc, gen, args, save_name, path, device='cpu'):

    TESTING_SERIES = 1000

    fake_noise = get_noise(TESTING_SERIES, args.noise_dim, device=device)
    fake = gen(fake_noise)
    fake = fake.cpu().detach().numpy()
    AR = []
    MA = []
    for i in range(len(fake)):
        mod = ARIMA(fake[i, :], order=(1, 0, 1))
        res = mod.fit(return_params=True)
        AR.append(res[1])
        MA.append(res[2])

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.axes_style("whitegrid")

    # Draw the density plot
    sns.distplot(AR, hist=True, kde=True,
                 kde_kws={'linewidth': 1, 'shade': True},
                 label='AR')
    # Draw the density plot
    sns.distplot(MA, hist=True, kde=True,
                 kde_kws={'linewidth': 1, 'shade': True},
                 label='MA')

    plt.axvline(0.5, 0, 10, linestyle='-')
    # plt.axvline(np.mean(AR), 0, max(AR), linestyle = '-')
    # plt.axvline(np.mean(MA), 0, max(MA))
    # Plot formatting
    plt.legend(prop={'size': 10}, title='Parameter')
    plt.title('Density Plot AR and MA paramters')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(path, save_name))
    plt.show()
