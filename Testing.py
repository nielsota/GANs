from Data import *
from utils import *
import seaborn as sns
import os
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)


def plot_timeseries(disc, gen, args, save_name, path,
                    conditional=False, device='cpu'):
    # Generate fake noise
    fake_noise = get_noise(args.batch_size, args.noise_dim, device=device)

    # Add labels to array if conditional GAN
    if conditional:
        params = np.arange(-0.95, 0.95, 0.1)
        labels = []
        for i in range(args.batch_size):
            # Draw parameters
            ar = np.random.choice(params)
            ma = np.random.choice(params)
            labels.append([ar, ma])
        labels = torch.tensor(labels)
        fake_noise = combine_noise_and_labels(fake_noise, labels)

    # Generate samples
    fake = gen(fake_noise)

    # Plot samples
    fig = make_timeseries_plots(fake)

    # Save samples
    fig.savefig(os.path.join(path, save_name))


def parameter_distribution(disc, gen, args, model_type, data_type,
                           path, conditional=False, device='cpu'):
    print("Started generating parameter distribution")
    save_name = model_type + '_' + data_type + '_parameter_distribution'

    # Set to target values
    ar_target = 0.5
    ma_target = 0.5

    # Number of testing series
    testing_series = 500

    # Generate fake noise
    fake_noise = get_noise(testing_series, args.noise_dim, device=device)

    # Append labels as channel to noise if this is variable arma
    if conditional:
        labels = torch.ones([testing_series, 2])
        labels[:, 0] = labels[:, 0] * ar_target
        labels[:, 1] = labels[:, 1] * ma_target
        fake_noise = combine_noise_and_labels(fake_noise, labels)

    # Generate fake examples
    fake = gen(fake_noise)
    fake = fake.cpu().detach().numpy()

    # Estimate and store ar and ma parameters
    ar = []
    ma = []
    for i in range(len(fake)):
        mod = ARIMA(fake[i, :], order=(1, 0, 1))
        res = mod.fit(return_params=True)
        ar.append(res[1])
        ma.append(res[2])

    # Plot estimates in graph
    f, axis = plt.subplots()
    print(" Finished generating paramter distribution")
    print('mean AR parameter estimates: {}'.format(np.mean(np.array(ar))))
    print('mean MA parameter estimates: {}'.format(np.mean(np.array(ma))))
    print('std AR parameter estimates: {}'.format(np.std(np.array(ar))))
    print('std MA parameter estimates: {}'.format(np.std(np.array(ma))))
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.axes_style("whitegrid")
    # Draw the density plot
    sns.distplot(ar, hist=True, kde=True,
                 kde_kws={'linewidth': 1, 'shade': True},
                 label='AR', ax=axis)
    # Draw the density plot
    sns.distplot(ma, hist=True, kde=True,
                 kde_kws={'linewidth': 1, 'shade': True},
                 label='MA', ax=axis)

    plt.axvline(ar_target, 0, 10, linestyle='-')
    plt.axvline(ma_target, 0, 10, linestyle='-')
    # plt.axvline(np.mean(AR), 0, max(AR), linestyle = '-')
    # plt.axvline(np.mean(MA), 0, max(MA))
    # Plot formatting
    plt.legend(prop={'size': 10}, title='Parameter')
    plt.title('Density Plot AR and MA parameters')
    plt.xlabel('Value')
    plt.ylabel('Density')
    f.set_size_inches(11.7, 8.27)
    print(os.path.join(path, save_name))
    f.savefig(os.path.join(path, save_name))
    plt.show()


def parameter_heatmap(disc, gen, args, model_type, data_type, path,
                      device='cpu'):
    # Create savenames
    save_name_ma = model_type + '_' + data_type + '_parameter_heatmap_ma'
    save_name_ar = model_type + '_' + data_type + '_parameter_heatmap_ar'
    save_name = model_type + '_' + data_type + '_parameter_heatmap'

    # Create space of parameters to test
    MA_range = np.arange(-0.95, 0.95, 0.1)
    AR_range = np.arange(-0.95, 0.95, 0.1)
    TESTING_SERIES = 128

    # Create matrices to save biases in
    map_ar = np.zeros([len(MA_range), len(AR_range)])
    map_ma = np.zeros([len(MA_range), len(AR_range)])

    # One column at a time
    for i, ar in enumerate(AR_range):

        # One row at a time
        for j, ma in enumerate(MA_range):

            # Generate noise
            fake_noise = get_noise(TESTING_SERIES, args.noise_dim, device=device)

            # Generate labels
            labels = torch.ones([TESTING_SERIES, 2])
            labels[:, 0] = labels[:, 0] * ar
            labels[:, 1] = labels[:, 1] * ma

            # Append labels to noise
            fake_noise = combine_noise_and_labels(fake_noise, labels)
            fake = gen(fake_noise)
            fake = fake.cpu().detach().numpy()

            # Do testing (n=TESTING_SERIES)
            AR = []
            MA = []
            for k in range(len(fake)):
                mod = ARIMA(fake[k, :], order=(1, 0, 1))
                res = mod.fit(return_params=True)
                AR.append(res[1])
                MA.append(res[2])

            # Compute mean estimate
            mean_ar = np.mean(AR)
            mean_ma = np.mean(MA)

            # Compute and store bias, positive mean overestimated
            map_ar[i, j] = (mean_ar - ar)
            map_ma[i, j] = (mean_ma - ma)

    # Convert to dataframe
    map_ar_df = pd.DataFrame(map_ar, columns=np.round(AR_range, 2), index=np.round(MA_range, 2))
    map_ma_df = pd.DataFrame(map_ma, columns=np.round(AR_range, 2), index=np.round(MA_range, 2))

    # Create colorpalette
    customPalette = sns.diverging_palette(230, 30, as_cmap=True)

    # Plot heatmap MA
    sns.heatmap(map_ma_df, cmap=customPalette)
    plt.title('Heatmap MA parameter bias')
    plt.xlabel('AR Parameter Value')
    plt.ylabel('MA Parameter Value')
    plt.savefig(os.path.join(path, save_name_ma))
    plt.show()

    # Plot heatmap AR
    sns.heatmap(map_ar_df, cmap=customPalette)
    plt.title('Heatmap AR parameter bias')
    plt.xlabel('AR Parameter Value')
    plt.ylabel('MA Parameter Value')
    plt.savefig(os.path.join(path, save_name_ar))
    plt.show()

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(map_ma_df, ax=axes[0], cmap=customPalette)
    axes[0].set_title('MA Parameter Bias')
    sns.heatmap(map_ar_df, ax=axes[1], cmap=customPalette)
    axes[1].set_title('AR Parameter Bias')
    plt.savefig(os.path.join(path, save_name))
    plt.show()


def residual_diagnostics(disc, gen, args, model_type, data_type, path,
                         conditional=False, device='cpu'):
    # Save name and checkpoint
    save_name = model_type + '_' + data_type + '_parameter_distribution'
    print("Started generating residual diagnostics")

    # Set to target values
    ar_target = 0.5
    ma_target = 0.5

    # Number of testing series
    testing_series = 2500

    # Generate fake noise
    fake_noise = get_noise(testing_series, args.noise_dim, device=device)

    # Append labels as channel to noise if this is variable arma
    if conditional:
        labels = torch.ones([testing_series, 2])
        labels[:, 0] = labels[:, 0] * ar_target
        labels[:, 1] = labels[:, 1] * ma_target
        fake_noise = combine_noise_and_labels(fake_noise, labels)

    # Generate fake examples
    fake = gen(fake_noise)
    fake = fake.cpu().detach().numpy()

    # Create arrays to store test statistic p-values
    jb = []
    lb_1 = []
    #lb_2 = []
    het = []

    # Estimate model and compute residual diagnostics for generated series
    for i in range(len(fake)):
        mod = ARIMA(fake[i, :], order=(1, 0, 1))
        res = mod.fit()
        #print(durbin_watson(res.resid))
        #print(acorr_breusch_godfrey(res))


        #lb_temp = np.squeeze((res.test_serial_correlation('ljungbox', lags=1)))
        lb_temp = np.squeeze(acorr_breusch_godfrey(res))
        jb_temp = np.squeeze(res.test_normality('jarquebera'))
        het_temp = res.test_heteroskedasticity('breakvar')

        #print(acorr_ljungbox(res.resid, lags=[i+1 for i in range(10)], return_df=True))

        # Jarque-Bera P-value
        jb.append(jb_temp[1])

        # LB lag 1 p-value
        #lb_1.append(lb_temp[1])
        lb_1.append(lb_temp[1])

        # LB lag 2 p-value
        #lb_2.append(lb_temp[1, 1])

        # Heteroskedasticity test p-value
        het.append(het_temp[0, 1])

    print("--------------------------------\n")
    print("% H rejected: " + str(sum([i < 0.05 * 3 for i in het]) / testing_series))
    print("% N rejected: " + str(sum([i < 0.05 * 3 for i in jb]) / testing_series))
    print("% S1 rejected: " + str(sum([i < 0.05 * 3 for i in lb_1]) / testing_series))
    #print("% S2 rejected: " + str(sum([i < 0.05 for i in lb_2]) / testing_series))
    print("\n--------------------------------\n")


################################################################################
############################## MAIN FUNCTION  ##################################
################################################################################


if __name__ == '__main__':
    print("models loading...")
    # save name example models/sin_generator
    models_path = os.path.join(str(os.getcwd()), 'fitted_models')
    figures_path = os.path.join(str(os.getcwd()), 'figures')
    gen = torch.load(os.path.join(models_path, '1_D_Conv' + '_' + 'arma_11_fixed' + '_generator.pth'))
    disc = torch.load(os.path.join(models_path, '1_D_Conv' + '_' + 'arma_11_fixed' + '_discriminator.pth'))

    parser = argparse.ArgumentParser(description=" Load arguments for script ")
    parser.add_argument('-nd', '--noise_dim', help='Noise dimension',
                        type=int, default=100)
    args = parser.parse_args()

    # parameter_heatmap(disc, gen, args,
    #                  device='cpu',
    #                  model_type='1_D_Conv',
    #                  data_type='arma_11_variable',
    #                  path=figures_path)

    residual_diagnostics(disc, gen, args,
                         device='cpu',
                         model_type='1_D_Conv',
                         data_type='arma_11_fixed',
                         path=figures_path,
                         conditional=False)
