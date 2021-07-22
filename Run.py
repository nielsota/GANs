from models.MLP_GANs import *
from models.CausalConvGan import *
from Training import *
from Testing import *
import sys


def run(path, args):
    n_epochs = args.n_epochs
    noise_dim = args.noise_dim
    batch_size = args.batch_size
    lr = args.lr
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    c_lambda = args.c_lambda
    crit_repeats = args.crit_repeats
    num_samples = args.num_samples
    len_samples = args.len_samples
    data_type = args.data_type
    model_type = args.model_type
    conditional = False

    # Load data into dataloader
    if data_type == 'sin':
        print('Using sin data')
        dataloader = load_sin_data(batch_size=batch_size, num_samples=num_samples,
                                   len_sample=len_samples)
    elif data_type == 'arma_11_fixed':
        print('Using ARMA(1,1) data w/ fixed parameters')
        dataloader = load_arima_data(batch_size=batch_size, num_samples=num_samples,
                                     len_sample=len_samples, dgp=data_type)
    elif data_type == 'arma_11_variable':
        print('Using ARMA(1,1) data w/ variable parameters')
        dataloader = load_arima_data(batch_size=batch_size, num_samples=num_samples,
                                     len_sample=len_samples, dgp=data_type)
        conditional = True
    else:
        sys.exit('invalid data type: choose sin or arma')

    # Create generator, discriminator and their optimizers
    if model_type == 'MLP':
        print('Using MLP models')
        gen = MLPGenerator(z_dim=noise_dim, len_sample=len_samples, hid_dim=128)
        disc = MLPDiscriminator(len_sample=len_samples)
    elif model_type == '1_D_Conv':
        _, y = next(iter(dataloader))
        num_labels = y.shape[1]
        print('Using 1-D Convolution models')
        gen = ConvGenerator(in_channels=1 + num_labels, z_dim=noise_dim, len_sample=100)
        disc = ConvDiscriminator(in_channels=1 + num_labels)
    else:
        sys.exit('invalid model type: choose MLP or 1_D_Conv')

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # Store variables for diagnostics
    generator_losses = []
    critic_losses = []

    for i in range(n_epochs):
        print("--------------- starting epoch {} --------------- \n".format(str(i + 1)))
        trainloop(gen, disc, gen_opt, disc_opt, noise_dim, dataloader,
                  c_lambda, crit_repeats, critic_losses, generator_losses,
                  conditional=conditional, device='cpu')
        print("--------------- finished epoch {} --------------- \n".format(str(i + 1)))

    # Save model
    generator_path = os.path.join(path, model_type + '_' + data_type + '_generator.pth')
    disc_path = os.path.join(path, model_type + '_' + data_type + '_discriminator.pth')
    print("--------------- models saved --------------- \n")

    torch.save(gen, generator_path)
    torch.save(disc, disc_path)


################################################################################
############################## MAIN FUNCTION  ##################################
################################################################################


if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description=" Load arguments for script ")

    # Instantiate arguments to be passed to parser
    # '--' makes argument not matter
    parser.add_argument('-e', '--n_epochs', help='Training epochs',
                        type=int, default=100)
    parser.add_argument('-nd', '--noise_dim', help='Noise dimension',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='Batch size',
                        type=int, default=128)
    parser.add_argument('-lr', '--lr', help='Learning rate',
                        type=float, default=0.0001)
    parser.add_argument('-b1', '--beta_1', help='Beta 1',
                        type=float, default=0.5)
    parser.add_argument('-b2', '--beta_2', help='Beta 2',
                        type=float, default=0.999)
    parser.add_argument('-cl', '--c_lambda', help='c lambda',
                        type=float, default=10)
    parser.add_argument('-cr', '--crit_repeats', help='Crit repeats',
                        type=int, default=5)
    parser.add_argument('-ns', '--num_samples', help='num_samples',
                        type=int, default=25000)
    parser.add_argument('-ls', '--len_samples', help='length samples',
                        type=int, default=100)
    parser.add_argument('-dt', '--data_type', help='ARMA or sin',
                        type=str, default="sin")
    parser.add_argument('-mt', '--model_type', help='MLP or 1_D_Conv',
                        type=str, default="MLP")
    parser.add_argument('-r', '--run', help='Run',
                        type=str2bool, default=True)
    parser.add_argument('-t', '--test', help='Test',
                        type=str2bool, default=True)
    # parse arguments
    args = parser.parse_args()
    # print(args)

    # make models directory
    makedirectory('fitted_models')

    # make figures directory
    makedirectory('figures')

    # specify save paths
    models_path = os.path.join(str(os.getcwd()), 'fitted_models')
    figures_path = os.path.join(str(os.getcwd()), 'figures')

    runbool = args.run
    testbool = args.test

    if runbool:
        run(models_path, args)

    if testbool:

        print("models loading...")
        # save name example models/sin_generator
        gen = torch.load(os.path.join(models_path, args.model_type + '_' + args.data_type + '_generator.pth'))
        disc = torch.load(os.path.join(models_path, args.model_type + '_' + args.data_type + '_discriminator.pth'))
        print("generating samples...")


        if args.data_type == 'arma_11_fixed':
            print("starting plotting samples...")
            #plot_timeseries(disc,
            #                gen,
            #                args,
            #                save_name=args.model_type + '_' + args.data_type + '_timeseries_example',
            #                path=figures_path,
            #                device='cpu',
            #                conditional=False)

            print("finished plotting samples...")

            parameter_distribution(disc, gen, args,
                                   device='cpu',
                                   model_type=args.model_type,
                                   data_type=args.data_type,
                                   conditional=False,
                                   path=figures_path)

        if args.data_type == 'arma_11_variable':
            plot_timeseries(disc,
                            gen,
                            args,
                            save_name=args.model_type + '_' + args.data_type + '_timeseries_example',
                            path=figures_path,
                            device='cpu',
                            conditional=True)

            #parameter_distribution(disc, gen, args,
             ##                      device='cpu',
              #                     model_type=args.model_type,
              #                     data_type=args.data_type,
              #                     conditional=True,
              #                     path=figures_path)
            parameter_heatmap(disc,
                              gen,
                              args,
                              device='cpu',
                              model_type=args.model_type,
                              data_type=args.data_type,
                              conditional=True,
                              path=figures_path)
    ################################################################################
    ################################################################################
