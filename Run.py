from models.MLP_GANs import *
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

    # Load data into dataloader
    if data_type == 'sin':
        dataloader = load_sin_data(batch_size=batch_size, num_samples=num_samples, len_sample=len_samples)
    elif data_type == 'arma_11_fixed':
        dataloader = load_arima_data(batch_size=batch_size, num_samples=num_samples, len_sample=len_samples)
    else:
        sys.exit('invalid data type: choose sin or arma')

    # Create generator, discriminator and their optimizers
    gen = MLPGenerator(z_dim=noise_dim, len_sample=len_samples, hid_dim=128)
    disc = MLPDiscriminator(len_sample=len_samples)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # Store variables for diagnostics
    generator_losses = []
    critic_losses = []

    for i in range(n_epochs):
        print("--------------- starting epoch {} --------------- \n".format(str(i)))
        trainloop(gen, disc, gen_opt, disc_opt, noise_dim, dataloader,
                  c_lambda, crit_repeats, critic_losses, generator_losses, device='cpu')
        print("--------------- finished epoch {} --------------- \n".format(str(i)))

    # Save model

    generator_path = os.path.join(path, data_type + 'generator.pth')
    disc_path = os.path.join(path, data_type + 'discriminator.pth')
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
                        type=int, default=25)
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
                        type=int, default=10000)
    parser.add_argument('-ls', '--len_samples', help='length samples',
                        type=int, default=100)
    parser.add_argument('-dt', '--data_type', help='ARMA or sin',
                        type=str, default="sin")
    parser.add_argument('-r', '--run', help='Run',
                        type=str2bool, default=True)
    # parse arguments
    args = parser.parse_args()
    print(args)

    runbool = args.run
    print(runbool)
    testbool = True

    # make models directory
    parent_dir = os.getcwd()
    directory = 'models'
    models_path = os.path.join(str(parent_dir), directory)
    if not os.path.exists(models_path):
        os.mkdir(models_path)
        print("Directory '% s' created" % directory)
    else:
        pass

    if runbool:
        run(models_path, args)

    # make figures directory
    parent_dir = os.getcwd()
    directory = 'figures'
    figures_path = os.path.join(str(parent_dir), directory)
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
        print("Directory '% s' created" % directory)
    else:
        pass

    if testbool:
        gen = torch.load(os.path.join(models_path, args.data_type + 'generator.pth'))
        disc = torch.load(os.path.join(models_path, args.data_type + 'discriminator.pth'))
        plot_timeseries(disc, gen, args, save_name=args.data_type + '_timeseries_example',
                        path=figures_path, device='cpu')
        #parameter_distribution(disc, gen, args, device='cpu',
        #                       save_name=args.data_type + '_parameter_distribution', path=figures_path)

    ################################################################################
    ################################################################################
