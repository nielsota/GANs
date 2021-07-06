from Losses import *
from Data import *
from tqdm.auto import tqdm


def trainloop(gen, disc, gen_opt, disc_opt, noise_dim, dataloader,
              c_lambda, crit_repeats, disc_losses, generator_losses, device='cpu'):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device).float()

        ### Update critic ###
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            # Zero out gradients
            disc_opt.zero_grad()

            # Generate sample and run generated and real through disc
            fake_noise = get_noise(cur_batch_size, noise_dim, device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_real_pred = disc(real)

            # Compute loss with gradient penalty
            # epsilon should have shape [B,1], then broadcasted to every time series
            epsilon = torch.rand(len(real), 1, device=device, requires_grad=True)
            gradient = compute_gradient(disc, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            disc_loss = critic_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += disc_loss.item() / crit_repeats

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

        # Save losses
        disc_losses += [mean_iteration_critic_loss]
        ### End Update critic ###

        ### Update generator ###

        # Zero out old gradients
        gen_opt.zero_grad()

        # Generate sample and run through disc
        fake_noise_2 = get_noise(cur_batch_size, noise_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)

        # Compute loss
        gen_loss = generator_loss(disc_fake_pred)

        # Compute gradients using back prop
        gen_loss.backward()

        # Take a step
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]
        ### End Update Generator ###
