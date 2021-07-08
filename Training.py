from Losses import *
from Data import *
from tqdm.auto import tqdm


def trainloop(gen, disc, gen_opt, disc_opt, noise_dim, dataloader,
              c_lambda, crit_repeats, disc_losses, generator_losses,
              conditional=False, device='cpu'):
    for real, labels in tqdm(dataloader):

        # Retrieve batch size, real data, and labels
        cur_batch_size = len(real)
        real = real.to(device).float()
        labels = labels.to(device).float()

        # Update critic
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):

            # Zero out gradients
            disc_opt.zero_grad()

            # Generate sample and run generated and real through disc
            fake_noise = get_noise(cur_batch_size, noise_dim, device=device)

            # Combine with labels if conditional GAN
            if conditional:
                fake_noise = combine_noise_and_labels(fake_noise, labels)

            # Input noise into generator
            fake = gen(fake_noise)

            # Combine fakes with labels if conditional GAN
            if conditional:
                fake = combine_noise_and_labels(fake, labels)
                real = combine_noise_and_labels(real, labels)

            # Input fake and real into discriminator
            disc_fake_pred = disc(fake.detach())
            disc_real_pred = disc(real)

            # Set correct shape for epsilon
            epsilon_shape = [len(real), 1]
            if conditional:
                epsilon_shape = [len(real), 1, 1]

            # Generate random epsilon in correct shape
            epsilon = torch.rand(*epsilon_shape, device=device, requires_grad=True)

            # Compute gradient wrt mixed images
            gradient = compute_gradient(disc, real, fake.detach(), epsilon)

            # Compute gradient penalty
            gp = gradient_penalty(gradient)

            # Reset real to only first channel (contains TS)
            if conditional:
                real = real[:, 0, :]

            # Compute loss
            disc_loss = critic_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += disc_loss.item() / crit_repeats

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

        # Save losses
        disc_losses += [mean_iteration_critic_loss]
        # End Update critic

        # Update generator

        # Zero out old gradients
        gen_opt.zero_grad()

        # Generate noise sample
        fake_noise_2 = get_noise(cur_batch_size, noise_dim, device=device)

        # combine noise with labels if conditional GAN
        if conditional:
            fake_noise_2 = combine_noise_and_labels(fake_noise_2, labels)

        # input noise into generator
        fake_2 = gen(fake_noise_2)

        # combine fakes with labels if conditional GAN
        if conditional:
            fake_2 = combine_noise_and_labels(fake_2, labels)

        # input fakes to discriminator
        disc_fake_pred = disc(fake_2)

        # Compute loss
        gen_loss = generator_loss(disc_fake_pred)

        # Compute gradients using back prop
        gen_loss.backward()

        # Take a step
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]
        # End Update Generator
