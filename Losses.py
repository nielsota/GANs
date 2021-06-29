import torch

################################################################################
########################## WASSERSTEIN LOSS W/ GP ## ###########################
################################################################################


# Compute gradient required for GP
def compute_gradient(critic, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(mixed_images)

    # grad_outputs: required when not a scalar output (mixed_score a [b,1,1,1] tensor)
    # Take the gradient of the scores with respect to the images

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradient


# Compute GP
def gradient_penalty(gradient):
    # Reshape images (flatten) to enable computation of norm
    # And since shape is [B, C, H , W] -> [B, -1]
    gradient = gradient.view(len(gradient), -1)

    # Dimension 0 is over rows (each col), dimension 1 is over cols (each row)
    # And since shape is [B, C, H , W]
    gradient_norm = gradient.norm(2, dim=1)

    # Estimate expected value
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


# Critic loss; learn W-distance
def critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = - torch.mean(crit_real_pred) + torch.mean(crit_fake_pred) + c_lambda * gp

    return crit_loss


# Discriminator loss
def generator_loss(crit_fake_pred):
    gen_loss = - torch.mean(crit_fake_pred)

    return gen_loss

################################################################################
################################################################################

