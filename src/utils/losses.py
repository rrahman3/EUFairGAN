import torch
import torch.nn.functional as F

def step_CrossEntropyLoss(y_pred, y_true, std_dev):
    std_samples = torch.normal(mean=0.0, std=std_dev.expand_as(y_pred))
    distorted_loss = F.cross_entropy(y_pred + std_samples, y_true, reduction='mean')
    # print(distorted_loss)
    return torch.exp(distorted_loss)

def BNN_CrossEntropyLoss(y_pred, y_true, y_var):
    # y_true = y_true.argmax(dim=1)
    std_dev = torch.sqrt(y_var)
    losses = [step_CrossEntropyLoss(y_pred, y_true, std_dev) for _ in range(100)]
    variance_loss = torch.mean(torch.stack(losses), dim=0)
    return torch.mean(torch.log(variance_loss))


def step_BCEWithLogitsLoss(y_pred, y_true, std_dev):
    std_samples = torch.normal(mean=0.0, std=std_dev.expand_as(y_pred))
    distorted_loss = F.binary_cross_entropy_with_logits(y_pred + std_samples, y_true, reduction='mean')
    return torch.exp(distorted_loss)

def BNN_BCEWithLogitsLoss(y_pred, y_true, y_var):
    std_dev = torch.sqrt(y_var)
    losses = [step_BCEWithLogitsLoss(y_pred, y_true, std_dev) for _ in range(100)]
    variance_loss = torch.mean(torch.stack(losses), dim=0)
    return torch.mean(torch.log(variance_loss))


def heteroscedastic_loss(y_pred, y_true, log_var):
    """
    Computes the heteroscedastic regression loss.

    Args:
        y_pred (Tensor): Predicted values (shape: [batch_size, 1]).
        y_true (Tensor): Ground truth values (shape: [batch_size, 1]).
        log_var (Tensor): Predicted log variance (shape: [batch_size, 1]).

    Returns:
        Tensor: The computed loss (a scalar).
    """
    # Compute the precision as the exponential of the negative log variance.
    precision = torch.exp(-log_var)
    
    # Compute the loss per sample: (1/2) * precision * squared_error + (1/2) * log_var.
    loss = 0.5 * precision * (y_true - y_pred) ** 2 + 0.5 * log_var
    
    # Return the mean loss over the batch.
    return loss.mean()

def discriminator_hinge_loss(D_real, D_fake):
    """
    Computes the hinge loss for the discriminator.

    Args:
    - D_real (torch.Tensor): Discriminator output on real images.
    - D_fake (torch.Tensor): Discriminator output on fake images.

    Returns:
    - torch.Tensor: Total discriminator loss.
    """
    loss_real = torch.mean(torch.relu(1.0 - D_real))  # max(0, 1 - D(x_real))
    loss_fake = torch.mean(torch.relu(1.0 + D_fake))  # max(0, 1 + D(G(z)))
    return loss_real + loss_fake  # Total loss for D

def generator_hinge_loss(D_fake):
    """
    Computes the hinge loss for the generator.

    Args:
    - D_fake (torch.Tensor): Discriminator output on fake images.

    Returns:
    - torch.Tensor: Generator loss.
    """
    return -torch.mean(D_fake)  # -D(G(z))