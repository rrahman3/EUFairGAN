import torch
import torch.nn.functional as F

def cross_entropy(y_true, y_pred, std_dev):
    std_samples = torch.normal(mean=0.0, std=std_dev.expand_as(y_pred))
    distorted_loss = F.cross_entropy(y_pred + std_samples, y_true, reduction='mean')
    # print(distorted_loss)
    return torch.exp(distorted_loss)

def crossentropy_loss(y_true, y_pred, y_var):
    # print(f'loss')
    # print(f'loss: {y_true.shape}, {y_pred.shape}, {y_var.shape}')
    # print(f'loss: {y_true.dtype}, {y_pred.dtype}, {y_var.dtype}')
    std_dev = torch.sqrt(y_var)
    
    losses = [cross_entropy(y_true, y_pred, std_dev) for _ in range(100)]
    variance_loss = torch.mean(torch.stack(losses), dim=0)
    # print(variance_loss)
    # return torch.mean(variance_loss)

    return torch.mean(torch.log(variance_loss))
    # std_samples = torch.normal(mean=0.0, std=std_dev.expand_as(y_pred))
    # distorted_loss = F.cross_entropy(y_pred, y_true + std_samples, reduction='mean')
    # return torch.exp(distorted_loss)



