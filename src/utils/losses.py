import torch
import torch.nn.functional as F

def step_CrossEntropyLoss(y_pred, y_true, std_dev):
    std_samples = torch.normal(mean=0.0, std=std_dev.expand_as(y_pred))
    distorted_loss = F.cross_entropy(y_pred + std_samples, y_true, reduction='mean')
    # print(distorted_loss)
    return torch.exp(distorted_loss)

def BNN_CrossEntropyLoss(y_pred, y_true, y_var):
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


