import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary


from .MedViT import MedViT_small as tiny
from .MedViT import MedViT_base as base
from .MedViT import MedViT_large as large

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = base()

model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=14, bias=True)

model = model.to(device)


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

# import medmnist
# from medmnist import INFO, Evaluator

# data_flag = 'breastmnist'
# data_flag = 'chestmnist'
# # [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, 
# # pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
# download = True

NUM_EPOCHS = 100
BATCH_SIZE = 32
lr = 0.005

# info = INFO[data_flag]
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])

# DataClass = getattr(medmnist, info['python_class'])


# from torchvision.transforms.transforms import Resize
# # preprocessing
# train_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.Lambda(lambda image: image.convert('RGB')),
#     torchvision.transforms.AugMix(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.Lambda(lambda image: image.convert('RGB')),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# # load the data
# train_dataset = DataClass(split='train', transform=train_transform, download=download)
# test_dataset = DataClass(split='test', transform=test_transform, download=download)

# # pil_dataset = DataClass(split='train', download=download)

# # encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# define loss function and optimizer
# if task == "multi-label, binary-class":
criterion = nn.BCEWithLogitsLoss()
# else:
#     criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

from src.dataloader.medical_dataset import NIHChestXrayDataset

train_dataset = NIHChestXrayDataset(metadata_file="data/nihcc_chest_xray/nihcc_chest_xray_training_samples.csv",
        image_dir="data/nihcc_chest_xray/xray_images/", 
        frac=1.00, isTest=False)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = NIHChestXrayDataset(metadata_file="data/nihcc_chest_xray/nihcc_chest_xray_testing_samples.csv",
        image_dir="data/nihcc_chest_xray/xray_images/", 
        frac=1.00, isTest=False)

test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
male_test_loader = data.DataLoader(test_dataset.filter_by_gender('male'), batch_size=32, shuffle=False, num_workers=4)
female_test_loader = data.DataLoader(test_dataset.filter_by_gender('female'), batch_size=32, shuffle=False, num_workers=4)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score, precision_score, recall_score

class MultiLabelEvaluator:
    def __init__(self, n_labels=14, threshold=0.5):
        self.n_labels = n_labels  # Number of labels per sample
        self.threshold = threshold  # Threshold for binary classification

    def apply_threshold(self, y_pred):
        """
        Apply threshold to the predicted probabilities to convert to binary labels.
        """
        return (y_pred >= self.threshold).astype(int)

    def evaluate(self, y_pred, y_true):
        """
        Evaluate model performance using various multi-label classification metrics.
        """
        # Apply threshold to convert probabilities to binary labels
        y_pred_bin = self.apply_threshold(y_pred)
        
        # Accuracy (Subset accuracy): exact match of all labels
        accuracy = accuracy_score(y_true, y_pred_bin)
        
        # Hamming loss: fraction of labels that are incorrectly predicted
        h_loss = hamming_loss(y_true, y_pred_bin)
        
        # Precision, Recall, F1 Score (Macro average across labels)
        precision = precision_score(y_true, y_pred_bin, average='macro')
        recall = recall_score(y_true, y_pred_bin, average='macro')
        f1 = f1_score(y_true, y_pred_bin, average='macro')
        
        # AUC-ROC (per label, then averaged)
        auc_roc = roc_auc_score(y_true, y_pred, average='macro')
        
        # Return all metrics as a dictionary
        return {
            "accuracy": accuracy,
            "hamming_loss": h_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        }


evaluator = MultiLabelEvaluator()
# evaluation
def test(model_pth=None, sensitive_group=None):
    if model_pth is not None:
        test_model = base()
        test_model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=14, bias=True)
        test_model.load_model(model_pth)
        test_model = test_model.to(device)

    test_model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    
    # data_loader = train_loader_at_eval if split == 'train' else test_loader
    if sensitive_group == 'male':
        data_loader = male_test_loader
    elif sensitive_group == 'female':
        data_loader = male_test_loader
    else:
        data_loader = test_loader
    num_batch = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = test_model(inputs)
            loss = criterion(outputs, targets)

            targets = targets.to(torch.float32)
            # outputs = outputs.softmax(dim=-1)
            outputs = torch.sigmoid(outputs)
            # outputs = (outputs > 0.5).float()

            print(f'batch: {num_batch}, loss: {loss}\n {targets[0]}, \n, {outputs[0]}')


            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)


        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        print(f'y_true.shape = {y_true.shape}')
        print(y_true[15])
        print(f'y_score.shape = {y_score.shape}')
        print(y_score[15])
        metrics = evaluator.evaluate(y_score, y_true)
        print(metrics)
        print(f"--------------------------Confusion matrix------------------------------:")
        for i in range(y_true.shape[1]):
            y_true_label = y_true[:, i]
            y_pred_label = y_score[:, i]
            
            cm = confusion_matrix(y_true_label, y_pred_label)
            print(f"Confusion matrix for label {i}:")
            print(cm)
        print(f"--------------------------F1 Score------------------------------:")
        f1 = f1_score(y_true, y_score, average='micro')
        print(f"F1 Score: {f1}")
        
        print(f"--------------------------AUC Score------------------------------:")
        # if len(np.unique(y_true)) == 2:
        #     auc = roc_auc_score(labels_binary, pred_sigmoid.detach().cpu().numpy(), multi_class='ovo')
        #     print(f"ROC AUC: {auc}")
        # else:
        #     print("Only one class present in y_true. Skipping AUC calculation.")

        # evaluator = Evaluator(data_flag, split)
        # metrics = evaluator.evaluate(y_score)
    
        # print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

# train
def train():
    loss_log = []
    epoch_log = []
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        print('Epoch [%d/%d]'% (epoch+1, NUM_EPOCHS))
        model.train()
        num_batch = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            
            loss_log.append(loss.item())
            epoch_log.append(epoch)
            print(f'epoch {epoch}, batch {num_batch}: loss {loss.item()}')
            loss.backward()
            optimizer.step()
            num_batch += 1
        
        # test('test')
        import pandas as pd
        df = pd.DataFrame({
            'epoch': epoch_log,
            'loss': loss_log
        })
        df.to_csv('outputs/medvit_base/train_log_medvit_base.csv')
        torch.save(model.state_dict(), f'outputs/medvit_base/medvit_mnist__base_wt{epoch}.pt')
        print('model saved')



# train()

        
# print('==> Evaluating ...')
# test('train')
# test('test')

