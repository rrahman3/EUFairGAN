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


from .BNN_MedViT import BNN_MedViT_small as tiny
from .BNN_MedViT import BNN_MedViT_base as base
from .BNN_MedViT import BNN_MedViT_large as large

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_pth = 'outputs/bnn_medvit_base_nihcccheset/medvit_mnist__base_wt74.pt'
model = base()
model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=14, bias=True)
model.load_model(model_pth)
model = model.to(device)


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

NUM_EPOCHS = 100
BATCH_SIZE = 32
lr = 0.005


from src.utils.losses import BNN_BCEWithLogitsLoss
criterion = BNN_BCEWithLogitsLoss
# else:
#     criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# from src.dataloader.medical_dataset import NIHChestXrayDataset

# train_dataset = NIHChestXrayDataset(metadata_file="data/nihcc_chest_xray/nihcc_chest_xray_training_samples.csv",
#         image_dir="data/nihcc_chest_xray/xray_images/", 
#         frac=1.0, isTest=False)
# train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# validation_dataset = NIHChestXrayDataset(metadata_file="data/nihcc_chest_xray/nihcc_chest_xray_validation_samples.csv",
#         image_dir="data/nihcc_chest_xray/xray_images/", 
#         frac=1.0, isTest=False)
# validation_loader = data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)

# test_dataset = NIHChestXrayDataset(metadata_file="data/nihcc_chest_xray/nihcc_chest_xray_testing_samples.csv",
#         image_dir="data/nihcc_chest_xray/xray_images/", 
#         frac=1.0, isTest=True)

from src.dataloader.chexpert_dataset import CheXpertDataset
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)

train_dataset = CheXpertDataset(metadata_file="data/CheXpert-v1.0/train.csv",
        image_dir="data/", 
        frac=0.5, isTest=False)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

validation_dataset = CheXpertDataset(metadata_file="data/CheXpert-v1.0/valid.csv",
        image_dir="data/", 
        frac=1.0, isTest=False)
validation_loader = data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

test_dataset = CheXpertDataset(metadata_file="data/CheXpert-v1.0/valid.csv",
        image_dir="data/", 
        frac=1.0, isTest=True)

age_threshold = 65
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
# male_test_loader = data.DataLoader(test_dataset.filter_by_gender('male'), batch_size=32, shuffle=False, num_workers=4)
# female_test_loader = data.DataLoader(test_dataset.filter_by_gender('female'), batch_size=32, shuffle=False, num_workers=4)
male_test_loader = data.DataLoader(test_dataset.filter_by_NIH_age(age_threshold, True), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
female_test_loader = data.DataLoader(test_dataset.filter_by_NIH_age(age_threshold, False), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

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

        # y_true = y_true.squeeze()
        # y_pred = y_pred.squeeze()

        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pred_bin[:, label])
            acc += label_acc
        accuracy = acc / y_true.shape[1]
            
        
        # Hamming loss: fraction of labels that are incorrectly predicted
        h_loss = hamming_loss(y_true, y_pred_bin)
        
        # Precision, Recall, F1 Score (Macro average across labels)
        precision = precision_score(y_true, y_pred_bin, average='macro')
        recall = recall_score(y_true, y_pred_bin, average='macro')
        f1 = f1_score(y_true, y_pred_bin, average='macro')
        
        # AUC-ROC (per label, then averaged)
        auc = 0
        # for i in range(y_true.shape[1]):
        #     label_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        #     auc += label_auc
        auc_roc = auc / y_true.shape[1]
        # auc_roc = roc_auc_score(y_true, y_pred, average='macro')

        
        print(f"--------------------------Confusion matrix------------------------------:")
        for i in range(y_true.shape[1]):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            y_pred_label = (y_pred_label >= 0.5).astype('float')
            
            cm = confusion_matrix(y_true_label, y_pred_label)
            print(f"Confusion matrix for label {i}: \n{cm}")
        
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
def test(test_model=None, model_pth=None, sensitive_group=None):
    if test_model == None and model_pth == None:
        print("---------------------------------------------------Fuck You------------------------------------------------")
        exit(1)

    if model_pth is not None:
        test_model = base()
        test_model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=14, bias=True)
        test_model.load_model(model_pth)
        test_model = test_model.to(device)

    test_model.train()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    variance_log = torch.tensor([]).to(device)
    
    # data_loader = train_loader_at_eval if split == 'train' else test_loader
    if sensitive_group == 'male':
        data_loader = male_test_loader
    elif sensitive_group == 'female':
        data_loader = female_test_loader
    elif sensitive_group == 'val':
        data_loader = test_loader
    else:
        data_loader = validation_loader
    
    num_batch = 0
    loss_log = []
    # variance_log = []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, variance = test_model(inputs)
            loss = criterion(outputs, targets, variance)
            loss_log.append(loss.item())
            variance_log = torch.cat((variance_log, variance), 0)

            targets = targets.to(torch.float32)
            # outputs = outputs.softmax(dim=-1)
            outputs = torch.sigmoid(outputs)
            # outputs = (outputs > 0.5).float()

            # print(f'batch: {num_batch}, loss: {loss}\n {targets[0]}, \n, {outputs[0]}')
            num_batch += 1


            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)


        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        variance_log = variance_log.detach().cpu().numpy()
        # print(f'y_true.shape = {y_true.shape}')
        # print(y_true[15])
        # print(f'y_score.shape = {y_score.shape}')
        # print(y_score[15])
        metrics = evaluator.evaluate(y_score, y_true)
        print(metrics)
        print(f'Aleatoric Uncertainty: {np.mean(variance_log)}')
        print(f"--------------------------Confusion matrix------------------------------:")
        for i in range(y_true.shape[1]):
            y_true_label = y_true[:, i]
            y_pred_label = y_score[:, i]
            y_pred_label = (y_pred_label >= 0.5).astype('float')
            
            cm = confusion_matrix(y_true_label, y_pred_label)
            print(f"Confusion matrix for label {i}:")
            print(cm)

        if sensitive_group == 'val':
            import pandas as pd
            val_log_file = 'outputs/bnn_medvit_base/validation_log_medvit_base.csv'
            df = pd.DataFrame({
                'loss': np.mean(np.array(loss_log)),
                'variance':np.mean(np.array(variance_log))
            }, index=[0])
            if os.path.exists(val_log_file):
                df_prev = pd.read_csv(val_log_file)
                df = pd.concat([df_prev, df], axis=0, ignore_index=True)
            df.to_csv(val_log_file)

# train
def train():
    loss_log = []
    epoch_log = []
    epoch_loss = 0
    for epoch in range(NUM_EPOCHS):

        if (epoch+1) % 1 == 0:
            print("-----------------------------------------Validation-----------------------------------------")
            test(test_model=model, sensitive_group='val')
            print("-----------------------------------------Validation-----------------------------------------")
            print("-----------------------------------------Male Samples-----------------------------------------")
            test(test_model=model, sensitive_group='male')
            print("-----------------------------------------Male Samples-----------------------------------------")
            print("-----------------------------------------Female Samples-----------------------------------------")
            test(test_model=model, sensitive_group='female')
            print("-----------------------------------------Female Samples-----------------------------------------")


        variance_log = torch.tensor([]).to(device)
        epoch_loss = 0
        print('Epoch [%d/%d]'% (epoch+1, NUM_EPOCHS))
        model.train()
        num_batch = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs, variance = model(inputs)
            
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets, variance)
            epoch_loss += loss.item()
            
            loss_log.append(loss.item())
            epoch_log.append(epoch)
            variance_log = torch.cat((variance_log, torch.mean(variance).unsqueeze(0)), 0)
            # print(f'epoch {epoch}, batch {num_batch}: loss: {loss.item()}, variance: {torch.mean(variance)}')
            loss.backward()
            optimizer.step()
            num_batch += 1
        variance_log = variance_log.detach().cpu().numpy()
        print(f'Epoch {epoch}, loss: {epoch_loss/num_batch}')
        print(f'Epoch {epoch}, Aleatoric Uncertainty: {np.mean(variance_log)}')

        import pandas as pd
        df = pd.DataFrame({
            'epoch': epoch_log,
            'loss': loss_log,
            # 'variance': variance_log.flatten()
        })
        df.to_csv('outputs/bnn_medvit_base/train_log_medvit_base.csv')
        torch.save(model.state_dict(), f'outputs/bnn_medvit_base/medvit_mnist__base_wt{epoch}.pt')
        print('model saved')






# train()

        
# print('==> Evaluating ...')
# test('train')
# test('test')

