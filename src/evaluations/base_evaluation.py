# evaluations/evaluator.py

import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class Evaluator:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics for a new epoch"""
        self.f1_scores = []
        self.auc_scores = []
        self.accuracies = []
        self.conf_matrices = []
        
        self.all_labels_direct = [] #contains the actual class labels , 1, 3, 4
        self.all_labels_binary = [] # contains in binary form [0, 0, 1, 0] for true class 2 #shpe=(None, 1, num_classes)
        self.all_predictions_probabilities = [] #contains all the classes probability
        self.all_predictions_driect = [] #contains the maxminum probaility class
        self.all_varience = []

    def update_metrics(self, y_true, y_pred, y_variance): # y_ture is the binary form of the y_true (None, 1, num_classes) # [[0, 0, 1, 0]] (1, 1, 4)

        # Get predictions
        pred_softmax = F.softmax(y_pred, dim=1)

        _, y_pred_argmax = torch.max(pred_softmax, 1)
        _, y_true_argmax = torch.max(y_true, 1)

        self.all_labels_direct.extend(y_true_argmax.cpu().numpy())
        self.all_predictions_driect.extend(y_pred_argmax.cpu().numpy())

        self.all_labels_binary.extend(y_true.detach().cpu().numpy())
        self.all_predictions_probabilities.extend(pred_softmax.detach().cpu().numpy())
        
        self.all_varience.extend(y_variance.detach().cpu().numpy())

        # epoch_variance += torch.mean(y_variance) * images.shape[0]

        # epoch_test_loss = epoch_test_loss / self.len
        # epoch_variance = epoch_variance / self.len
        # y_pred_argmax = y_pred_argmax.detach().cpu().numpy()
        # y_true_argmax = y_true_argmax.detach().cpu().numpy()

        # y_pred = y_pred.detach().cpu().numpy()
        # y_true = y_true.detach().cpu().numpy()



    def compute_epoch_metrics(self):
        """Combine batch metrics to get epoch metrics."""

        self.auc_scores = roc_auc_score(self.all_labels_binary, self.all_predictions_probabilities, multi_class='ovo')
        
        self.f1_scores = f1_score(self.all_labels_direct, self.all_predictions_driect, average='macro')
        
        self.accuracies = accuracy_score(self.all_labels_direct, self.all_predictions_driect)
        
        self.conf_matrices = confusion_matrix(self.all_labels_direct, self.all_predictions_driect)

        self.epoch_metrics = {
            'variance': np.mean(self.all_varience),
            'accuracy': self.accuracies,
            'f1_score': self.f1_scores,
            'auc': self.auc_scores,
            'confusion_metrix': self.conf_matrices
        }
        return self.epoch_metrics

    def print_metrics(self):
        print(f"F1 Score: {self.epoch_metrics['f1_score']:.4f}", end='\t')
        print(f"AUC: {self.epoch_metrics['auc']:.4f}", end='\t')
        print(f"Accuracy: {self.epoch_metrics['accuracy']:.4f}", end='\t')
        print(f"Aleatoric Uncertainty: {self.epoch_metrics['variance']:.5f}", end='\n')
        print(f"Confusion Metrics\n{self.epoch_metrics['confusion_metrix']}", end="\n\n")


