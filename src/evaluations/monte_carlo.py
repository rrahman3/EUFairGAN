from src.evaluations.evaluator import Evaluator
import numpy as np
import torch
from scipy.special import softmax

epsilon = 0.000001
class MonteCarloPrediction:
    def __init__(self, model, dataloader, N=10):
        self.model = model
        self.dataloader = dataloader
        self.N = N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_metrics = MonteCarloEvaluator()


    def asfsdgd(self):
        with torch.no_grad():
            for batch, (images, genders, y)  in enumerate(self.dataloader):
                self.model.train()
                images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {genders.shape}, {y.shape}')

                # Compute prediction and loss
                y_pred_N = np.empty((images.shape[0], self.N, y.shape[-1]))
                y_var_N = np.empty((images.shape[0], self.N, 1))
                for i in range(1, self.N+1):
                    # print(i)
                    y_pred, y_var = self.model(images, genders)
                    # print(y_pred)
                    y_pred = y_pred.detach().cpu().numpy()
                    y_pred = y_pred[:, np.newaxis, :]
                    # print(y_pred)
                    y_var = y_var.detach().cpu().numpy()
                    y_var = y_var[:, np.newaxis, :]

                    # Store the predictions and variances at the current iteration index (i-1)
                    y_pred_N[:, i - 1, :] = y_pred[:, 0, :]  # Using [0] to get rid of the added dimension
                    y_var_N[:, i - 1, :] = y_var[:, 0, :]
                    # print(y_pred.shape, y_var.shape)
                    # print(y_pred_N.shape, y_var_N.shape)
                    # print(len(y_pred_N), len(y_var_N))
                    # if i==1:
                    # print(y_pred_N[0])
                    # print(y_var_N)
                # y_pred_N = (batch_size, N, num_classes)

                y_pred_mean = np.mean(np.array(y_pred_N), axis=1) #(batch, num_classes)
                y_var_mean = np.mean(np.array(y_var_N), axis=1) #(batch, 1)
                # print(y_pred_mean.shape, y_var_mean.shape)
                # print(y_pred_mean[0])
                y_pred_softmax = softmax(y_pred_mean, axis=1)
                # print(y_pred_softmax.shape)
                # print(np.sum(y_pred_softmax, axis=1))
                epistemic_uncertainty = np.apply_along_axis(self.predictive_entropy, axis=1, arr=y_pred_softmax) #(batch_size, 1), epistemic uncertainty of each user
                aleatoric_uncertainty = y_var_mean #(batch, 1)
                # print(epistemic_uncertainty.shape, np.mean(epistemic_uncertainty), epistemic_uncertainty)
                # print(aleatoric_uncertainty.shape, np.mean(aleatoric_uncertainty), aleatoric_uncertainty)
                # print(prediction_variances.shape)

                self.evaluation_metrics.update_metrics(y_true=y, y_pred=y_pred_softmax)
                self.evaluation_metrics.update_variance(variance=aleatoric_uncertainty)
                self.evaluation_metrics.update_entropy(entropy=epistemic_uncertainty)

            epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
            self.evaluation_metrics.print_metrics()
            

    def predictive_entropy(self, prob):
        return -1 * np.sum(np.log(prob+epsilon) * (prob)) # entropy of class distribution
    

# evaluations/evaluator.py

import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class MonteCarloEvaluator:
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
        self.all_entropy = []

    def update_metrics(self, y_true, y_pred): # y_ture is the binary form of the y_true (None, 1, num_classes) # [[0, 0, 1, 0]] (1, 1, 4)

        _, y_pred_argmax = torch.max(y_pred, 1)
        _, y_true_argmax = torch.max(y_true, 1)

        self.all_labels_direct.extend(y_true_argmax.cpu().numpy())
        self.all_predictions_driect.extend(y_pred_argmax.cpu().numpy())

        self.all_labels_binary.extend(y_true.detach().cpu().numpy())
        self.all_predictions_probabilities.extend(y_pred.detach().cpu().numpy())
    
    def update_variance(self, variance):
        self.all_varience.extend(variance)

    def update_entropy(self, entropy):
        self.all_entropy.extend(entropy)

    def compute_epoch_metrics(self):
        """Combine batch metrics to get epoch metrics."""

        self.auc_scores = roc_auc_score(self.all_labels_binary, self.all_predictions_probabilities, multi_class='ovo')
        
        self.f1_scores = f1_score(self.all_labels_direct, self.all_predictions_driect, average='macro')
        
        self.accuracies = accuracy_score(self.all_labels_direct, self.all_predictions_driect)
        
        self.conf_matrices = confusion_matrix(self.all_labels_direct, self.all_predictions_driect)

        self.epoch_metrics = {
            'variance': np.mean(self.all_varience),
            'entropy': np.mean(self.all_entropy),
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
        print(f"Epistemic Uncertainty: {self.epoch_metrics['entropy']:.5f}", end='\n')
        print(f"Confusion Metrics\n{self.epoch_metrics['confusion_metrix']}", end="\n\n")


